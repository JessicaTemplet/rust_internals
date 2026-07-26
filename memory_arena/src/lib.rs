//! A bump-allocating memory arena, built directly on `std::alloc` with no
//! external crates.
//!
//! The idea: instead of one heap allocation per value, the arena grabs a
//! large raw block of memory up front and hands out slices of it by just
//! moving a pointer forward ("bumping" it) on every `alloc` call. That
//! makes individual allocations extremely cheap (a few arithmetic ops,
//! no locking, no calling into the system allocator), at the cost of only
//! being able to free everything at once, when the arena itself drops or
//! is explicitly `reset`.
//!
//! This is the same technique crates like `bumpalo` and `typed-arena`
//! are built on. Reimplementing it here means being explicit about the
//! parts a safe `Vec`-of-`Box` never has to think about: manual layout
//! and alignment math, raw pointer bookkeeping, and running destructors
//! by hand since bump-allocated memory doesn't participate in Rust's
//! normal drop glue.

use std::alloc::{alloc, dealloc, handle_alloc_error, Layout};
use std::cell::{Cell, RefCell};
use std::ptr::NonNull;

const DEFAULT_CHUNK_SIZE: usize = 4096;

/// One contiguous block of raw heap memory owned by the arena, with a
/// bump pointer (`len`) tracking how much of it is in use.
struct Chunk {
    ptr: NonNull<u8>,
    layout: Layout,
    len: Cell<usize>,
}

impl Chunk {
    fn new(size: usize) -> Chunk {
        let layout = Layout::array::<u8>(size).expect("arena chunk size overflowed");
        // Safety: `layout` has a non-zero size (see `alloc_layout` below,
        // which never requests a zero-size chunk), which is what `alloc`
        // requires.
        let raw = unsafe { alloc(layout) };
        let ptr = match NonNull::new(raw) {
            Some(p) => p,
            None => handle_alloc_error(layout),
        };
        Chunk {
            ptr,
            layout,
            len: Cell::new(0),
        }
    }

    fn capacity(&self) -> usize {
        self.layout.size()
    }

    /// Tries to carve out room for `layout` from the unused tail of this
    /// chunk, respecting `layout`'s required alignment. Returns `None`
    /// if there isn't enough room left, without mutating any state.
    ///
    /// This does its arithmetic on the pointer itself (via `add` and
    /// `align_offset`) rather than round-tripping through `usize`, so it
    /// stays within Rust's strict-provenance rules instead of relying on
    /// the looser "exposed provenance" model that plain `as usize` /
    /// `as *mut u8` casts fall back on.
    fn try_alloc(&self, layout: Layout) -> Option<NonNull<u8>> {
        // Safety: `self.len.get()` never exceeds `self.capacity()` (that
        // invariant is maintained below), so this offset lands at or
        // before one-past-the-end of this chunk's single allocation,
        // which is exactly what `add` requires.
        let current = unsafe { self.ptr.as_ptr().add(self.len.get()) };

        let padding = current.align_offset(layout.align());
        if padding == usize::MAX {
            // align_offset's contract allows this sentinel when no
            // aligned offset could be computed; vanishingly unlikely for
            // ordinary heap pointers, but handled rather than assumed
            // away.
            return None;
        }

        let used = self.len.get().checked_add(padding)?.checked_add(layout.size())?;
        if used > self.capacity() {
            return None;
        }
        self.len.set(used);

        // Safety: `used <= self.capacity()`, so `current` advanced by
        // `padding` lands at or before one-past-the-end of this chunk's
        // allocation.
        let aligned = unsafe { current.add(padding) };
        NonNull::new(aligned)
    }
}

/// A destructor to run when the arena drops or resets, paired with the
/// pointer it needs to run on. Type-erased since the arena stores values
/// of many different types side by side in the same chunks.
type DropEntry = (NonNull<u8>, unsafe fn(*mut u8));

/// A growable bump-allocation arena.
///
/// `alloc` takes `&self`, not `&mut self` — allocation only needs to
/// mutate the arena's internal bookkeeping (via `Cell`/`RefCell`), not
/// the objects it has already handed out, so callers can hold many
/// simultaneous `&mut T` references into the arena as long as each
/// points to a distinct allocation. That's sound because every
/// allocation carves out a disjoint region of a chunk; two calls to
/// `alloc` can never return overlapping memory.
pub struct Arena {
    chunks: RefCell<Vec<Chunk>>,
    drops: RefCell<Vec<DropEntry>>,
}

impl Arena {
    pub fn new() -> Arena {
        Arena {
            chunks: RefCell::new(Vec::new()),
            drops: RefCell::new(Vec::new()),
        }
    }

    /// Moves `value` into the arena and returns a mutable reference to
    /// it, borrowed from the arena rather than owned by the caller.
    pub fn alloc<T>(&self, value: T) -> &mut T {
        let layout = Layout::new::<T>();
        let ptr = self.alloc_layout(layout).cast::<T>();

        // Safety: `ptr` is freshly carved out of a chunk, correctly
        // aligned and sized for `T` (guaranteed by `alloc_layout`), and
        // not yet initialized, so writing `value` into it is valid and
        // doesn't drop or alias anything.
        unsafe {
            ptr.as_ptr().write(value);
        }

        if std::mem::needs_drop::<T>() {
            // Safety of this cast: invoked only from `run_drops`, exactly
            // once per registered entry, on a pointer that still points
            // at a live, initialized `T` (nothing else in this module
            // ever frees or overwrites arena memory before that point).
            unsafe fn drop_glue<T>(ptr: *mut u8) {
                std::ptr::drop_in_place(ptr.cast::<T>());
            }
            self.drops.borrow_mut().push((ptr.cast::<u8>(), drop_glue::<T>));
        }

        // Safety: `ptr` is uniquely owned at this point (this is the
        // only reference to it in existence), and the returned
        // reference's lifetime is tied to `&self` by lifetime elision,
        // so the borrow checker won't allow `reset` (which needs
        // `&mut self`) to run while this reference, or any other
        // previously returned reference, is still alive.
        unsafe { &mut *ptr.as_ptr() }
    }

    fn alloc_layout(&self, layout: Layout) -> NonNull<u8> {
        if let Some(ptr) = self
            .chunks
            .borrow()
            .last()
            .and_then(|chunk| chunk.try_alloc(layout))
        {
            return ptr;
        }

        // The current chunk (if any) doesn't have room; grow. New chunks
        // are at least `DEFAULT_CHUNK_SIZE`, or double the requested
        // size for anything larger than that, so one oversized
        // allocation doesn't leave a chunk permanently undersized for
        // everything after it.
        let chunk_size = DEFAULT_CHUNK_SIZE.max(layout.size().saturating_mul(2));
        let chunk = Chunk::new(chunk_size);
        let ptr = chunk
            .try_alloc(layout)
            .expect("a freshly allocated chunk must fit the allocation it was sized for");
        self.chunks.borrow_mut().push(chunk);
        ptr
    }

    /// Runs every registered destructor and rewinds all chunks back to
    /// empty, so the arena's memory can be reused for new allocations
    /// without returning it to the system allocator. Requires `&mut
    /// self`: since every live reference returned by `alloc` borrows
    /// `&self`, the borrow checker refuses to call `reset` while any of
    /// them are still reachable, which is exactly what makes this safe.
    pub fn reset(&mut self) {
        self.run_drops();
        for chunk in self.chunks.borrow().iter() {
            chunk.len.set(0);
        }
    }

    /// Number of chunks currently backing this arena. Mostly useful for
    /// tests and diagnostics, to confirm growth is actually happening.
    pub fn chunk_count(&self) -> usize {
        self.chunks.borrow().len()
    }

    fn run_drops(&mut self) {
        // Reverse order mirrors the order a stack of local variables
        // would unwind in, which is a reasonable default when values
        // allocated later may (informally) depend on values allocated
        // earlier.
        for (ptr, drop_fn) in self.drops.borrow_mut().drain(..).rev() {
            // Safety: see the comment on `drop_glue` above.
            unsafe { drop_fn(ptr.as_ptr()) };
        }
    }
}

impl Default for Arena {
    fn default() -> Arena {
        Arena::new()
    }
}

impl Drop for Arena {
    fn drop(&mut self) {
        self.run_drops();
        for chunk in self.chunks.borrow_mut().drain(..) {
            // Safety: `chunk.layout` is exactly the layout that was
            // passed to `alloc` when this chunk's memory was obtained
            // (see `Chunk::new`), which is what `dealloc` requires.
            unsafe { dealloc(chunk.ptr.as_ptr(), chunk.layout) };
        }
    }
}

// Safety: an `Arena` owns its chunk allocations exclusively, and every
// reference `alloc` hands out borrows `&self` (or `&mut self`, for
// `reset`), so the borrow checker guarantees no such reference can
// outlive a move of the `Arena` itself. Moving the whole arena, chunks
// and all, to another thread is therefore sound.
unsafe impl Send for Arena {}

// Deliberately not `unsafe impl Sync`: the bump pointer in each `Chunk`
// is a plain `Cell<usize>`, which is not safe to mutate from two threads
// concurrently, and `Cell` already makes `Arena` `!Sync` automatically,
// so no explicit opt-out is required.


#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell as StdCell;
    use std::rc::Rc;

    #[test]
    fn stores_and_returns_correct_values() {
        let arena = Arena::new();
        let a = arena.alloc(41);
        let b = arena.alloc("hello");
        let c = arena.alloc(vec![1, 2, 3]);

        assert_eq!(*a, 41);
        assert_eq!(*b, "hello");
        assert_eq!(*c, vec![1, 2, 3]);
    }

    #[test]
    fn respects_type_alignment() {
        #[repr(align(64))]
        struct Aligned(u8);

        let arena = Arena::new();
        for i in 0..8u8 {
            let value = arena.alloc(Aligned(i));
            assert_eq!((value as *mut Aligned as usize) % 64, 0);
            assert_eq!(value.0, i);
        }
    }

    #[test]
    fn allocates_many_disjoint_values_across_chunk_growth() {
        let arena = Arena::new();
        let mut refs: Vec<&mut i32> = Vec::new();

        // DEFAULT_CHUNK_SIZE is 4096 bytes; 2000 i32s is ~8000 bytes, so
        // this forces at least one chunk growth.
        for i in 0..2000 {
            refs.push(arena.alloc(i));
        }

        for (i, r) in refs.iter().enumerate() {
            assert_eq!(**r, i as i32);
        }
        assert!(arena.chunk_count() > 1, "expected chunk growth, got {} chunk(s)", arena.chunk_count());
    }

    /// A value that records into a shared counter when dropped, so tests
    /// can confirm the arena actually runs destructors instead of just
    /// leaking bump-allocated memory.
    struct Tracked(Rc<StdCell<i32>>);

    impl Drop for Tracked {
        fn drop(&mut self) {
            self.0.set(self.0.get() + 1);
        }
    }

    #[test]
    fn drop_glue_runs_when_arena_drops() {
        let counter = Rc::new(StdCell::new(0));
        {
            let arena = Arena::new();
            for _ in 0..5 {
                arena.alloc(Tracked(counter.clone()));
            }
            assert_eq!(counter.get(), 0, "destructors shouldn't run before the arena drops");
        }
        assert_eq!(counter.get(), 5);
    }

    #[test]
    fn reset_runs_drops_and_allows_reuse() {
        let counter = Rc::new(StdCell::new(0));
        let mut arena = Arena::new();

        for _ in 0..3 {
            arena.alloc(Tracked(counter.clone()));
        }
        arena.reset();
        assert_eq!(counter.get(), 3);

        // The chunk memory should still be usable after a reset.
        let value = arena.alloc(99);
        assert_eq!(*value, 99);

        for _ in 0..2 {
            arena.alloc(Tracked(counter.clone()));
        }
        drop(arena);
        assert_eq!(counter.get(), 5);
    }
}

