# rust_internals

Reference implementations of two fundamental systems programming building
blocks, written from scratch against the Rust standard library with no
external dependencies. The goal is to make the underlying machinery
explicit -- the parts that crates like `bumpalo` or `rayon` abstract away.

Both crates are heavily commented and come with thorough test suites.

---

## Crates

### `memory_arena`

A bump-allocating memory arena built directly on `std::alloc`.

Instead of one heap allocation per value, the arena grabs a large raw
block of memory up front and hands out slices of it by advancing a pointer
forward on every `alloc` call. Individual allocations cost a few arithmetic
ops with no locking and no system allocator call. The tradeoff is that
memory can only be freed all at once, when the arena resets or drops.

This is the same technique `bumpalo` and `typed-arena` are built on.
Implementing it here means being explicit about what those crates handle
for you: manual `Layout` and alignment math via `align_offset` on raw
pointers (staying within Rust's strict-provenance rules), type-erased drop
glue registered per allocation so destructors still run correctly, and the
`&self` vs `&mut self` split that lets the borrow checker enforce safety
around `reset`.

```rust
let arena = Arena::new();
let x = arena.alloc(42);
let s = arena.alloc(String::from("hello"));
// x and s are &mut references valid for the lifetime of the arena
```

### `thread_pool`

A fixed-size thread pool built on `std::sync` and `std::thread`.

Three primitives do all the coordination: a `Mutex`-guarded `mpsc::Receiver`
as the shared job queue (so multiple workers can safely pull from one
channel), a `Condvar` paired with a job counter for `join()`, and a
per-job one-shot channel so callers can get typed results back out.

Jobs that panic are caught via `catch_unwind` and logged; the worker stays
alive and picks up the next job. Dropping the pool sends a shutdown token
per worker after all queued jobs, so no work is abandoned on drop.

```rust
let pool = ThreadPool::new(4);
let handle = pool.execute(|| expensive_computation());
let result = handle.join().unwrap();
```

---

## Running the examples

```sh
cargo run --example basic -p memory_arena
cargo run --example basic -p thread_pool
```

## Running the tests

```sh
cargo test
```

Both crates run their full test suites entirely offline with no setup.