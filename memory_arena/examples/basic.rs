//! Demonstrates the arena allocating a singly linked list -- the kind of
//! pointer-heavy structure that normally requires Rc or RefCell, but here
//! uses plain borrowed references because everything lives in the arena.
use memory_arena::Arena;

struct Node<'a> {
    value: i32,
    next: Option<&'a Node<'a>>,
}

fn main() {
    let arena = Arena::new();

    let mut head: Option<&Node> = None;
    for value in (1..=5).rev() {
        head = Some(arena.alloc(Node { value, next: head }));
    }

    print!("list:");
    let mut cursor = head;
    while let Some(node) = cursor {
        print!(" {}", node.value);
        cursor = node.next;
    }
    println!();

    println!("chunks used: {}", arena.chunk_count());
}