//! Demonstrates the thread pool: typed results, panic recovery,
//! fire-and-forget jobs, and clean shutdown on drop.
use thread_pool::ThreadPool;

fn main() {
    let pool = ThreadPool::new(4);
    println!("started a pool with {} workers", pool.worker_count());

    let handles: Vec<_> = (1u64..=8).map(|n| pool.execute(move || n * n)).collect();
    let squares: Vec<u64> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    println!("squares: {squares:?}");

    let panicking = pool.execute(|| -> u32 { panic!("deliberate failure for the demo") });
    match panicking.join() {
        Ok(_) => unreachable!(),
        Err(e) => println!("as expected, that job failed: {e}"),
    }

    let recovered = pool.execute(|| 2 + 2).join().unwrap();
    println!("pool still works after a panic: 2 + 2 = {recovered}");

    for i in 0..5 {
        pool.execute(move || println!("fire-and-forget job {i} running"));
    }
    pool.join();
    println!("all fire-and-forget jobs finished");
}
