//! A generic thread pool built entirely on the standard library: no
//! `rayon`, no `tokio`, no external crates.
//!
//! Three synchronization primitives do all the coordination work:
//!   - a `Mutex`-guarded `mpsc::Receiver` as the shared job queue, so
//!     every worker thread can pull the next job off the same channel
//!   - a `Condvar` paired with a `Mutex<usize>` job counter, so callers
//!     can block on `join()` until every submitted job has finished
//!   - a fresh one-shot `mpsc::channel` per submitted job, so
//!     `execute`'s caller can get a typed result back out
//!
//! Unlike the memory arena in this same folder, none of this needs
//! `unsafe`: threads, mutexes, and channels are all safe APIs. The
//! "low-level" part here is building the pool abstraction itself, by
//! hand, out of those pieces, rather than being unsafe in the Rust
//! sense.

use std::panic::{self, AssertUnwindSafe};
use std::sync::{mpsc, Arc, Condvar, Mutex};
use std::thread;

type Job = Box<dyn FnOnce() + Send + 'static>;

enum Message {
    NewJob(Job),
    Shutdown,
}

struct SharedState {
    active_jobs: Mutex<usize>,
    all_done: Condvar,
}

struct Worker {
    thread: Option<thread::JoinHandle<()>>,
}

impl Worker {
    fn spawn(id: usize, queue: Arc<Mutex<mpsc::Receiver<Message>>>, shared: Arc<SharedState>) -> Worker {
        let thread = thread::Builder::new()
            .name(format!("worker-{id}"))
            .spawn(move || Worker::run(id, &queue, &shared))
            .expect("failed to spawn worker thread");
        Worker { thread: Some(thread) }
    }

    fn run(id: usize, queue: &Mutex<mpsc::Receiver<Message>>, shared: &SharedState) {
        loop {
            // The lock is only held long enough to dequeue one message;
            // `recv()` blocks while holding it, which is what lets
            // multiple workers safely share one `Receiver`, but the job
            // itself runs after the guard is dropped so a slow job never
            // stops other workers from picking up the next one.
            let message = {
                let queue = queue.lock().expect("job queue mutex poisoned");
                queue.recv()
            };

            let job = match message {
                Ok(Message::NewJob(job)) => job,
                Ok(Message::Shutdown) | Err(_) => break,
            };

            // A job that panics shouldn't take a whole worker thread down
            // with it; catch the unwind, log it, and keep looping.
            if let Err(payload) = panic::catch_unwind(AssertUnwindSafe(job)) {
                eprintln!("worker {id}: job panicked: {}", panic_message(&payload));
            }

            let mut count = shared.active_jobs.lock().expect("active job counter mutex poisoned");
            *count -= 1;
            if *count == 0 {
                shared.all_done.notify_all();
            }
        }
    }
}

fn panic_message(payload: &Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "non-string panic payload".to_string()
    }
}

/// A fixed-size pool of worker threads that run submitted jobs.
///
/// `execute` and `join` both take `&self`, so a pool is typically shared
/// across threads behind an `Arc<ThreadPool>`. The job sender is kept
/// behind a `Mutex` specifically to make that sound: `mpsc::Sender`
/// implements `Send` but not `Sync`, so touching it through a bare
/// shared reference from multiple threads at once is not something the
/// compiler would otherwise allow.
pub struct ThreadPool {
    workers: Vec<Worker>,
    sender: Mutex<mpsc::Sender<Message>>,
    shared: Arc<SharedState>,
}

impl ThreadPool {
    /// Creates a pool with `size` worker threads. Panics if `size` is
    /// zero, since a pool with no workers can never make progress on
    /// anything submitted to it.
    pub fn new(size: usize) -> ThreadPool {
        assert!(size > 0, "thread pool size must be at least 1");

        let (sender, receiver) = mpsc::channel::<Message>();
        let receiver = Arc::new(Mutex::new(receiver));
        let shared = Arc::new(SharedState {
            active_jobs: Mutex::new(0),
            all_done: Condvar::new(),
        });

        let workers = (0..size)
            .map(|id| Worker::spawn(id, Arc::clone(&receiver), Arc::clone(&shared)))
            .collect();

        ThreadPool {
            workers,
            sender: Mutex::new(sender),
            shared,
        }
    }

    /// Submits a job to the pool and returns a handle that can be used
    /// to wait for and collect its result.
    pub fn execute<F, T>(&self, job: F) -> JobHandle<T>
    where
        F: FnOnce() -> T + Send + 'static,
        T: Send + 'static,
    {
        let (result_tx, result_rx) = mpsc::channel::<T>();

        {
            let mut count = self.shared.active_jobs.lock().expect("active job counter mutex poisoned");
            *count += 1;
        }

        let wrapped: Job = Box::new(move || {
            let result = job();
            // If the caller dropped the JobHandle, `result_rx` is gone
            // and this send fails; the result is simply discarded, same
            // as dropping a `JoinHandle` without joining it.
            let _ = result_tx.send(result);
        });

        self.sender
            .lock()
            .expect("job sender mutex poisoned")
            .send(Message::NewJob(wrapped))
            .expect("all worker threads have already shut down");

        JobHandle { receiver: result_rx }
    }

    /// Blocks the calling thread until every job submitted so far has
    /// finished running (whether it returned normally or panicked).
    /// Jobs submitted concurrently from another thread while `join` is
    /// waiting may or may not be waited for too.
    pub fn join(&self) {
        let mut count = self.shared.active_jobs.lock().expect("active job counter mutex poisoned");
        while *count != 0 {
            count = self.shared.all_done.wait(count).expect("condvar wait poisoned");
        }
    }

    pub fn worker_count(&self) -> usize {
        self.workers.len()
    }
}

impl Drop for ThreadPool {
    fn drop(&mut self) {
        // Send one Shutdown token per worker. Because the job queue is a
        // FIFO channel, these land after every job already submitted via
        // `execute`, so workers drain all real work first and only see
        // Shutdown once they'd otherwise go idle - dropping the pool
        // does not abandon queued jobs.
        if let Ok(sender) = self.sender.lock() {
            for _ in &self.workers {
                let _ = sender.send(Message::Shutdown);
            }
        }

        for worker in &mut self.workers {
            if let Some(thread) = worker.thread.take() {
                let _ = thread.join();
            }
        }
    }
}

/// A handle to a job's eventual result, returned by `ThreadPool::execute`.
pub struct JobHandle<T> {
    receiver: mpsc::Receiver<T>,
}

/// Returned by `JobHandle::join` when the job panicked instead of
/// returning a value.
#[derive(Debug)]
pub struct JobPanicked;

impl std::fmt::Display for JobPanicked {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "the worker running this job panicked before producing a result")
    }
}

impl std::error::Error for JobPanicked {}

impl<T> JobHandle<T> {
    /// Blocks until the job finishes, returning its result, or
    /// `Err(JobPanicked)` if the job panicked instead of completing
    /// normally.
    pub fn join(self) -> Result<T, JobPanicked> {
        self.receiver.recv().map_err(|_| JobPanicked)
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex as StdMutex;
    use std::time::Duration;

    #[test]
    fn executes_and_returns_result() {
        let pool = ThreadPool::new(2);
        let handle = pool.execute(|| 2 + 2);
        assert_eq!(handle.join().unwrap(), 4);
    }

    #[test]
    fn runs_many_jobs_and_join_waits_for_all_of_them() {
        let pool = ThreadPool::new(4);
        let counter = Arc::new(AtomicUsize::new(0));

        for _ in 0..200 {
            let counter = Arc::clone(&counter);
            pool.execute(move || {
                counter.fetch_add(1, Ordering::SeqCst);
            });
        }

        pool.join();
        assert_eq!(counter.load(Ordering::SeqCst), 200);
    }

    #[test]
    fn job_panic_is_reported_and_pool_keeps_working() {
        let pool = ThreadPool::new(2);

        let failed = pool.execute(|| -> i32 { panic!("boom") });
        assert!(failed.join().is_err());

        // The pool should still be usable after a panicking job.
        let ok = pool.execute(|| 41 + 1);
        assert_eq!(ok.join().unwrap(), 42);
    }

    #[test]
    fn join_blocks_until_slow_jobs_finish() {
        let pool = ThreadPool::new(3);
        let results = Arc::new(StdMutex::new(Vec::new()));

        for i in 0..6 {
            let results = Arc::clone(&results);
            pool.execute(move || {
                thread::sleep(Duration::from_millis(10));
                results.lock().unwrap().push(i);
            });
        }

        pool.join();
        assert_eq!(results.lock().unwrap().len(), 6);
    }

    #[test]
    fn dropping_the_pool_waits_for_queued_jobs() {
        let results = Arc::new(StdMutex::new(Vec::new()));
        {
            let pool = ThreadPool::new(2);
            for i in 0..10 {
                let results = Arc::clone(&results);
                pool.execute(move || {
                    results.lock().unwrap().push(i);
                });
            }
            // `pool` drops here and should block until all 10 jobs, not
            // just whichever ones happened to start first, have run.
        }
        assert_eq!(results.lock().unwrap().len(), 10);
    }

    #[test]
    fn pool_can_be_shared_across_threads_via_arc() {
        let pool = Arc::new(ThreadPool::new(4));
        let mut outer_handles = Vec::new();

        for t in 0..4 {
            let pool = Arc::clone(&pool);
            outer_handles.push(thread::spawn(move || {
                let job = pool.execute(move || t * 10);
                job.join().unwrap()
            }));
        }

        let mut results: Vec<u64> = outer_handles.into_iter().map(|h| h.join().unwrap()).collect();
        results.sort_unstable();
        assert_eq!(results, vec![0, 10, 20, 30]);
    }
}


