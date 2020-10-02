#!/usr/bin/env python
# coding: utf-8

# ### Multithreading and multiprocessing APIs in Python
# 
# * In a previous notebook I discussed the core ideas of multithreading, multiprocessing, green threads, and asynchronous programming: ["Notes on asynchronous programming"](https://www.kaggle.com/residentmario/notes-on-asynchronous-programming). This notebook is a refrence specifically to the Python API for multithreading and multiprocessing.
# 
# ## Multithreading
# 
# This section based on the following Real Python tutorial: https://realpython.com/intro-to-python-threading/.

# In[ ]:


import logging
import threading
import time

def thread_function(name):
    logging.info("Thread %s: starting", name)
    time.sleep(2)
    logging.info("Thread %s: finishing", name)

format = "%(asctime)s: %(message)s"
logging.basicConfig(
    format=format, level=logging.INFO, datefmt="%H:%M:%S"
)

print("Main: before creating thread")
x = threading.Thread(target=thread_function, args=(1,))
print("Main: before running thread")
x.start()
print("Main: wait for the thread to finish")
x.join()
print("Main: all done")


# In[ ]:


x = threading.Thread(target=thread_function, args=(1,))
x.start()


# * Threads wrap functions. `thread.start` causes a thread to start executing a function, whilst `join` blocks until the thread is done executing.
# * Python's default programming model has the main thread wait for all spawned threads to finish executing before exiting. It is possible to not wait on a thread to finish by setting the `daemon` flag instead.
# * The simplest way to manage concurrency in threads is to `start` a list of threads and then `join` them sequentially. This is fine, but there is a built-in that does this for you: `concurrent.futures.ThreadPoolExecutor`. That looks thus:
# 
#     ```python
#     with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
#         executor.map(thread_function, range(3))
#     ```
# 
# * There is some confusing aspects of how this API works and is run. In particular, errors in the function signature of `thread_function` will be swallowed without an error being raised.
# * The GIL held by a thread will always be released during I/O operations, `time.sleep`, and certain computationally intensive operations in e.g. `numpy` (there might be other operations but these are the two that I'm aware of). These are scenarios in which the thread gives the GIL up voluntarily. In other cases, the thread manager will release the GIL, and a new thread will apply it; these are free to occur at any time step in the code.
# so these are the occassions during which race conditions are possible.
# * Data is **thread-safe** if:
#   * It is data local to the thread, e.g. it is not shared memory.
#   * It is shared memory but the data structure itself is thread-safe. For example, a queue, or an append-only data structure with no total ordering.
# * To avoid race conditions during non thread-safe segments, take a lock with `threading.Lock()`. You can acquire and release this lock using a context manager or using `l.acquire()` and then `l.release()`. It has a `blocking=True` argument and a `timeout`. Using a non-blocking lock will cause the lock to return `False` if the lock cannot be acquired (because it is held by another thread), allowing you to do the required logic to avoid deadlocks yourself.
# * That gets us to the next, more complex example:

# In[ ]:


import random 
SENTINEL = object()

class Pipeline:
    """
    Class to allow a single element pipeline between producer and consumer.
    """
    def __init__(self):
        self.message = 0
        self.producer_lock = threading.Lock()
        self.consumer_lock = threading.Lock()
        self.consumer_lock.acquire()

    def get_message(self, name):
        logging.debug("%s:about to acquire getlock", name)
        self.consumer_lock.acquire()
        logging.debug("%s:have getlock", name)
        message = self.message
        logging.debug("%s:about to release setlock", name)
        self.producer_lock.release()
        logging.debug("%s:setlock released", name)
        return message

    def set_message(self, message, name):
        logging.debug("%s:about to acquire setlock", name)
        self.producer_lock.acquire()
        logging.debug("%s:have setlock", name)
        self.message = message
        logging.debug("%s:about to release getlock", name)
        self.consumer_lock.release()
        logging.debug("%s:getlock released", name)

def producer(pipeline):
    """Pretend we're getting a message from the network."""
    for index in range(10):
        message = random.randint(1, 101)
        logging.info("Producer got message: %s", message)
        pipeline.set_message(message, "Producer")

    # Send a sentinel message to tell consumer we're done
    pipeline.set_message(SENTINEL, "Producer")

def consumer(pipeline):
    """Pretend we're saving a number in the database."""
    message = 0
    while message is not SENTINEL:
        message = pipeline.get_message("Consumer")
        if message is not SENTINEL:
            logging.info("Consumer storing message: %s", message)

format = "%(asctime)s: %(message)s"
logging.basicConfig(format=format, level=logging.INFO,
                    datefmt="%H:%M:%S")

pipeline = Pipeline()
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
    executor.submit(producer, pipeline)
    executor.submit(consumer, pipeline)


# * In this code sample the `pipeline` object represents all of the shared state: a queue, effectively, with a `SENTINEL` value representing end-of-queue. A thread having to wait for a lock allows a context switch to another thread, and this is used to the benefit of this program (which has two threads, a producer and a consumer) by having one thread execute a function which acquires the lock it needs, do some work, release the lock needed by its compatriot, and exit.
#   
#   The now-exited function never released the lock that the next iteration of that function needs to execute, so the next iteration of the function blocks, yielding executing to its companion. The companion function does some work and reverses the locks, blocks itself, releases execution to the previous function, which is now unblocked, and so on and so forth until the entire queue is consumed.
#   
#   This system of dueling locks is an academically interesting example. In real cases you would use a queue, which is the next example.

# In[ ]:


import concurrent.futures
import logging
import queue
import random
import threading
import time

def producer(queue, event):
    """Pretend we're getting a number from the network."""
    while not event.is_set():
        message = random.randint(1, 101)
        logging.info("Producer got message: %s", message)
        queue.put(message)

    logging.info("Producer received event. Exiting")

def consumer(queue, event):
    """Pretend we're saving a number in the database."""
    while not event.is_set() or not queue.empty():
        message = queue.get()
        logging.info(
            "Consumer storing message: %s (size=%d)", message, queue.qsize()
        )

    logging.info("Consumer received event. Exiting")

format = "%(asctime)s: %(message)s"
logging.basicConfig(format=format, level=logging.INFO,
                    datefmt="%H:%M:%S")

pipeline = queue.Queue(maxsize=10)
event = threading.Event()
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
    executor.submit(producer, pipeline, event)
    executor.submit(consumer, pipeline, event)

    time.sleep(0.1)
    logging.info("Main: about to set event")
    event.set()


# `Queue` is a thread-safe object which may freely be shared between threads (as here; `enqueue` and `dequeue` are locked operations, presumably?). The main thing introduced in this example which is new to me is the multithreading `Event` API, which allows you to share a once-only-incremented boolean flag between threads. This is used here to shut down the producer operations in the main thread, and to prevent the consumer from stopping too soon (if it outstrips the producer due to scheduling and empties out the queue).
# 
# 
# * Here are a few other useful constructs in the Python threading module:
#   * **Semaphore** &mdash; an atomic counter. A semaphore can be aquired as many times as the value it is initialized, after which further attempts at acquisition will block.
#   * **Timer** &mdash; Best-effort function scheduler.
#   * **Barrier** &mdash; A construct that will block when `wait()` is called on it until a certain threshold number of threads have called `wait()`. At that point all of the threads will be released at once.
#   * `threading.local` &mdash; Marks a variable in the global scope copy-on-init for all threads. The variable is now global in that thread, but each thread has its own copy of it.

# ## Multiprocessing
# 
# This section based on the following Real Python tutorial: https://realpython.com/python-concurrency/.
# 
# * The multiprocessing API is a near-copy of the thread API in many ways, but with some differences and expansions to the API.
# * In multithreading the main thread creates all of the shared state that the child threads work with. In multiprocessing this connection is less direct; processes do not share memory (at least, not inside of the Python process) and must incur IPC costs (e.g. serializing and deserializing objects, and transfering them to one another over sockets) in order to communicate with one another. This, plus the heavyweight nature of processes (versus more lightweight threads), is the chief reason disadvantage of multiprocessing.
# 
#   The chief advantage of multiprocessing is freedom from the GIL (so true parallelization, as opposed to mere concurrency, is possible).
# 
# * In the multiprocessing API the `Thread` analogue is `Process`. A trivial example of a multiprocess program is:
# 
#     ```python
#     from multiprocessing import Process
# 
#     def f(name):
#         print('hello', name)
# 
#     p = Process(target=f, args=('bob',))
#     p.start()
#     p.join()
#     ```
# 
# * Here's an example of a slightly less trivial multiprocess program:
# 
# ```python
# import multiprocessing
# import time
# 
# session = None
# 
# 
# def set_global_session():
#     global session
#     if not session:
#         session = requests.Session()
# 
# 
# def download_site(url):
#     with session.get(url) as response:
#         name = multiprocessing.current_process().name
#         print(f"{name}:Read {len(response.content)} from {url}")
# 
# 
# def download_all_sites(sites):
#     with multiprocessing.Pool(initializer=set_global_session) as pool:
#         pool.map(download_site, sites)
# 
# 
# if __name__ == "__main__":
#     sites = [
#         "https://www.jython.org",
#         "http://olympus.realpython.org/dice",
#     ] * 80
#     start_time = time.time()
#     download_all_sites(sites)
#     duration = time.time() - start_time
#     print(f"Downloaded {len(sites)} in {duration} seconds")
# ```
# 
# * Notice the use of `initializer`, which is passed a function which performs some operations when the process is initialized. The initializer function initializes a `global session` object, which is a `requests.Session`, which is just a TCP connection reuse mechanic. However we have to use the `global` keyword here so that the session object created is global in scope with respect to the process; otherwise the `session` object would be limited to its declarative functional scope, which would render it invisible to our workhorse function, `download_site`.
# 
# 
# * Since we're in process land now, we have to pay attention to how the process is started. Multiprocessing supports three different start methods, which may be toggled using `mp.set_start_method()`:
#     * `'spawn'`&mdash;The parent process starts a new child process. The child process is given copies of the subset of of the parent's state which is deemed relevant to its execution: objects and variables yes, file descriptors no.
#     * `'fork'`&mdash;The parent Python process starts a new child Python process which is a full copy of the parent process. Fork, unlike spawn, is a UNIX semantic: forking is the original (and, historically, only) way of starting a child process in UNIX (you would then use exec to switch to a different process in-place). Because of this forking is a UNIX syscall and (thanks to the copy-on-write optimization) fast. However there are cases where this operation is unsafe (for the reason why see text later in this notebook).
#     * `'forkserver'`&mdash;An interesting intermediate between the two. A single-threaded server process is started, and that server is used as a surrogate process for spawning child processes.
# * At the present time:
#   * Fork is faster than spawn.
#   * Spawn is unsafe on macOS.
#   * The default mode on macOS is spawn, but forkserver is recommended.
#   * Windows only supports spawn.
# 
# * There are two high-level primatives for IPC: the `mp.Queue`, and the `mp.Pipe`. The latter is essentially a two-sided queue, and is multiprocess-safe so long as processes communicating over a pipe do so by working on different ends of it.
# * Shared state can to be created explicitly by wrapping a value in a `multiprocessing.Value` or an array in `multiprocessing.Array`. This will set that data as shared memory. These have a `lock` attr, set to `True` by default, which is used to synchronize access (`with counter.get_lock(): counter += 1`); the lock may be disabled, which will render the shared state racey (but this may be fine for certain use cases).
# * An alternative way of creating and using shared state is with a server process. `mp.Manager` is a context manager that lets you create shared memory which is managed by an independent server process. This server process is single-threaded, so it serializes all operations made on memory it manages, allowing safe operations on shared memory objects without locking. Example utilization included in the Python docs:
# 
# ```python
# from multiprocessing import Process, Manager
# 
# def f(d, l):
#     d[1] = '1'
#     d['2'] = 2
#     d[0.25] = None
#     l.reverse()
# 
# if __name__ == '__main__':
#     with Manager() as manager:
#         d = manager.dict()
#         l = manager.list(range(10))
# 
#         p = Process(target=f, args=(d, l))
#         p.start()
#         p.join()
# 
#         print(d)
#         print(l)
# ```
# 
# This prints:
# 
# ```python
# {0.25: None, 1: '1', '2': 2}
# [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
# ```
# 
# * The last important primitive is the `Pool`. This is much like the thread pool. Most multiprocessing programs are written utilizing pools and queues.
# 
# ## Why is fork unsafe on macOS
# * As mentioned above, the fork mode in `multiprocessing` is unsafe on macOS. Why? [This article explains](https://blog.phusion.nl/2017/10/13/why-ruby-app-servers-break-on-macos-high-sierra-and-what-can-be-done-about-it/).
# * The most important point is that **forking a process immediately kills all threads except for the one doing the forking**. Because this is done at the OS level, other threads may be in inconsistent states at the time at which this occurs. If they are holding a global lock, that lock will not be released before the thread exits. Any child that attempts to access the resource protected by that lock will deadlock.
# 
#   In Python's case, the following resources can cause this problem:
#   * File descriptors, which may take an OS-level write lock.
#   * `threading.lock` objects.
# 
#   The article cites the following generic example:
#   
#   > An application spawns a thread X. This thread is allocating a memory buffer. The memory allocator locks a global mutex.While thread X is allocating a memory buffer and also has a lock on the global mutex, the application's main thread forks. The child process also allocates some memory. But the memory allocation mutex is locked, and X is gone, so it will never be unlocked. The child process deadlocks forever.
# 
# * In the High Sierra release Apple started to enforce a rule that processes may not initialize a new Objective-C API inside of a forked process. This has raised many errors that users were making with said locks from silent to loud, which is good.
# * However it also invalidates one of *the* most common multiprocess application design patterns: **preforking**. Preforking is the design pattern of having a parent process, on startup, preemtively fork a bunch of processes before any work comes in. The child processes then do all of the actual work of the application; the parent process merely routes data as necessary. With this change, any APIs that rely on Objective-C APIs but initializes them after the fork will immediately fail. Services that rely on doing this (usually by calling through the dependency chain on something that requires Objective-C) no longer work.
# * Ruby ran into this problem and added a patch that initializes Objective-C on init. Python did not due this, and instead made using spawn the best practice.
# 
# 
# ## Mixing the two styles
# * Multithreading and multiprocessing can be combined in one program. This makes correctness even harder to achieve. You *must not* spawn any threads in any processes until the processes are done initializing. Otherwise see the above. Once all of the forks have occurred, *then* you can start spawning threads.
