#!/usr/bin/env python
# coding: utf-8

# # Signals, Interrupts, and Docker
# 
# ## Signals
# 
# * Signals are how the Linux kernel and/or processes tell processes to do things.
# * Every signal has a default action. The process may choose to overwrite that action by defining a **signal handler**, except in the case of `SIGSTOP` (which stops a process) and `SIGKILL` (which kills a process immediately at the kernel level).
# * The two most important signals are `SIGTERM` and `SIGKILL`. `SIGTERM` is a polite shutdown command. `SIGKILL` is a hard stop shutdown command. For example on AWS SageMaker, container processes will recieve a `SIGTERM` first, then a `SIGKILL` 30 seconds later (by default, configurable, in some context I don't remember the exact details of).
# * There is a raft of different types of signals, most of which are pretty context-specific.
# 
# ## Interrupts
# 
# * An interrupt is to hardware what a signal is to the operating system. It occurs when the hardware wants to register an event with the operating system. The operating system must immediately suspend execution and deal with the interrupt.
# * An example of an interrupt is a keyboard interrupt, e.g. triggered by `Ctrl + Z` or `Ctrl + C`. Linux bubbles this up as a `SIGSTOP` and `SIGINTERRUPT`, respectively. Something similar happens in some other interrupt cases.
# * Hardware interrupts used to be used for keyboard input and mouse movement and the like as well, however these days signal polling at the level of the software driver in the operating system is used instead.
# 
# ## Docker
# 
# * Finally we get to Docker, and some key points about signals that are necessary to keep in mind when dealing with Docker.
# * This section referenced from: https://engineeringblog.yelp.com/2016/01/dumb-init-an-init-for-docker.html and https://hynek.me/articles/docker-signals/.
# 
# 
# * Docker has two forms of syntax for commands, a "shell form" which is convenient to write and gets run via `bin/sh -c`, and an "exec form" that is JSON array structured (with weird syntactical nuisances, in my experience) but which is executed directly.
# * Every command that is run in a shell is run via `fork` by default, meaning that it gets spawned in a new subprocess. This results in the following process tree:
# 
# ```
# docker run (on the host machine)
#     /bin/sh (PID 1, inside container)
#         python my_server.py (PID ~2, inside container)
# ```
# 
# * If you use the exec syntax instead, you recieve the following process tree:
# 
# ```
# docker run (on the host machine)
#     python my_server.py (PID 1, inside container)
# ```
# 
# * PID 1 has special significance in Linux; it is known as the "init" process and has certain special responsibilities, such as adopting orphans and reaping zombies.
# * On a regular machine this will be some sort of `init` process, but Docker doesn't have any `init` process overhead, and instead sends your process straight to PID 1.
# * Linux has a special rule for PID 1. Typically, processes that recieve a `SIGTERM` and do not have a signal handler defined will fall back to exiting immediately. PID 1 processes that recieve a `SIGTERM` and do not have a signal handler defined will instead do nothing...the signal will bounce right off the process.
# * When Docker recieves `SIGTERM`, it forwards the signal to the container and then exits immediately. It does not wait for the container to die, it just assumes it will.
# * The shell will queue signals, and ignore them until after the currently executing child process is finished running.
# 
# 
# * Knowing all this, what is the behavior vis-a-vis signals in these two cases?
# * In the shell form, the PID 1 process is `/bin/sh`, which has a `SIGTERM` signal handler defined but queues it unti the child process exits. Since the `my_server.py` process is a long-running process, this never happens, so `SIGTERM` signals (and all other signals really) are effectively ignored.
# * In the exec form, the PID 1 process is `python my_server.py`. Trappable signals that are not `SIGTERM` are processed as normal, but assuming you do not define a `SIGTERM` handler, due to the special-casing of PID 1, `SIGTERM` signals are ignored.
# 
# 
# * How does Docker stop containers? How does that interact with these two scenarios?
# * When `docker run` receives `SIGTERM` (e.g. via ), it forwards the signal to the container and then exits, even if the container itself never dies.
# * When `docker stop` executes, it sends `SIGTERM`, waits ten seconds, then sends `SIGKILL` if it hasn't stopped.
# * So if (as in these default cases) your container isn't responding to `SIGTERM`, your container can only be stopped via `SIGKILL` via `docker stop`.
# 
# 
# * The solution is many-fold.
# * First of all, never use the shell form, ever.
# * The PID 1 special-casing is basically a design error on Docker's part. Docker now supports running `tini`, a proper initialization process, in PID 1, by passing the `--init` flag to `docker run`. That results in the following process tree (assuming you follow the other advice):
# 
#     ```
#     docker run (on the host machine)
#         tini (PID 1, inside container)
#             python my_server.py (PID ~2, inside container)
#     ```
# 
# * `tini` has a `SIGTERM` signal handler. It propogates the signal to the process group of its children, then exits itself once they are done exiting. This is the behavior you want!
# * If you do not have support for the `--init` flag in your environment, you can also install it into the container yourself, then use it as your `ENTRYPOINT`: `ENTRYPOINT ["/tini", "-v", "--", "/app/bin/docker-entrypoint.sh"]`.
# 
# 
# * There are other cases that you need to keep in mind:
# 
# 
# * The problem re-occurs if you run a shell script in exec form (e.g. `RUN ["/bin/sh", "-c", "run.sh"]`). Shell scripts have a specific command for this, called `exec` (surprise surprise), which causes the shell to replace its own process with the process you prefix with `exec` (e.g. `exec python app.py`). Obviously this will only work for the very last line of the script, but that's where your long-running process is going to be right?
# * If you start a subshell in your shell, you will once again lose the ability to receive signals. A lot of different pieces of syntactic sugar launch subshells. The obvious command is `($CMD)`, which executes a command in a subshell. `$(CMD)`, which pipes the result of running `CMD` in a subshell to the current command, is similar. The pipe trick `|` also uses subshells...and so on.
# * So when using shell commands in a Docker script, try to avoid syntactic sugar for long-running bits of process where interuptability is important.
# * The full list for Bash is: https://unix.stackexchange.com/questions/442692/is-a-subshell
# 
# 
# * Recall that keyboard interrupts are `SIGINT` events, not `SIGTERM` events. If `docker run` in the foreground, then `Ctrl + C`, that sends `SIGINT` to the container. The default action for keyboard interrupts is process-dependent; Python will raise them as an error and exit out; you can define your own cleanup code for this case by wrapping your code in an `except KeyboardInterrupt` block.
# 
# 
# * Interesting Docker bonus feature. You can set e.g. `STOPSIGNAL SIGINT` in your Dockerfile to change what command Docker will send when you run `docker stop`.
