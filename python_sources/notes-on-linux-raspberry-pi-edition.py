#!/usr/bin/env python
# coding: utf-8

# ## Filesystem
# 
# * Linux has an official filesystem standard, the [Filesystem Hierarchy Standard](https://en.wikipedia.org/wiki/Filesystem_Hierarchy_Standard).
# * The Filesystem Hierarchy Standard is a trailing standard, not a leading standard. Although Linux distros conform to the major points of the standard, and indeed have to to maintain compatibility, there are some small differences in behaviors between distributions. And more prominently, there are new additions or new experimental features that are present in many distros that have not yet made it into the standard.
# 
#   Here, we list the interesting and non-obvious ones.
# 
# * `/bin` &mdash; contains binary executables (e.g. command-line executables like `ls` et. al.) that are runable in single-user mode.
# 
#   **Single-user mode** is a special boot mode which boots the machine into a single superuser. This mode disables network resources, so it can be used for security purposes (like, say, depriving a known computer virus from access to the Internet in the process of trying to quarantine and erase it). It is also often used to perform administrative tasks that require sole ownership over a shared resource. The Wikipedia article on this subject lists running `fsck` (a filesystem consistency tool) on a networked resource (e.g. an NFS server).
#   
#   Single-user model is initialized via boot options, and is Linux's equivalent to Windows safe mode, as it initializes only a very safe subset of processes (leaving out, for instance, the X Window Server; you only get terminal in full-screen mode).
#   
#   Needless to say, only the superuser can write into this folder, and you really shouldn't ever need to do it unless you're doing something at the sysadmin level.
#   
# * `/dev` &mdash; short for device; this is a primary location for devices mounted onto the filesystem as regular read/write I/O pipes.
# 
#   Device mounts are a useful abstraction because they allow you to use the standard Linux commands for reading to and writing from a file with devices (or partitions of devices, or even software-defined devices that don't actually correspond to any hardware) besides the boot device. This fits with Linux's general "everything is a file" philosophy. Some examples of devices include:
#   
#   * The boot device (or boot device partition), mounted to `/boot`.
#   * The developer tools in-memory (on-RAM via `tmpfs`) device, mounted to `/dev`, which includes gems like `/dev/null` and `/dev/random` and `/dev/urandom` (`/dev/urandom` is "unlimited random", which is non-blocking, unlike `/dev/random`; it's theoretically less safe than `/dev/random` but not mainstream attack against it has been discovered as of yet).
#   
#   * TODO: explore the tools available in `/dev` further.
# 
# * `/etc` &mdash; short for etcetera; used for files meant to be available system wide. For example, a fresh install of RASPIAN has an `etc/matplotlibrc` file, and Apache installs its config files in `/etc/apache2`.
# * `/media` &mdash; Mount point for media devices, e.g. CD-ROMs.
# * `/mnt` &mdash; Mount point for explicitly temporal temporary filesystems. These are distinguishable from things mounted in `dev/`, which are supposed to be persistant in nature.
# * `/proc` &mdash; Provides various process information for the OS at large (e.g. `locks`) and for specific processes by PID (e.g. `status` for the root process, which is a folder called `1`). This is a special filesystem space managed by a Linux kernel on-the-fly using a kernel subsystem internally called `procfs`.
# * `/root` &mdash; Home directory for the root user. The home directory for the current user is `home/$username`.
# * `/run` &mdash; A RAM-mounted (`tmpfs`) space meant as a scratch space for files written by daemons, especially during the startup process. Used to store things like lockfiles and the like. `systemd` for example writes a pile of stuff into here.
# 
#   * TODO: sockets and named pipes, what are those?
# 
# * `/sbin` &mdash; For super-user only binary executables that can be run in single-user mode.
# * `/tmp` &mdash; The primary temporary file space for userland processes.
# * `/usr` &mdash; The current user's directory, which constitutes the primary body of data on the machine. More deets below.
# * `/var` &mdash; Short for "variable", meant for files that change constantly. More deets below.
# 
# ### The usr directory
# 
# * The `/usr` directory is terra firma for most applications files on Linux. It has may subdivisions.
# * `/local` mirrors the hierarchy of the `/usr` bin, which is user-independent, for user-dependent things. E.g. writing to `usr/bin/` would make a binary available for all users whilst writing to `usr/local/bin` would make it available to the current user only.
# * `/bin` &mdash; contains non-essential binaries (e.g. binaries that are not available in single-user mode) that are multiuser. There is also `/bin/local` (e.g. `/usr/local/bin`) for binary executables that are non-essential and single-user, and a similar `/usr/local/sbin` for non-essential single-user priviledged binaries, and a `/usr/sbin/` because you know.
# * `/include` &mdash; Shared space for header files. To avoid collisions, non-OS header files are almost always placed under a subdirectory, e.g. `numpy`. There is also `/usr/local/include`.
# * `/lib` &mdash; Libraries. A more general partition than `/include` I suppose. There is also `/usr/local/lib`.
# 
# ### The var directory
# 
# * This directory contains files whose contents is expected to vary during normal operations, but which is expected to be preserved between runs.
# * `/cache` &mdash; Application caches.
# * `/var/lib` &mdash; Persistant state information, e.g. databases.
# * `/var/lock` &mdash; Lock files.
# * `/var/log` &mdash; Log files.
# * `/var/tmp` &mdash; Temporary data that is meant to be preserved between reboots.
# 
# ### File system usage notes
# 
# * One thing that the Linux file hierarchy doesn't make clear is the boundary between "stuff installed in `/usr/local` and `/home/$username/.local` (alias `~/.local`). This is because the spec sucked on this point, and the standard that eventually emerged is as follows:
# 
#   > /usr/local is a place where software usable by all users can be installed by an administrator.
#   >
#   > ~/.local/bin is a place where a user can install software for their own use.
#   >
#   > There is some messiness in the history of the directory structure used in different distros and communities, but this covers the basic differences.
# 
#   The `/home/$username.local` folder has its own `bin`, `lib`, and `share` directories, just as `/` (root), `usr/` (all user), and `usr/local` (current user) all do. It is meant to be the place where you, the user, install stuff. For example, on my Pi, `python3 -m pip install virtualenv` installed the `virtualenv` binary in `~/.local/bin`.
#   

# ## Inter-process communication: sockets
# 
# * Sockets and ports are the generalized abstraction used in Linux (and pretty much everywhere else) for network connectivity. A **port** is a number that is assigned to a connection point on the computer "switchboard". A **socket** is a process component that acts on traffic that gets sent to a port that the socket is bound to.
# * There are two types of sockets on a modern Linux machine.
# 
#   The first and vastly less common type is the Unix domain socket. This is a socket type that is constrained to the local machine, and is designed to play well with how Unix works (e.g. it is file-based, takes advantage of userland permissions, etcetera). The second and predominant socket type is the **Berkeley socket**. Berkeley sockets are the socket API used across all of Linux, macOS, and Windows for connecting a machine to a network.
#   
#   These are two distinct subsystems. Because Berkeley sockets are much more useful and also instantly portable, they're generally what people mean when they talk about "sockets". Network programming on a local machine level means dealing with sockets.
# 
# * Sockets are a transport layer standard, so they are made to work with a variety of transport layer protocols, which in practice means either TCP or UDP. When specifying what a socket should listen to, you need to set the transport protocol you are using and the port number you are using. Port numbers below 1024 are reserved and require superuser privileges to listen to. Two well-known port numbers are 80 for HTTP traffic and 443 for TLS/HTTPS traffic.
# 
#   The next layer over the transport layer is the Internet layer, which almost always means IP. Although you can omit identification and send and recieve anonymous packets with a raw port, in practice TCP and UPD almost always mean TCP/IP or UDP/IP. So in the Python sockets API for example you can set your expected traffic type to IPv4 or IPv6, and the library will handle de-encapsulating the data payload and reading out the IP address identifier for you.
# 
# * Working with a socket generally proceeds thusly. First you create a **listener socket** and bind it to a certain IP address and port number on that machine. Connection requests (`SYN` packets) bound for that address are forwarded to the listener socket, which is able to then attach that connection to a new **connected socket**, which handles the rest of the handshake and the ensuing two-way communication. After spinning off the new socket, the listener socket goes back to waiting for new connection requests.

# ## Inter-process communication: files
# 
# * Files are the simplest way of performing IPC: write some data to a file, and let other processes read from that file to find out the current state of the system.
# 
#   The "everything is a file" Unix philosophy is very conducive to this organizational format. You don't even need to read from disk (slow) to do this, as you can mount a filesystem partition in `tmpfs` RAM (or use one of the RAM filesystem partitions that Linux provides by default) to make it a read operation from memory instead (fast).
# * There are synchronous and asynchronous file APIs, both of which are POSIX standards. However so-called AIO is [apparently not useful](https://stackoverflow.com/questions/87892/what-is-the-status-of-posix-asynchronous-i-o-aio) because it's better done in the software layer for all practical applications, using an in-memory buffer constructed by the application.
# * That being the case, let's limit ourselves to just synchronous file sematics and APIs.
# 
# 
# * You can open multiple file descriptors against a file. Linux takes an **advisory lock** out on a file while it is being read from or written to (by creating an empty lockfile, which is an instantaneous atomic operation on the filesystem); programs may choose to disregard this lock and perform their operation anyway.
# * Concurrent read-writes are atomic IFF all of the writes are performed on append mode (`O_APPEND`) and the underlying filesystem is not NFS below version 4. [Source](https://stackoverflow.com/questions/1154446/is-file-append-atomic-in-unix). But there are minor bugs in certain edge cases, see [Dan Luu's blog post on this subject](https://danluu.com/deconstruct-files/).
# * A program may also choose to take a mandatory lock on a file, which guarantees exclusive access to that file for that program.

# ## Memory-mapped files
# 
# * **Memory-mapped files** are files on disk that are mapped into RAM. A memory-mapped file is created by executing a POSIX `mmap` command on a file on disk. Memory-mapped files are intrisincally lazy; bytes are transferred to RAM page-by-page as programs read to or write from the file. This is known as **demand paging**. The fules can be persisted or non-persisted, depending on what your need is. If the file is persisted, the actual write-back operation is similarly delayed until deallocation time (`munmap`).
# * Memory-mapped files are a way of performing IPC using files with OS-managed disk persistence.
# * Memory-mapped files used for IPC (e.g. used by multiple processes) are a typical example of **shared memory**.
# * TODO: play with memory-mapped files on the Pi using the [mmap module in Python](https://docs.python.org/3.7/library/mmap.html).

# ## `fsynch` and flushing to disk
# 
# * The `fsynch` is a blocking sysop that flushes all in-memory data with a corresponding disk footprint to disk. This includes memory-mapped files, pending changes on file buffers, etcetera. A completed `fsych` basically guarantees that if the system were to crash, all of the disk-footprinted data currently in the system will be retained.
# * However `fsynch` is a notably slow operation. It goes through and flushes every single one of Linux's many layers of caches to disk; a time-consuming proposition. Thus while it's *safe*, it's not *fast*.

# ## Message passing (conceptual)
# 
# * **Message passing** is the technique of performing a programming action by sending a message to another process, as opposed to doing so by executing a function directly, in the current thread. Message passing is the fundamental primitive of distributed programming. The degree of message passing involved in the programming model used by a service makes up the difference between a monolithic service (which has very little) and a microservice (which has a lot of it).
# * The fundamental theory of the microservice is **encapsulation**&mdash;the idea that services that are responsible for specific well-defined pieces of functionality, and that those other services can rely on the encapsulating service without caring about its implementation specifics.
# * The **actor model** is the most academically prominent theoretical description of message passing.
# 
# 
# * RPC (**remote procedure call**) is a way of thinking about distributing work by applying message passing over remote machines. It has the following components:
#   1. The client calls the client stub. The call is a local procedure call, with parameters pushed on to the stack in the normal way.
#   2. The client stub packs the parameters into a message and makes a system call to send the message. Packing the parameters is called marshalling.
#   3. The client's local operating system sends the message from the client machine to the server machine.
#   4. The local operating system on the server machine passes the incoming packets to the server stub.
#   5. The server stub unpacks the parameters from the message. Unpacking the parameters is called unmarshalling.
#   6. Finally, the server stub calls the server procedure. The reply traces the same steps in the reverse direction.
# * This is a very general way of thinking of process distribution. Basically any microservice implements a bajillion RPC calls.
# * REST is a methodology for interprocess communication that is specifically designed to work well in web contexts. A well-designed distributed system on e.g. Kubernetes may be an implementation of REST on RPC. The two concepts are compatible with one another, but separate. It definitely doesn't make sense to talk about "RPC versus REST".
# 
# 
# ## Inter-process communication: pipes
# * Pipes are used to send the of `STDOUT` one one program as input to another program. They are supposedly very well-implemented, all things considered, in terms of how they manage their throughput.
# * **Anonymous pipes** are created using the `|` character, and are a standard part of Linux operation pipelining.
# * Linux has the capacity to persist transient anonymous pipes as **named pipes**. Named pipes are files on the filesystem which can be connected to certain output actions using standard Linux commands. Inputs connected that are then connected to the pipe then follow the program defined in the pipe in execution.
# * Named pipes are basically persistant function definitions for Bash scripts. Not a recommended technique for anything else, but it's nice to know that they exist.
