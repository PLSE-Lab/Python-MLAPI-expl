#!/usr/bin/env python
# coding: utf-8

# ## Reference on debugging Docker and Kubernetes
# 
# This page is a collection of things I've learned which are helpful for debugging Docker and/or Kubernetes systems.
# 
# * Recall that the chracteristics of a **container** are defined by an **image**.
# * An image contains many intermediate layers, each of which constitutes its own image. That image is not tagged, but is nevertheless accessible to various `docker` CLI commands via its SHA-256 identifier. This identifier is available in the following places:
#   * Via print-out to `stdout` during `docker build`.
#   * Via `docker image ls --all`.
#   
# ### Opening a bash interpreter in a Docker image
# * I know of two useful ways of opening a Bash interpreter inside of a Docker image.
# * The first is `docker run -ti --entrypoint=/bin/bash ID`, where `ID` is the SHA-256 identifier or the image tag. What this command does:
#   * `docker run` runs an image, e.g. starts a new container up based on this image definition.
#   * `-ti` does "Allocate a pseudo-tty" and "Keep STDIN open even if not attached". The `i` flag is short for "interactive mode". I don't understand what is going on under the hood well enough to say why these are named the way they are.
#   * `--entrypoint=/bin/bash` overrides the `ENTRYPOINT` in the `Dockerfile` with your own.
#   
#     Recall that the `ENTRYPOINT` is command-line argument you want to get executed at `run` time. Replacing the entrypoint with `/bin/bash` means that a `bash` process is opened as the last step of the shell.
#   * The `--entrypoint` parameter does not support command parameterization, merely specifying the executable. Entrypoint arguments are passed in the usual way, by specifying them at the end. For example, to launch the shell with no interactive customizations, one would run e.g. `docker run -ti --entrypoint=/bin/bash de6de243001b --noprofile --norc`.
# * This command is advantageous when the Docker container you are trying to examine does not persist, e.g. its `ENTRYPOINT` script is not a long-lived process. In this case, the `exec` form will not work.
# * As in many other Docker commands, order matters! The `-ti` (or `-it`) and `--entrypoint` flags must appear ahead of the `ID` argument.
# 
# 
# * The second form is `docker exec -ti ID /bin/bash`. The `exec` command "executes a command" in a running container; it doesn't start up a new one. This is therefore used for stepping into running containers. So the `ID` in this case is the container ID of an already-running container, accessible via e.g. `docker container ls`.
# 
# 
# * Are there any differences between the bash shell you start this way and others? Yes!
#   * These Docker commands launch bash in **interactive mode**. Interactive mode is the bash mode used when the shell yields to user input (e.g. opening a terminal), as opposed to system input (e.g. running a script).
#   * There are two files controlling bash behavior: `~/.bash_profile`, which is run immediately before every login into a bash shell; and `~/.bashrc`, which is run after the login challenge succeeds, or, if there is no login, at startup time. Commands from these files are *only* run when the shell is interactive. [See here](https://medium.com/@kingnand.90/what-is-the-difference-between-bash-profile-and-bashrc-d4c902ac7308#:~:targetText=Answer%3A%20.,executed%20for%20login%20shells%2C%20while%20.&targetText=bashrc%20is%20also%20run%20when,configure%20that%20in%20the%20preferences.) for more details.
#   * This means that anything in your `~/.bash_profile` or `~/.bashrc` will only be run when you exec into the container, as here. It will not be run when you run `bash` via the system inside of the container. Some more details [here](https://unix.stackexchange.com/questions/257571/why-does-bashrc-check-whether-the-current-shell-is-interactive).
# * To force a non-interactive bash shell to launch in interactive mode, execute it with the `-i` flag.
# * To force an interactive bash shell to launch in non-interactive mode, execute it with the `--noprofile` and `--norc` flag.
# 
# ### Container logs
# * Anything that gets written to STDOUT by a container is send to logs by Kubernetes, populating `kubectl logs [container-name]` as a result.
# * If you are not in a Kubernetes environment, or potentially if you are, you can view that same log stream with `docker logs`.
# * If you have configured an alternative log consumer you may have to use that as these code paths will not work
# 
# ### Intermediate containers
# * The complete list of intermediate containers built during the process of building the final container is tagged and available for interaction. To see the list run  `docker history CONTAINER`. You can then target any of these intermediate containers with a `run` instruction to debug from there.
# 
# ### Opening a bash interpreter in a container in a Kubernetes cluster
# * To execute a command directly on a container, use `kubectl exec [pod-name] [command]`. Thus you can use `kubectl exec -ti [pod-name] bash` to open a Bash interpreter in a pod container. Since containers under management by Kubernetes are expected to be long-lived, this is the primary interaction mechanism.
