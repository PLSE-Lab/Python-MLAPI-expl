#!/usr/bin/env python
# coding: utf-8

# This notebook contains my notes on the [Kubernetes production best practices checklist](https://learnk8s.io/production-best-practices/). This is a recommendable checklist providing a running tally of concerns that should be addressed when working with Kubernetes systems. This notebook can be considered a follow-up to my previous notebook [Notes on Docker packaging with Python](https://www.kaggle.com/residentmario/notes-on-docker-packaging-with-python), which contains a some notes on a series of blog posts explaining best practices around the package management of Python processes within Docker images.
# 
# Without further ado, let's break down the list.
# 
# ## Health checks
# 1. Containers have readiness probes
#    * Readiness probes are run throughout the lifecycle of the container, at set intervals and with set response timers. Failing a readiness probe will take the container out of the service routing table until such time as it passes the readiness probe again.
#    * Kubelet by default assumes that contains that have finished startup are ready to recieve traffic, and are always healthy so long as they have not crashed. If the container isn't actually ready to recieve traffic until later on, this will result in traffic hitting the container too early.
#    * Well-designed services enforce non-null readiness probes in all applications, even those which are indeed instantaneously available.
#    * The actual **readiness probe** definition is an instruction that looks something like this:
#    ```
#    readinessProbe:
#   exec:
#     command:
#     - cat
#     - /tmp/healthy
#   initialDelaySeconds: 5
#   periodSeconds: 5
#   ```
# 
# 2. Containers crash when there's a fatal error
#    * An example of a bad behavior would be having only application-level restarts in the container that toggle the readiness probe, without ever actually allowing the container to crash and be kubectl-restarted.
#    * The failure cases being overly aggressive about your restart policy can create is documented in ["Kubernetes Liveness and Readiness Probes Revisited"](https://blog.colinbreck.com/kubernetes-liveness-and-readiness-probes-revisited-how-to-avoid-shooting-yourself-in-the-other-foot/#letitcrash).
# 
# 3. Configure a passive liveness probe
#    * The **liveness probe** is designed to restart the container when the container gets fatally deadlocked. For example, if it is processing an infinitely hard request.
#    * Every service should at a minimum have a passive liveness probe.
#    * A particularly bad failure scenario is a container deadlocking which has a readiness probe but no liveness probe. This will cause the container to leave the service rotation, but to keep running forever, until or unless the container process itself fails. This is a waste of resources. Always configure a passive liveness probe.
# 
# 4. Liveness probe values are not the same as readiness
#    * If a container fails a liveness check and a readiness check simultaneously, it will detach from the service and begin shutdown simultaneously. This can cause dropped connections, as the live connections may not be rebalanced fast enough. And hence, transient service failures. 
#    * Liveness probes should run at different time offsets from readiness probes in order to avoid this problem. That way, if there is a state failure which causes both probes to fail, the detatch and restart procedures will be handled at separate times.

# ## Apps are independent
# 
# 5. The readiness probe doesn't include dependencies to services such as: databases, database migrations, APIs, and third party services.
#    
#    * The reasoning for this is obvious: it creates additional error surface for your probes, creating the potential for failures which have nothing to do with your application to cause your application to look unhealthy.
#    * It also creates an external uptime dependency.
#    * Personally, I don't think this is a hard and fast rule. I think it's fine to rely on external resources in your health checks so long as those resources have extremely high uptime guarantees, but it should be avoided.
# 
# 6. The app retries connecting to dependent services.
#    * Kubernetes applications are meant to be built in such a way that they may be launched in any order. Applications should be not fail because dependent applications are not up yet. Retried should be used instead.
# 
# ## Graceful shutdown
# 
# 7. The app doesn't shut down on SIGTERM, but it gracefully terminates connections.
#    * A single SIGTERM is the termination signal used by the Kubernetes control pane to shut down a container instance.
#    * This SIGTERM is not checked or followed up upon. Once the SIGTERM is delivered, it is assumed that termination will eventually occur. Recall that setting the container up in such a way that it actually terminates on SIGTERM delivery is actually work.
#    * Awareness of the SIGTERM is also not propogated to the entire system right away. It may take time for the network daemon to become aware of the termination state of the container (this is particularly true in larger runtime environments), during which time the container will continue to recieve new inbound connection requests.
#    * **Graceful termination** means correctly refusing any inbound requests recieved after the termination signal is recieved, closing all current connections as soon as the remaining work on them is complete, and *only then* actually fully terminating.
#    * How exactly this is implemented in application context specific.
# 
# 8. The app still processes incoming requests in the grace period.
#    * Basically the same bullet point as the above. Basically, do better than dropping the connection outright.
# 
# 9. The CMD in the `Dockerfile` forwards the SIGTERM to the process.
#    * As described in my previous set of notes! Again, see bullet point 7.
# 
# 10. Close all idle keep-alive sockets.
#     * If IPC to the container image is via HTTP keep-alive connection, that tunneled connection all the way to the target container will be reused for all future traffic.
#     * Connection reuse in this form is good networking practice for long-lived services, but it gets in the way of connection profile management against the pod as a whole: the Kubernetes connection router cannot redirect (on container shutdown) the connections elsewhere.
#     * To provide graceful termination at shutdown time you need to end idle keep-alive connections yourself, in the container service definition.
#     
# ## Fault tolerance
# 11. Run more than one replica for your deployment.
#     * Recall some terminology: a set of *containers* are managed inside of a *pod*, which defines some useful unit of service. A pod may be deployed individually, but does not provide any fault tolerance on its own; pod lifecycles are meant to be managed in production using *deployments*, which consider their responsible pod-sets as *replicas*.
#     * Thus this bullet point boils down to: run more than one instance of your pod at all times.
#     * This is necessary to gaurantee that the failure of a single pod will not cause the entire application to go down.
# 
# 12. Avoid placing pods on the same nodes.
#     * It is still possible to experience pod-failure related downtime with multiple pods for a service up, if a node failure occurs and the pods happen to be colocated on a node. Default pod assignment is random, less resource utilization balancing, so node colocation is very much possible for deployments with very few replica sets.
#     * This potential failure surface area can be managed away using placement rules known as [pod anti-affinity rules](https://cloudmark.github.io/Node-Management-In-GKE/#pod-anti-affinity-rules).
#     * This is a very cool feature, but a fairly advanced use case, e.g. there are many profiles of services for which this form of a fault tolerance is not hugely impactful.
# 
# 13. Set pod disruption budgets
#     * Pod disruption budgets are a maximum value on how many pods must be kept online.
#     * Pods are taken down when draining nodes. Setting a pod disruption budget sets a ceiling on how quickly and simultaneously nodes can be drained (for upgrades and whatever), which will increase rebalancing time, cost, and complexity, but also protect the remaining pods from possible oversubscription.
#     * Pod disruption budgets are thus useful for preventing hardware maintenance and other background rebalancing tasks from causing service downtime during periods of high load.
#     * The recommended reading on this subject is [the official documentation for this feature](https://kubernetes.io/docs/concepts/workloads/pods/disruptions/).
# 
# ## Resource utilization
# 14. Set memory limits for all containers, and consider setting CPU limits
#     * Resource limits as input to Kubernetes fall into two brackets: the minimum resources required to run a pod, and the maximum resources a pod is allowed to consume. Resource requirements are used during pod scheduling to correctly distribute resources across the containers. Maximum resources are used to throttle application resource utilization within well-specified boundaries.
#     * The statement declaring a set of resource limits looks something like this:
#     ```
#     resources:
#       requests:
#         cpu: 50m
#         memory: 50Mi
#       limits:
#         cpu: 100m
#         memory: 100Mi
#     ```
#     * There are three types of allocatable resources: RAM, ephemeral storage (which is new-ish), and CPU cycles. For a reference on how the resource allocation cycle works, see ["Understanding resource limits in kubernetes: memory"](https://medium.com/@betz.mark/understanding-resource-limits-in-kubernetes-memory-6b41e9a955f9), ["Understanding resource limits in kubernetes: cpu time"](https://medium.com/@betz.mark/understanding-resource-limits-in-kubernetes-cpu-time-9eff74d3161b), and the Kubernetes O'Reily book.
#     * Containers that exceed their memory limit are terminated (since memory is a non-compressible resource, and can't otherwise be reclaimed once allocated) whilst containers that exceed their compute limit are throttled (since CPU time is a compressible resource). 
#     * Setting memory limits (both minimum and maximum) is always a good idea. It guards against memory leaks.
#     * Setting CPU limits (maximum in particular) may not be a good idea. It's often difficult in practice to set CPU limits which balance resource utilization throttling against available compute. E.g. a limit may throttle a process that would otherwise have been able to use additional slack CPU capacity.
# 
# 15. Set CPU requests limit to 1 CPU or below.
#     * "Unless the container is computationally expensive". This bullet point seems not to actually state anything?
# 
# 16. Disable CPU limits, unless you have a really good use case.
#     * See bullet point 14.
# 
# 17. The namespace has a limit range.
#     * It may be good practice to set default resource allocation limits on the namespace level. These will apply equivalently to everything in the namespace.
# 
# 18. Set an appropriate Quality Of Service (QoS) for pods.
#     * **Quality of Service** is a Kubernetes feature used by the Kubernetes resource allocator to manage the conditions under which pods will be evicted from a node which is considered overcommitted (hardware load balancing).
#     * Quality of Service measures the attention that Kubernetes pays to pod resourcing. There are three QoS values:
#       * Best Effort (used if no resource requests or limits are set; "do whatever")
#       * Guaranteed (used if the resource requests and limits are all set, and all set to the same value, e.g. the mimimum and maximum resource allocation is the same value; these pods will be statitically block allocated).
#       * Burstable (used in all other instances, e.g. all partial resource limit instances; represents pods which are expected to experience occassional "burst traffic").
#     * These labels are applied standardly to pods, but you may override them./
#     * The Quality of Service level indicates how eager Kubernetes is to perform protective eviction: `Guaranteed > Burstable > Best Effort`.
#     * Kind of like preemptabe job priority in the MapReduce job grid.
# 
# ## Tagging resources
# 19. Resources have technical labels defined.
# 20. Resources have busines labels defined.
# 21. Resources have security labels defined.
#     * Labels are essential for performing human-logic processes on compute infrastructure.
#     * This list has some example labels that may be handy, but defining the exact set is up to your and your users' requirements.
# 
# ## Logging
# 22. The application logs to `stdout` and `stderr`.
#     * So-called *passive* logging is considered the best practice: have the application write to `stdout` and/or `stderr`, without having any awareness of the log aggregator, and have the log aggregator redirect that to output to the log aggregation space. This is part of the 12 factor app manifesto.
# 23. Avoid sidecars for logging (if you can).
#     * A sidecar is a secondary process inside of a container that provides some additional working supporting the main application.
#     * Sidecars are ocassionally attractive in logging because you may want to perform a transform on the application output before streaming to logs.
#     * But this is considered an anti-pattern in all but advanced use cases. You should have your application write the logs to stream in the correct format to begin with.
# 
# ## Scaling
# 24. Containers do not store any state in their local filesytem.
#     * Or more concretely, local filesystem space within the container should only be used as scratch space.
#     * Statefuleness should be backed out to PersistantVolume claims or to external API services.
# 25. Use the Horizontal Pod Autoscalar for apps with variable usage patterns.
#     * For applications with usage patterns that are highly volatile, the HPA provides a good solution to the problem of matching pod volume to job volume, and is considered a best practice up until you need/want a more complex scaling policy.
# 26. Don't use the Vertical Pod Autoscalar.
#     * The VPA on the other hand is still in beta and is considered highly imperfect, and shouldn't be used in production (maybe by the time you are reading this, that has changed).
#     * Prefer horizontal pod scaling. This is the appropriate choice for most application settings anyway.
# 27. Use the Cluster Pod Autoscalar if it's helpful.
#     * Analogous to the vertical and horizontal scalars, there is a cluster scalar, which can be useful for workloads that are especially prone to load scaling.
# 
# ## Configuration and secrets
# 28. Externalize all configuration.
#     * As much configuration as possible should be maintained outside of the application code.
#     * It should live instead of configuration settings shipped alongside the container definitions in the infrastructure.
# 29. Mount secrets as volumes, not environment variables
#     * Secrets need to be mounted into containers using `tmpfs`, not environment variables.
#     * Environment variables are not secret. If the environment variable is written in using a `Runfile` command, that command will be accessible in the intermediate build layers of the container, and it will be printed as output to the running user during the container definition process.
#     * Kubernetes has well-defined secrets management best practices that should be followed.
# 
# ## Namespace limits
# 30. Namespaces have limit ranges.
# 31. Namespaces have resource quotas.
#     * These are two features which limit the resource consumption of containers within a namespace.
#     * *Limit ranges* are default resource request values and limit values (see "Resource utilization" for an explainer on what these are) that are applied to every individual container included in a namespace. *Resource quotas* are cumulative resource request values set for *every* container in a namespace.
#     * These two namespace-based resource limits are the methodology of choice for setting resource utilization limits for human-labelled service groupings. For example, they allow you to set the maximum utilization for "the pods making up Service A" to 20 CPU-equivalents.
#     * Incorporating some amount of namespace limiting as a sanity check is considered a best practice. Like the United States debt ceiling, its existence helps to avoid unchecked accidental resource overconsumption.
# 
# ## Pod security policies
# 32. Enable pod security policies
#     * There are a *lot* of pod security policy options, and what they do and when they are useful is a highly involved topic. Some subset of the pod security policies should be enabled, following the principle of least privilege, but a full description of this feature is out of scope for this notebook.
# 33. Disable priviledged containers
#     * Kubernetes supports a special priviledged run mode that has almost unmitigated access to the underlying hardware.
#     * Large security attack surface, shouldn't be used unless absolutely required.
#     * When is this required? This may be required in the case of GPU compute.
# 34. Use a read-only filesystem in containers, where appropriate
#     * Using a read-only filesystem, though generally only easy for relatively simple services, closes a lot of attack surface area security-wise, making it a worthwile investment.
# 35. Prevent containers from running as root
#     * Actually you need to do this anyway to get graceful termination, as if you run the container as root SIGTERM will not propogate.
# 36. Limit Linux capacities
#     * Ah yes, Linux capacities! Linux capacities are profile-assigned priviliges that were introduced to Linux as a way of moving the RBAC part of the OS permissions model beyond "superuser or not".
#     * In theory you should be using the minimum set of capacities that accomplishes the task the container is responsible for (though this may be highly annoying to put into practice; and I found the corresponding man pages to be extremely confusing about what is and isn't under a capacity).
# 37. Prevent privilege escalation (disable `setuid` and `setgid` binaries)
#     * To help prevent trivial privilege escalation attacks, make sure to use an OS configuration that doesn't have or hasn't turned on `setuid` and `setgid` turned on.
#     * This bullet point should go further, to the following: use a bare OS, if you can. Less is more (unless you're debugging, yeah, I know).

# ## Network policies
# 38. Enable network policies.
#     * Network policies define firewall rules about which pods can network with each other.
#     * By default every container in the service lattice can see every other node in the lattice on the internal network, so long as it is able to discover the service.
#     * Network policies need to be set up to keep network visibility tuned to just the node combinations the service actually needs.
# 39. There's a conservative network policy in each namespace.
#     * Namespaces can have default network policies attached to them (as with default resource profiles, mentioned earlier in this list).
#     * It's a good idea to have a conservative network policy set as the default one.
# 
# ## RBAC policies
# 
# TODO
