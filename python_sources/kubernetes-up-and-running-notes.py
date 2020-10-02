#!/usr/bin/env python
# coding: utf-8

# My notes on the book "Kubernetes: Up & Running". A real page-turner!
# 
# ## 1-3&mdash;Containers and clusters
# 
# * A **container image** is a binary file that contains all of the dependencies necessary to run a certain application inside of a certain OS container.
# * Containers are stored and retrieved from a **container registry**.
# * The Docker image format is the de facto standard. There is now an Open Container Image format that is to be the new standard but which is still in its infancy.
# * Container definitions are based on a linear sequence of filesystem layers.
# * To achieve this they use an **overlay filesystem**. There are several implementations of such filesystems, including `aufs`, `overlay`, and `overlay2`.
# * TODO: investigate the properties of these filesystems further.
# 
# 
# * Container image formats are typically bundled with a **container configuration file**, which determines how the container environment should be set up. Concerns typical for configuration are:
#   * Networking concerns
#   * [Namespace isolation](https://en.wikipedia.org/wiki/Linux_namespaces)
#   * Resource constraints ([cgroups](https://en.wikipedia.org/wiki/Cgroups))
#   * `syscall` restrictions ([`seccomp`](https://docs.docker.com/engine/security/seccomp/))
# * The configuration file is bundled with the container root filesystem in Docker.
# * There are commonly two types of containers. *System containers* seek to emulate some container definition useful for setting up a container on. *Application containers* run a single application, typically using a system container as a base layer.
# * Kuberbetes has three services it deploys by default: a loader balancer, a DNS service, and a web UI.
# 
# 
# ## 4&mdash;`kubectl`
# 
# * **Namespaces** can be used to manage object groups, like e.g. distinguishing between `prod` and `stage` within the same cluster. To do things only with objects in a namespace run `kubectl --namespace "foo"`.
# * **Contexts** can be used to change things more permanently (unclear how).
# * Instances of the various resource types in Kubernetes are known as **objects**. You can list all objects of a particular resource type using `kubectl get [resource, [resource-name]]`. E.g. `kubectl get services` to list installed services.
# * By default `kubectl get` provides human-readable output. You can get the exact configuration as a JSON or YAML fragment using the `-o` flag.
# * You can omit the header (for use with e.g. Unix pipes) via `--no-headers`.
# * You can subset fields using e.g. `--template={.foo.bar}`. This is configured via the JSONPath query DSL.
# 
# ## 5&mdash;Pods
# 
# * Containers in the same pod share a number of resources:
#   * IP address
#   * Port space
#   * Hostname (UTS namespace) (TODO: what?)
#   * Communicate using native interprocess communication: System V IPC or POSIX message queues (TODO: what are those?)
# * Conceptually, containers in different pods are effectively on different servers.
# * Pods are the smallest unit managed by Kubernetes.
# 
# 
# * Pod manifests are persisted to `etcd`.
# * Termination grace period for pods (e.g. `SIGTERM`) is 30 seconds. After that `SIGKILL` is sent.
# * You can use port-forwarding to give yourself access to pod without exposing it to the public Internet using e.g. `kubectl port-forward foo 8080:8080`.
# * There also a `kubectl cp` command if you really need it.
# 
# 
# * Kubernetes comes with a variety of health checks.
# * There is a basic, always present process probe, which ensures that the main process of your application is always running. If the process probe returns a negative result (e.g. your process crashed) Kubernetes will restart the pod for you (but it will back off after some number of restart attemtps).
# * A user-writable **liveness probe** allows you to check if the pod is actually live and performing as expected. Liveness problems are configured, interestingly, using a YAML fragment in the pod manifest. If a pod fails a liveness probe, it gets killed and recreated.
# * Finally there is a user-writable **readiness probe**. If a pod fails this test it is merely removed from the load balancing rotation until such time as it succeeds the probe again.
# * Avaiable probe types: HTTP/S requests, TCP socket requests, and `exec` script executions (which determine success based on execution error code).
# * Besides service liveness, readiness probes can also be used to perform graceful shutdowns (have the server wait for all connections to close, and only _then_ shut down).
# 
# 
# * The pod manifest allows specifying CPU, memory, etc. minimum *requests* on a per-container basis.
# * Limits are soft. If more of a resource is available than requested, more of it will be allocated.
# * However a resource will never be allocated *less* of a resource than requested in the minimum.
# * Resources are shared equally in proportion to the containers in the pod.
# * Limits go into effect at pod start time. E.g. if two containers are splitting a core equally, they will each be allocated that split then, and cannot "flex" about that percentage to e.g. claim disk space the other container is not using (although hardware-level bursts may still provide resource bonuses for short bursts of busy times).
# * CPU is allocated in terms of percentages of cores.
# * CPU allocation uses the `cpu-shares` Linux feature.
# * Memory is allocated in terms of raw memory values. If memory is fully subscribed, because there is no easy way to deallocate memory (aside from telling the OS to send memory reclaims yourself I guess), an offending container will be killed and restarted with lower memory ceiling.
# * Other resources are also supported, like e.g. GPU.
# 
# 
# * There is a separate interface for specifying maximum *limits*. These work the way you expect them to.
# 
# 
# * Disk memory allocation is modeled using `volumes`.
# * There are generally three patterns for using disk:
#   * `emptyDir`, which provides an empty directory with a specifiable maximum storage size that persists for the length of the current session.
#   * Truly fully persistant storage via raw block storage (e.g. NFS) or communication with storage APIs (e.g. S3).
#   * Access to the underlying node host. This is provided via the ability to mount a chunk of the volume of the underlying host instance. But it's fragile.
# * Volumes are mounted on a container-by-container basis within a pod.
# * A container need not mount every volume, and volumes need not be mounted at the same location throughout.
# 
# 
# ## 6&mdash;Labels and Annotations
# * Historical experiences have shown that no automatic tagging of system components is sufficient for engineering purposes.
# * Plus, your understanding of your system is constantly changing.
# * So it's important to be able to assign selectors at any time yourself. Labels and annotations meet this need.
# * **Labels** are arbitrary key-value pairs which are meant to be used for identifying information and may ber assigned to arbitrary objects. Labels are the fundamental of groupings.
# * Labels may have have an optional prefix which corresponds with a DNS name (TODO: why?).
# * You can apply labels in the template for a pod or replica set, or via `kubectl label deployments foo "something=whatever"` at the CLI for deployments. 
# * To delete a label you do something funky, add a `-` at the end. E.g. `kubectl label deployments foo "something=whatever-"`.
# * The `--selector` flag to `get` allows you to specifying labels and boolean combinations of labels which you would like to get info on. It supports a variety of lightweight boolean ops, specified as a string.
# * Label selectors sytax in YAML/JSON templates use a `matchExpressions` syntax, which allows you to specify the same boolean selection operations as a `dict` fragment.
# * **Annotations** meanwhile are keyed blob data fragments that may be attached to an object. They are meant to be used by e.g. tools that abstract on top of Kubernetes.
# * Annotations are specified using arbitary-length string key-value pairs in the `metadata` header on the object definition.
# * These is some overlap between annotation and label functionality, but by and by you should be using labels for most things.
# 
# 
# ## 7&mdash;Service Discovery
# * One of the two hard problems in computer science: naming things!
# * DNS as standardly deployed on the Internet is not appropriate for Kubernetes, becuase Kubernetes is a much more dynamic and smaller system than the Internet. This leads to issues like:
#   * Figuring out how long to set TTLs for is problematic.
#   * Clients will round-robin a choice when returned many A records, but not reliably; and this isn't necessarily the right load balancing policy regardless.
#   * DNS is creaky past 20-30 A records, and you would need significantly more than that on a large deployment.
# * Kubernetes uses DNS for name resolution, but does so via the `Service` object API.
# 
# 
# * The `kubectl expose` command exposes a deployment you've launched (via e.g. `kubectl run`) to the wider Kubernetes service network. It does so by creating a `Service` object for that deployment, and registering that service object (with its name and its tags) in the service list.
# * Other pods may query the Kubernetes DNS service for the IP address and port of a deployment or deployments of interest.
# * IP addresses are *cluster IP* addresses. A cluster IP address is a **virtual IP address**; it doesn't correspond with an actual physical interface, but instead acts as a router, load balancing requests into the necessary pods and deployments appropriately.
# * Because the cluster IP address is virtual, it is permanent, and may safely be cached in DNS with no expiry.
# * Because routing is network-local, this design adds minimal latency to network transit times.
# * However, the address being virtual also means that the deployment is only accessible from within the cluster, e.g. by other Kubernetes-managed services.
# * You can make a service publicly reachable by exposing a `NodePort` instead. A `NodePort` is specified with a static port, and once the service is launched, requests to that port on the local machine will be routed to the cluster IP, which in turn will route into the pod.
# * You can add a third level of indirection: a `LoadBalancer`. How this works exactly is a cloud provider -specific implementation detail?
# 
# 
# * The Kubernetes DNS service exposing all of these things is provided to all pods running on the cluster.
# * DNS names look something like `alpaca-prod.default.svc.cluster.local`. The name components are:
#   * `alpaca-prod`: The name of the service in question.
#   * `default`: The namespace the service is running in.
#   * `svc`: Identifies that this is a service. Currently only services may be exposed via Kubernetes, but in the future maybe other things will be possible.
#   * `cluster.local`: The base domain name for the cluster. May be changed by administrators, mostly cosmetic?
# * Going forward, you can get the service by sending an HTTP request against that name.
# * Note: this is different from the more laborious way I stitched these things together when I built the subway application deploy, where I routed A to B myself using environment variables, for no particular reason really...
# 
# 
# * Some applications want to be able to use services without going through the cluster IP (e.g. they want to avoid load balancing, or they want to maintain a session with a specific pod). This is done using `Endpoint` objects. Every `Service` contains an `Endpoint` that contains the IP address for that service.
# * The Kubernetes API may be used to get pod IP addresses, and even to subscribe to updates on that pod's IP address (via `--watch`).
# * This is essentially an alternative to using DNS proxy routing. Instead of using DNS to perform service discovery, we use a message mesh.
#   * Advantage: more performant if you are designing a service to work on Kubernetes from the start.
#   * Disadvantage: DNS is a well-known and travelled system, and most existing applications expect to work through it.
# 
# 
# * You can also do service discovery manually, by querying the Kubernetes API for pod IP addresses. But this is hard to sync, which is why `Service` objects were invented.
# * All of this is managed by the `kube-proxy` service that Kubernetes launches for you when you start up a cluster. The actual packet routing is handled via changes to the underlying operating system's `iptables` rules.
# * A caution against using environment variables like I did: totally doable, not all that recommended. It has the major disadvantage that it requires standing up services in a certain order, and that order is easy to mess up.
# 
# 
# ## 8&mdash;ReplicaSets
# * A `ReplicaSet` is a pod manager. It ensures that the right number of pods is kept running.
# * `ReplicaSet` objects are meant to handle scale-out and self-healing concerns. They are distinguishable from `Service` objects, which handle service discovery.
# * Each `ReplicaSet` consists of some number of homogenous containers that come from a single image.
# * You may create a deployment that launches pods, but best practice is to create deployments that launch replica sets of pods.
# 
# 
# * The "correct" way to scale is using `kubectl apply` to do so after editing the source file. You should avoid using `kubectl scale` because it desyncs the state of the cluster from the state of the files defining the cluster...which makes debugging much harder.
# 
# 
# * `ReplicaSet` objects may be configured to automatically scale horizontally. This uses a `kubectl autoscale` command.
# * There is not currently a solution for vertical scaling.
# * Advanced feature, and it can clash with hand management.
# 
# 
# * How deployments, services, and replica sets interact. First, you write a `deployment.yml` definition for a `Deployment` that contains a number of `ReplicaSet` groups managing the pods that make up the group. You `kubectl run` to stand this up. Then, you `kubectl expose deployment` to stand the service up. Once you've done this for all of your cluster deployment components, you're ready to go.
# * Reliance on pods that don't make sense to include in the present deployment (e.g. ones managed by other teams) should be the boundary between deployments.
# 
# 
# ## 9&mdash;DaemonSets
# * `DaemonSet` objects are like `ReplicaSet` objects but for services that need to run on every node. Things like log collectors and the like make sense to define using `DaemonSet` objects instead of `ReplicaSet` ones.
# * `DaemonSet` objects may limit their pod deployments to only certain nodes using label selection criteria.
# * `DaemonSet` build and teardowns can be managed using the same rolling updates that deployments use. But this is not the default behavior for backwards compatibility. You have to set that behavior yourself.
# 
# 
# ## 10&mdash;Jobs
# * `Job` objects provide execution, e.g. the ability to run a pod until termination only once. Trying to deploy an exiting container on a pod otherwise would cause restarts, backoffs, and fails. Jobs work by launching a container, then passing some command line arguments to that container `ENTRYPOINT`.
# * There are generally three patterns for jobs that you want to run.
#   * *One-shot jobs* are pretty much just script executions, but extrapolated to the Kubernetes level of abstraction.
#     * Scheduling a job is done via `kubectl run`, same as with other object run-schedulings.
#     * Pass `--restart=Never` to get job restarting behavior (or `OnFailure`).
#     * Pass `-i` to cause `kubectl` to output logs from the first pod in the job to STDOUT (the terminal).
#     * Pass `--` followed by blah to pass command line arguments to the container image.
#     * Note that `kubectl` may miss the first few lines of output (really? weird).
#   * *Parallelized jobs* can run embarrasingly parallelizable workloads.
#     * To do this you specify a `parallelism` and `completions` in the job definition. Up to `parallelism` containers will be run, and new containers will be created until `completions` many finish successfully.
#   * *Work queue jobs* can run parallel workloads up until at least one container exits.
#     * To set up a work queue, specify a `parallelism` but not a `completions` in the job definition.
#     
# 
# ## 11&mdash;ConfigMaps and Secrets
# * `ConfigMap` is used to specify configuration that is not scoped to the container definition.
# * It can be passed to a container image in one of three ways, all controlled by the deployment definition: mounted to a file on the filesystem, passed as environment variables, or injected into CLI commands (which optionally runs at runtime).
# 
# 
# * `Secret` objects can be used to manage confidential information.
# * Secrets have only one way of getting exposed to the container: being mounted as a file.
# * Specifically, secrets are written to a `tempfs` RAM partition that is mounted as a pseudo-disk inside of the container.
# * You can then access that file to access the secret.
# 
# 
# * A special type of secret is the *image pull secret*, which is used for validation of user auth when attempting to pull from private Docker registries.
# * To create such a secret pass `--docker-username`, `--docker-password`, and `--docker-email` to the `kubectl create secret foo` command.
# * To consume this secret, pass it into the `imagePullSecrets` field.
# 
# 
# * Note that there is currently no way for an image to know that a config map update has taken place. You are responsible for dealing with that yourself.
# 
# 
# ## 12&mdash;Deployment
# * Uses a rolling update strategy.
# * As with other resources, prefer to edit a file and then `apply` -ing it (declaratively making changes // using the reconciliation loop) rather than imperitively making changes.
# * There are `Recreate` and `RollingUpdate` strategies. The former will tear down before building back up, so it's strongly not recommended in production, but acceptable in experimental use cases.
# * There are a variety of parameters which may be used to tweak how rolling updates are preferred: `maxUnavailable`, `maxSurge`, `minReadySeconds`, `progressDeadlineSeconds`.
# * In order to maintain undegraded service whilst performing a rolling update in the context of a data input change, your software must be able to accept both new and old data formats, as services will be addressed by a mixture of old and new systems for the duration of the upgrade.
# * This is important!
# 
# 
# ## 13&mdash;Data
# * Data is often the most technically difficult part of managing a containerized service.
# * Containerization is a native fit for stateless applications, but any reasonable complex application will have state persisted somewhere. There are a number of things about persistant state that are challenging to manage from a Kubernetes perspective:
#   * You rarely get to have a "clean start"; instead that state is already living on some other system somewhere, and needs to be migrated or adapted to.
#   * Data is not swappable. One disk drive containing your data cannot be automatically swapped for an empty one of the same size.
#   * The Kubernetes API is designed to make building stateless aplication surface area easy first and foremost, so working with persistant data "feels" clunky.
# * The book outlines a number of strategies for working with your data.
# 
# 
# * The recommended way of working with completely external storage, e.g. an existing database, is to wrap that storage in a Kubernetes `Service` object anyway, and have that object handle directing traffic as appropriate.
# * This has the notable advantage (compared to other techniques) that if you want to replace your external service with an internal cluster-managed one, it's painless to do so, since you're already using the right path. This isn't just fanciful either: it's very useful for testing.
# * To initialize an external service, instantiate a `Service` object with no pod specified.
# * If the external service has a publicly accessible `CNAME`, you pass that to the `externalName` field. The `Service` will automatically handle publishing a stable internal IP address that routes to this `CNAME`.
# * If the external service only has an IP address, you create a `Service` without even an `externalName`, and then handle routing to that IP by creating an `Endpoint` resource for it manually.
# 
# 
# * A *reliable singleton* provides an alternative that managed completely inside of Kubernetes and *relatively* simple. However it is not HA.
# * This is the `PersistantVolume` and `PersistantVolumeClaim` approach.
# * Many cloud providers allow you to specify a `Provisioner`, which sets up *dynamic volume claims*&mdash;the cloud provider will handle giving and scaling the memory required for your application for you.
# 
# 
# * The most advanced way of managing persistant memory in Kubernetes is with **stateful sets** (`StatefulSet` objects).
# * Stateful sets are a variant of the replica set. It has some special linearized behaviors:
#   * Each replica has a uniquely indexed persistant hostname, e.g. `db-0` on.
#   * Each replica is created in low-to-high order, and creation will block on the previous replica completing.
#   * Similarly, replicas are torn down high-to-low.
# * You still need to populate a service object for this replica set, but there isn't a single IP address associated with load-balancing the requests anymore.
# * Instead, you create a "headless" `Service` object (via `clusterIP: None`). This service object will:
#   * Provide the full list of A records pointing to the actual service IP address when DNS queried with the service name, instead of providing a virtual IP address that does load balancing behind the scene.
#   * Populate an additional A record for every single pod in the set, e.g. `db-0` on.
# * Note that actually configuring the services running in the pods for replication is up to you. You can do with configuration scripts, but the "how" in going about doing that is going to be database-specific.
# * You may also need to stand up persistant storage. Since a `StatefulSet` generates a lot of pods, getting that storage requires not a persistant volume claim but a persistant volume claim *template*. 
# * Note that the `StatefulSet` will need to have access to a pool of volume objects that it can claim from at creation time, otherwise it will fail for lack of room.
