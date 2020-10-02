#!/usr/bin/env python
# coding: utf-8

# These notes are based on the following blog posts:
# * https://medium.com/@Alibaba_Cloud/elasticsearch-distributed-consistency-principles-analysis-1-node-b512e2b839f8
# * https://medium.com/@Alibaba_Cloud/elasticsearch-distributed-consistency-principles-analysis-2-meta-af9fa7275fb6
# * https://medium.com/@Alibaba_Cloud/elasticsearch-distributed-consistency-principles-analysis-3-data-a98cc436bc6b
# 
# * ElasticSearch is based on the Lucene information retrieval library.
# * It uses its own distributed architecture, and doesn't rely on external components like e.g. ZooKeeper.
# 
# ### ES Cluster Composition
# * Nodes in an ES cluster are boolean combinations of master-eligible/non-master-eligible and data/non-data. Master eligible nodes may become the master nodes for the cluster via election, non-master nodes may not. Data nodes hold data partitions.
# * Any node may forward to any other node. Hence a non-master non-data node may act as a proxy.
# * Node discovery and master election is handled by an internal ZenDiscovery module.
# * Every node in the cluster is connected to a list of network ports that are set in the config. By default the recommendation is that all nodes be connected to the list of master-eligible nodes.
# * This provides full connectivity at one network hop of distance (so long as at least one master eligible node is respondant!).
# * Quorums are used to avoid split-brain. ES seems to only allow majority quorums.
# * Master elections may be launched by master-eligible nodes that discover that no nodes it is connected to are connected to master, and that it is connected to sufficiently many nodes to be able to trigger an election.
# * Master-eligible nodes are promoted to master if they are at the head of a two-order sort: network metadata version (which is sequential), and node ID (which is random).
# * If a master-eligible node selects itself, it waits until it gets a majority quorum from the other nodes, and updates the cluster state to point to itself.
# * If a master-eligible node selects another node, it waits until that node joins the vote _as well as_ until the majority of nodes have voted for that node, before sending an upgrade approval to that node. If a timeout occurs while waiting for the join, the node will initialize a new voting cycle. If the other master-eligible node was already holding its own election, that node will take that upgrade request as a vote for itself, and proceed as normal. If the other master-eligible node rejects the upgrade, the original node will initialize a new election.
# * This algorithm is problematic because it doesn't prevent double-voting: a node may try to vote for another node, time out, and send a new vote to another node it has now deemed to be even higher priority. This _will_ lead to the election of multiple master nodes.
# * This is a problem solved in e.g. Raft, which introduces election terms to deal with this problem.
# 
# ### MasterFaultDetection and NodesFaultDetection
# * There are two fault detection systems.
# * Masters scan for other nodes in the cluster. If a master finds a node that fails a ping it performs a remove node operation, which causes the other nodes it's connected to to rebalance the data shards.
# * Non-masters scan for connections to master. If it detects it is not connected, pending `cluster_state` information will be cleared, and a rejoin will be initialized to try and reconnect to master. Additionally, if the conditions for voting are met, a new election will be initiated.
# * A master will voluntarily step down if it detects that its neighborhood is no longer a majority quorum.
# 
# ## Scaling
# * Scaling down or up on data nodes means rebalancing shards. It's a pretty simple operation.
# * Master-eligible scaling is more complicated.
# * In order to retain majority quorum rules, the master must increment master-eligible node information. This is a synchronous call to the current master (or to the master pending election, if such a thing is happening).
# * Write safety requires that the upgrade request be written to a log. The master then confirms the change in the cluster meta. The problem is that if there is a reboot, the logged parameters will be used for the initial master election, rather than the in-memory cluster meta value. The newly elected master will only then attempt to upgrade using the cluster meta, if accessible. This can cause boundary conditional data loss.
# * As a result it's a good idea to only apply master node scaling at safe times, to avoid dropping actions.
# * This section ends with a comparison to ZooKeeper and Raft.
# 
# ## How Does Master Manage the Cluster
# * Master must actively manage the cluster. It does so through the publication and circulation of new cluster states.
# * Cluster states are declarative. They are consumed by connected nodes, which perform the network and compute tasks necessary to converge the system to the new state.
# * ES makes the decision to (asynchronously?) write master metadata receipts to disk on data nodes, in order to provide integrity in case there is only a single master node (so that master meta may still be recovered).
# 
# * Cluster updates are ensured to be well-ordered through true total ordering: e.g. many threads may send cluster updates to the master, but master will linearize these cluste updates in a single sequential queue. Personal comment: this is a pretty serious bandwidth restriction!
# 
# * 2PC (two phase commit) is used to garauntee that state updates from master are rollback-safe and recoverable, e.g. in case of a network partition. First, master sends the cluster state transcript to all nodes, and waits for at least a majority quorum to ACK. It only then sends an execute command.
# * Thus each state update requires two majority-quorum network round trips.
# * 2PC of course is simple but flawed, as a consistency garuantee goes...the simplest case where this fails is that nodes queue the update in memory before sending an ACK, so if they crash the update instruction is lost. There are other race conditions also.
# 
# ## Current Issues
# * The ES index is divided into shards. Each shard has multiple copies: one of these is designated the primary, and the rest are replicas. Replicas are backed up from the primary for write requests, but both the main shard and the replicas accept read requests.
# * Shard writes are concurrently synchronized to the other replica shards (cosistency over availability), of which a certain minimum number must ACK before the primary shard returns an OK to the client.
# * The primary is configured with a minimum number of shards it must be able to replicate to before it can start writing. This pre-write check can be used to induce additional consistency, but of course shards can fail during the write process so it's not a true garuantee.
# * What's interesting is that the primary blocks until _all_ nodes explicitly success or fail. An asychronous model was used in the past, but had the issues you'd expect it to have (data loss), so they swapped to a strongly consistent sharding model.
# 
# * There is weak read-your-own-write consistency. If a node fails to replicate, this is reported by the primary shard node to the master. The master then removes that node from the replica list (potentially replacing it with another node), and sends this updated status as a new cluster state message. Once that message reaches the node that a user is trying to read on, that node knows not to use the legacy shard. However, until then, the user is still able to read legacy data. This is nevertheless weakly consistent because normal use of ES requires a page refresh anyway. In normal operations, the state update will _probably_ outrace the network request...though this can change if the system is stressed.
# 
# * ES makes the unusual decision to write to Lucene in-memory _before_ writing to the transaction log. This is because messages may fail validation in Lucene, which would cause inconsistency in the transaction log, which now requires a corrective log item or a delete op, both of which suck in what's supposed to be an append-only log.
# * Hence the Lucene write happens first. This of course creates a race condition: the data may end up in Lucene and not in the transcation log due to a system crash. This is recoverable however.
# 
# ## Et al
# * The rest of the blog post highlights PacificA, the algorithm ES bases its shard replication algorithms on.
# * For this, it's best to read the PacificA paper.
