#!/usr/bin/env python
# coding: utf-8

# * **Data analytics** functions at Uber include (1) building dashboards to monitor business metrics (2) making automated decisions (such as trip pricing and fraud detection) based on aggregated metrics (3) making ad hoc queries to diagnose and troubleshoot business operations issues.
# * These issues are all distinct from **machine learning** and model management (recall the data science hierarchy of needs)
# * Dashboards and decision support systems rely on **real-time analytics systems** to do their work.
# * Time series aggregates are by far the most common dimensionality for analytics use.
# 
# * Some keywords used that I am not 100% familiar with follow:
# * **Key-based deduplication** &mdash; enforcing strict per-key single-storage (e.g. the database accepts the same record N times, but only stores it once).
# * **Upsert** &mdash; (1) inserting a new record or (2) updating an existing record. Should be atomic, e.g. without concurrency issues.
# * **Inverted index** &mdash; a map of values to document locations. Used in full-text search engines, since it makes lookups on a specific word $O(1)$.
#     * Inverted indices allow filtering but are not optimized for time-series search.
# 
# * AresDB is columnar for time-series query efficiency and storage efficiency.
# * Strings are stored as enums, though this is a design decision they'd like to overcome.
# * AresDB has a metadata store which describes checkpoints and a DDL schema, which is on disk.
# * Backups are also handled via disk.
# * Most data is stored in "host memory", e.g. RAM. The data is serialized to the GPU for parallel processing _at query time_ (interesting decision).
# * Data is stored in terms of fact tables, which are infinite timestamped append-mainly list, and dimension tables describing formats.
# 
# * A "live store" stores uncompressed and unsorted columnar vectors, partitioned into "live batches" of configured capacity. New batches are created at ingestoon, while old batches are purged after their records are archived.
# * A primary key index is used for deduplication and update lookups. The primary key indexes both the batch and the position within the batch. All types are fixed-width, so that can be immediately resolved to a memory address.
# * Nullity is stored as a (one byte per field) mask vector.
# 
# * An archival store handles persisting data via fact tables. Archive batches are ID-ed (and organized) with the Unix epoch timestamp for the particular day, e.g. they are daily.
# * Vectors in archival storage are sorted according to a user-configured ordered list of fields.
# * Vectors are run-length compressed, but only low-cardinality columns are encoded, as the compression effect for high-cardinality columns would be negligible.
# * This means storing an additional count vector as well as the original value value and null vectors.
# 
# * Clients ingest data via an API that requires POST-ing an upsert batch. The message format used is a custom serialized bunary format (why not use one of any of the existing messaging protocols? Weird).
# * Step-by-step:
#    1. The upsert batch is written to a log for recovery purposes.
#    2. Timestamps are checked; late records (records which have event times older than the archived cut-off event time) are sent to a backfill queue.
#    3. The primary key index is used to look up batches in live storage that non-late facts should be applied to (incuding partially empty batches or empy batches for append operations for new facts), and the updates/inserts are performed.
# * This operation is atomic, e.g. fully consistent.
# * This operation makes note of **late arriving facts**: facts which update values which are unexpectantly past. This is a data warehouse indexing concern; e.g. if you have _ever_ reset the index you lose the abiliy to naively modify record values to account for late arrivals.
# 
# * Archival is an occassional process which is applied to all data in live memory that is within a certain time range.
# * Note that this introduces an archiving delay into the architecture, e.g. we do not completely clear the in-memory live data structure on archival. Although you can certainly configure things that way, if you'd like.
# 
# * Backfill is performed from a backfill queue by a backfill process that is configurable to be time and/or size thresholded (Uber does both).
# * Backfilling is asychronous, unlike ingestion by the live store (which is synchronous in order to be upsert atomic), and relatively more expensive computationally.
# * During massive backfill loads, the client will be blocked from normal operations in order to allow the backfill run to clear.
# * Backfilling is **idempotent**: making the same call multiple times will create the same result. This is in contrast to regular upsert workloads, which are non-idempotent.
# 
# * Uses AQL, a non SQL JSON fragment based query language mean to be more easily programmatically editable than SQL.
# * Queries are transformed into an AST of operations on the database side.
# * A pre-filter is applied before send to GPU for cheap filtering operations, namely on sorted columns.
# * Two CUDA streams are used to parallalize data streaming and computation and thus speed up operations.
# * The Thrust library from nVidia is used to manage computation.
# * Each GPU thread given a workload position within the input vector, which it handles computations for and writes to the corresponding position in the output vector.
# * The operations are performed using what Uber describes as a one operator per kernel model. Can't find too much info on this online but basically what you do is:
#   1. Linearize the AST into an iterator of operators.
#   2. Allocate stratch space vectors for non-leaf non-root nodes.
#   3. Operators are launched by Thrust on the GPU.
#   4. The root node, once evaluated, emits the final results.
# 
# * AresDB is a fully in-memory database (besides recovery logs obviously).
# * It manages a set of memory buffers for various tasks, like e.g. the backfill queue, which can be user-configured to be of a certain size.
# * There is a user-configurable ability to expand previously archived data into memory, if doing so would meaningfully speed up the detected workloads and memory space is still available on the machine.
# * Naturally there is an eviction policy for sending in-memory data back to archive.
# * GPUs are managed via a device manager that handles job scheduling. GPUs are treated on two dimensions: number of threads and amount of device memory.
# * There is no caching strategy on GPU memory, but they'd like to add it one day. Difficult due to dynamic nature of queries.
# * AresDB is non-distributed, for now.
