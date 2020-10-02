#!/usr/bin/env python
# coding: utf-8

# These notes from ["Arrow - memory layout"](https://arrow.apache.org/docs/memory_layout.html).
# 
# ### First some high-level design goals
# 
# * Arrow defines a physical memory layout intended for low-overhead data interchange amongst different systems and protocols that handle (1) flat columnar data (a la columnar databases) and/or (2) nested columnar data (e.g. tree data a la document stores).
# * It is intended to provide `O(1)` access to top-level slots, with linear access behavior (`O(n)`) for increasingly nested fields.
# * It is designed to be fully parquet serializable/deserializable.
# * It uses a static message block size (word size?) of 8 bytes. So all buffers are a multiple of 8 in size. The general recommendation is to use a 64-byte boundary.
# * Arrays are immutable once built. So append operations require a rebuild, which is slow. Remember this is designed to be an in-memory data transfer format!
# * Null slots are supported.
# 
# 
# * The last design goal is more involved to understand:
# 
# > Arrays are relocatable (e.g. for RPC/transient storage) without pointer swizzling. Another way of putting this is that contiguous memory regions can be migrated to a different address space (e.g. via a memcpy-type of operation) without altering their contents.
# 
# * **Pointer swizzling** is a memory management implementation detail.
# * A piece of memory that implements self-referential pointers cannot be relocated ("memcpy-ed") to another memory address immediately, because the pointer will still point to the old location whilst the actual data will be in the new.
# * This technique deals with this problem by detecting absolute memory pointers, turning them into offset-based pointers, performing the move, and then desugaring the pointers again.
# * Why have absolute pointers instead of relative ones? Because they are more efficient to scan to, as you can random-access an absolute in-memory pointer in `O(1)` time. But for an offset pointer you will have to perform at least two reads, maybe more: one to get the offset info, and one to jump to the required offset position.
# * Why support memcopy? It's a useful convenience, and importantly necessary for efficient garbage collection, as it enables "moving GC" -- having a garbage collector pack in-memory objects, thus defragmenting RAM and enabling more allocations overall.
# * An alternative would be to not support internal references, e.g. to only support relative references. This seems to be what Arrow does.
# 
# 
# * What does it mean to have buffers with 8 byte boundaries? Well, in practical terms it means that all buffers start at a memory address divisible by 8, and are padded to a length that is a multiple of 8. This alignment achieves a number of important effects. It allows elements in numeric arrays to be retrieved in one read operation, since you know the address of the struct and the offset of the memory entry and can just jump to the right place in memory. It can also partially used cache lines: cache lines are a power of two in size, so it's more likely that your memory access will align with the starts and ends of cache lines, preventing the need for additional memory accesses.
# * Why use 64 byte alignment, as recommended by Arrow? It's what Intel recommends for data structures over 64 bytes in size. They say this because it is the largest SIMD instruction registry size (at time of writing of the Arrow spec, in 2016), so it's the largest byte boundary size that is parallelizable via the on-chip SIMD instruction sets (64 bytes == 512 bits). The largest size possible is appropriate because most data in Arrow is expected to be large.
# 
# 
# ### Types
# 
# * Arrays have a known fixed max length of 2^32 - 1 (a signed int). The signed part is for compatibility with Java, which does it this way also. Arrays are the primitive C style array: non-fixed length, but fixed type (may itself be Array) and fixed slot size.
# * The null count is also stored as part of the array structure. Since it can be as large as the array, it is also a 2^31 - 1 max-size signed int32.
# * There is a nullity bitmap. Fully non-null arrays may optionally choose not to allocate it.
# * Arrays may have child arrays. The child arrays have their own null bitmaps and other array assets which are independent of the parent.
# 
# 
# * There is a list type that allows flexibly sizes sets of things of one type. An Arrow list is not like a Python list, every item in the list must be the same known type, but those items may not be a fixed length.
# * A list has two parts. One part is a values array, another part is the offsets array. The offets array encodes start positions for every entry in a signed in32. The first element and last element of the offsets array is 0 and the length of the array, respectively.
# 
# 
# * Structs are implemented as dual lists and/or arrays. One list or array stores the key values (with e.g. a `List<Char>` type for strings), one list or array stores the values (with e.g. an `Array<int32>` type).
# 
# 
# * There are two union types: dense unions, and sparse unions. These union types extend support for multiple-typed lists. E.g. it's not possible to have a `List<{char, int32}>`, e.g. a list which contains some entries which are characters and some entries which are integers. But it is possible to have a `DenseUnion<{char, int32}>`. A `DenseUnion` is essentially a Python list.
# * A dense union achieves this by, again, backing off to the primitives. A DenseUnion has an offsets buffer just like `List`, an integer-based types buffer of 8-bit signed integers (limiting the total number of types in the union to 127), and an offsets buffer describing the offset for the corresponding element in the list into the coresponding offset buffer.
# * The high level interface equivalent of a Python list is a `DenseUnion` of structs.
# * There is also a `SparseUnion` which omits the offsets array (advanced use case, sparse and memory-wasting but convenient for...something).
# 
# 
# * Finally, there is support for dictionary encoding:
# 
# ```
# type: List<String>
# 
# [
#  ['a', 'b'],
#  ['a', 'b'],
#  ['a', 'b'],
#  ['c', 'd', 'e'],
#  ['c', 'd', 'e'],
#  ['c', 'd', 'e'],
#  ['c', 'd', 'e'],
#  ['a', 'b']
# ]
# ```
# 
# Becomes:
# 
# ```
# data List<String> (dictionary-encoded, dictionary id i)
# indices: [0, 0, 0, 1, 1, 1, 0]
# 
# dictionary i
# 
# type: List<String>
# 
# [
#  ['a', 'b'],
#  ['c', 'd', 'e'],
# ]
# ```
# 
# * However this connection is handled at the metadata level? Unclear.
# 
# 
# ### Metadata
# * Metadata is written and read using `google/flatbuffers`.
# * Metadata is meant to communicate three things:
#   * Logical array types.
#   * Schemas for table-like Arrow collections
#   * Data headers indicating the physical locations of memory buffers.
# * There is a `Schema` message type that defines metadata and interface over any number of physical arrays. A schema consists of a sequence of `Fields`:
# 
# ```
# table Field {
#   // Name is not required, in i.e. a List
#   name: string;
#   nullable: bool;
#   type: Type;
# 
#   // Present only if the field is dictionary encoded
#   dictionary: DictionaryEncoding;
# 
#   // children apply only to Nested data types like Struct, List and Union
#   children: [Field];
# 
#   // User-defined metadata
#   custom_metadata: [ KeyValue ];
# }
# ```
# 
# * The `type` is the logical type of the field (as opposed to the physical types). Nested types, such as List, Struct, and Union, have a sequence of child fields.
# * So `table Schema { [...list of Field objects...] }
# * Notice that this is where the dictionary encoding flag is set.
# 
# 
# ### IPC
# * IPC is handled via a file with the following layout:
# 
# ```
# <metadata_size: int32>
# <metadata_flatbuffer: bytes>
# <padding>
# <message body>
# ```
# 
# * Write to `parquet`?
# * Use `Flight`?
# 
# 
# ### Pyarrow, memory management and I/O
# * The `Buffer` object is the primary tool for memory management in Apache Arrow. It enables interaction with both owned and shared memory.
# * A `Buffer` can be created from any object by calling `pa.py_buffer(data)`, e.g. `pa.py_buffer(b'abc')`.
# * The resulting object is a zero-copy view on the bytes of the target.
# * Exernal memory can be referenced via `pa.foreign_buffer(...)`.
# * Use `buf.to_pybytes()` to convert to a bytestring, which copies data.
# * A glocal memory allocation number is maintained and accessible. Memory may be allocated for any of a number of pools, and memory which is allocated but not used is garbage collected in the standard Python way.
# * Buffers can be made resizable via e.g. `buf.resize(2048)`.
# 
# 
# * Messages may be read as files via any of the subclasses of `NativeFile`, which implements the same interface as the standard Python file class.
# * There are a lot of options: `OSFile`, `MemoryMappedFile` (memory-mapped only), `BufferReader` (for reading buffers as files), `BufferOutputStream` (for writing data into memory using the `Buffer` interface), `FixedSizeBufferWriter` (for writing to already-allocated buffers), `HdfsFile` (for HDFS files, e.g. parquet), `PythonFile` for Python file handlers, and `CompressedInputStream` and `CompressedOutputStream`.
# * Do `pa.input_steam(<str>)` and `pa.output_stream(<str>)` to do reading and writing.
# * To read from a file, do ops on a `pa.OSFile` or `pa.memory_map`. The difference between the two is that the former allocates memory on read whilst the latter does not (e.g. copy versus view).
# 
# 
# ### Pyarrow, data types and the in-memory data model
# * Buffers are composed with data structures that expose the semantics of Arrow storage.
# 
# 
# * Type metadata is instanced by `pyarrow.DataType` objects.
# * Arrow logical types (which are different from physical types) can be instantiated using wrapper classes, e.g. `pa.int32()` or `pa.timestamp('ms')`. Logical type-ness is a metadata annotation; the underlying data often has the same physical type, e.g. `int64`.
# 
# 
# * Fields can be built using a `Field` type: `pa.field('int32_field', pa.int32())`. Note the optional metadata component: the name of the field.
# * Lists can be built e.g. like so: `t6 = pa.list_(t1)`.
# * Structs collect fields: `pa.struct([pa.field('s0', pa.int32()), pa.field('s1', pa.int32()), ...])`. You can also omit `field` here, for convenience.
# * Schemas do the same: `pa.schema([('field1', t1), ...])`
# * The array interface is very numpy-like. You can have it do type interference for you via `pa.array([1, 2, None, 3])`, or specify it explicitly via `pa.array([1, 2, None, 3], type=pa.uint16())`. Of course `len(arr)` and `arr.null_count` are $O(1)$ lookups, thanks to the `arrow` bookkeeping discussed above. An array is immutable once defined, and you can do zero-copy slices (view) in the normal Python way: `arr[1:3]`.
# * To init an array of lists or structs, you need to pass the type, as they can't be inferred. `pa.array([{'x': 1, 'y': True}, {'x': 2, 'y': False}], type=pa.struct([('x', pa.int8())])`.
# 
# 
# * Union arrays. First, sparse unions:
# 
# ```python
# # the 0-position fixed-typed array, which must be the same length as the caller
# xs = pa.array([5,6,7])
# # the 1-position fixed-type array, same length restriction
# ys = pa.array([False, False, True])
# # the types array, which tells the UnionArray which child array to draw from by indx
# types = pa.array([0, 1, 1], type=pa.int8())
# # construct the array
# union_arr = pa.UnionArray.from_sparse(types, [xs, ys])
# # this will have the type Union<0: int, 1: bool>.
# # the actual contents are [5, False, True]
# # notice how we are using 6 records to store 3 values...that is the "Sparse" part, and its the opposite of what you think :)
# ```
# 
# * If you want to specify a `DenseUnion`, you have to do a bit more work in the definition. Specifically, you additionally have to specify an `int32` array which states which offset in the child array contains the value you want.
# 
# ```python
# xs = pa.array([5, 6, 7])
# ys = pa.array([False, True])
# types = pa.array([0, 1, 1, 0, 0], type=pa.int8())
# offsets = pa.array([0, 0, 1, 1, 2], type=pa.int32())
# union_arr = pa.UnionArray.from_dense(types, offsets, [xs, ys])
# ```
# 
# * Dictionary encoded data is provided via a `Dictionary`.
# 
# ```python
# indices = pa.array([0, 1, 0, 1, 2, 0, None, 2])
# dictionary = pa.array(['foo', 'bar', 'baz'])
# dict_array = pa.DictionaryArray.from_arrays(indices, dictionary)
# ```
# 
# * A `RecordBatch` is a collection of equal length arrays.
# 
# ```python
# data = [
#      pa.array([1, 2, 3, 4]),
#      pa.array(['foo', 'bar', 'baz', None]),
#      pa.array([True, None, False, True])
#        ]
# batch = pa.RecordBatch.from_arrays(data, ['f0', 'f1', 'f2'])
# ```
# 
# * Finally there is a `Table` class. This is a Python-level convenience and not part of the Arrow spec. It is used for organizing data.
# 
# ```python
# batches = [batch] * 5
# table = pa.Table.from_batches(batches)
# ```
# 
# ### Pyarrow, serialization and deserialization
# * OK, let's bring `Buffer` and the various data types together and see how they interact when reading from and writing to files.
# 
# ```python
# sink = pa.BufferOutputStream()
# writer = pa.RecordBatchStreamWriter(sink, batch.schema)
# 
# # later...
# restored_data = pa.deserialize(buf)
# ```
# 
# * See the reference for more information.
# 
# ### Pyarrow, parquet
# * See the docs.
# 
# ### Pyarrow, CUDA integration
# * Via `numba`! Interestingly enough.
# 
# ### Wes's comments
# * From http://wesmckinney.com/blog/apache-arrow-pandas-internals/, http://wesmckinney.com/blog/arrow-columnar-abadi/, http://wesmckinney.com/blog/python-parquet-update/
# 
# 
# * The 10 things that `pandas` doesn't do quite right.
# 1. Internals too far from "the metal"
#    * Wes points to the Python string implementation in particular. `pandas` (`numpy`?) simply wraps Python native strings in a container. Those strings are allocated in the Python heap, which does not provide locality, e.g. the objects are placed all over memory, resulting in many disk seeks and slow performance.
#    * Arrow makes columnar string storage properly contiguous.
# 2. No support for memory-mapped datasets
#    * `pandas` must load data completely into memory to be processed.
#    * The `pandas` `BlockManager` is too complex to be used with memory-mapped files.
#    * A **memory-mapped file** is a disk resource that has been loaded into virtual memory as a file. When, as in the usual case, that virtual memory is the operating system's page cache, which is on RAM, that results in the typical RAM-over-disk speedup. The program can now operate on the fast in-memory copy of the bytes, which become consistent with the bytes on disk again at save time.
#    * If a file is huge and doesn't fit into RAM, or if there is contention for RAM resources, memory-mapped files can still useful because they can just cache the most commonly accessed pages in the file, while falling back to loading other resources from disk.
#    * Note that if the file is very small, memory mapping is potentially wasteful, as page sizes are fixed (to 64K?) and underfull pages mean less memory allocation overall.
#    * In `pandas` you are forced to convert into an in-memory format every time you create a `DataFrame`, and retain that copy in memory. In Arrow you can memory-map a file into memory, skipping the copy operation completely, and gaining the ability to partial-load huge datasets into memory.
#    * This means that you can, for example, read 1 MB from the middle of 1 TB of memory successfully, using the `Arrow` API (and not some file scanning API) to do so.
#    * Note that since this is using the page cache and zero-copy, the format of the store is relevant. A compressed storage format like `parquet` will we paged without changes, leading to efficient allocation but less efficient access (as decompression with be required before read-out). `plasma` and `feather` are two examples of formats, both well-integrated into `arrow`, which (among other things?) attempt to ameliorate the delta.
# 3. Poor performance in database and file ingest / export
#    * Again because of the copy and modify operations required. Every time you exported a `pandas` `DataFrame` to a database or a file you have to read the entire set of records, translate them into the canonical format, and write that. Every time you import records, the opposite occurs.
#    * Arrow addresses this use case in part with an inbuilt concept of a record batch stream, which models a database pointer streaming out records.
#    * Arrow also just has a memory format which is more efficiently laid out, making it a better and more efficient match for the deserialization operations that must be performed when you read a file into an Arrow representation in memory.
# 4. The missing data problem
#    * `numpy` does not natively support a missing data type, which is a ubiquitious need. `pandas` invented its own type, which leads to a lot of typing problems and API complexity (which I couldn't figure out when I dove into the codebase...).
#    * Arrow supports missing data using a true bit mask, which is probably the "correct" way of doing things.
# 5. Memory allocation blow-ups
#    * `pandas` does not actually own its memory, it leaves that to Python and to `numpy`, which means that certain operations can cause memory blow-ups in hard-to-track and hard-to-know ways.
#    * Arrow owns its own memory.
#    * Furthermore, Arrow has a concept of a **memory pool**, and of subpools associated with that memory pool. Objects are assigned to and tracked using pools, so memory footprint is well understood. This allows you to concretely track algorithm memory usage.
#    * All memory in Arrow is either immutable or copy-on-write (memory mapping).
#    * This obviates the need for **defensive copying**: copying memory to new memory when performing operations to avoid causing interop problems. Which is something `pandas` has to do for memory interop.
# 6. Categoricals
#    * Arrow has first-class categorical support via dictionary encoding.
#    * Pandas has this too now though, but it's less interop because it was added much later, it's mainly just a `pandas` thing.
# 7. Groupby-apply not parallelized
#    * `pandas` could not parallelize the common `groupby("...").apply(f)` pattern, due to the combination of the issues in this list.
#    * Arrow makes this parallelization easy. It is more natively capable of this pattern.
# 8. Append performance
#    * In `pandas` all data must reside in the same `numpy` array, which leads to problems.
#    * Table columns in Arrow can be chunked. Appending to a chunked table is a zero-copy operation, because you are always appending to the last incomplete chunk. The chunk is not written to the store until full, then the in-memory object is cleared and re-reading for more write-in. This means there is no new memory allocation involved.
#    * Table columns in Arrow can also be streamed (see the "database pointer" idea earlier in this document). Appending to a stream is a similarly efficient, you use the same chunk buffer.
#    * Appending in `pandas` is slow because it might e.g. change the types of columns, which leads to reallocation and badness.
# 9. New data types
#    * Arrow comes with good out-of-the-box support for constructing new logical data types out of its physical ones (you do this to build a higher level of API).
# 10. Query planning
#    * `pandas` has no query planning, so expressions like `a[a.b > 5].sum()` require create a new temporary copy of the `b` column of `a`. The `query` DSL is faster and better here specifically because it can do query planning whereas this expression cannot!
#    * Arrow is supposed to have in-built query planning support. It uses Dask for this.
#    
# ### Comments on compression formats
# * There are compression formats that work and matter for an in-memory data format. Arrow supports some limited compression, but it needs to support more to match the performance of mature systems like e.g. Vertica. This is one of the commentaries from this blog post: http://dbmsmusings.blogspot.com/2017/10/apache-arrow-vs-parquet-and-orc-do-we.html.
#    
# ### Apache Arrow Flight design
# * From https://www.slideshare.net/JacquesNadeau5/apache-arrow-flight-overview
# * Arrow Flight is in-memory. But not all processes can be located, and in fact modern microservice architectures built on e.g. Kubernetes are very distributed. These applications communication in RPCs, so for these architectures it's important that Arrow have some kind of RPC support.
# * You can use an existing RPC format like gRPC, but it's not tuned for performance on and API semantics for working with data.
# * Arrow Flight provides a batch streaming API that piggybacks on gRPC. It has PUT and GET, both initialited by the client.
# * A **flight** is composed of streams. Each stream has a `FlightEndpoint`, which is an opaque stream ticket and some location metadata. The system can use that location metadata to improve performance with e.g. multiplexing.
# * The rest is pretty unreadable.
# * Is this all we have to go off of? Really? Christ.
