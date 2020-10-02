#!/usr/bin/env python
# coding: utf-8

# ## Notes on Apache Parquet
# The following is a transcription of what I gleaned from [their "Documentation" page](https://parquet.apache.org/documentation/latest/).
# 
# ### Overview
# `parquet` is a modern columnar data serialization format that was originally created for use in the Hadoop ecosystem. It is an open source implementation of a scheme originated at Google and made public in the [Dremel paper](https://ai.google/research/pubs/pub36632).  It supports complex nested data types, efficient compression and encoding schemes, and backwards and forward reverse compatibility using the Apache Thrift message protocol.
# 
# ### Algorithm
# Parquet files are defined using a record shredding and assembly algorithm.
# 
# First a word on the format. Parquet file defintion include *groups* and *fields*. Groups are repeatable named pointers that point deeper into the stack. *Fields* are key-value pairs that hinge off of a group. The root group is the document root. Fields and groups alike can be required, optional, or repeated. So e.g. here is an example `parquet` file serialization format:
# 
# ```
# message Document {
#   required int64 DocId;
#   optional group Links {
#     repeated int64 Backward;
#     repeated int64 Forward; }
#   repeated group Name {
#     repeated group Language {
#       required string Code;
#       optional string Country; }
#     optional string Url; }}
# ```
# 
# Here's the corresponding document example:
# 
# ```
# DocId: 10
# Links
#   Forward: 20
#   Forward: 40
#   Forward: 60
# Name
#   Language
#     Code: 'en-us'
#     Country: 'us'
#   Language
#     Code: 'en'
#   Url: 'http://A'
# Name
#   Url: 'http://B'
# Name
#   Language
#     Code: 'en-gb'
#     Country: 'gb'
# ```
# 
# Serializing a record is a depth-first tree traversal. Whenever you reach a non-zero leaf, record to the column store the value, the repetition level (initially zero, but this number will grow if there are multiple instances of this field in the record tree), and the definition level. When traversal bottoms out an an empty field, no value is written, but the definition level for the last defined level for all leaf nodes that could appear below the node does get written.
# 
# The vaue and repetition level are obvious, but the definition level is more nuanced. The insight is that in a tree following a schema, the non-presence of a parent field immediately implies the non-presence of any and all child fields as well. For example:
# 
# ```python
# {a: null}  # definition level 0
# {a: {b: null}}  # definition level 1
# {a: {b: {c: "foo"}  # definition level 2
# ```
# 
# When `parquet` serializes a document fragment, as a natural consequence of its depth-first traversal order it stores the maximum depth of each group alongside that group's name. When `parquet` deserializes a column of data (corresponding with a field), it sequences through the groups in depth-first order. Any parent groups that have maximum definition levels that are too low for the desired columnar value to appear do not need to be parsed any further and can be skipped in the iteration order.
# 
# This allows `parquet` to skip the bulk of the tree parsing operations when the column being requested is sparse (as would be typical for some Hadoop workflows). 
# 
# Note that required fields are obviously non-sparse, and so this machinery does not kick in. Similarly, repetition levels are only necessary for fields which may be repeated, as they inform the construction algorithm when to stop looking for more values with the same field.
# 
# ### Serialization format
# `parquet` serializes to file using the following top-down scheme:
# 
# ```
# 4-byte magic number "PAR1"
# <Column 1 Chunk 1 + Column Metadata>
# <Column 2 Chunk 1 + Column Metadata>
# ...
# <Column N Chunk 1 + Column Metadata>
# <Column 1 Chunk 2 + Column Metadata>
# <Column 2 Chunk 2 + Column Metadata>
# ...
# <Column N Chunk 2 + Column Metadata>
# ...
# <Column 1 Chunk M + Column Metadata>
# <Column 2 Chunk M + Column Metadata>
# ...
# <Column N Chunk M + Column Metadata>
# File Metadata
# 4-byte length in bytes of file metadata
# 4-byte magic number "PAR1"
# ```
# 
# The `PAR1` sequence is a [file signature](https://en.wikipedia.org/wiki/List_of_file_signatures). The data is divided into individual columns and groups of rows (row chunks), which major ordering on the chunks (e.g. group row-major ordering) and minor ordering on the columns. Data from the same chunk is colocated on the file, e.g. file access against chunks is disk seek free.
# 
# The file metadata includes the location of all metadata start locations, making row chunk access by offset efficient (requires a disk seek and not a file scan). It is included at the tail end of the file in order to allow single-pass writing. The expected reader behavior is to access the file metadata first, determine desired row chunks, and then access those chunks sequentially.
# 
# In a distributed setting data of interest may be partitioned into multiple `parquet` files in storage, and the read process may be parallelized via an external coordinator (e.g. a Hive metastore).
# 
# Here is the detailed format diagram:
# 
# ![](https://i.imgur.com/rf4RTgd.png)
# 
# Notes:
# * The file as a whole is a sequence of Thrift n+1 messages, not a Thrift message in and of itself.
# * Repetition levels (columnar value repetition counts) and definition levels (columnar null or not-null) are run-length encoded. Notice that nulls are not encoded in the data values.
# * The footer contains a lot of metadata partitioned by row group and includes optional key-value metadata.
# * The footer includes its length at the bottom, again to allow disk seek.
# 
# ### Types
# Physical types are:
# 
# * BOOLEAN: 1 bit boolean
# * INT32: 32 bit signed ints
# * INT64: 64 bit signed ints
# * INT96: 96 bit signed ints
# * FLOAT: IEEE 32-bit floating point values
# * DOUBLE: IEEE 64-bit floating point values
# * BYTE_ARRAY: arbitrarily long byte arrays.
# 
# `parquet` provides a selection of logical types which map on top of these: https://github.com/apache/parquet-format/blob/master/LogicalTypes.md.
