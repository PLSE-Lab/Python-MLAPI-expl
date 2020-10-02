#!/usr/bin/env python
# coding: utf-8

# # Building High Performance Data Pipelines with tf.Data and Google Cloud Storage
# 
# This article goes through the steps of building a high performance data input pipeline using Tensorflow and Google Cloud Storage.
# The concepts and techniques are evolved at each step, going from the slowest to the fastest solution.
# 
# This article uses the Stanford Dogs Dataset [1] with ~20000 images and 120 classes.
# 
# [1] https://www.kaggle.com/jessicali9530/stanford-dogs-dataset

# ## Benchmark function
# 
# The benchmark will measure the number of images ingested (read) per second from Cloud Storage to the host virtual machine. 
# There are several ways to implement this calculation, but a simple function was used to iterate through the dataset and measure the time.
# 
# The following python function ('timeit' function) from Tensorflow documentation [1] (as of 03/18/2020 - version 2.1) is used.
# Since tf.data.Dataset implements \__iter__, it is possible to iterate on this data to observe the progression.
# 
# [1] https://www.tensorflow.org/tutorials/load_data/images#performance

# In[ ]:


# First let's import Tensorflow
import tensorflow as tf


# In[ ]:


# Now import some additional libraries
from numpy import zeros
import numpy as np
from datetime import datetime


# In[ ]:


# Benchmark function for dataset
import time
default_timeit_steps = 1000
BATCH_SIZE = 1

# Iterate through each element of a dataset. An element is a pair 
# of image and label.
def timeit(ds: tf.data.TFRecordDataset, steps: int = default_timeit_steps, 
           batch_size: int = BATCH_SIZE) -> None:
    
    start = time.time()
    it = iter(ds)
    
    for i in range(steps):
        batch = next(it)
        
        if i%10 == 0:
            print('.',end='')
    print()
    end = time.time()
    
    duration = end-start
    print("{} batches: {} s".format(steps, duration))
    print("{:0.5f} Images/s".format(batch_size*steps/duration))


# ## Let's create the Dataset using tf.data - Reading images individually
# 
# All the images are located in a bucket in Google Cloud Storage (example: gs://cloud_bucket/label/image.jpeg).
# Labels are extracted (parsed) from the image name.
# 
# In this first step, the dataset is created from the images file paths (gs://...), and labels are extracted and one-hot encoded.
# 
# This dataset maps each image in the bucket individually.

# In[ ]:


# Global variables

# Paths where images are located
FILENAMES = 'gs://tf-data-pipeline/*/*.jpg'

# Paths where labels can be parsed
FOLDERS = 'gs://tf-data-pipeline/*'

# Image resolution and shape
RESOLUTION = (224,224)
IMG_SHAPE=(224,224,3)

# tf.data AUTOTUNE
AUTOTUNE = tf.data.experimental.AUTOTUNE


# In[ ]:


# Get labels from folder's name and create a map to an ID
def get_label_map(path: str) -> (dict, dict):
    #list folders in this path
    folders_name = tf.io.gfile.glob(path)

    labels = []
    for folder in folders_name:
        labels.append(folder.split(sep='/')[-1])

    # Generate a Label Map and Interted Label Map
    label_map = {labels[i]:i for i in range(len(labels))}
    inv_label_map = {i:labels[i] for i in range(len(labels))}
    
    return label_map, inv_label_map


# In[ ]:


# One hot encode the image's labels
def one_hot_encode(label_map: dict, filepath: list) -> dict:
    labels = dict()
    
    for i in range(len(filepath)):
        encoding = zeros(len(label_map), dtype='uint8')
        encoding[label_map[filepath[i].split(sep='/')[-2]]] = 1
        
        labels.update({filepath[i]:list(encoding)})
    
    return labels


# In[ ]:


label_map, inv_label_map = get_label_map(FOLDERS)


# In[ ]:


list(label_map.items())[:5]


# In[ ]:


# List all files in bucket
filepath = tf.io.gfile.glob(FILENAMES)
NUM_TOTAL_IMAGES = len(filepath)


# In[ ]:


# Split the features (image path) from labels
dataset = one_hot_encode(label_map, filepath)
dataset = [[k,v] for k,v in dataset.items()]

features = [i[0] for i in dataset]
labels = [i[1] for i in dataset]


# In[ ]:


# Create Dataset from Features and Labels
dataset = tf.data.Dataset.from_tensor_slices((features, labels))


# In[ ]:


# Example of one element of the dataset
# At this point we have a dataset containing the path and labels of an image
print(next(iter(dataset)))


# Next we define some preprocessing functions to:
#  - Read the data from Cloud Storage
#  - Decode JPEG
#  - Convert image to a range between 0 and 1, as float
#  - Resize image

# In[ ]:


# Download image bytes from Cloud Storage
def get_bytes_label(filepath, label):
    raw_bytes = tf.io.read_file(filepath)
    return raw_bytes, label


# In[ ]:


# Preprocess Image
def process_image(raw_bytes, label):
    image = tf.io.decode_jpeg(raw_bytes, channels=3)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.image.resize(image, (224,224))
    
    return image, label


# #### Building the dataset
# 
# From the Dataset already built with image paths and labels, the preprocessing functions are applied for each element to download the bytes from Cloud Storage and apply some transformations to the images.
# These steps are only performed when the dataset is iterated.
# 
# At this point, all the steps are executed while streaming the data, including:
#  - IO intensive operations like download de images (get_bytes_label)
#  - CPU intensive operations like decode and resize the image (process_image)
# 
# Some observations for the code below:
#  - "num_parallel_calls = tf.data.experimental.AUTOTUNE" was used to let tensorflow runtime decide the best parametrization for its functions.
#  - "dataset.cache" was implemented, but as we are reading a large amount of data, this may not fit into memory and become impossible to use.
#  - "dataset.prefetch" allows buffering of elements in order to increase performance.

# In[ ]:


# Map transformations for each element inside the dataset
# Maps are separated as IO Intensive and CPU Intensive
def build_dataset(dataset, batch_size=BATCH_SIZE, cache=False):
    
    dataset = dataset.shuffle(NUM_TOTAL_IMAGES)
    
    # Extraction: IO Intensive
    dataset = dataset.map(get_bytes_label, num_parallel_calls=AUTOTUNE)

    # Transformation: CPU Intensive
    dataset = dataset.map(process_image, num_parallel_calls=AUTOTUNE)
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size=batch_size)
    
    if cache:
        if isinstance(cache, str):
            dataset = dataset.cache(filename=cache)
        else:
            dataset = dataset.cache()
    
    # Pipeline next iteration
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    
    return dataset


# In[ ]:


# Apply transformations to the dataset with images paths and labels
train_ds = build_dataset(dataset)


# ## Benchmark baseline
# 
# Let's first create a baseline for our benchmark with a local cache to understand how fast we can go with this process.
# To do that, read a single file, cache it in memory and repeate forever.
# With this dataset, let's run our benchmark for 20000 steps.

# In[ ]:


local_ds = train_ds.take(1).cache().repeat()


# When we cache after taking an element, the preprocess won't be repeated.

# In[ ]:


timeit(local_ds, 20000, batch_size=1)


# This test achieved a pick throughput of ~10500 images per second.

# ## First Attempt, without caching
# 
# In this first benchmark no caching mecanism is used and the images are read one by one from the bucket.
# 
# The biggest problem here is to read 1000's of files one by one.
# Since there are thousands of images, this process can take longer. From the tensorflow documentation: 
# > In a real-world setting, the input data may be stored remotely (for example, GCS or HDFS). A dataset pipeline that works well when reading data locally might become bottlenecked on I/O when reading data remotely because of the following differences between local and remote storage:
# 
# >  - Time-to-first-byte: Reading the first byte of a file from remote storage can take orders of magnitude longer than from local storage.
# >  - Read throughput: While remote storage typically offers large aggregate bandwidth, reading a single file might only be able to utilize a small fraction of this bandwidth.
# 
# 
# Let's call our "timeit" function to measure the time needed for the load. 

# In[ ]:


# Iterate through this dataset for 1000 steps.
timeit(train_ds, batch_size=1, steps=1000)


# Reading files individually took a long time and is far from an ideal this throughput.

# ## Ok, let's put some local cache in action
# 
# tf.data.Dataset implements a cache function. 
# 
# If no parameter is passad to the cache function, it uses the memory of the host to cache the data.
# The problem is if your dataset is bigger than your host memory and you can't cache the epoch in memory. In this case the cache won't help and we still have an IO bottleneck.
# It is also possible to cache the images in a local storage for reuse in future epochs.
# 
# First let's test the throughput using cache in memory and than in as a local file.
# Note that we need to pass at least twice through the dataset in order to the cache to have any effect.

# In[ ]:


# Memory
train_cache_ds = build_dataset(dataset, cache=True)
timeit(train_cache_ds, batch_size=1, steps=50000)


# This execution exausted the memory of my host VM with 16GB of RAM and gave the following error.
# 
# >ResourceExhaustedError: OOM when allocating tensor with shape[688,813,3] and type float on /job:localhost/replica:0/task:0/device:CPU:0 by allocator mklcpu
# 	 [[{{node convert_image/Cast}}]]

# In[ ]:


# Local Cache File
train_local_cache_ds = build_dataset(dataset, cache='./dog.tfcache', batch_size=1)
timeit(train_local_cache_ds, batch_size=1, steps=50000)


# ### Any performance improvement?
# 
# Using the memory of the host VM as a cache mechanism, we exausted all the resources without improving the throughtput of the dataset.\
# While using a local storage we could cache all the data, but no performance gain was perceived.
# 
# To solve this problem we can follow some best practices for designing a performant TensorFlow data input pipeline (from the Tensorflow documentation [1]):
# 
#  - Use the prefetch transformation to overlap the work of a producer and consumer.
#  - Parallelize the data reading transformation using the interleave transformation.
#  - Parallelize the map transformation by setting the num_parallel_calls argument.
#  - Use the cache transformation to cache data in memory during the first epoch
#  - Vectorize user-defined functions passed in to the map transformation
#  - Reduce memory usage when applying the interleave, prefetch, and shuffle transformations.
#  
# But before we continue, let's do some tracing to understand what is going on.
# 
# I would add another factor:
#  - Bundle your data, preprocessed if possible, in TFRecord files.
# 
# [1] https://www.tensorflow.org/guide/data_performance
# 

# #### tip: Performance analysis with Tensorboard
# 
# If you want to go deeper and investigate why the performance of your benchmark if not going well, you can trace the tensorflow ops to see whats going on. \
# As we are not running a model training loop, we can start tracing individually for this operation.

# In[ ]:


tf.summary.trace_off()
tf.summary.trace_on(graph=False, profiler=True)

train_ds = build_dataset(dataset)
timeit(train_ds, steps=1000)

tf.summary.trace_export('Data Pipeline', profiler_outdir='/home/jupyter/tensorflow-data-pipeline/logs/')


# In[ ]:


# Load the TensorBoard notebook extension.
get_ipython().run_line_magic('load_ext', 'tensorboard')


# In[ ]:


# Start tensorboard inside one cell
get_ipython().run_line_magic('tensorboard', '--logdir=/home/jupyter/tensorflow-data-pipeline/logs')


# <table style="width:100%">
#   <tr>
#     <th>High Level View</th>
#     <th>Zoom View</th> 
#   </tr>
#   <tr>
#     <td><img src="https://storage.cloud.google.com/renatoleite-nb/images/trace1.png"></td>
#     <td><img src="https://storage.cloud.google.com/renatoleite-nb/images/trace2.png"></td>
#   </tr>
# </table>

# Two threads were created to read the files in parallel.
# We won't go into much details on Tensorboard, but it would be useful to analyse the time each operation took to execute.
# 
# The next step is to bundle together all the images in a TFRecord file, so let's do it.

# ## Using TF.Record for speedup de reading process
# 
# Up to now the images were read one by one, which proved to be a very inefficient process. \
# To mitigate this problem, one solution is to preprocess and write the images and labels to TFRecord files.
# 
# We can get the motivation on why creating TFRecode files with our images would be a good idea:
# 
# > To read data efficiently it can be helpful to serialize your data and store it in a set of files (100-200MB each) that can each be read linearly. This is especially true if the data is being streamed over a network. This can also be useful for caching any data-preprocessing.
# 
# > The TFRecord format is a simple format for storing a sequence of binary records.
# 
# In the following steps, the images are preprocessed and written to TFRecords.
# The following steps are followed:
#  - Read the data from Cloud Storage
#  - Decode the JPEG and resize the image
#  - Encode the JPEG
#  - Serialize the images into Bytes (tf.train.BytesList) and Labels into Ints (tf.train.Int64List)
#  - Create a tf.Example with this two components and return a serialized string.

# In[ ]:


# Function to download bytes from Cloud Storage
def get_bytes_label_tfrecord(filepath, label):
    raw_bytes = tf.io.read_file(filepath)
    return raw_bytes, label


# In[ ]:


# Preprocess Image
def process_image_tfrecord(raw_bytes, label):
    image = tf.io.decode_jpeg(raw_bytes, channels=3)
    image = tf.image.resize(image, (224,224), method='nearest')
    image = tf.io.encode_jpeg(image, optimize_size=True)
    
    return image, label


# In[ ]:


# Read images, preprocess and return a dataset
def build_dataset_tfrecord(dataset):
    
    dataset = dataset.map(get_bytes_label_tfrecord, num_parallel_calls=AUTOTUNE)
    dataset = dataset.map(process_image_tfrecord, num_parallel_calls=AUTOTUNE)
    
    return dataset


# In[ ]:


def tf_serialize_example(image, label):
    
    def _bytes_feature(value):
        """Returns a bytes_list from a string / byte."""
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _float_feature(value):
        """Returns a float_list from a float / double."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    def _int64_feature(value):
        """Returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))    
    
    def serialize_example(image, label):
        
        feature = {
            'image': _bytes_feature(image),
            'label': _int64_feature(label)
        }

        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        
        return example_proto.SerializeToString()
    
    tf_string = serialize_example(image, label)

    return tf_string


# The following function shards the dataset into batches of images and labels.
# For each shard, the images and labels are serialized and written to the TFRecord file.
# 
# The TFRecordWriter allows the compression of files to some formats. One chosen here is GZIP.

# In[ ]:


# Create TFRecord with `n_shards` shards
def create_tfrecord(ds, n_shards):

    for i in range(n_shards):
        batch = map(lambda x: tf_serialize_example(x[0],x[1]), ds.shard(n_shards, i)
                    .apply(build_dataset_tfrecord)
                    .as_numpy_iterator())
        
        with tf.io.TFRecordWriter('output_file-part-{i}.tfrecord'.format(i=i), 'GZIP') as writer:
            print('Creating TFRecord ... output_file-part-{i}.tfrecord'.format(i=i))
            for a in batch:
                writer.write(a)


# In[ ]:


# We sharded into 4 files with 130MB each.
# If the dataset is bigger, you can create more shards
create_tfrecord(dataset, 4)


# # Consuming the TFRecord and Re-Running the Benchmark
# 
# The TFRecords are saved in the local filesystem. To continue our benchmark, it is necessary to copy the files to a bucket in Cloud Storage. 
# The files were copied to the following path:

# In[ ]:


TFRECORDS = 'gs://renatoleite-nb/tfrecords/*'


# To read the Serialized data inside each TFRecord, it is necessary to pass a description of the features (image and label) previously encoded as a tf.feature. 
# To do so, create a dictionary describing each component we will read.

# In[ ]:


# Create a description of the features.
feature_description = {
    'image': tf.io.FixedLenFeature([], tf.string),
    'label': tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True)
}


# Then a function can parse an example from the TFRecord, using the description created before.

# In[ ]:


@tf.function
def _parse_function(example_proto):
    # Parse the input `tf.Example` proto using the dictionary above.
    return tf.io.parse_single_example(example_proto, feature_description)


# First all the files were listed inside the specified bucketand created a dataset using ".from_tensorf_slices", but it would be possible to create a TFRecordDataset directly from this listing.  
# The reason this was done is because the dataset with listing is used later.

# In[ ]:


# List all the TFRecords and create a dataset from it
filenames = tf.io.gfile.glob(TFRECORDS)
filenames_dataset = tf.data.Dataset.from_tensor_slices(filenames)


# In[ ]:


# Preprocess Image
@tf.function
def process_image_tfrecord(record):  
    image = tf.io.decode_jpeg(record['image'], channels=3)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    
    label = record['label']
    
    return image, label


# Note that our preprocess function don't resize the image anymore.\
# This is because we store the images in the TFRecord files already resized.
# 
# The TFRecordDataset has a flag "num_parallel_reads" to parallelize the number of reads by the runtime. \
# This flag is set to AUTOTUNE to let Tensorflow decide how many threads are necessary to optimize the process.

# In[ ]:


# Create a Dataset composed of TFRecords (paths to bucket)
@tf.function
def get_tfrecord(filename):
    return tf.data.TFRecordDataset(filename, compression_type='GZIP', num_parallel_reads=AUTOTUNE)


# The new function to build the dataset has the following changes:
#  - Use of "interleave" to parallelize the opening of files. From tensorflow documentation:
#  > To mitigate the impact of the various data extraction overheads, the tf.data.Dataset.interleave transformation can be used to parallelize the data loading step, interleaving the contents of other datasets (such as data file readers). The number of datasets to overlap can be specified by the cycle_length argument, while the level of parallelism can be specified by the num_parallel_calls argument. Similar to the prefetch transformation, the interleave transformation supports tf.data.experimental.AUTOTUNE which will delegate the decision about what level of parallelism to use to the tf.data runtime.
#  
#  - Use of the "\_parse_function" to extract and deserialize the image and label.
#  - Lighter version of preprocess, without resizing it (images are stored in TFRecord file already resized).

# In[ ]:


def build_dataset_test(dataset, batch_size=BATCH_SIZE):
    
    dataset = dataset.interleave(get_tfrecord, num_parallel_calls=AUTOTUNE)
    
    # Transformation: IO Intensive 
    dataset = dataset.map(_parse_function, num_parallel_calls=AUTOTUNE)

    # Transformation: CPU Intensive
    dataset = dataset.map(process_image_tfrecord, num_parallel_calls=AUTOTUNE)
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size=batch_size)
    
    # Pipeline next iteration
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    
    return dataset


# In[ ]:


test_ds = build_dataset_test(filenames_dataset, batch_size=32)


# In[ ]:


timeit(test_ds, steps=20000, batch_size=32)


# This new benchmark gives us around 2100 images per second, a much better version of the original pipeline developed (reading images individually).   
# To speedup the training process and utilized better your resources like GPUs and TPUs, it is critical to build a very efficient data pipeline. otherwise this can quickly become a bottleneck in you training loop.
# 
# We could also try to cache the data at different stages, like the example bellow (from tensorflow documentation), but I am assuming the data won't fit into the host VM memory, so it is needed to read Cloud Storage each epoch.
# 
# > dataset.map(time_consuming_mapping).cache().map(memory_consuming_mapping)

# ## Batch before we Map!
# 
# One last performance optimization we can try is to batch the elements before applying the map transformation.
# This technique is called "Vectorizing maps" which is recommended to user-defined function (that is, have it operate over a batch of inputs at once) and apply the batch transformation before the map transformation. 
# 
# Let's redefine the build_dataset and the preprocess map transformation:

# In[ ]:


def build_dataset_test(dataset, batch_size=BATCH_SIZE):
    
    dataset = dataset.interleave(get_tfrecord, num_parallel_calls=AUTOTUNE)

    dataset = dataset.batch(batch_size=batch_size)
    dataset = dataset.map(_parse_function, num_parallel_calls=AUTOTUNE)
    dataset = dataset.map(process_image_tfrecord, num_parallel_calls=AUTOTUNE)

    dataset = dataset.repeat()
    # Pipeline next iteration
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    
    return dataset


# Notice that it is necessary to redefine the \_parce_function and process_image_tfrecord to receive a batch of elements and process all of them.

# In[ ]:


@tf.function
def _parse_function(example_proto):
    
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True)
    }
    
    # Parse the input `tf.Example` proto using the dictionary above.
    return tf.io.parse_example(example_proto, feature_description)


# In[ ]:


# Preprocess Image
@tf.function
def process_image_tfrecord(record):
    
    image = tf.map_fn(tf.io.decode_jpeg, record['image'], dtype=tf.uint8)
    image = tf.map_fn(lambda image: 
                      tf.image.convert_image_dtype(image, dtype=tf.float32), image, dtype=tf.float32)
    
    label = record['label']
    
    return image, label


# In[ ]:


test_ds = build_dataset_test(filenames_dataset, batch_size=32)


# In[ ]:


timeit(test_ds, steps=20000, batch_size=32)


# With this new approach we could improve by ~15% the throughput performance of our input pipeline, reaching ~2300 images/sec.
