#!/usr/bin/env python
# coding: utf-8

# # Data processing
# 
# This Notebooks performs 2 tasks:
# Create TF Records, which is a TPU-friendly format to take advantage of TPU's
# Export the generated files to GCS, to stream them to our models

# In[ ]:


import os, time
import pandas as pd
import tensorflow as tf
import transformers as ppb
import tokenizers
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
import seaborn as sns
import glob
sns.set()
import logging
logging.getLogger().setLevel(logging.NOTSET)
print(tf.version.VERSION)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # Set global variables
# 
# Set maximum sequence length, model and path variables.

# In[ ]:


SEQUENCE_LENGTH = 128
MODEL_NAME = 'jplu/tf-xlm-roberta-large' # The model you want to use here
DATA_PATH =  "../input/jigsaw-multilingual-toxic-comment-classification"
AUTO = tf.data.experimental.AUTOTUNE


# # Datasets
# Read and extract relevant columns from datasets

# In[ ]:


train1_df = pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv")
train2_df = pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-unintended-bias-train.csv")
train2_df.toxic = train2_df.toxic.round().astype(int)
print("Training Dataset 1 with {} samples".format(len(train1_df)))
print("Training Dataset 2 with {} samples".format(len(train2_df)))
train_df = pd.concat([train1_df[['id','comment_text','toxic']], train2_df[['comment_text','toxic']]])
print("Training Dataset with {} samples".format(len(train_df)))
del train1_df
del train2_df

validation_df = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/validation.csv')
print("Validation Dataset with {} samples".format(len(validation_df)))
test_df = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/test.csv')
print("Test Dataset with {} samples".format(len(test_df)))


# # Tokenizer

# In[ ]:


def get_tokenizer(model_name=MODEL_NAME):
    tokenizer = ppb.AutoTokenizer.from_pretrained(model_name)
    print(type(tokenizer))
  
    return tokenizer

tokenizer = get_tokenizer()


# # Preprocessing
# 
# Process individual sentences for input to BERT using the tokenizer, and then prepare the entire dataset. The same code will process the other training data files, as well as the validation and test data.

# In[ ]:



def process_sentence(sentence, max_seq_length, tokenizer):
    """Helper function to prepare data for any BERT model. Converts sentence input examples
    into the form ['input_ids', 'input_mask', 'segment_ids']."""
    # Tokenize, and truncate to max_seq_length if necessary.
    input_ids = tokenizer.encode(sentence)
    if len(input_ids) > max_seq_length :
        input_ids = input_ids[:max_seq_length]

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    pad_length = max_seq_length - len(input_ids)
    input_ids.extend([0] * pad_length)
    input_mask.extend([0] * pad_length)

    # We only have one input segment.
    segment_ids = [0] * max_seq_length

    return (input_ids, input_mask, segment_ids)

def _int64_feature(value):
    """Returns a single element int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _int64_feature_list(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    v = tf.train.Int64List(value=value)
    return tf.train.Feature(int64_list=v)

def serialize_example(f0, f1, f2, y):
    """
    Creates a tf.Example message ready to be written to a file.
    """
    # Create a dictionary mapping the feature name to the tf.Example-compatible
    # data type.    
    feature = {
      'f0': _int64_feature_list(f0), # input_word_ids
      'f1': _int64_feature_list(f1), # masks
      'f2': _int64_feature_list(f2), # input_type
      'y':  _int64_feature(y), # target
    }

    # Create a Features message using tf.train.Example.
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def preprocess_and_save_tfrec(dataframe, filename, text_label='comment_text', seq_length=SEQUENCE_LENGTH, records_per_file=200000, verbose=True):
    """Preprocess a CSV to the expected TF Dataset in TFRecord format, and save the result.
    Google recommends each file to be ~100MB, so 200K samples per file is about 100MB"""
    processed_filename = (filename.rstrip('.csv') + "-processed-seqlen{}.csv".format(SEQUENCE_LENGTH))

    start = time.time()
    
    for i in range(0,len(dataframe), records_per_file):
        processed_df = dataframe.iloc[i:i + records_per_file]

        input_word_ids, input_mask, all_segment_id = (zip(*processed_df[text_label].apply(lambda x: process_sentence(x, SEQUENCE_LENGTH, tokenizer))))
        y = processed_df['toxic'].values

        with tf.io.TFRecordWriter(filename+'_'+str(i)+'.tfrec') as writer:
          for k in range(len(processed_df)):
            example = serialize_example(input_word_ids[k], input_mask[k], all_segment_id[k], y[k])
            writer.write(example)

        if verbose:
            print('Processed {} examples in {}'.format(i + len(processed_df), time.time() - start))
    return
  
# Process the datasets.
print('Transforming data and generating files, this could take a while...')
preprocess_and_save_tfrec(train_df.iloc[:30000], 'train') # remove the .iloc to process all data
preprocess_and_save_tfrec(validation_df, 'validation')


# # Example of how to read a TFRecord
# We need to instantiate a TFRecordDataset object from a 

# In[ ]:


filenames = glob.glob('*.tfrec')
raw_dataset = tf.data.TFRecordDataset(filenames)
raw_dataset


# In[ ]:


for raw_record in raw_dataset.take(1):
  example = tf.train.Example()
  example.ParseFromString(raw_record.numpy())
  print(example)


# # Export to GCS

# This part is for creating and uploading automatically your generated files from Kaggle to GCS. You can obviously do it manually by downloading all your files and uploading to GCS using the web console. If you do not have a nice bandwidth you might want to use the following scrip.  
# 
# First thing you need to generate is a credentials.json file from your GCS IAM section.
# 
# **NOTE: DON'T MAKE THIS INFORMATION PUBLIC!!, otherwise anyone with this data could access your GC account**

# This part uses GCS API: 
# [https://googleapis.dev/python/storage/latest/index.html](http://)

# In[ ]:


import json
from google.oauth2 import service_account
from google.cloud import storage
import glob

# Copy here your credentials.json data
auth = {
# your auth data comes here. This is sensitive data
}

with open('auth.json', 'w') as json_file:
  json.dump(auth, json_file)

credentials = service_account.Credentials.from_service_account_file('auth.json')


# In[ ]:


# Utility functions to manage GCS

def create_bucket(dataset_name):
    """Creates a new bucket. https://cloud.google.com/storage/docs/ """
    bucket = storage_client.create_bucket(dataset_name)
    print('Bucket {} created'.format(bucket.name))

def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket. https://cloud.google.com/storage/docs/ """
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    print('File {} uploaded to {}.'.format(
        source_file_name,
        destination_blob_name))
    
def list_blobs(bucket_name):
    """Lists all the blobs in the bucket. https://cloud.google.com/storage/docs/"""
    blobs = storage_client.list_blobs(bucket_name)
    for blob in blobs:
        print(blob.name)
        
def download_to_kaggle(bucket_name,destination_directory,file_name):
    """Takes the data from your GCS Bucket and puts it into the working directory of your Kaggle notebook"""
    os.makedirs(destination_directory, exist_ok = True)
    full_file_path = os.path.join(destination_directory, file_name)
    blobs = storage_client.list_blobs(bucket_name)
    for blob in blobs:
        blob.download_to_filename(full_file_path)


# Time to generate a bucket in your Google Cloud Storage project

# In[ ]:


GCS_PROJECT = 'YOUR-GOOGLE-CLOUD-PROJECT'

storage_client = storage.Client(project=GCS_PROJECT, credentials=credentials)
bucket_name = 'your-bucket-name' # this must be unique accross all bucket names, so it could be something like your_name_something_else       
try:
    create_bucket(bucket_name)
except:
    print('Probably your bucket already existed or the name you selected is already taken')


# Copy all generated .tfrec files (from Kaggle output directory) to your GCS bucket

# In[ ]:


files = glob.glob('*.tfrec')
for f in files:
    upload_blob(bucket_name, f, f)
    
print('Listing data in {} :'.format(bucket_name))   
list_blobs(bucket_name)


# Now you can open your Google Cloud Platform console and check for your bucket with your files

# # Read from GCS
# These are some helper functions to deserialize and read data from TF Records

# In[ ]:


BATCH_SIZE = 128
def read_labeled_tfrecord(raw_example):
    LABELED_TFREC_FORMAT = {
        "f0": tf.io.FixedLenFeature([SEQUENCE_LENGTH], tf.int64),
        "f1": tf.io.FixedLenFeature([SEQUENCE_LENGTH], tf.int64), 
        "f2": tf.io.FixedLenFeature([SEQUENCE_LENGTH], tf.int64), 
        "y":  tf.io.FixedLenFeature([], tf.int64),  
    }

    example = tf.io.parse_single_example(raw_example, LABELED_TFREC_FORMAT)
    input_ids = tf.cast(example['f0'], tf.int64)
    input_mask = tf.cast(example['f1'], tf.int64)
    segment_ids =  tf.cast(example['f2'], tf.int64)
    y = tf.cast(example['y'], tf.int64)
    return (input_ids,input_mask,segment_ids) , y # returns a dataset of (image, label) pairs

def load_dataset(filenames, ordered = False):
    # Read from TFRecords. For optimal performance, reading from multiple files at once and
    # Diregarding data order. Order does not matter since we will be shuffling the data anyway
    
    ignore_order = tf.data.Options()
    if not ordered:
        ignore_order.experimental_deterministic = False # disable order, increase speed
        
    gs_path = tf.io.gfile.glob(filenames)
    dataset = tf.data.TFRecordDataset(gs_path, num_parallel_reads = AUTO) # automatically interleaves reads from multiple files
    dataset = dataset.with_options(ignore_order) # use data as soon as it streams in, rather than in its original order
    dataset = dataset.map(read_labeled_tfrecord, num_parallel_calls = AUTO) # returns a dataset of (image, label) pairs if labeled = True or (image, id) pair if labeld = False
    return dataset

def get_training_dataset(dataset, validation=False):
    if not validation:
        return dataset.repeat().shuffle(2048).batch(BATCH_SIZE).prefetch(AUTO)
    else:
        return dataset.batch(BATCH_SIZE).cache().prefetch(AUTO)
    
# Usage would be:
# get_training_dataset( load_dataset( filenames = 'gs://your_bucket_name/train*.tfrec'))

