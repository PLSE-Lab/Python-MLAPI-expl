#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os, json, time, sys, math
import numpy as np
import pandas as pd 
from matplotlib import pyplot as plt
from PIL import Image

if 'google.colab' in sys.modules:
    get_ipython().run_line_magic('tensorflow_version', '2.x')
import tensorflow as tf

print("Tensorflow version " + tf.__version__)
AUTO = tf.data.experimental.AUTOTUNE 

start_time = time.time()


# ### create data.csv
# [Link for understand data and create data.csv](https://www.kaggle.com/seraphwedd18/herbarium-consolidating-the-details)

# In[ ]:


df = pd.read_csv('../input/data-for-datatse-herbarium/data.csv')
df1 = df.copy()
df1['file_name'] = df['file_name'].map(lambda x: x.split('/')[-1])
df1.head()


# In[ ]:


TRAIN_PATTERN = '/kaggle/input/herbarium-2020-fgvc7/nybg2020/train/images/*/*/*.jpg'
TEST_PATTERN = '/kaggle/input/herbarium-2020-fgvc7/nybg2020/test/images/*/*/*.jpg'


# In[ ]:


shard_size = 32


# In[ ]:


def display_from_dataset(dataset):
    plt.figure(figsize=(13,13))
    subplot=331
    for i, (image, label) in enumerate(dataset):
        plt.subplot(subplot)
        plt.axis('off')
        plt.imshow(image.numpy().astype(np.uint8))
        plt.title(label.numpy().decode("utf-8"), fontsize=16)
        subplot += 1
        if i==8:
            break
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()
    
def decode_jpeg_and_label(filename):
    bits = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(bits)
    label = tf.strings.split(tf.expand_dims(filename, axis=-1), sep='/')
    label = label.values[-1]
    return image, label

def resize_and_crop_image(image, label):
    w = tf.shape(image)[0]
    h = tf.shape(image)[1]
    tw = w//5
    th = h//5
    resize_crit = (w * th) / (h * tw)
    image = tf.cond(resize_crit < 1,
                    lambda: tf.image.resize(image, [w*tw/w, h*tw/w]), # if true
                    lambda: tf.image.resize(image, [w*th/h, h*th/h])  # if false
                   )
    nw = tf.shape(image)[0]
    nh = tf.shape(image)[1]
    image = tf.image.crop_to_bounding_box(image, (nw - tw) // 2, (nh - th) // 2, tw, th)
    return image, label

def recompress_image(image, label):
    height = tf.shape(image)[0]
    width = tf.shape(image)[1]
    image = tf.cast(image, tf.uint8)
    image = tf.image.encode_jpeg(image, optimize_size=True, chroma_downsampling=False)
    return image, label, height, width


# In[ ]:


filenames = tf.data.Dataset.list_files(TRAIN_PATTERN) 
dataset   = filenames.map(decode_jpeg_and_label, num_parallel_calls=AUTO)
dataset   = dataset.map(resize_and_crop_image, num_parallel_calls=AUTO) 
dataset   = dataset.map(recompress_image, num_parallel_calls=AUTO)
dataset   = dataset.batch(shard_size)
dataset   = dataset.prefetch(AUTO)


# In[ ]:


get_ipython().system('mkdir tfrecords')


# In[ ]:



def _bytestring_feature(list_of_bytestrings):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=list_of_bytestrings))

def _int_feature(list_of_ints): # int64
    return tf.train.Feature(int64_list=tf.train.Int64List(value=list_of_ints))

def _float_feature(list_of_floats): # float32
    return tf.train.Feature(float_list=tf.train.FloatList(value=list_of_floats))
  

def to_tfrecord(tfrec_filewriter, img_bytes, label, family, genus, category_id, width, height):
    one_hot_family = np.eye(310)[family]
    one_hot_genus = np.eye(3678)[genus]
    one_hot_category_id = np.eye(32094)[category_id]

    feature = {
      "image": _bytestring_feature([img_bytes]),
      "label":  _bytestring_feature([label]),
        
      "family": _int_feature([family]),
      "genus": _int_feature([genus]),
      "category_id": _int_feature([category_id]),
        
      "one_hot_family": _float_feature(one_hot_family.tolist()),
      "one_hot_genus": _float_feature(one_hot_genus.tolist()),
      "one_hot_category_id": _float_feature(one_hot_category_id.tolist()),
        
      "size":  _int_feature([width, height])
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))

print("Writing TFRecords")
stoped = 0
for shard, (image, label, height, width) in enumerate(dataset):
    if stoped > 0:
        break
    stoped +=1
    shard_size = image.numpy().shape[0]
    filename   = './tfrecords/' + "{:02d}-{}.tfrec".format(shard, shard_size)
  
    with tf.io.TFRecordWriter(filename) as out_file:
        for i in range(shard_size):
            lbl         = label.numpy()[i]
            family      = df1.loc[df1.file_name == lbl.decode('utf-8')]['family'].values[0]
            genus       = df1.loc[df1.file_name == lbl.decode('utf-8')]['genus'].values[0]
            category_id = df1.loc[df1.file_name == lbl.decode('utf-8')]['category_id'].values[0]
            
            example = to_tfrecord(out_file,
                            image.numpy()[i],
                            lbl,
                            family, 
                            genus, 
                            category_id,
                            height.numpy()[i],
                            width.numpy()[i])
            out_file.write(example.SerializeToString())
        print("Wrote file {} containing {} records".format(filename, shard_size))


# In[ ]:


def read_tfrecord(example):
    features = {
        "image": tf.io.FixedLenFeature([], tf.string), 
        "label": tf.io.FixedLenFeature([], tf.string),
        
        "family": tf.io.FixedLenFeature([], tf.int64), 
        "genus": tf.io.FixedLenFeature([], tf.int64), 
        "category_id": tf.io.FixedLenFeature([], tf.int64), 
        
        "one_hot_family": tf.io.VarLenFeature(tf.float32) ,
        "one_hot_genus": tf.io.VarLenFeature(tf.float32) ,
        "one_hot_category_id": tf.io.VarLenFeature(tf.float32) ,

        "size": tf.io.FixedLenFeature([2], tf.int64) 
    }
    example = tf.io.parse_single_example(example, features)
    width = example['size'][0]
    height  = example['size'][1]
    image = tf.image.decode_jpeg(example['image'], channels=3)
    image = tf.reshape(image, [width,height, 3])
    
    label = example['label']
    
  
    return image, label

option_no_order = tf.data.Options()
option_no_order.experimental_deterministic = False

filenames = tf.io.gfile.glob('/kaggle/working/tfrecords/' + "*.tfrec")
dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)
dataset = dataset.with_options(option_no_order)
dataset = dataset.map(read_tfrecord, num_parallel_calls=AUTO)
dataset = dataset.shuffle(300)


# In[ ]:


display_from_dataset(dataset)


# In[ ]:


end_time = time.time()
total = end_time - start_time
h = total//3600
m = (total%3600)//60
s = total%60
print("Total time spent: %i hours, %i minutes, and %i seconds" %(h, m, s))

