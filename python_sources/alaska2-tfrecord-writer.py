#!/usr/bin/env python
# coding: utf-8

# ## About This Notebook
# 
# This is the general process for writing tfrecords for the ALASKA2 competition. 
# 
# Please note that kaggle kernels can only store 5gb, so you need to fork it many times to split the data into datsets of 5gb. 
# 
# If you use the public option, you won't be consuming your private data storage space. 
# 
# The dataframe preparation is taken from https://www.kaggle.com/hooong/train-inference-gpu-baseline. 
# 
# I recommend reading https://www.tensorflow.org/tutorials/load_data/tfrecord#writing_a_tfrecord_file for more information if you plan on doing fancy stuff. 
# 
# Just remember that the augmentations that you can do on the TPU are limited. Most augmentations libraries do not work on the TPU (albumentations for example). You may have to pre augment your data. 
# 

# In[ ]:




import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf 


# In[ ]:


def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def serialize_example(bits, label):
    feature = {
        'bits': _bytes_feature(bits),
        'label' : _int64_feature(label)
      }

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def tf_serialize_example(bits, label):
    tf_string = tf.py_function(
        serialize_example,
        (bits, label),  # pass these args to the above function.
        tf.string)      # the return type is `tf.string`.
    return tf.reshape(tf_string, ()) 

def read_image(filename, label=None):
    bits = tf.io.read_file(filename)
    if label is None:
        return bits
    else:
        return bits, label
    


# In[ ]:


from glob import glob 
import random
from sklearn.model_selection import GroupKFold
import pandas as pd 

dataset = []

for label, kind in enumerate(['Cover', 'JMiPOD', 'JUNIWARD', 'UERD']):
    for path in glob('../input/alaska2-image-steganalysis/Cover/*.jpg'):
        dataset.append({
            'kind': kind,
            'image_name': path.split('/')[-1],
            'label': label
        })

random.shuffle(dataset)
dataset = pd.DataFrame(dataset)

gkf = GroupKFold(n_splits=5)

dataset.loc[:, 'fold'] = 0
for fold_number, (train_index, val_index) in enumerate(gkf.split(X=dataset.index, y=dataset['label'], groups=dataset['image_name'])):
    dataset.loc[dataset.iloc[val_index].index, 'fold'] = fold_number


dataset['local_path'] = dataset.apply(lambda x : '../input/alaska2-image-steganalysis/%s/%s' % (x.kind, x.image_name), axis = 1)
dataset = dataset.sort_values('image_name')


# In[ ]:


from tqdm.notebook import tqdm 

start = 0

# 30 records at ~150mb each 
for i in tqdm(range(start * 30, start *30 + 30)):
    
    df = dataset.iloc[1500 * i : 1500 * (i + 1)]

    ds = tf.data.Dataset.from_tensor_slices((df.local_path.values, df.label.values))

    ds = ds.map(read_image)
    ds = ds.map(tf_serialize_example)

    def generator():
        for features in ds:
            yield features

    serialized_ds = tf.data.Dataset.from_generator(generator, output_types=tf.string, output_shapes=())


    serialized_ds

    filename = '%05d.tfrecord' % i
    writer = tf.data.experimental.TFRecordWriter(filename)
    writer.write(serialized_ds)


# In[ ]:


raw_dataset = tf.data.TFRecordDataset(filename)

# Create a description of the features.
feature_description = {
    'bits': tf.io.FixedLenFeature([], tf.string, default_value=''),
    'label' : tf.io.FixedLenFeature([], tf.int64, default_value=0),
}

def _parse_function(example_proto):
  # Parse the input `tf.Example` proto using the dictionary above.
  return tf.io.parse_single_example(example_proto, feature_description)

def split(features):
    return features['bits'], features['label']

def decode_image(bits, label=None, image_size=(512, 512)):
    
    image = tf.image.decode_jpeg(bits, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(image, image_size)
    
    if label is None:
        return image
    else:
        return image, label

p = raw_dataset.map(_parse_function)
p = p.map(split)
p = p.map(decode_image)

for a in p.take(1):
    print(a)

