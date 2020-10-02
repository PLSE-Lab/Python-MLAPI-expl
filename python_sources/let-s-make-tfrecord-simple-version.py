#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


from __future__ import absolute_import, division, print_function
 
import tensorflow as tf
print(tf.VERSION)
  
tf.enable_eager_execution()
 
AUTOTUNE = tf.data.experimental.AUTOTUNE
 
import IPython.display as display
import matplotlib.pyplot as plt


# In[ ]:


df = pd.read_csv('../input/train.csv')
 
f = df['file_name']
id = df['category_id']
 
all_image_paths = ['../input/train_images/' + fname for fname in f]
all_image_labels = [i for i in id]


# In[ ]:


all_image_paths[0]


# In[ ]:


get_ipython().system('ls ../input/train_images/5998cfa4-23d2-11e8-a6a3-ec086b02610b.jpg -al')


# In[ ]:


all_image_paths[0:1]


# In[ ]:


get_ipython().run_cell_magic('time', '', "\ndef my_fn(img):\n  a = tf.io.read_file(img)\n  b = tf.image.decode_jpeg(a)\n  c = tf.image.resize_images(b, (192,192))\n  d = tf.dtypes.cast(c, tf.uint8)\n  e = tf.image.encode_jpeg(d)\n  return e\n    \nds = tf.data.Dataset.from_tensor_slices(all_image_paths)\n\nds2 = ds.map(my_fn)\n\ndds = ds2.map(tf.io.serialize_tensor)\n\ntfrec = tf.data.experimental.TFRecordWriter('images.tfrec')\ntfrec.write(dds)")


# In[ ]:


get_ipython().system('ls -al')


# In[ ]:


ds = tf.data.TFRecordDataset('images.tfrec')

def parse(x):
  result = tf.io.parse_tensor(x, out_type=tf.string)
#  result = tf.reshape(result, [28, 28, 3])
  return result

ds = ds.map(parse, num_parallel_calls=AUTOTUNE)
ds


# In[ ]:


import matplotlib.pyplot as plt

for i in ds.take(10):
  display.display(display.Image(i.numpy()))


# In[ ]:




