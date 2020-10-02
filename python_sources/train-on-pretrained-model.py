#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


get_ipython().run_cell_magic('time', '', '\nfrom __future__ import absolute_import, division, print_function\n\nimport tensorflow as tf\nimport IPython.display as display\n\ntf.enable_eager_execution()\nprint(tf.VERSION)\n\nAUTOTUNE = tf.data.experimental.AUTOTUNE')


# In[3]:


get_ipython().run_cell_magic('time', '', "\nimg_size = 10\n\ndf = pd.read_csv('../input/iwildcam-2019-fgvc6/train.csv')\n\nf = df['file_name']\nid = df['category_id']\n\nall_image_paths = ['../input/iwildcam-2019-fgvc6/train_images/' + fname for fname in f]\nall_image_labels = [i for i in id]\n\npaths_labels = dict(zip(all_image_paths[0:img_size], all_image_labels[0:img_size]))")


# In[4]:


get_ipython().run_cell_magic('time', '', "\nmobile_net = tf.keras.applications.DenseNet121(weights='imagenet', input_shape=(192, 192, 3), include_top=False)\nmobile_net.trainable=False")


# In[5]:


get_ipython().run_cell_magic('time', '', "\nds = tf.data.TFRecordDataset('../input/let-s-make-tfrecord-simple-version/images.tfrec')\n\ndef parse(x):\n  result = tf.io.parse_tensor(x, out_type=tf.string)\n  result = tf.image.decode_jpeg(result, channels=3)\n  result = tf.dtypes.cast(result, tf.float32)\n#  result = 2 * (result/255.) - 1\n  result = result/255.\n#  result = tf.reshape(result, [28, 28, 3])\n  return result\n\nds = ds.map(parse, num_parallel_calls=AUTOTUNE)\n#ds = ds.map(change_range)")


# In[6]:


lables = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels, tf.int64))

ds = tf.data.Dataset.zip((ds, lables))


# In[7]:


ds = ds.repeat()
ds = ds.batch(32)


# In[8]:


model = tf.keras.Sequential([
  mobile_net,
  tf.keras.layers.GlobalAveragePooling2D(),
  tf.keras.layers.Dense(1024, activation='relu'),
  tf.keras.layers.Dense(23, activation='softmax')])


# In[9]:


model.compile(optimizer=tf.keras.optimizers.Adam(0.001), 
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=["accuracy"])


# In[10]:


#%%time

#model.fit(ds, epochs=10, steps_per_epoch=5000)


# In[11]:


model.fit(ds, epochs=5, steps_per_epoch=60)


# In[12]:


get_ipython().system('ls -al')


# In[13]:


checkpoint_path = "cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)


# In[14]:


# include the epoch in the file name. (uses `str.format`)

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path, verbose=1, save_weights_only=True,
    period=1)

model.save_weights(checkpoint_path.format(epoch=0))
model.fit(ds, epochs = 5, steps_per_epoch = 60, callbacks = [cp_callback],
          verbose=1)


# In[15]:


get_ipython().system('ls -al')


# In[20]:


latest = tf.train.latest_checkpoint(checkpoint_dir)
latest


# In[21]:


model.load_weights(latest)


# In[ ]:


model.fit(ds, epochs = 5, steps_per_epoch = 60, callbacks = [cp_callback],
          verbose=1)


# In[19]:


get_ipython().system('ls -al')


# In[ ]:




