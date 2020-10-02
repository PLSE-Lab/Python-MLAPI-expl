#!/usr/bin/env python
# coding: utf-8

# # Setup environment

# In[1]:


get_ipython().system('pip install -q --upgrade pip')
get_ipython().system('pip install -q -U tensorflow-gpu==2.0.0-alpha0')


# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from IPython.display import Image, display
from sklearn.model_selection import train_test_split

AUTOTUNE = tf.data.experimental.AUTOTUNE


# # Inspect data

# In[3]:


train_csv = pd.read_csv('../input/train.csv')
train_csv.head()


# In[4]:


# Cactus
display(Image('../input/train/train/0004be2cfeaba1c0361d39e2b000257b.jpg'))
display(Image('../input/train/train/000c8a36845c0208e833c79c1bffedd1.jpg'))

# No cactus
display(Image('../input/train/train/002134abf28af54575c18741b89dd2a4.jpg'))
display(Image('../input/train/train/0024320f43bdd490562246435af4f90b.jpg'))


# In[5]:


images = ['../input/train/train/' + fname for fname in train_csv['id']]
labels = train_csv['has_cactus'].tolist()


# In[6]:


X_train, X_dev, y_train, y_dev = train_test_split(images, labels, test_size=0.1, random_state=42)
n_train = len(X_train)
n_dev = len(X_dev)


# In[7]:


IMAGE_SIZE = 96

def preprocess_image(fname, label=None):
  img = tf.io.read_file(fname)
  img = tf.image.decode_jpeg(img)
  img = tf.cast(img, tf.float32)
  img = (img / 127.5) - 1
  img = tf.image.resize(img, size=(IMAGE_SIZE, IMAGE_SIZE))
  if label is not None:
    return img, label
  else:
    return img


# In[8]:


BATCH_SIZE = 32 #@param

ds_train = (tf.data.Dataset.from_tensor_slices((X_train, y_train))
            .map(preprocess_image, num_parallel_calls=AUTOTUNE)
            .shuffle(n_train)
            .batch(BATCH_SIZE)
            .prefetch(buffer_size=AUTOTUNE)
           )
ds_dev  = (tf.data.Dataset.from_tensor_slices((X_dev, y_dev))
           .map(preprocess_image, num_parallel_calls=AUTOTUNE)
           .shuffle(n_dev)
           .batch(BATCH_SIZE)
           .prefetch(buffer_size=AUTOTUNE)
          )


# # Model

# In[9]:


base_model = tf.keras.applications.MobileNetV2(
    input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3),
    include_top = False,
    weights = 'imagenet'
)
base_model.trainable = False
base_model.summary()


# In[10]:


# My layers
pooling_layer = tf.keras.layers.GlobalMaxPooling2D()
final_layer = tf.keras.layers.Dense(units=1, activation='sigmoid')


# In[11]:


model = tf.keras.Sequential([
    base_model,
    pooling_layer,
    final_layer
])


# In[12]:


learning_rate = 0.0001
model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=learning_rate),
    loss='binary_crossentropy',
    metrics=['accuracy']
)


# # Training

# In[14]:


initial_epochs = 32
steps_per_epoch = (tf.math.ceil(n_train/BATCH_SIZE))
model.fit(
    ds_train.repeat(),
    epochs=initial_epochs,
    steps_per_epoch=steps_per_epoch,
    validation_data=ds_dev
)


# # Fine Tune

# In[15]:


base_model.trainable = True
len(base_model.layers)


# In[16]:


fine_tune_at = 100
for layer in base_model.layers[:fine_tune_at]:
  layer.trainable =  False


# In[19]:


model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=learning_rate/10),
    loss='binary_crossentropy',
    metrics=['accuracy']
)


# In[21]:


fine_epochs = 32
total_epochs = initial_epochs + fine_epochs
model.fit(
    ds_train.repeat(),
    epochs=total_epochs,
    initial_epoch=initial_epochs,
    steps_per_epoch=steps_per_epoch,
    validation_data=ds_dev
)


# # Prediction

# In[ ]:


test_image_names = tf.io.gfile.listdir('../input/test/test/')
n_test = len(test_image_names)
test_image_paths = list(map(lambda s: '../input/test/test/' + s, test_image_names))


# In[ ]:


ds_test = (tf.data.Dataset.from_tensor_slices(test_image_paths)
            .map(preprocess_image, num_parallel_calls=AUTOTUNE)
            .batch(BATCH_SIZE)
            .prefetch(buffer_size=AUTOTUNE)
           )


# In[ ]:


predictions = model.predict(ds_test)


# In[ ]:


final_df = pd.DataFrame()
final_df['id'] = test_image_names
final_df['has_cactus'] = predictions


# In[ ]:


final_df.to_csv('submission.csv', index=False)

