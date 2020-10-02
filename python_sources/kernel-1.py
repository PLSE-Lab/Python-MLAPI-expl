#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pathlib
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.models as km
import tensorflow.keras.layers as kl
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import IPython.display as display
tf.enable_eager_execution()


# In[2]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/sample_submission.csv')
AUTOTUNE = tf.data.experimental.AUTOTUNE


# In[3]:


train_image_names = train['id']
ytrain = train['has_cactus']


# In[4]:


train_image_names.values


# In[5]:


train_image_paths = '../input/train/train/'+ train_image_names.values


# In[6]:


train_image_paths


# In[7]:


image = train_image_paths[0]


# In[8]:


ytrain.values


# In[46]:


def preprocess_image(image):
    image = tf.image.decode_jpeg(image,channels=3)
    image = tf.image.resize_images(image,[32,32])
    image /= 255.0
    return image


# In[47]:


def load_and_preprocess_image(path):
    image = tf.read_file(path)
    return preprocess_image(image)


# In[48]:


path_ds = tf.data.Dataset.from_tensor_slices(train_image_paths)


# In[49]:


ds = path_ds.map(load_and_preprocess_image,num_parallel_calls=AUTOTUNE)


# In[50]:


labels_ds = tf.data.Dataset.from_tensor_slices(ytrain)


# In[51]:


ds_label_ds = tf.data.Dataset.zip((ds,labels_ds))


# In[52]:


ds_label_ds = ds_label_ds.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=len(train)))


# In[53]:


ds_label_ds = ds_label_ds.batch(30)
ds_label_ds = ds_label_ds.prefetch(buffer_size=AUTOTUNE)


# In[54]:


ds_label_ds


# In[55]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator


# In[60]:


model = km.Sequential([
    kl.Conv2D(filters=32, kernel_size=3,padding='same',input_shape=(32,32,3),activation=tf.nn.relu),    
])


# In[61]:


model.add(kl.Conv2D(32, (3, 3)))
model.add(kl.Activation('relu'))
model.add(kl.MaxPooling2D(pool_size=(2, 2)))
model.add(kl.Dropout(0.25))

model.add(kl.Conv2D(64, (3, 3), padding='same'))
model.add(kl.Activation('relu'))
model.add(kl.Conv2D(64, (3, 3)))
model.add(kl.Activation('relu'))
model.add(kl.MaxPooling2D(pool_size=(2, 2)))
model.add(kl.Dropout(0.25))

model.add(kl.Conv2D(64, (3, 3), padding='same'))
model.add(kl.Activation('relu'))
model.add(kl.Conv2D(64, (3, 3)))
model.add(kl.Activation('relu'))
model.add(kl.MaxPooling2D(pool_size=(2, 2)))
model.add(kl.Dropout(0.25))

model.add(kl.Flatten())
model.add(kl.Dense(512))
model.add(kl.Activation('relu'))
model.add(kl.Dropout(0.5))
model.add(kl.Dense(2))
model.add(kl.Activation('softmax'))


# In[62]:


model.compile(optimizer='adam',loss=tf.keras.losses.sparse_categorical_crossentropy,
             metrics=['accuracy'])


# In[63]:


model.fit(ds_label_ds,epochs=5,steps_per_epoch=len(train)//5)


# In[64]:


test.shape


# In[65]:


test_image_names = test['id']


# In[66]:


test_image_paths = '../input/test/test/'+test_image_names


# In[73]:


test_image_paths.values


# In[74]:


Xtest = []


# In[75]:


import cv2


# In[76]:


for path in test_image_paths:
    image = cv2.imread(path)
    Xtest.append(image)


# In[77]:


Xtest = np.reshape(Xtest,newshape=(-1,32,32,3))
Xtest = Xtest / 255.0


# In[78]:


test_ds = tf.data.Dataset.from_tensor_slices(Xtest)


# In[79]:


test_ds = test_ds.batch(30)


# In[80]:


pre = model.predict(test_ds,steps=len(test))


# In[81]:


pre.shape


# In[82]:


pre_ = np.argmax(pre,axis=1)


# In[83]:


pre_


# In[84]:


test.has_cactus = pre_


# In[86]:


test.to_csv('submission_7.csv',index=False)


# In[ ]:




