#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from cv2 import cv2
import zipfile
import os
import matplotlib.pyplot as plt


# In[ ]:


TEST_DIR = '../input/dogs-vs-cats-redux-kernels-edition/test.zip'
TRAIN_DIR = '../input/dogs-vs-cats-redux-kernels-edition/train.zip'


# now we extract the images (i have no idea on how to work with elements of zip files without extracting them)

# In[ ]:


#you can execute this  only once
with zipfile.ZipFile(TRAIN_DIR,'r') as trainfile:
    trainfile.extractall()
with zipfile.ZipFile(TEST_DIR,'r') as trainfile:
    trainfile.extractall()


# Now lets see if the files are extracted or not

# In[ ]:


get_ipython().system('ls')


# In[ ]:


testdir ='test/'
traindir = 'train/'


# now lets get the full path of train images and test images

# In[ ]:


test_images = [testdir+i for i in os.listdir(testdir)]
all_images = [traindir+i for i in os.listdir(traindir)]

limit = int( 0.8* len(all_images))

train_images = all_images[0:limit]
validation_images = all_images[limit:]


# small visualization

# In[ ]:


img = cv2.imread(train_images[1])
plt.imshow(img)


# lets create a function which will read the image using opencv library

# In[ ]:


rows, columns = 160,160


# In[ ]:


def getallimages(path):
    actualdata = np.ndarray((len(path),rows,columns,3),dtype=np.uint8)
    for index , file in enumerate(path):
        img = cv2.imread(file)
        img= cv2.resize(img, (rows, columns), interpolation=cv2.INTER_CUBIC)
#         img = tf.cast(img,tf.float32)
#         img = (img/127.5) - 1
#         img = tf.image.resize(img, (rows, columns))
        actualdata[index] = img
    return actualdata
train = getallimages(train_images)
test = getallimages(test_images)


# In[ ]:


validation = getallimages(validation_images)


# In[ ]:


test.shape


# we need some labels too so lets add y which will be our label 

# In[ ]:


label = [1 if 'dog' in i else 0 for i in train_images]
validation_label = [1 if 'dog' in i else 0 for i in validation_images]

validation_label[:10]


# In[ ]:


image_shape = (rows,rows,3)


# In[ ]:


type(train)


# In[ ]:


base_model = tf.keras.applications.ResNet101(
    weights = 'imagenet', include_top=False, input_shape=image_shape)


# **We freeze the base layer ie its weights are not going to be retraning the model**

# In[ ]:


base_model.trainable=False


# In[ ]:


base_model.summary()


# In[ ]:


model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
   
    tf.keras.layers.Dense(1,activation='sigmoid')
    
])


# In[ ]:


model.summary()


# we will create a training batch now
# 

# In[ ]:


# train_dataset = tf.data.Dataset.from_tensor_slices((train,label))
# validation_ds = tf.data.Dataset.from_tensor_slices((validation,validation_label))

# BATCH_SIZE = 64
# SHUFFLE_BUFFER_SIZE = 10000
# validation_batches = validation_ds.batch(BATCH_SIZE)
# train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)


# In[ ]:


base_learning_rate = 0.001
# you can use this tootf.keras.optimizers.RMSprop(learning_rate=base_learning_rate
model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])


# In[ ]:


epochs = 10
validation_steps=20

# model.evaluate(validation,validation_label, steps = validation_steps)


# In[ ]:


model.fit(x=np.array(train),y=np.array(label),validation_data=(np.array(validation),np.array(validation_label)) ,batch_size=128,epochs=epochs,shuffle=True)


# In[ ]:


prediction  = model.predict_proba(test,verbose=1)


# In[ ]:


plt.xlabel(prediction[4][0])
plt.imshow(test[4])


# In[ ]:


# model.save('dgvscat.h5)


# In[ ]:



test_id = [i.split('/')[1][:-4] for i in test_images]

predictions_df = pd.DataFrame({'id': test_id, 'label': prediction[:,0]})
predictions_df
predictions_df.to_csv("submission.csv", index=False,header=True)

