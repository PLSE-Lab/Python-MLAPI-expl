#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# Importing necessary libraries

# In[ ]:


import shutil
import glob
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


# Reading list_attr_celeba.csv and setting image id as index for easier access when loading images

# In[ ]:


import pandas as pd
data = pd.read_csv('/kaggle/input/celeba-dataset/list_attr_celeba.csv')
data.set_index('image_id',inplace = True)
data.head()


# Taking only Male column as only it is useful for our project

# In[ ]:


gender = data.Male


# Training and test arrays. In the training and test arrays we store images and in the label arrays we store labels

# In[ ]:


raw_train = []
raw_train_labels = []
raw_test = []
raw_test_labels = []


# Function for loading images
# 
# * Here data is the array of images which are laoded
# * no of images is the no of images we load from the dataset
# * labels are labels for each image respectively
# * inp and outp are the partition in which take take data from. Here for training we take images in between 0 50000 images in files

# In[ ]:


def get_images(data,no_of_images,labels,inp,outp):
    path = '/kaggle/input/celeba-dataset/img_align_celeba/img_align_celeba'
    files = os.listdir(path)
    mcount, fcount = 0, 0
    for i in files[inp:outp]:
        if gender[i] == 1:
            mcount = mcount + 1
            if mcount == no_of_images:
                continue
        elif gender[i] == -1:
            gender[i] = 0
            fcount = fcount + 1
            if fcount == no_of_images:
                continue
        img = cv2.imread(os.path.join(path,i))
        data.append(img)
        labels.append(gender[i])
        if len(data) == 2 * no_of_images:
            return data, labels
        if len(data) % 100 == 0:
            print(len(data),'images')


# Loading the images and labels in raw_train, raw_train_labels. We need to process them before use so the name.

# In[ ]:


raw_train,raw_train_labels = get_images(raw_train,2500,raw_train_labels,0,50000)
raw_test,raw_test_labels = get_images(raw_test,500,raw_test_labels,50000,100000)


# Checking their length

# In[ ]:


len(raw_train),len(raw_train_labels),len(raw_test),len(raw_test_labels)


# Converting the arrays to tensorflow datasets, as in this format we have various good methods we can use to handle data.

# In[ ]:


train_data = tf.data.Dataset.from_tensor_slices((raw_train,raw_train_labels))
test_data = tf.data.Dataset.from_tensor_slices((raw_test,raw_test_labels))


# Formatting the images
# 
# * here we are casting image to float
# * standarding the values between 0 and 1
# * resizing the image to (160,160) 

# In[ ]:


IMG_SIZE = 160
def format_example(image,label):
    image = tf.cast(image,dtype = tf.float32)
    image = (image / 255) - 1
    image = tf.image.resize(image,(IMG_SIZE,IMG_SIZE))
    return image, label
train_data = train_data.map(format_example)
test_data = test_data.map(format_example)


# Here with the help of methods in tf.data.datasets
# 
# * we are shuffing the data
# * we making batches out of them

# In[ ]:


BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 2000
train_data = train_data.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
test_data = test_data.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)


# As we are using transfer learning, MobileNetV2 as our base model. 
# 
# we pass include top = False as we do not use its last layers which is used to classfy on 1000 classes. We set trainable = False as we donot train the base model because we are using less data.

# In[ ]:


IMG_SHAPE = (IMG_SIZE,IMG_SIZE,3)
base_model = tf.keras.applications.MobileNetV2(input_shape = IMG_SHAPE,include_top = False,weights = 'imagenet')
base_model.trainable = False


# Let's check the shape of output shape of the image batches in training data.

# In[ ]:


for image_batch, label_batch in train_data.take(1):
    pass
print(image_batch.shape)


# Here we check the output shape of the base model

# In[ ]:


print(base_model(image_batch).shape)


# Let's add global average pooling layer after the base model, to average its output and make the output feedable to dense layers, and check whether it is working properly by checking its output shape.

# In[ ]:


global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
print(global_average_layer(base_model(image_batch)).shape)


# Let's add a 256 neuron as our first hidden layer and dropout layer with 0.5 dropout. Check their output.

# In[ ]:


hidden_layer1 = tf.keras.layers.Dense(256,activation = 'relu')
dropout1 = tf.keras.layers.Dropout(0.5)
print(dropout1(hidden_layer1(global_average_layer(base_model(image_batch)))).shape)


# Let's add a 128 neuron hidden layer as our second layer and Check its output

# In[ ]:


hidden_layer2 = tf.keras.layers.Dense(128,activation = 'relu')
print(hidden_layer2(dropout1(hidden_layer1(global_average_layer(base_model(image_batch))))).shape)


# Let's add a final prediction layer with 1 neuron with sigmoid activation as output layer

# In[ ]:


prediction_layer = tf.keras.layers.Dense(1)
print(prediction_layer(hidden_layer2(dropout1(hidden_layer1(global_average_layer(base_model(image_batch)))))).shape)


# Now as all these layers are working properly we make a sequential model out of them and name it model

# In[ ]:


model = tf.keras.Sequential([
     base_model,
     global_average_layer,
     hidden_layer1,
     dropout1,
     hidden_layer2,
     prediction_layer
])


# Let's use learning rate 0.0001 and rmsprop as optimizer

# In[ ]:


base_learning_rate = 0.0001
model.compile(optimizer = tf.keras.optimizers.RMSprop(lr = base_learning_rate),
              loss = 'binary_crossentropy', metrics = ['accuracy'])


# Here is the summary of our model

# In[ ]:


model.summary()


# Let's calculate steps per epoch and fix validation steps.Let's also how our model performs before training

# In[ ]:


num_train = 10000
num_test = 2500
initial_epochs = 20
steps_per_epochs = round(num_train) //  BATCH_SIZE
validation_steps = 4

loss0, accuracy0 = model.evaluate(test_data, steps = validation_steps)


# Let's add callbacks to save best model weights and load them to evaluate on test set

# In[ ]:


callbacks = tf.keras.callbacks.ModelCheckpoint('/kaggle/working/best_model.h5',save_best_only = True,monitor = 'val_accuracy',mode = max,verbose = 1)


# Let's train the model

# In[ ]:


history = model.fit(train_data,epochs = initial_epochs, validation_data = test_data,callbacks = [callbacks])


# In[ ]:


model.load_weights('best_model.h5')


# In[ ]:


model.evaluate(test_data)


# So, we got 86.5% percent accuracy on the test set.

# so, that is it. We simply got 87% accuracy approximately. This is power of transfer learning.

# If you have liked the kernel, Please give upvote.
