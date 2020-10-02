#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
from tensorflow import keras
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D
from keras.layers import MaxPool2D
from keras.layers.core import Lambda
from keras.layers.core import Dense
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.utils.np_utils import to_categorical
from keras.models import  Sequential
from keras.preprocessing import image
from keras.optimizers import Adam

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


# create the training & test sets
train = pd.read_csv("../input/train.csv")
test= pd.read_csv("../input/test.csv")
print("Train size:{}\nTest size:{}".format(train.shape, test.shape))


# ### <span style="color:red">NOTE</span>: *Train colms are 785, where test colms are 784 because first colm is the Labels colm*. Let's see:

# In[ ]:


# The output variable is an integer from 0 to 9. 
#This is a multiclass classification problem.
train['label']


# ### <span style="color:red">Split Labels and images</span> and <span style="color:green">convert data from pandas dataframe to numpy array </span>:
# ***pd.dataframe.values*** is used to convert dataframe in numpy array
# 

# In[ ]:


X_train = train.drop(['label'], axis=1).values.astype('float32') # all pixel values
y_train = train['label'].values.astype('int32') # only labels i.e targets digits
X_test = test.values.astype('float32')


# ## Data Visualization:
# Lets look at some  images from data set with their labels.

# In[ ]:


#Convert train datset to (num_images, img_rows, img_cols) format 
X_train = X_train.reshape(X_train.shape[0], 28, 28)
for j,i in enumerate(range(15,20)):
    plt.subplot(1,5,j+1)
    plt.imshow(X_train[i], cmap=plt.get_cmap('gray'))
    plt.axis('off')
    plt.title(y_train[i]);


# ### <span style="color:red">Change dimensions for CNN</span> and <span style="color:green">expand 1 more dimension to store channels</span>:
# CNN layers takes data shape as (num_images, img_rows, img_cols, n_channels)

# In[ ]:


X_train = X_train.reshape(X_train.shape[0], 28, 28,1)
X_test = X_test.reshape(X_test.shape[0], 28, 28,1)
print("Train size:{}\nTest size:{}".format(X_train.shape, X_test.shape))


# ## Data preprocessing:
# ---
# ### 1. <span style="color:red">Feature Standardization:</span>
# * It is important preprocessing step.
# * It is used to centre the data around **zero mean and unit variance**.

# In[ ]:


mean_px = X_train.mean().astype(np.float32)
std_px = X_train.std().astype(np.float32)
def standardize(x): 
    return (x-mean_px)/std_px


# ### 2. <span style="color:red">One Hot encoding of labels:</span>
# ---
# * A one-hot vector is a vector which is 0 in most dimensions, and 1 in a single dimension. 
# * In this case, the nth digit will be represented as a vector which is 1 in the nth dimension. 
#     * For example, 3 would be [0, 0, 0, 1, 0, 0, 0, 0, 0, 0].

# In[ ]:


y_train = to_categorical(y_train)
print("Number of classes:{}".format(y_train.shape[1]))
print("This is 3 (starting from zero):{}".format(y_train[9])) #starting from zero


# # CNN Model Structure:

# In[ ]:


model = Sequential([
    Lambda(standardize, input_shape=(28,28,1)),
    Conv2D(32,(3,3), activation='relu'),
    BatchNormalization(axis=1),
    Dropout(.5),
    Conv2D(32,(3,3), activation='relu'),
    BatchNormalization(axis=1),
    MaxPool2D(),
    Conv2D(64,(3,3), activation='relu'),
    BatchNormalization(axis=1),
    Dropout(.5),
    Conv2D(64,(3,3), activation='relu'),
    MaxPool2D(),
    Flatten(),
    BatchNormalization(),
    Dropout(.5),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(.5),
    Dense(10, activation='softmax')
    ])


# # Data Augmentation: <small>(important step)</small>
# * Data augmentation means increasing the number of data points. 
# * In terms of images, it may mean that increasing the number of images in the dataset. 
# * In terms of traditional row/column format data, it means increasing the number of rows or objects.
# 
# <span style="color:orange">But why?</span>
# 
# The answer to why is simple:
# * We do not have large amount of data.
# * The more the data, the better our CNN model will be, in principle.
# 
# <span style="color:orange">OK, cool. How to do it?</span>
# 
# In images, you can:
# * rotate the original image, 
# * change lighting conditions, 
# * crop it differently, etc.
# 
# <span style="color:red">NOTE</span>: be careful about your choice of data augmentation.
# 
# **Keras provides easy way for Data Augmentation:**
# 

# In[ ]:


gen = image.ImageDataGenerator(rotation_range=10, 
                               width_shift_range=0.10, 
                               shear_range=0.5,
                               height_shift_range=0.10, 
                               zoom_range=0.10
                              )


# ## Set rest of the things:

# In[ ]:


# Complile keras model
model.compile(Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
# Learning Rate
model.optimizer.lr=0.001
# Get data batches
batches = gen.flow(X_train, y_train, batch_size=64)


# ### <span style="color:red">NOTE:</span>
# No evaluation is performed here.
# Hmm. <span style="color:orange">Why?</span>
# 
# You need all the images in the training set to train because data is already limited and for better accuracy.
# 

# ## Start training:
# **10 epochs -> test acc: .99542**
# 
# **15 epochs -> test acc: .99714**
# 
# I'm gonna run the  training for epochs=1 because it'll take too long to run here.. 
# 
# But, don't worry i'm not lying, you'll get .99714 accuracy on test set after training.
# 
# (May be accuracy fluctuate a little because of random weights initialization, but i'll be around .997xx). 
# 
# Honestly, my GTX 1060 6GB (~10mins/epoch) is better than kaggle's Tesla K80 (~28mins/epoch).
# Run the below cell to train.

# In[ ]:


# We can create plots from the history object returned by fit_generator() or fit ()
history = model.fit_generator(generator=batches, steps_per_epoch=batches.n, epochs=1) # change epochs=15


# ## Time to test our work:

# In[ ]:


predictions = model.predict_classes(X_test, verbose=0)
# Setting submission file according to Competition guidelines
submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),
                          "Label": predictions})
submissions.to_csv("submission.csv", index=False, header=True)


# ### That's it. 
# Fork this notebook and download and run it on your pc.
# ### <span style="color:purple">If you like this kernel, please upvote :)</span>

# In[ ]:




