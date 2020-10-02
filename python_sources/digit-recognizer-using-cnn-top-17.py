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

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # Importing Library

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop


# # Load Data

# In[ ]:


train = pd.read_csv("../input/digit-recognizer/train.csv")
test = pd.read_csv("../input/digit-recognizer/test.csv")


# In[ ]:


train.columns


# In[ ]:


print(train.shape)
train.head()


# In[ ]:


print(test.shape)
test.head()


# Keeping dependent variable in y_train and independent variable in x_train

# In[ ]:


y_train = train["label"].astype('float32')
x_train = train.drop(labels = ["label"],axis = 1).astype('float32')

y_train.value_counts()


# In[ ]:


x_train.shape, y_train.shape


# # Missing Value

# In[ ]:


x_train.isnull().any().describe()


# In[ ]:


test.isnull().any().describe()


# 
# There is no missing values in the train and test dataset. So we can safely go ahead.

# # Normalization

# We can divide each pixel with 255.0 to get NORMALIZATION

# In[ ]:


x_train = x_train / 255.0
test = test / 255.0


# # Reshaping

# Input data are in 1-D, we re- shape into 3-D matrix. 
# For RGB images, there is 3 channels, we would have reshaped 784px vectors to 28x28x3 3D matrices.
# For Black and White image we re-shape into 28x28x1

# In[ ]:


#Reshaping the data for 2D CNN input into (-1,28,28,1) size
x_train = np.array(x_train).reshape(-1,28,28,1)
test = np.array(test).reshape(-1,28,28,1)
print(x_train.shape)
print(test.shape)


# In[ ]:


# Some examples
g = plt.imshow(x_train[8][:,:,0])
print(y_train[8])


# # Label Encoding

# In[ ]:


# Encode labels to one hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])
y_train = keras.utils.to_categorical(y_train, 10)


# # CNN Model

# Forming Model

# In[ ]:


model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu', input_shape = (28,28,1)))
model.add(BatchNormalization())
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))


model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(BatchNormalization())
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))


model.add(Flatten())
model.add(Dense(512, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation = "softmax"))


# Compile Model

# In[ ]:


optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)


# In[ ]:


model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])


# Getting the summary of Model

# In[ ]:


model.summary()


# Data Augmentation

# In[ ]:


datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images


datagen.fit(x_train)


# Fit the Model

# In[ ]:


model.fit_generator(datagen.flow(x_train,y_train, batch_size = 86),epochs = 30, steps_per_epoch = x_train.shape[0]//64)
                            


# Prediction based on test data.

# In[ ]:


pred = model.predict(test)

# Convert predictions classes to one hot vectors 
final_pred = np.argmax(pred,axis = 1) 


# In[ ]:


final_pred


# In[ ]:


sample_sub = pd.read_csv('../input/digit-recognizer/sample_submission.csv')


# In[ ]:


sample_sub = pd.DataFrame({"ImageId": list(range(1,len(final_pred)+1)),
                         "Label": final_pred})
sample_sub.to_csv('CNN_submission1.csv', index=False)
sample_sub.head()

