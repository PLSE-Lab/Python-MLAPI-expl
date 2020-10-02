#!/usr/bin/env python
# coding: utf-8

# # Fashion MNIST - 3D CNN using Tensorflow,Keras
# 
# *  The objective of this kernel is to define,compile & evaluate a 3 layer convolutional neural network(CNN) model 
# 
# *  Visualise the validation accuracy and validation loss
# 

# In[ ]:


# Import Libraries

import pandas as pd

import numpy as np

import warnings

warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import keras

from keras.models import Sequential

from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout

from keras.optimizers import Adam

from keras.callbacks import TensorBoard

import os

print(os.listdir("../input"))


# **Create dataframes for train and test datasets**

# In[ ]:


train_df = pd.read_csv('/kaggle/input/fashion-mnist_train.csv',sep=',')
test_df = pd.read_csv('/kaggle/input/fashion-mnist_test.csv', sep = ',')


# Now it is observed that the first column is the label data and because it has 10 classes so it is going to have from 0 to 9.The remaining columns are the actual pixel data.Here as you can see there are about 784 columns that contain pixel data.
# Here each row is a different image representation in the form pixel data.
# 
# Now let us split the train data into x and y arrays where x represents the image data and y represents the labels.
# To do that we need to convert the dataframes into numpy arrays of float32 type which is the acceptable form for tensorflow and keras.

# In[ ]:


train_data = np.array(train_df, dtype = 'float32')
test_data = np.array(test_df, dtype='float32')


# Now let us slice the train arrays into x and y arrays namely x_train,y_train to store all image data and label data respectively.
# i.e 
# 
# - x_train contains all the rows and all columns except the label column and excluding header info .
# - y_train contains all the rows and first column and excluding header info .
# 
# Similarly slice the test arrays into x and y arrays namely x_train,y_train to store all image data and label data respectively.
# i.e 
# - x_test contains all the rows and all columns except the label column and excluding header info .
# - y_test contains all the rows and first column and excluding header info .
# ####  Important Note : Since the image data in x_train and x_test is from 0 to 255 ,  we need to rescale this from 0 to 1.To do this we need to divide the x_train and x_test by 255 

# In[ ]:


x_train = train_data[:,1:]/255
y_train = train_data[:,0]
x_test= test_data[:,1:]/255
y_test=test_data[:,0]


# Now we are gonna split the training data into validation and actual training data for training the model and testing it using the validation set. This is achieved using the train_test_split method of scikit learn library.

# In[ ]:


x_train,x_validate,y_train,y_validate = train_test_split(x_train,y_train,test_size = 0.2,random_state = 12345)


# 
# Now let us visualise the sample image how it looks like in 28 * 28 pixel size

# In[ ]:


image = x_train[13,:].reshape((28,28))
plt.imshow(image)
plt.show()


# As you can observe above the shape of shoe from the sample image
# 
# ### Create the 3D Convolutional Neural Networks (CNN)
# 
# - #### Define the model
# - #### Compile the model
# - #### Fit the model
# 
# First of all let us define the shape of the image before we define the model

# In[ ]:


image_rows = 28
image_cols = 28
batch_size = 512
image_shape = (image_rows,image_cols,1) # Defined the shape of the image as 3d with rows and columns and 1 for the 3d visualisation


# Now we need to do more formating on the x_train,x_test and x_validate sets.

# In[ ]:


x_train = x_train.reshape(x_train.shape[0],*image_shape)
x_test = x_test.reshape(x_test.shape[0],*image_shape)
x_validate = x_validate.reshape(x_validate.shape[0],*image_shape)


# In[ ]:


print("x_train shape = {}".format(x_train.shape))
print("x_test shape = {}".format(x_test.shape))
print("x_validate shape = {}".format(x_validate.shape))


# - #### Define the model 

# In[ ]:


name = '1_Layer'
cnn_model_1 = Sequential([
    Conv2D(32, kernel_size=3, activation='relu', input_shape=image_shape, name='Conv2D-1'),
    MaxPooling2D(pool_size=2, name='MaxPool'),
    Dropout(0.2, name='Dropout'),
    Flatten(name='flatten'),
    Dense(32, activation='relu', name='Dense'),
    Dense(10, activation='softmax', name='Output')
], name=name)

name = '2_Layer'
cnn_model_2 = Sequential([
    Conv2D(32, kernel_size=3, activation='relu', input_shape=image_shape, name='Conv2D-1'),
    MaxPooling2D(pool_size=2, name='MaxPool'),
    Dropout(0.2, name='Dropout-1'),
    Conv2D(64, kernel_size=3, activation='relu', name='Conv2D-2'),
    Dropout(0.25, name='Dropout-2'),
    Flatten(name='flatten'),
    Dense(64, activation='relu', name='Dense'),
    Dense(10, activation='softmax', name='Output')
], name=name)

name='3_layer'
cnn_model_3 = Sequential([
    Conv2D(32, kernel_size=3, activation='relu', 
           input_shape=image_shape, kernel_initializer='he_normal', name='Conv2D-1'),
    MaxPooling2D(pool_size=2, name='MaxPool'),
    Dropout(0.25, name='Dropout-1'),
    Conv2D(64, kernel_size=3, activation='relu', name='Conv2D-2'),
    Dropout(0.25, name='Dropout-2'),
    Conv2D(128, kernel_size=3, activation='relu', name='Conv2D-3'),
    Dropout(0.4, name='Dropout-3'),
    Flatten(name='flatten'),
    Dense(128, activation='relu', name='Dense'),
    Dropout(0.4, name='Dropout'),
    Dense(10, activation='softmax', name='Output')
], name=name)

cnn_models = [cnn_model_1, cnn_model_2, cnn_model_3]


# **the model summaries**

# In[ ]:


for model in cnn_models:
    model.summary()


#  **train the models and save results to a dict**

# In[ ]:


history_dict = {}
for model in cnn_models:
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=Adam(),
        metrics=['accuracy']
    )
    
    history = model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=50, verbose=1,
        validation_data=(x_validate, y_validate)
    )
    
    history_dict[model.name] = history


# ### Plot the Accuracy and Loss

# In[ ]:


fig,(ax1,ax2)=plt.subplots(2,figsize=(8,6))
for history in history_dict:
    val_acc = history_dict[history].history['val_acc']
    val_loss = history_dict[history].history['val_loss']
    ax1.plot(val_acc, label=history)
    ax2.plot(val_loss, label=history)
ax1.set_ylabel('Validation Accuracy')
ax2.set_ylabel('Validation Loss')
ax1.set_xlabel('Epochs')
ax1.legend()
ax2.legend()
plt.show()  


# As you can see in the above graph as the no of convolution layers increases the accuracy is increasing and loss keep decreasing
# 
# ## If you like this kernel greatly appreciate an Upvote.

# In[ ]:




