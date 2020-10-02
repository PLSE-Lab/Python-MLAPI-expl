#!/usr/bin/env python
# coding: utf-8

# ## Visualize a Convolutional Neural Network
# This kernel is used for demonstrating the visualization of layers in a Convolutional Neural Network, which uses data from the Digit Recognizer. For kernel that actually solves the problem on the Digit Recognizer, feel free to check out my another kernel, [Fast and Easy CNN for starters in Keras 0.99471
# ](https://www.kaggle.com/codeastar/fast-and-easy-cnn-for-starters-in-keras-0-99471) .
# 
# You can also check out my [blog](http://www.codeastar.com/visualize-convolutional-neural-network/) for more details on the related topic.
# 

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# In[ ]:


#load the traing and testing datasets
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')


# In[ ]:


#import required modules
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import RMSprop
from keras.callbacks import ReduceLROnPlateau
import matplotlib.pyplot as plt
from random import randrange


# In[ ]:


#define our CNN model
def cnn_model(result_class_size):
    model = Sequential()
    model.add(Conv2D(30, (5, 5), input_shape=(28,28,1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(15, (3, 3), activation='relu'))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(result_class_size, activation='softmax'))   
    model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])
    return model


# Prepare the training materials and start to train our CNN model.

# In[ ]:


df_train_x = df_train.iloc[:,1:]  #get 784 pixel value columns after the first column
df_train_y = df_train.iloc[:,:1]  #get the lable column


# In[ ]:


arr_train_y = np_utils.to_categorical(df_train_y['label'].values)
model = cnn_model(arr_train_y.shape[1])
model.summary()


# In[ ]:


df_train_x = df_train_x / 255
df_test = df_test / 255
 
#reshape training X and texting X to (number, height, width, channel)
arr_train_x_28x28 = np.reshape(df_train_x.values, (df_train_x.values.shape[0], 28, 28, 1))


# In[ ]:


#train only 3 epochs for demo purpose
model.fit(arr_train_x_28x28, arr_train_y, epochs=3, batch_size=100)


# How does our trained CNN see the world? By using following filters: 

# In[ ]:


#get_weights [x, y, channel, nth convolutions layer ]
weight_conv2d_1 = model.layers[0].get_weights()[0][:,:,0,:]

col_size = 6
row_size = 5
filter_index = 0
fig, ax = plt.subplots(row_size, col_size, figsize=(12,8))
for row in range(0,row_size): 
  for col in range(0,col_size):
    ax[row][col].imshow(weight_conv2d_1[:,:,filter_index],cmap="gray")
    filter_index += 1


# We pick an image from the testing dataset. 

# In[ ]:


test_index = randrange(df_test.shape[0])
test_img = arr_train_x_28x28[test_index]
plt.imshow(test_img.reshape(28,28), cmap='gray')
plt.title("Index:[{}]".format(test_index))
plt.show()


# Then let our trained CNN recognize the image and get the outputs from each layer.

# In[ ]:


from keras.models import Model
layer_outputs = [layer.output for layer in model.layers]
activation_model = Model(inputs=model.input, outputs=layer_outputs)
activations = activation_model.predict(test_img.reshape(1,28,28,1))


# Let's write a function to display outputs in defined size and layer. 

# In[ ]:


def display_activation(activations, col_size, row_size, act_index): 
    activation = activations[act_index]
    activation_index=0
    fig, ax = plt.subplots(row_size, col_size, figsize=(row_size*2.5,col_size*1.5))
    for row in range(0,row_size): 
      for col in range(0,col_size):
        ax[row][col].imshow(activation[0, :, :, activation_index], cmap='gray')
        activation_index += 1


# First, we visualize the first convolutional layer with 30 filters.

# In[ ]:


#conv2d_1
display_activation(activations, 6, 5, 0)


# Then the pooling layer.

# In[ ]:


#max_pooling2d_1
display_activation(activations, 6, 5, 1)


# And our second convolutional layer with 15 filters.

# In[ ]:


#conv2d_2
display_activation(activations, 5, 3, 2)


# We have a dropout layer which its dropping rate is set to 25% of inputs. 

# In[ ]:


#dropout_1
display_activation(activations, 5, 3, 3)


# Finally, we get the outputs from the last fully connected layer.

# In[ ]:


act_dense_3  = activations[7]
act_dense_3


# In[ ]:


y = act_dense_3[0]
x = range(len(y))
plt.xticks(x)
plt.bar(x, y)
plt.show()

