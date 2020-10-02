#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#this is my second basic project on kaggle 
#formally going to use convolution neural network 
#basically this project can help me a lot for understanding CNN and ANN 


#simple importing the library
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #for data visualization
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


#importing the simple library for creating neural network 
#Using keras 
#using tensorflow backend 
#importing sequential library 

from keras.models import Sequential 
from keras.layers import Dense , Dropout , Lambda, Flatten

from keras.optimizers import Adam ,RMSprop
from sklearn.model_selection import train_test_split
from keras import  backend as K
from keras.preprocessing.image import ImageDataGenerator

#after running this cell the backend of tensorflow will be activated


# In[ ]:


#importing the training dataset 
train = pd.read_csv("../input/train.csv")
print(train.shape)
train.head()


# In[ ]:


#importing the test dataset 

test= pd.read_csv("../input/test.csv")
print(test.shape)
test.head()


# In[ ]:


#converting all the values of training and testing dataset into floating values
X_train = (train.iloc[:,1:].values).astype('float32') # features excluding label of images  
y_train = train.iloc[:,0].values.astype('int32') #labels of images 

X_test = test.values.astype('float32') 


# In[ ]:


#checking the value of training and testing dataset
X_train

y_train


# In[ ]:


#Convert training datset to (num_images, img_rows, img_cols) format
X_train = X_train.reshape(X_train.shape[0], 28, 28)


# In[ ]:


# data visualizing of the images 
for i in range(6, 9):
    plt.subplot(330 + (i+1))
    plt.imshow(X_train[i], cmap=plt.get_cmap('gray'))
    plt.title(y_train[i]);


# In[ ]:


#expand one more dimension in array of x_train as 1 for colour channel gray
X_train = X_train.reshape(X_train.shape[0], 28, 28,1)
X_train.shape	


# In[ ]:


# same doing with testing dataset
X_test = X_test.reshape(X_test.shape[0], 28, 28,1)
X_test.shape


# In[ ]:


# Feature Standardization
# It is important preprocessing step. 
#It is used to centre the data around zero mean and unit variance.
mean_px = X_train.mean().astype(np.float32)
std_px = X_train.std().astype(np.float32)
def standardize(x): 
    return (x-mean_px)/std_px


# In[ ]:


# One Hot encoding of labels.
# A one-hot vector is a vector which is 0 in most dimensions, and 1 in a single dimension. In this case, the nth digit will be represented as a vector which is 1 in the nth dimension.

# For example, 3 would be [0,0,0,1,0,0,0,0,0,0].

from keras.utils.np_utils import to_categorical
y_train= to_categorical(y_train)
num_classes = y_train.shape[1]
num_classes


# In[ ]:


# plotting the first 10 0 & 1 after one hot encoding
plt.title(y_train[9])
plt.plot(y_train[9])
plt.xticks(range(10));


# In[ ]:


#knowing that when creating neural networks 
#it's standard practice to create a 'random seed' so that you can get producible results in your models
#it is designing phase of neural network architecture 
seed = 43
np.random.seed(seed)


# Lets create a simple model from Keras Sequential layer
# ****Linear model****
# 
# > we import lambda layer it is used to perform simple arithmetic operations.
# 
# > The first layer of the model is defines for the input dimension of our data such as rows, columns, colour channel format .
# 
# > flatten will transform input into 1d array
# 
# > Dense is used for fully connected layer that means all neurons in previous layers will be connected to all neurons in fully connected layer
# 
# > Here it's 10, since we have to output 10 different digit labels.
# 
# >Here we use softmax as our activation function of neuron 
# 
# 

# In[ ]:


from keras.models import  Sequential
from keras.layers.core import  Lambda , Dense, Flatten, Dropout
from keras.callbacks import EarlyStopping
from keras.layers import BatchNormalization, Convolution2D , MaxPooling2D


# In[ ]:


model= Sequential()
model.add(Lambda(standardize,input_shape=(28,28,1)))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))
print("input shape ",model.input_shape)
print("output shape ",model.output_shape)


# **Compiling phase of network**
# 
# Before making network ready for training we have to make sure to add below things:
# 
# >A loss function: to measure how good the network is
# 
# >An optimizer: to update network as it sees more data and reduce loss value
# 
# >Metrics: to monitor performance of network
# 
# 

# In[ ]:


from keras.optimizers import RMSprop
model.compile(optimizer=RMSprop(lr=0.001),
 loss='categorical_crossentropy',
 metrics=['accuracy'])


# Importing imagedatagenerator from preprocessing module of keras 
# 

# In[ ]:


from keras.preprocessing import image
gen = image.ImageDataGenerator()


# ****Cross validation phase ****
# 
# >In this phase we are now going to split dataset of x train ,ytrain 
# 
# > Batching the X_train , y_train and X_val , Y_val  for generator for the next step .
# 

# In[ ]:


from sklearn.model_selection import train_test_split
X = X_train
y = y_train
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.10, random_state=42)
batches = gen.flow(X_train, y_train, batch_size=64)
val_batches=gen.flow(X_val, y_val, batch_size=64)


# ****Creating Epoch phase****
# 
# IN this we use fit_generator method for fitting our traning neural network 

# In[ ]:


history=model.fit_generator(generator=batches, steps_per_epoch=batches.n, epochs=3, 
                    validation_data=val_batches, validation_steps=val_batches.n)


# In[ ]:


history_dict = history.history
history_dict.keys()


# ******Visualization of some relation******
# 
# Visualizing the realation between epoch and loss
# Visualizing the realation between epoch and accuracy

# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, len(loss_values) + 1) #considering all epoch one by one one

# "bo" is for "blue dot"
plt.plot(epochs, loss_values, 'bo')
# b+ is for "blue crosses"
plt.plot(epochs, val_loss_values, 'b+')
plt.xlabel('Epochs')
plt.ylabel('Loss')

plt.show()


# In[ ]:


plt.clf()   # clear figure
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']

plt.plot(epochs, acc_values, 'bo')
plt.plot(epochs, val_acc_values, 'b+')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')

plt.show()


# **Fully Connected Model******
# Neurons in a fully connected layer have full connections to all activations in the previous layer, as seen in regular Neural Networks. Adding another Dense Layer to model.

# In[ ]:


def get_fc_model():
    model = Sequential([
        Lambda(standardize, input_shape=(28,28,1)),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(10, activation='softmax')
        ])
    model.compile(optimizer='Adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# In[ ]:


fc = get_fc_model()
fc.optimizer.lr=0.01


# In[ ]:


#simple using fit_generator method 
history=fc.fit_generator(generator=batches, steps_per_epoch=batches.n, epochs=1, 
                    validation_data=val_batches, validation_steps=val_batches.n)


# **Adding more dense layers**
# Adding more dense layer for increasing accuracy 
# Adding Layers in cnn is very easy 
# now creating a function for cnn model
# Commenting this because it will take lot of a time 

# In[ ]:


#def get_cnn_model():
#    model = Sequential([
 #       Lambda(standardize, input_shape=(28,28,1)),
  #      Convolution2D(32,(3,3), activation='relu'),
   #     Convolution2D(32,(3,3), activation='relu'),
    #    MaxPooling2D(),
     #   Convolution2D(64,(3,3), activation='relu'),
      #  Convolution2D(64,(3,3), activation='relu'),
       # MaxPooling2D(),
#        Flatten(),
 #       Dense(512, activation='relu'),
  #      Dense(10, activation='softmax')
   #     ])
 #   model.compile(Adam(), loss='categorical_crossentropy',
#                  metrics=['accuracy'])
 #   return model


# In[ ]:


#model= get_cnn_model()
#model.optimizer.lr=0.01


# In[ ]:


#history=model.fit_generator(generator=batches, steps_per_epoch=batches.n, epochs= 1, 
 #                   validation_data=val_batches, validation_steps=val_batches.n)
#Sorry very time taking as we have added upto 2 sdense layer


# ****Submitting the prediction for kaggle ****
# We have to train with whole dataset now 

# In[ ]:


fc.optimizer.lr=0.01
gen = image.ImageDataGenerator()
batches = gen.flow(X, y, batch_size=64)
history=fc.fit_generator(generator=batches, steps_per_epoch=batches.n, epochs=3)


# In[ ]:


#creating a submission dataset for kaggle submission 
predictions = fc.predict_classes(X_test, verbose=0)

subm =pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),
                         "Label": predictions})
subm.to_csv("DR.csv", index=False, header=True)


# In[ ]:




