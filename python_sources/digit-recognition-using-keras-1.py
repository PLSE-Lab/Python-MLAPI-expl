#!/usr/bin/env python
# coding: utf-8

# The complete notebook is inspired and adopted by the kernal [https://www.kaggle.com/poonaml/deep-neural-network-keras-way](http://) from Poonam Ligade.

# In[ ]:


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


# In[ ]:


import warnings
warnings.filterwarnings("ignore")


# **import libraries**

# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


from keras.models import Sequential
from keras.layers import Dense, Dropout, Lambda, Flatten
from keras.optimizers import Adam, RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K


# **load train and test data**

# In[ ]:


train = pd.read_csv("../input/train.csv")
print(train.shape)
(train.head(5))


# In[ ]:


test = pd.read_csv("../input/test.csv")
print(test.shape)
test.head(5)


# In[ ]:


x_train = train.iloc[:,1:].values.astype("float32")
y_train = train.iloc[:,0].values.astype("int32")
y_train = y_train.reshape(y_train.shape[0],1)
x_test = test.values.astype("float32")


# In[ ]:


x_train.shape, y_train.shape, x_test.shape


# **Data Visualization**

# In[ ]:


x_train = x_train.reshape(x_train.shape[0], 28, 28)

for i in range(6,9):
    plt.subplot(330 + (i+1))
    plt.imshow(x_train[i], cmap=plt.get_cmap('gray'))
    plt.title(y_train[i])


# In[ ]:


x_train = x_train.reshape(x_train.shape[0], 28,28,1)


# In[ ]:


x_train.shape


# ****Preprocessing the digit images****

# **feature standardization**

# In[ ]:


mean_value = x_train.mean().astype("float32")
std_value = x_train.mean().astype("float32")

def standardize(x):
    return (x-mean_value)/std_value


# **one hot encoding of labels**

# In[ ]:


from keras.utils.np_utils import to_categorical

y_train = to_categorical(y_train)
print(y_train.shape)
num_classes = y_train.shape[1]
print(num_classes)


# In[ ]:


# plot any label (for ex 10th label)

plt.title(y_train[9])
plt.plot(y_train[9])
plt.xticks(range(10))
plt.show()


# **Designing neural network architecture**

# In[ ]:


# seed for reproducibility

seed = 43
np.random.seed(seed)


# **Linear Model**

# In[ ]:


from keras.models import Sequential
from keras.layers.core import Dropout, Dense, Lambda, Flatten
from keras.callbacks import EarlyStopping
from keras.layers import BatchNormalization, MaxPooling2D, Convolution2D


# Lets create a simple model from Keras Sequential layer.
# 
# Lambda layer performs simple arithmetic operations like sum, average, exponentiation etc.
# 
# In 1st layer of the model we have to define input dimensions of our data in (rows,columns,colour channel) format. (In theano colour channel comes first)
# 
# Flatten will transform input into 1D array.
# Dense is fully connected layer that means all neurons in previous layers will be connected to all neurons in fully connected layer. In the last layer we have to specify output dimensions/classes of the model. Here it's 10, since we have to output 10 different digit labels.

# In[ ]:


model = Sequential()
model.add(Lambda(standardize, input_shape=(28,28,1)))
model.add(Flatten())
model.add(Dense(10, activation="softmax"))

print("input shape : ", model.input_shape)
print("output shape : ", model.output_shape)


# **compile network**

# Before making network ready for training we have to make sure to add below things:
# 
# A loss function: to measure how good the network is
# 
# An optimizer: to update network as it sees more data and reduce loss value
# 
# Metrics: to monitor performance of network

# In[ ]:


from keras.optimizers import RMSprop
model.compile(optimizer=RMSprop(lr=0.001), loss="categorical_crossentropy", metrics=["accuracy"])


# In[ ]:


from keras.preprocessing import image
gen = image.ImageDataGenerator()


# **cross validation**

# In[ ]:


from sklearn.model_selection import train_test_split
x = x_train
y = y_train
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)

batches = gen.flow(x_train, y_train, batch_size=64)
val_batches = gen.flow(x_val, y_val, batch_size=64)


# In[ ]:


history = model.fit_generator(generator=batches, steps_per_epoch=batches.n, epochs=3, validation_data=val_batches, validation_steps=val_batches.n)


# In[ ]:


history_dict = history.history
history_dict


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

loss_values = history_dict["loss"]
val_loss_values = history_dict["val_loss"]
epochs = range(1, len(loss_values)+1)

plt.plot(epochs, loss_values, "-bo")
plt.plot(epochs, val_loss_values, "-b+")
plt.xlabel("Epochs")
plt.ylabel("Losses")
plt.show()


# In[ ]:


acc_values = history_dict["acc"]
val_acc_values = history_dict["val_acc"]
plt.plot(epochs, acc_values, "-ro")
plt.plot(epochs, val_acc_values, "-r+")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.show()


# **Fully connnected model**

# Neurons in a fully connected layer have full connections to all activations in the previous layer, as seen in regular Neural Networks. Adding another Dense Layer to model.

# In[ ]:


def get_fc_model():
    model = Sequential([Lambda(standardize, input_shape=(28,28,1)),Flatten(),Dense(512, activation='relu'),Dense(10, activation='softmax')])
    model.compile(optimizer='Adam', loss='categorical_crossentropy',metrics=['accuracy'])
    return model

fc = get_fc_model()
fc.optimizer.lr=0.01


# In[ ]:


history = fc.fit_generator(generator=batches, steps_per_epoch=batches.n, epochs=1, validation_data=val_batches, validation_steps=val_batches.n)


# **Convolutional neural network**

# In[ ]:


from keras.layers import Convolution2D, MaxPooling2D


# In[ ]:


def get_cnn_model():
    model = Sequential( [Lambda(standardize, input_shape=(28,28,1)), Convolution2D(32,(3,3), activation="relu"), Convolution2D(32,(3,3), activation='relu'), MaxPooling2D(), Convolution2D(64,(3,3), activation='relu'), Convolution2D(64,(3,3), activation='relu'), MaxPooling2D(), Flatten(), Dense(512, activation='relu'), Dense(10, activation='softmax') ] )
    model.compile(Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# In[ ]:


model = get_cnn_model()
model.optimizer.lr=0.01


# In[ ]:


history = model.fit_generator(generator=batches, steps_per_epoch=batches.n, epochs=1, validation_data=val_batches, validation_steps=val_batches.n)


# **Data Augmentation**
# 
# It is tehnique of showing slighly different or new images to neural network to avoid overfitting. And to achieve better generalization. In case you have very small dataset, you can use different kinds of data augmentation techniques to increase your data size. Neural networks perform better if you provide them more data.
# 
# Different data aumentation techniques are as follows:
# 
# Cropping
# Rotating
# Scaling
# Translating
# Flipping
# Adding Gaussian noise to input images etc.

# In[ ]:


gen = ImageDataGenerator( rotation_range=8, width_shift_range=0.08, shear_range=0.3, height_shift_range=0.08, zoom_range=0.08 )
batches = gen.flow(x_train, y_train, batch_size=64)
val_batches = gen.flow(x_val, y_val, batch_size=64)


# In[ ]:


model.optimizer.lr=0.001
history = model.fit_generator( generator=batches, steps_per_epoch=batches.n, epochs=1, validation_data=val_batches, validation_steps=val_batches.n )


# **Batch Normalization**

# Batch Normalization helps to fine tune hyperparameters more better and train really deep neural networks.

# In[ ]:


from keras.layers.normalization import BatchNormalization


# In[ ]:


#def get_bn_model():
#    model = Sequential( [ Lambda(standardize, input_shape=(28,28,1)), Convolution2D(32,(3,3), activation='relu'),BatchNormalization(axis=1),Convolution2D(32,(3,3), activation='relu'),MaxPooling2D(),BatchNormalization(axis=1),Convolution2D(64,(3,3), activation='relu'),BatchNormalization(axis=1),Convolution2D(64,(3,3), activation='relu'),MaxPooling2D(),Flatten(),BatchNormalization(),Dense(512, activation='relu'),BatchNormalization(),Dense(10, activation='softmax') ] )
#    model.compile(Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
#    return model


# In[ ]:


#model = get_bn_model()
#model.optimizer.lr=0.01
#history=model.fit_generator(generator=batches, steps_per_epoch=batches.n, epochs=1, validation_data=val_batches, validation_steps=val_batches.n)


# In[ ]:


#model.optimizer.lr=0.01
#gen = image.ImageDataGenerator()
#batches = gen.flow(X, y, batch_size=64)
#history=model.fit_generator(generator=batches, steps_per_epoch=batches.n, epochs=3)


# **Reference**

# The complete notebook is inspired and adopted by the kernal [https://www.kaggle.com/poonaml/deep-neural-network-keras-way](http://) from Poonam Ligade.
# 

# In[ ]:




