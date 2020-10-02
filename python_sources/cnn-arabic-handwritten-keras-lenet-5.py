#!/usr/bin/env python
# coding: utf-8

# # CNN Arabic Handwritten using Keras and <br>LeNet-5 Architecture

# <img align=left width=500 src='https://upload.wikimedia.org/wikipedia/commons/thumb/5/50/Birmingham_Quran_manuscript.jpg/1920px-Birmingham_Quran_manuscript.jpg'/>

# ## **Table content**
# - [Goal](#goal)
# - [Data Preparation](#preparation)
# - [Architecture](#architecture)
# - [Implementation](#implementation)
# - [Conclusion](#conclusion)

# <a id='goal'></a>
# # Goal
# 
# The goal of this notebook is to use LeNet-5 in order to train a model to recognize arabic handwritten characters.<br>
# <br>
# ### What is LeNet ?
# LeNet is a convolutional neural network structure proposed by Yann LeCun et al. in 1998. In general, LeNet refers to lenet-5 and is a simple convolutional neural network. Convolutional neural networks are a kind of feed-forward neural network whose artificial neurons can respond to a part of the surrounding cells in the coverage range and perform well in large-scale image processing.
# <br>
# ### Reference
# https://en.wikipedia.org/wiki/LeNet

# <a id='preparation'></a>
# # Data Preparation

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[ ]:


import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Flatten, Dense
from tensorflow.keras.optimizers import SGD


# In[ ]:


train_images = pd.read_csv('../input/ahcd1/csvTrainImages 13440x1024.csv')
train_label = pd.read_csv('../input/ahcd1/csvTrainLabel 13440x1.csv')

test_images = pd.read_csv('../input/ahcd1/csvTestImages 3360x1024.csv')
test_label = pd.read_csv('../input/ahcd1/csvTestLabel 3360x1.csv')


# In[ ]:


X_train = train_images.to_numpy()
y_train = train_label.to_numpy()

X_test = test_images.to_numpy()
y_test = test_label.to_numpy()


# In[ ]:


nr_classes = len(np.unique(train_label))


# In[ ]:


X_train = X_train.reshape(13439, 32, 32, 1)
y_train = tf.keras.backend.one_hot(y_train, nr_classes)
y_train = y_train.numpy().reshape(13439, 28)

X_test = X_test.reshape(3359, 32, 32, 1)
y_test = tf.keras.backend.one_hot(y_test, nr_classes)
y_test = y_test.numpy().reshape(3359, 28)


# <a id='architecture'></a>
# # Architecture

# <img align=left src="https://dpzbhybb2pdcj.cloudfront.net/elgendy/v-8/Figures/05-06_img_0032.png" />
# <br><br><br><br><br><br><br><br><br>
# We use the original mnist architecture but instead of numbers, we train with arabic alphabet.

# <a id='implementation'></a>
# # Implementation

# In[ ]:


# Instantiate an empty sequential model
model = Sequential()
# C1 Convolutional Layer
model.add(Conv2D(filters = 6, kernel_size = 5, strides = 1, activation = 'tanh',
input_shape = (32,32,1), padding = 'same'))
 
# S2 Pooling Layer
model.add(AveragePooling2D(pool_size = 2, strides = 2, padding = 'valid'))
 
# C3 Convolutional Layer
model.add(Conv2D(filters = 16, kernel_size = 5, strides = 1,activation = 'tanh',
padding = 'valid'))
# S4 Pooling Layer
model.add(AveragePooling2D(pool_size = 2, strides = 2, padding = 'valid'))
 
# C5 Convolutional Layer
model.add(Conv2D(filters = 120, kernel_size = 5, strides = 1,activation = 'tanh',
padding = 'valid'))
 
# Flatten the CNN output to feed it with fully connected layers
model.add(Flatten())
 
# FC6 Fully Connected Layer
model.add(Dense(units = 84, activation = 'tanh'))
 
# FC7 Output layer with softmax activation
model.add(Dense(units = 28, activation = 'softmax'))
 
# print the model summary
model.summary()


# In[ ]:


def lr_schedule(epoch):
    # initiate the learning rate with value = 0.0005
    lr = 5e-4
    # lr = 0.0005 for the first two epochs, 0.0002 for the next three epochs,
    # 0.00005 for the next four, then 0.00001 thereafter.
    if epoch > 2:
        lr = 2e-4
    elif epoch > 5:
        lr = 5e-5
    elif epoch > 9:
        lr = 1e-5
    return lr


# In[ ]:


model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=lr_schedule(0)), metrics=['accuracy'])


# In[ ]:


hist = model.fit(X_train, y_train, batch_size=124, epochs=500,
validation_data=(X_test, y_test), verbose=0, shuffle=True)


# <a id='conclusion'></a>
# # Conclusion

# In[ ]:


plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

