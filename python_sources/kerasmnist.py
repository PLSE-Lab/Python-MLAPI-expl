#!/usr/bin/env python
# coding: utf-8

# # Major Imports

# In[ ]:


from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import matplotlib.pyplot as plt
from time import time
from keras.callbacks import TensorBoard
import tensorflow as tf
import pandas as pd
import numpy as np


# # Some Pre-Processing Steps
# 
# ### Batch SIze = 200
# ### Epochs = 30

# In[ ]:


batch_size = 200
num_classes = 10
epochs = 30

# input image dimensions
img_rows, img_cols = 28, 28

train_data = pd.read_csv("../input/train.csv")
train_data = np.array(train_data)
X_train = train_data[:,1:]
Y_train = train_data[:,0]

test_data = pd.read_csv("../input/test.csv")
X_test = np.array(test_data)


if K.image_data_format() == 'channels_first':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols,1)
    input_shape = (img_rows, img_cols, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print('x_train shape:', X_train.shape)
print('x_test shape:', X_test.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = keras.utils.to_categorical(Y_train, num_classes)


Y_train.shape


# # Network Initialization
# 
# ### 3 Convolution Layers of size 3x3, 5x5 and 3x3 having rectified linear Activation function followed by Max Pool Layer of size 2x2  after 2nd and 3rd Convolution Layer followed by dense layer with dropout of 0.25.

# In[ ]:


model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))

model.add(Conv2D(32, kernel_size=(5, 5),
                 activation='relu',))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(200, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
model.summary()


# # Fit Model and compute accuracy score 

# In[ ]:


#cb=keras.callbacks.TensorBoard(log_dir='/tmp/mnist_demo/2', histogram_freq=0, batch_size=200, write_graph=True, write_grads=True, write_images=True,embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, update_freq='epoch')
history = model.fit(X_train, Y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1, shuffle = True,
          validation_split= 0.1)


# # Error vs Epoch

# In[20]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Error vs Epochs')
plt.ylabel('Error')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')
plt.show()


# ## Predictions

# In[21]:


Y_test = model.predict(X_test)
results = np.argmax(Y_test,axis = 1)
results = pd.Series(results,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)
submission.to_csv("submission.csv",index=False)


# In[ ]:




