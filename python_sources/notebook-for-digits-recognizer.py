#!/usr/bin/env python
# coding: utf-8

# Notebook for Digits Recognizer

# In[ ]:


import numpy as np
import pandas as pd
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt

batch_size = 128
nb_classes = 10
nb_epoch = 1

# input image dimensions
img_rows, img_cols = 28, 28
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)

df1 = pd.read_csv('../input/train.csv')
df2 = pd.read_csv('../input/test.csv')


# In[ ]:


print(len(df1))
#print(df1.columns.values)

y_train = np.array(df1['label'].astype(float))
X_train = np.array(df1.drop(['label'], 1).astype(float))

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.2, random_state = 4)
X_test = np.array(df2).reshape(28000, 784)

print(y_train.shape, X_train.shape)
print(y_val.shape, X_val.shape)
print(K.image_dim_ordering())
print(X_test.shape)


# In[ ]:


X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
X_val = X_val.reshape(X_val.shape[0], img_rows, img_cols, 1)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

X_train = X_train.astype('float32')
X_val = X_val.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_val /= 255
X_test /= 255
print(X_train.shape, 'X_train shape')
print(X_train.shape[0], 'train samples')
print(X_val.shape[0], 'validation samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_val = np_utils.to_categorical(y_val, nb_classes)


# In[ ]:


model = Sequential()

model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1], border_mode='valid', input_shape=input_shape))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=0, validation_data=(X_val, Y_val))
score = model.evaluate(X_val, Y_val, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

y_pred = model.predict_classes(X_test)


# In[ ]:


print(y_pred[:25])
#print(y_test[0:25])

fig, axes = plt.subplots(12,12, figsize = (28,28))
fig.subplots_adjust(hspace = 0.1, wspace = 0.1)

print(axes.shape)

# Plot the impages starting from i = 1
for i, ax in enumerate(axes.flat):
    a = i+100
    im = np.reshape(X_test[a], (28,28))
    ax.imshow(im, cmap = 'binary')
    ax.text(0.95, 0.05, 'Predict={0}'.format(y_pred[a]), ha='right', transform = ax.transAxes, color = 'blue', size=20)
    ax.set_xticks([])
    ax.set_yticks([])

