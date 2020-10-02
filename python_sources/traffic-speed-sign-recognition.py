#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from matplotlib import pyplot
import tensorflow as tf


# In[ ]:


# load dataset
(x1_train, y1_train), (x1_test, y1_test) = tf.keras.datasets.mnist.load_data()


# In[ ]:


x_train = []
y_train = []
for i in range(len(x1_train)):
    if y1_train[i] != 0:
        x_train.append(x1_train[i])
        y_train.append(y1_train[i])
x_train = np.array(x_train)
y_train = np.array(y_train)


# In[ ]:


x_test = []
y_test = []
for i in range(len(x1_test)):
    if y1_test[i] != 0:
        x_test.append(x1_test[i])
        y_test.append(y1_test[i])
x_test = np.array(x_test)
y_test = np.array(y_test)


# In[ ]:


# summarize loaded dataset
print('Train: X=%s, y=%s' % (x_train.shape, y_train.shape))
print('Test: X=%s, y=%s' % (x_test.shape, y_test.shape))


# In[ ]:


# plot first few images
for i in range(9):
    pyplot.subplot(330 + 1 + i)
    pyplot.imshow(x_train[i], cmap=pyplot.get_cmap('gray'))
# show the figure
pyplot.show()


# In[ ]:


x_train = x_train.reshape(-1, 28,28, 1)
x_test = x_test.reshape(-1, 28,28, 1)
x_train.shape, x_test.shape


# In[ ]:


x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train = x_train / 255.
x_test = x_test / 255.


# In[ ]:


print(np.unique(y_train))


# In[ ]:


from keras.utils import to_categorical
train_y_one_hot = to_categorical(y_train, num_classes=10)
test_y_one_hot = to_categorical(y_test, num_classes=10)


# In[ ]:


print('Original label:', y_train[0])
print('After conversion to one-hot:', train_y_one_hot[0])


# In[ ]:


from sklearn.model_selection import train_test_split
train_X,valid_X,train_label,valid_label = train_test_split(x_train, train_y_one_hot, test_size=0.2, random_state=13)


# In[ ]:


import keras
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU


# In[ ]:


batch_size = 64
epochs = 20
num_classes = 10


# In[ ]:


model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',input_shape=(28,28,1),padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D((2, 2),padding='same'))
model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
model.add(LeakyReLU(alpha=0.1))                  
model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
model.add(Flatten())
model.add(Dense(128, activation='linear'))
model.add(LeakyReLU(alpha=0.1))                  
model.add(Dense(num_classes, activation='softmax'))


# In[ ]:


model.summary()


# In[ ]:


model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])


# In[ ]:


train = model.fit(train_X, train_label, batch_size=60,epochs=20,verbose=1,validation_data=(valid_X, valid_label))


# In[ ]:


import os
filenames = (os.listdir("../input/"))
verify = []
for i in filenames:
    if "10" in i:
        verify.append(1)
    if "20" in i:
        verify.append(2)
    if "30" in i:
        verify.append(3)
    if "40" in i:
        verify.append(4)
    if "50" in i:
        verify.append(5)
    if "60" in i:
        verify.append(6)
    if "90" in i:
        verify.append(9)
verify = np.array(verify)


# In[ ]:


from keras.preprocessing.image import load_img, img_to_array
from numpy import asarray
from PIL import Image
verify_x =[]
for i in filenames:
    photo = load_img("../input/"+i, grayscale=True, target_size=(28,28))
    photo = img_to_array(photo)
    verify_x.append(photo)
verify_x = asarray(verify_x)


# In[ ]:


pred_x = model.predict(verify_x)
pred_x = np.argmax(np.round(pred_x),axis=1)
pred_x.shape


# In[ ]:


img_no = 2
plt.imshow(verify_x[img_no].reshape(28,28),cmap='gray')
print(pred_x[img_no]*10," KMPH")


# In[ ]:




