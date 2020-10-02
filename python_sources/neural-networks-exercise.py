#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils
from keras.datasets import mnist
import matplotlib.pyplot as plt


# In[ ]:


import copy

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_test_copy = copy.deepcopy(X_test)


# In[ ]:


plt.subplot(221)
plt.imshow(X_train[0], cmap=plt.get_cmap('gray'))
plt.subplot(222)
plt.imshow(X_train[1], cmap=plt.get_cmap('gray'))
plt.subplot(223)
plt.imshow(X_train[2], cmap=plt.get_cmap('gray'))
plt.subplot(224)
plt.imshow(X_train[3], cmap=plt.get_cmap('gray'))
# show the plot
plt.show()


# In[ ]:


seed = 42
np.random.seed(seed)


# In[ ]:


num_pixels = X_train.shape[1] * X_train.shape[2]
print(num_pixels)
print(X_train.shape)


# In[ ]:


X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')


# In[ ]:


X_train.shape


# In[ ]:


# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255


# In[ ]:


y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]


# In[ ]:


y_train.shape


# In[ ]:


y_train[0]


# In[ ]:


def baseline_model():
    model = Sequential()
    model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))
    model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))
	# Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# In[ ]:


# build the model
model = baseline_model()
# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200, verbose=2)
# Final evaluation of the model

scores = model.evaluate(X_test, y_test, verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))


# In[ ]:


loss,acc = model.evaluate(X_test, y_test, verbose=0)
print(loss,acc)


# In[ ]:


X_test.shape


# In[ ]:


predict = np.array(X_test[0])


# In[ ]:


model.predict(np.array([X_test[0]]).reshape(1,784))


# In[ ]:


plt.imshow(X_test_copy[0], cmap=plt.get_cmap('gray'))
# show the plot
plt.show()

