#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D, Input
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.utils import np_utils
import numpy as np


# In[ ]:


import os
print(os.listdir("../input"))


# In[ ]:


X_train = np.load("../input/pokemon/X_train.npy")
Y_train = np.load("../input/pokemon/Y_train.npy")
x_test = np.load("../input/pokemon/X_test.npy")


# In[ ]:


Y_train = np_utils.to_categorical(Y_train)


# In[ ]:


print ((X_train.shape, Y_train.shape))


# In[ ]:


model = Sequential()

# 1st Convolutional Layer
model.add(Conv2D(filters=96, input_shape=(227,227,3), kernel_size=(11,11), strides=(4,4), padding='valid'))
model.add(Activation('relu'))
# Pooling 
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
# Batch Normalisation before passing it to the next layer
model.add(BatchNormalization())

# 2nd Convolutional Layer
model.add(Conv2D(filters=256, kernel_size=(11,11), strides=(1,1), padding='valid'))
model.add(Activation('relu'))
# Pooling
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
# Batch Normalisation
model.add(BatchNormalization())


# In[ ]:


# 3rd Convolutional Layer
model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
model.add(Activation('relu'))
# Batch Normalisation
model.add(BatchNormalization())

# 4th Convolutional Layer
model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
model.add(Activation('relu'))
# Batch Normalisation
model.add(BatchNormalization())

# 5th Convolutional Layer
model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid'))
model.add(Activation('relu'))
# Pooling
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
# Batch Normalisation
model.add(BatchNormalization())


# In[ ]:


# Passing it to a dense layer
model.add(Flatten())
# 1st Dense Layer
model.add(Dense(4096, input_shape=(227*227*3,)))
model.add(Activation('relu'))
# Add Dropout to prevent overfitting
model.add(Dropout(0.4))
# Batch Normalisation
model.add(BatchNormalization())

# 2nd Dense Layer
model.add(Dense(4096))
model.add(Activation('relu'))
# Add Dropout
model.add(Dropout(0.4))
# Batch Normalisation
model.add(BatchNormalization())

# 3rd Dense Layer
model.add(Dense(1000))
model.add(Activation('relu'))
# Add Dropout
model.add(Dropout(0.4))
# Batch Normalisation
model.add(BatchNormalization())

# Output Layer
model.add(Dense(10))
model.add(Activation('softmax'))

model.summary()


# In[ ]:


adam = Adam(lr = 0.00003)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])


# In[ ]:


from keras.callbacks import ModelCheckpoint


# In[ ]:


checkpoint = ModelCheckpoint("model.h5", save_best_only=True,monitor='val_acc')
hist = model.fit(X_train, Y_train, epochs = 100, validation_split = 0.2, shuffle = True, batch_size = 64,callbacks = [checkpoint])


# In[ ]:


from keras.models import load_model
r_model = load_model("model.h5")


# In[86]:


y_predict = r_model.predict_classes(x_test)


# In[87]:


print(y_predict.shape)


# In[88]:


y_predict


# In[89]:


y_test = np.load("../input/pokemon1/Y_test.npy")


# In[90]:


print(y_test.shape)


# In[91]:


y_test


# In[92]:


accuracy = np.mean(y_predict == y_test)
accuracy


# In[93]:


print(y_predict.dtype)

np.savetxt('f1.csv',y_predict)


# In[ ]:




