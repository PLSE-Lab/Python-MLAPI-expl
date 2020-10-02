#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# # Loading Data

# In[3]:


data_train = pd.read_csv("../input/emnist-bymerge-train.csv", header=None)
data_test = pd.read_csv("../input/emnist-bymerge-test.csv", header=None)


# # Data Exploration

# In[4]:


print(data_train.shape)
print(data_test.shape)


# In[5]:


print(data_train.head())


# In[ ]:


x_train = data_train.loc[:,1:].as_matrix()
y_train = np.array(data_train[0])
x_test = data_test.loc[:,1:].as_matrix()
y_test = np.array(data_test[0])


# In[7]:


print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


# In[ ]:


import gc
del [[data_train, data_test]]
gc.collect()


# In[9]:


print('max = ', y_train.max(),"\t"," min = ", y_train.min())


# # Exploratory Visualization

# In[10]:


mapping = np.loadtxt('../input/emnist-bymerge-mapping.txt',dtype=int, usecols=(1), unpack=True)
print(mapping)


# In[11]:


char_labels={}
for i in range(y_train.min(),y_train.max()+1):
    char_labels[i] = chr(mapping[i])
print(char_labels)


# ## Visualize Random 20 Training Images

# In[12]:


random_array = np.random.randint(x_train.shape[0], size=(2,10))
plt.figure(figsize=[10,3])
for i in range(10):
    plt.subplot(2,10,i+1)
    a = int(y_train[random_array[0,i]])
    plt.title(char_labels[a])
    plt.imshow(x_train[random_array[0,i]].reshape(28,28).squeeze().T, cmap='gray')
    plt.axis('off')
    plt.subplot(2,10,i+11)
    b = int(y_train[random_array[1,i]])
    plt.title(char_labels[b])
    plt.imshow(x_train[random_array[1,i]].reshape(28,28).squeeze().T, cmap='gray')
    plt.axis('off')


# # Data Preprocessing
# ## Rescale the Images by Dividing Every Pixel in Every Image by 255

# In[ ]:


x_train = x_train / 255.0
x_test = x_test / 255.0


# ## Normalize Images

# In[ ]:


from sklearn.preprocessing import normalize
x_train = normalize(x_train)
x_test = normalize(x_test)


# ## Encode Categorical Integer Labels Using a One-Hot Scheme

# In[ ]:


from keras.utils import np_utils

# one-hot encode the labels
y_train = np_utils.to_categorical(y_train, num_classes=47)
y_test = np_utils.to_categorical(y_test, num_classes=47)


# In[9]:


y_train.shape


# In[10]:


y_train[0]


# ## Reshape images data from row to 32 X 32

# In[ ]:


x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)


# In[12]:


print(x_train[0].shape)
x_train[0]


# # CNN Model

# ## Define the Model Architecture

# In[ ]:


from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import BatchNormalization
from keras.layers import Conv2D, MaxPooling2D

# define the model
cnn_model = Sequential()
cnn_model.add(Conv2D(filters=64, kernel_size=3, padding='valid', activation='relu', input_shape=(28, 28, 1)))
cnn_model.add(MaxPooling2D(pool_size=2))
#cnn_model.add(BatchNormalization())
cnn_model.add(Dropout(0.3))
cnn_model.add(Conv2D(filters=32, kernel_size=3, padding='valid', activation='relu',))
cnn_model.add(MaxPooling2D(pool_size=2))
#cnn_model.add(BatchNormalization())
#cnn_model.add(Dropout(0.3))
cnn_model.add(Flatten())
cnn_model.add(Dense(128, activation='relu'))
cnn_model.add(Dropout(0.2))
cnn_model.add(BatchNormalization())
cnn_model.add(Dense(64, activation='relu'))
cnn_model.add(Dropout(0.2))
cnn_model.add(Dense(47, activation='softmax'))

# summarize the model
cnn_model.summary()


# ## Compile the Model

# In[ ]:


# compile the model
cnn_model.compile(loss='categorical_crossentropy', optimizer='Nadam', metrics=['accuracy'])


# ## Train the Model

# In[ ]:


from keras.callbacks import ModelCheckpoint, EarlyStopping

# train the model
checkpointer = ModelCheckpoint(filepath='EMNIST_CNN.cnn_model.best.hdf5',
                               verbose=1, save_best_only=True)
earlystopper = EarlyStopping(patience=3, verbose=1)
hist = cnn_model.fit(x_train, y_train, batch_size=128, epochs=3, 
                     validation_split=0.2, callbacks=[checkpointer, earlystopper], verbose=1, shuffle=True)


# In[16]:


# load the weights that yielded the best validation accuracy
cnn_model.load_weights('EMNIST_CNN.cnn_model.best.hdf5')


# In[17]:


# evaluate test accuracy
score = cnn_model.evaluate(x_test, y_test, verbose=0)
accuracy = 100*score[1]
# print test accuracy
print('Test accuracy: %.4f%%' % accuracy)


# In[ ]:




