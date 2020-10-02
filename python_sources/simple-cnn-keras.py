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
import matplotlib.image as mpimg
import glob
import scipy
import cv2

import keras


# In[3]:



import random


# **Exploration**
# 

# In[4]:



train_data = pd.read_csv('../input/train.csv')


# In[5]:



train_data.shape


# In[6]:



train_data.head()


# In[7]:



train_data.has_cactus.unique()


# In[8]:



train_data.has_cactus.hist()


# In[9]:



train_data.has_cactus.value_counts()


# In[10]:



train_data.has_cactus.plot()


# **Model**

# In[11]:



def image_generator(batch_size = 16, all_data=True, shuffle=True, train=True, indexes=None):
    while True:
        if indexes is None:
            if train:
                if all_data:
                    indexes = np.arange(train_data.shape[0])
                else:
                    indexes = np.arange(train_data[:15000].shape[0])
                if shuffle:
                    np.random.shuffle(indexes)
            else:
                indexes = np.arange(train_data[15000:].shape[0])
            
        N = int(len(indexes) / batch_size)
       

        # Read in each input, perform preprocessing and get labels
        for i in range(N):
            current_indexes = indexes[i*batch_size: (i+1)*batch_size]
            batch_input = []
            batch_output = [] 
            for index in current_indexes:
                img = mpimg.imread('../input/train/train/' + train_data.id[index])
                batch_input += [img]
                batch_input += [img[::-1, :, :]]
                batch_input += [img[:, ::-1, :]]
                batch_input += [np.rot90(img)]
                
                temp_img = np.zeros_like(img)
                temp_img[:28, :, :] = img[4:, :, :]
                batch_input += [temp_img]
                
                temp_img = np.zeros_like(img)
                temp_img[:, :28, :] = img[:, 4:, :]
                batch_input += [temp_img]
                
                temp_img = np.zeros_like(img)
                temp_img[4:, :, :] = img[:28, :, :]
                batch_input += [temp_img]
                
                temp_img = np.zeros_like(img)
                temp_img[:, 4:, :] = img[:, :28, :]
                batch_input += [temp_img]
                
                batch_input += [cv2.resize(img[2:30, 2:30, :], (32, 32))]
                
                batch_input += [scipy.ndimage.interpolation.rotate(img, 10, reshape=False)]
                
                batch_input += [scipy.ndimage.interpolation.rotate(img, 5, reshape=False)]
                
                for _ in range(11):
                    batch_output += [train_data.has_cactus[index]]
                
            batch_input = np.array( batch_input )
            batch_output = np.array( batch_output )
        
            yield( batch_input, batch_output.reshape(-1, 1) )


# In[ ]:





# In[12]:


model = keras.models.Sequential()
model.add(keras.layers.Conv2D(64, (5, 5), input_shape=(32, 32, 3)))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.LeakyReLU(alpha=0.3))

model.add(keras.layers.Conv2D(64, (5, 5)))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.LeakyReLU(alpha=0.3))

model.add(keras.layers.Conv2D(128, (5, 5)))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.LeakyReLU(alpha=0.3))

model.add(keras.layers.Conv2D(128, (5, 5)))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.LeakyReLU(alpha=0.3))

model.add(keras.layers.Conv2D(256, (3, 3)))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.LeakyReLU(alpha=0.3))

model.add(keras.layers.Conv2D(256, (3, 3)))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.LeakyReLU(alpha=0.3))

model.add(keras.layers.Conv2D(512, (3, 3)))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.LeakyReLU(alpha=0.3))

model.add(keras.layers.Flatten())


model.add(keras.layers.Dense(100))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.LeakyReLU(alpha=0.3))

model.add(keras.layers.Dense(1, activation='sigmoid'))


# In[13]:



model.summary()


# In[14]:



opt = keras.optimizers.Adam(0.0001)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])


# In[15]:



model.fit_generator(image_generator(), steps_per_epoch= train_data.shape[0] / 16, epochs=30)


# In[16]:



keras.backend.eval(model.optimizer.lr.assign(0.00001))


# In[17]:



model.fit_generator(image_generator(), steps_per_epoch= train_data.shape[0] / 16, epochs=15)


# In[ ]:





# In[18]:



indexes = np.arange(train_data.shape[0])
N = int(len(indexes) / 64)   
batch_size = 64

wrong_ind = []
for i in range(N):
            current_indexes = indexes[i*64: (i+1)*64]
            batch_input = []
            batch_output = [] 
            for index in current_indexes:
                img = mpimg.imread('../input/train/train/' + train_data.id[index])
                batch_input += [img]
                batch_output.append(train_data.has_cactus[index])
            
            batch_input = np.array( batch_input )
#             batch_output = np.array( batch_output )

            model_pred = model.predict_classes(batch_input)
            for j in range(len(batch_output)):
                if model_pred[j] != batch_output[j]:
                    wrong_ind.append(i*batch_size+j)


# In[19]:



len(wrong_ind)


# In[20]:



indexes = np.arange(train_data.shape[0])
N = int(len(indexes) / 64)   
batch_size = 64

wrong_ind = []
for i in range(N):
            current_indexes = indexes[i*64: (i+1)*64]
            batch_input = []
            batch_output = [] 
            for index in current_indexes:
                img = mpimg.imread('../input/train/train/' + train_data.id[index])
                batch_input += [img[::-1, :, :]]
                batch_output.append(train_data.has_cactus[index])
            
            batch_input = np.array( batch_input )

            model_pred = model.predict_classes(batch_input)
            for j in range(len(batch_output)):
                if model_pred[j] != batch_output[j]:
                    wrong_ind.append(i*batch_size+j)


# In[21]:



wrong_ind


# In[22]:



indexes = np.arange(train_data.shape[0])
N = int(len(indexes) / 64)   
batch_size = 64

wrong_ind = []
for i in range(N):
            current_indexes = indexes[i*64: (i+1)*64]
            batch_input = []
            batch_output = [] 
            for index in current_indexes:
                img = mpimg.imread('../input/train/train/' + train_data.id[index])
                batch_input += [img[:, ::-1, :]]
                batch_output.append(train_data.has_cactus[index])
            
            batch_input = np.array( batch_input )

            model_pred = model.predict_classes(batch_input)
            for j in range(len(batch_output)):
                if model_pred[j] != batch_output[j]:
                    wrong_ind.append(i*batch_size+j)


# In[23]:



wrong_ind


# In[ ]:





# In[24]:



get_ipython().system('ls ../input/test/test/* | wc -l')


# In[25]:



test_files = os.listdir('../input/test/test/')


# In[26]:



len(test_files)


# In[27]:



batch = 40
all_out = []
for i in range(int(4000/batch)):
    images = []
    for j in range(batch):
        img = mpimg.imread('../input/test/test/'+test_files[i*batch + j])
        images += [img]
    out = model.predict(np.array(images))
    all_out += [out]


# In[ ]:





# In[28]:



all_out = np.array(all_out).reshape((-1, 1))


# In[29]:



all_out.shape


# In[30]:



sub_file = pd.DataFrame(data = {'id': test_files, 'has_cactus': all_out.reshape(-1).tolist()})


# In[31]:



sub_file.to_csv('sample_submission.csv', index=False)


# In[ ]:





# In[ ]:




