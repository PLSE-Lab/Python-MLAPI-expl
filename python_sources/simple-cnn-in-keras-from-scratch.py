#!/usr/bin/env python
# coding: utf-8

# In[55]:


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


# In[56]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from keras.models import Sequential
from keras.layers import Convolution2D,Dense,Flatten,Dropout,MaxPool2D
from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator
import cv2
import glob
from tqdm import tqdm


# In[57]:


df = pd.read_csv('../input/train.csv')


# In[58]:


df.head()


# In[59]:


im = cv2.imread('../input/train/train/'+df['id'][0])


# In[60]:


im.shape


# In[61]:


train_datagen = ImageDataGenerator(rescale=1./255,validation_split=0.15,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)


# In[62]:


df['has_cactus'] = df['has_cactus'].astype(str)


# In[63]:


train_generator = train_datagen.flow_from_dataframe(df,directory='../input/train/train/',subset='training',x_col='id',y_col = 'has_cactus',target_size = (32,32),class_mode='binary')
test_generator = train_datagen.flow_from_dataframe(df,directory='../input/train/train/',subset='validation',x_col='id',y_col = 'has_cactus',target_size = (32,32),class_mode='binary')


# In[64]:


model = Sequential()


# In[65]:


model.add(Convolution2D(32,(3,3),activation='relu',input_shape = (32,32,3)))
model.add(Convolution2D(32,(3,3),activation='relu'))
model.add(MaxPool2D(2,2))
model.add(Convolution2D(64,(3,3),activation='relu'))
model.add(Convolution2D(64,(3,3),activation='relu'))
model.add(MaxPool2D(2,2))
model.add(Convolution2D(128,(3,3),activation='relu'))
model.add(MaxPool2D(2,2))


# In[66]:


model.add(Flatten())
model.add(Dense(512,activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(1,activation = 'sigmoid'))


# In[67]:


model.compile(optimizer='adam',loss = 'binary_crossentropy',metrics=['accuracy'])


# In[ ]:


history = model.fit_generator(train_generator,steps_per_epoch=2000,epochs=10,validation_data=test_generator,validation_steps=64)


# In[ ]:


test = glob.glob('../input/test/test/*.jpg')


# In[ ]:


submission = pd.read_csv('../input/sample_submission.csv')
submission.head()


# In[ ]:


test_path = '../input/test/test/'
test_images_names = []

for filename in os.listdir(test_path):
    test_images_names.append(filename)
    
test_images_names.sort()

images_test = []

for image_id in tqdm(test_images_names):
    images_test.append(np.array(cv2.imread(test_path + image_id)))
    
images_test = np.asarray(images_test)
images_test = images_test.astype('float32')
images_test /= 255


# In[ ]:


prediction = model.predict(images_test)


# In[ ]:


predict = []
for i in tqdm(range(len(prediction))):
    if prediction[i][0]>0.5:
        answer = prediction[i][0]
    else:
        answer = prediction[i][0]
    predict.append(answer)


# In[ ]:


submission['has_cactus'] = predict


# In[ ]:


submission.head(50)


# In[ ]:


submission.to_csv('sample_submission.csv',index = False)


# In[ ]:




