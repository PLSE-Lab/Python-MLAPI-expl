#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import matplotlib.pyplot as plt
from keras import optimizers, Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.utils import to_categorical
from tqdm import tqdm
import cv2
from sklearn.model_selection import train_test_split
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


dirpath = '/kaggle/input/thanos-or-grimace/'


# In[ ]:


pixel = 100


# In[ ]:


grimace = os.listdir(dirpath+'grimace')


# In[ ]:


len(grimace)


# In[ ]:


img_test = cv2.imread(dirpath+'grimace/'+grimace[0])
img_test = cv2.cvtColor(img_test,cv2.COLOR_BGR2RGB)
plt.imshow(img_test)
plt.show()
print(img_test.shape)


# In[ ]:


thanos = os.listdir(dirpath+'thanos')


# In[ ]:


len(thanos)


# In[ ]:


data = []


# In[ ]:


for i in tqdm(range(len(grimace))):
    path = dirpath+'grimace/'+grimace[i]
    if 'jpg' in path:
        img = cv2.imread(path)
        img = img/255
        img = cv2.resize(img,(pixel,pixel))
        data.append([img,0])


# In[ ]:


for i in tqdm(range(len(thanos))):
    path = dirpath+'thanos/'+thanos[i]
    if 'jpg' in path:
        img = cv2.imread(path)
        img = img/255
        img = cv2.resize(img,(pixel,pixel))
        data.append([img,1])


# In[ ]:


data = np.array(data)


# In[ ]:


data.shape


# In[ ]:


data[456][1]


# In[ ]:


for i in tqdm(range(5)):
    np.random.shuffle(data)


# In[ ]:


X = []
Y = []
for i in tqdm(range(data.shape[0])):
    X.append(data[i][0])
    Y.append(data[i][1])

X = np.array(X)
Y = np.array(Y)
Y = np.reshape(Y,(data.shape[0],1))
    


# In[ ]:


X.shape


# In[ ]:


Y.shape


# In[ ]:


Y = to_categorical(Y)


# In[ ]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)


# In[ ]:


print(X_train.shape,'\t',Y_train.shape)
print(X_test.shape,'\t',Y_test.shape)


# In[ ]:


# Fitting CNN

model = Sequential()
model.add(Conv2D(32, kernel_size=5, activation='relu', input_shape=(pixel,pixel,X_train.shape[3])))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(32, kernel_size=4, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(2, activation='softmax'))


# In[ ]:


model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])


# In[ ]:


history = model.fit(X_train, Y_train,validation_split = 0.25, epochs=15,verbose=1)


# In[ ]:


model.evaluate(X_test,Y_test)


# In[ ]:


plt.plot(history.history['accuracy'])
plt.plot(history.history['loss'])
plt.plot(history.history['val_accuracy'])
plt.plot(history.history['val_loss'])
plt.xlabel('Epochs')
plt.ylabel('Values for Accuracy and Loss')
plt.legend(['Training Accuracy','Training Loss','Validation Accuracy','Validation Loss'])


# In[ ]:


model.summary()


# In[ ]:





# In[ ]:




