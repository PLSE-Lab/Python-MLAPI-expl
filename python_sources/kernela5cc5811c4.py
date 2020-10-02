#!/usr/bin/env python
# coding: utf-8

# In[133]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[134]:


d_train = pd.read_csv('../input/train.csv')
d_test = pd.read_csv('../input/test.csv')


# In[135]:


plt.imshow(d_train.iloc[0, 1:].values.reshape((28,28)), cmap='gray');


# In[136]:


y_train = d_train['label']
X_train = d_train.drop("label", axis=1)
X_test = d_test


# In[137]:


X_train_sc = X_train.astype('float32') / 255.0
X_test_sc = X_test.astype('float32') / 255.0


# In[138]:


from keras.utils.np_utils import to_categorical


# In[139]:


y_train_cat = to_categorical(y_train)


# In[140]:


from keras.models import Sequential
from keras.layers import Dense


# In[141]:


model = Sequential()
model.add(Dense(512, input_dim=784, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax')) # output
model.compile(loss='categorical_crossentropy',
optimizer='rmsprop',
metrics=['accuracy'])


# In[142]:


h = model.fit(X_train_sc, y_train_cat, batch_size=128,
epochs=10, verbose=1,
validation_split=0.1)


# In[143]:


plt.plot(h.history['acc'])
plt.plot(h.history['val_acc'])
plt.legend(['Training', 'Validation'])
plt.title('Accuracy')
plt.xlabel('Epochs');


# In[144]:


y_pred = model.predict_classes(X_test_sc)


# In[145]:


a = np.array([y_pred]).T
a.shape


# In[146]:


b = np.array([np.arange(1,28001)]).T
b.shape


# In[147]:


output = np.hstack((b , a))
output.shape


# In[150]:


df = pd.DataFrame(output, columns=['ImageId', 'Label'])


# In[152]:


df.to_csv('digits.csv', index=False)


# In[ ]:




