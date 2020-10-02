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


data = pd.read_csv("/kaggle/input/breast-cancer-wisconsin-data/data.csv")


# In[ ]:


data


# In[ ]:


y= data['diagnosis'].map({'M':0,'B':1})


# In[ ]:


x= data.drop(columns=['id','diagnosis','Unnamed: 32'])


# In[ ]:


x


# In[ ]:


import tensorflow as tf


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


#Applying train_test_split

X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size=0.33)
N ,D =X_train.shape


# In[ ]:


# Scalling the data

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)


# In[ ]:


### Now creating the model

model = tf.keras.models.Sequential([tf.keras.layers.Input(shape=(D,)),
                                    tf.keras.layers.Dense(1,activation='sigmoid')
])


# In[ ]:


# Compiling the model

model.compile(optimizer='adam',
             loss='binary_crossentropy',
             metrics=['accuracy'])


# In[ ]:


r= model.fit(X_train,Y_train,validation_data=(X_test,Y_test),epochs=100)


# In[ ]:


# train and test score
print("Train score",model.evaluate(X_train,Y_train))
print("Test score",model.evaluate(X_test,Y_test))


# In[ ]:


import matplotlib.pyplot as plt
plt.plot(r.history['loss'],label='loss')
plt.plot(r.history['val_loss'],label='val_loss')
plt.legend()
plt.show()


# In[ ]:


x=scaler.fit_transform(x)
cl = pd.DataFrame(model.predict_classes(x),columns=['class'])


# In[ ]:


data['class']=cl


# In[ ]:


pd.DataFrame([data['id'],data['class']]).T.to_csv('Submit.csv',index=False)


# In[ ]:





# In[ ]:




