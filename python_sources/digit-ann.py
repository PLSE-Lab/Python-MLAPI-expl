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


import pandas as pd
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense


# In[ ]:


df =  pd.read_csv("/kaggle/input/digit-recognizer/train.csv")


# In[ ]:


df.shape


# In[ ]:


test= pd.read_csv("/kaggle/input/digit-recognizer/test.csv")


# In[ ]:


df.head()


# In[ ]:


x = df.iloc[:,1:].values
y= df.iloc[:,0].values


# In[ ]:


from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.1)


# In[ ]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[ ]:


from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)


# In[ ]:


model = Sequential()
model.add(Dense(units=784,activation='relu',input_dim = 784))
from keras.layers import Dropout


# In[ ]:


model.add(Dense(units = 128, activation = 'relu'))
model.add(Dense(units = 128, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(units = 64, activation = 'relu'))
model.add(Dense(units = 64, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(units = 32, activation = 'relu'))
model.add(Dense(units = 32, activation = 'relu'))
model.add(Dropout(0.05))


# In[ ]:


model.add(Dense(units = 10, activation = 'softmax'))


# In[ ]:


model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])


# In[ ]:


model.summary()


# In[ ]:


model.fit(X_train, y_train, validation_data = (X_test, y_test), batch_size = 28, epochs = 10)


# In[ ]:


acc = model.evaluate(X_test, y_test)
acc


# In[ ]:


test_x = test.iloc[:,:].values


# In[ ]:


test_pred = model.predict(test_x)


# In[ ]:



results = test_pred.argmax(axis=1)


# In[ ]:


results


# In[ ]:


results = pd.Series(results,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)
submission.to_csv("submission.csv",index=False)


# In[ ]:




