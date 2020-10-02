#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))


# In[ ]:


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
sample = pd.read_csv("../input/sample_submission.csv")


# In[ ]:


train = train.iloc[:,0:31]
train.head(10)


# In[ ]:


test = test.iloc[:,0:30]
test.head(5)


# In[ ]:


sample.head(5)


# In[ ]:


from sklearn.preprocessing import LabelEncoder 
labelencoder = LabelEncoder()
train['diagnosis'] = labelencoder.fit_transform(train['diagnosis'])
train.head()


# In[ ]:


x = train.drop('diagnosis', axis=1)
y = train['diagnosis']


# In[ ]:


x.head(5)


# In[ ]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)


# In[ ]:


print(x.shape)
print(y.shape)


# In[ ]:


print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)


# In[ ]:


from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import accuracy_score


# In[ ]:


classify = Sequential()
classify.add(Dense(units=15,activation='relu', input_dim=30))
classify.add(Dense(units=1,activation='sigmoid'))
classify.compile(optimizer='adam',loss="binary_crossentropy",metrics=['binary_accuracy'])


# In[ ]:


classify.fit(x_train,y_train,batch_size=150,epochs=200)


# In[ ]:


y_pred = classify.predict(x_test)
y_pred = y_pred > 0.5


# In[ ]:


acc = accuracy_score(y_test,y_pred)
print("The Accuracy is", acc)

