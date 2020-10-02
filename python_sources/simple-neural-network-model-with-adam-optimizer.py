#!/usr/bin/env python
# coding: utf-8

# **Updates:**
# 
# 1. Changed Model from Classification to Regression in order to gain better Accuracy. Thanks for feedback @Aditya Soni.

# In[ ]:


import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense


# In[ ]:


train = pd.read_csv("../input/train.csv")
train.head()


# In[ ]:


train.isnull().sum().sum()


# In[ ]:


#There are no null values in the data provided


# In[ ]:


test = pd.read_csv("../input/test.csv")
test.head()


# In[ ]:


X_train = train.drop(['ID_code', 'target'], axis = 1)
X_train.head()


# In[ ]:


y_train = train['target']
y_train.head()


# In[ ]:


X_test = test.drop(['ID_code'], axis = 1)
X_test.head()


# In[ ]:


test.isnull().sum().sum()


# In[ ]:


X_train.var()


# In[ ]:


# As the variance is high, we need to scale out the features


# In[ ]:


ss = StandardScaler()
X_train_Scaled = ss.fit_transform(X_train)


# In[ ]:


X_test_Scaled = ss.transform(X_test)


# In[ ]:


y_train.value_counts(dropna=False)


# In[ ]:


encoder = LabelEncoder()
encoder.fit(y_train)
encoded_Y = encoder.transform(y_train)


# In[ ]:


model = Sequential()
model.add(Dense(200, activation='relu'))
model.add(Dense(25, activation='relu'))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')


# In[ ]:


model.fit(X_train_Scaled, encoded_Y, epochs=200, batch_size = 32, shuffle=True, verbose=1)


# In[ ]:


pred = model.predict(X_test_Scaled)


# In[ ]:


result = pd.DataFrame({"ID_code": test['ID_code'], "target": pred[:,0]})
result.head()


# In[ ]:


result.to_csv("submission1.csv", index=False)

