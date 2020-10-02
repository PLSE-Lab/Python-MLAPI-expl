#!/usr/bin/env python
# coding: utf-8

# In[30]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


train_data = pd.read_csv('../input/pubg-finish-placement-prediction/train_V2.csv')
print(train_data.shape)
train_data.head()


# In[ ]:


train_data = train_data.drop(['Id','groupId','matchId'],1)
train_data.head()


# In[ ]:


train_data.winPlacePerc = train_data.winPlacePerc.fillna(train_data.winPlacePerc.median())


# In[ ]:


train_data.isna().any()


# In[ ]:


X_train = train_data.drop('winPlacePerc',1)
Y_train = train_data.winPlacePerc


# In[ ]:


cat_var = train_data.matchType
X_train = X_train.drop('matchType',1)
cat_var = pd.get_dummies(cat_var)
cat_var.head()


# In[ ]:


X_train = pd.concat((X_train,cat_var),1)
print(X_train.shape)


# In[ ]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)


# In[ ]:


Y_train.head()


# In[ ]:


from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization

# create model
model = Sequential()
model.add(Dense(40, input_dim=40, kernel_initializer='normal', activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(100, kernel_initializer='normal', activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(100, kernel_initializer='normal', activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(100, kernel_initializer='normal', activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(100, kernel_initializer='normal', activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(50, kernel_initializer='normal', activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(20, kernel_initializer='normal', activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(10, kernel_initializer='normal', activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(5, kernel_initializer='normal', activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(1, kernel_initializer='normal'))

model.summary()


# In[ ]:


# Compile model
model.compile(loss='mean_squared_error', optimizer='adam')


# In[ ]:


from keras.callbacks import ModelCheckpoint
checkpoint = ModelCheckpoint(
    'model.h5', 
    monitor='val_loss', 
    verbose=1, 
    save_best_only=True, 
    save_weights_only=False,
    mode='auto'
)


# In[ ]:


history = model.fit(X_train,Y_train,epochs=2,batch_size=64,validation_split=0.3,callbacks=[checkpoint])


# In[ ]:


# model.load_weights('../input/pubg-nn/model.h5')


# In[ ]:


test_data = pd.read_csv('../input/pubg-finish-placement-prediction/test_V2.csv')
test_data.head()


# In[ ]:


test_data = test_data.drop(['Id','groupId','matchId'],1)
test_data.head()


# In[ ]:


cat_var = test_data.matchType
X_test = test_data.drop('matchType',1)
cat_var = pd.get_dummies(cat_var)
cat_var.head()


# In[ ]:


print(X_test.shape)


# In[ ]:


X_test = pd.concat((X_test,cat_var),1)
print(X_test.shape)


# In[ ]:


X_test = scaler.fit_transform(X_test)


# In[ ]:


pred = model.predict(X_test, verbose=1)


# In[ ]:


pred


# In[ ]:


sam_sub = pd.read_csv('../input/pubg-finish-placement-prediction/sample_submission_V2.csv')
print(sam_sub.shape)
sam_sub.head()


# In[ ]:


_id = sam_sub.Id.values


# In[ ]:


_id = _id.reshape(-1,1)
pred = pred.reshape(-1,1)
print(_id.shape)
print(pred.shape)


# In[ ]:


output = np.array(np.concatenate((_id, pred), 1))


# In[ ]:


output = pd.DataFrame(output,columns = ["Id","winPlacePerc"])


# In[ ]:


output.to_csv('submission.csv',index = False)

