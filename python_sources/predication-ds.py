#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import os


# In[ ]:


os.listdir('../input')


# In[ ]:


sales_data = pd.read_csv('../input/competitive-data-science-predict-future-sales/sales_train.csv')
test_data = pd.read_csv('../input/competitive-data-science-predict-future-sales/test.csv')
item_cat = pd.read_csv('../input/competitive-data-science-predict-future-sales/item_categories.csv')
items = pd.read_csv('../input/competitive-data-science-predict-future-sales/items.csv')
shops = pd.read_csv('../input/competitive-data-science-predict-future-sales/shops.csv')


# In[ ]:


sales_data.info()


# In[ ]:


sales_data.describe()


# In[ ]:


sales_data.isnull().sum()


# In[ ]:


sales_data.isna().sum()


# In[ ]:


print(test_data.info())
print('*'*100)
print(test_data.describe())
print('*'*100)
print(test_data.isnull().sum())
print('*'*100)
print(test_data.isna().sum())


# In[ ]:


print(item_cat.info())
print('*'*100)
print(item_cat.describe())
print('*'*100)
print(item_cat.isnull().sum())
print('*'*100)
print(item_cat.isna().sum())


# In[ ]:


print(items.info())
print('*'*100)
print(items.describe())
print('*'*100)
print(items.isnull().sum())
print('*'*100)
print(items.isna().sum())


# In[ ]:


sales_data.dtypes


# In[ ]:


sales_data['date'] = pd.to_datetime(sales_data['date'],format='%d.%m.%Y')


# In[ ]:


dFrame = sales_data.pivot_table(index=['shop_id','item_id'],values=['item_cnt_day'],columns = ['date_block_num'],fill_value=0,aggfunc=sum)


# In[ ]:


dFrame.reset_index(inplace=True)


# In[ ]:


dFrame.head()


# In[ ]:


dFrame = pd.merge(test_data,dFrame,on = ['item_id','shop_id'],how='left')


# In[ ]:


dFrame.fillna(0,inplace=True)


# In[ ]:


dFrame.head()


# In[ ]:


dFrame.drop(['ID','shop_id','item_id'],inplace=True,axis=1)
dFrame.head()


# In[ ]:



Xtrain = np.expand_dims(dFrame.values[:,:-1],axis = 2)
Ytrain = dFrame.values[:,-1:]
Xtest = np.expand_dims(dFrame.values[:,1:],axis = 2)


# In[ ]:


def sepraterPattern(i,n):
    if(i==1):
        print('*'*n)
    else:
        print('-'*n)


# In[ ]:


print("X Train : ", Xtrain.shape)
sepraterPattern(1,30)
print("Y Train : ", Ytrain.shape)
sepraterPattern(1,30)
print("X Test : ", Xtest.shape)
sepraterPattern(1,30)


# In[ ]:


from keras.models import Sequential
from keras.layers import LSTM,Dense,Dropout


# In[ ]:


model  = Sequential()
model.add(LSTM(units=64,input_shape=(33,1)))
model.add(Dropout(0.25))
model.add(Dense(1))
model.compile(loss = 'mse',optimizer = 'SGD', metrics = ['mean_squared_error'])


# In[ ]:


model.summary()


# In[ ]:


model.fit(Xtrain,Ytrain,batch_size = 4096,epochs = 15)


# In[ ]:


sample_submission = pd.read_csv('../input/competitive-data-science-predict-future-sales/sample_submission.csv')


# In[ ]:


sample_submission.describe()


# In[ ]:


sample_submission.head()


# In[ ]:


submissionModel = model.predict(Xtest)


# In[ ]:


submissionModel = submissionModel.clip(0,20)


# In[ ]:


# creating dataframe with required columns 
submission = pd.DataFrame({'ID':test_data['ID'],'item_cnt_month':submissionModel.ravel()})
# creating csv file from dataframe
submission.to_csv('submissionFiles.csv',index = False)


# In[ ]:




