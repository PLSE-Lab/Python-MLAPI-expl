#!/usr/bin/env python
# coding: utf-8

#     The problem at hand is about predicting next month sale based on shop_id and item_id columns. We will use LSTM approach to solve this problem. First we will import libraries

# In[ ]:


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import pandas as pd
import numpy as np


#     Now lets import all the datasets and see data shape.

# In[ ]:


train=pd.read_csv('../input/sales_train.csv')
print('Training set shape:',train.shape)
#Training set imported


# In[ ]:


test=pd.read_csv('../input/test.csv')
print('Testing set shape',test.shape)
#testing set imported


# In[ ]:


items_cats=pd.read_csv('../input/item_categories.csv')
print('Item categories:',items_cats.shape)


# In[ ]:


items=pd.read_csv('../input/items.csv')
print('Items set shape',items.shape)


# In[ ]:


shops=pd.read_csv('../input/shops.csv')
print('Shops set shape',shops.shape)


# Now lets have a look at columns of training set

# In[ ]:


train.columns.values


# Now we will see number of rows in 

# In[ ]:


shops_train=train.groupby(['shop_id']).groups.keys()
len(shops_train)


# In[ ]:


item_train=train.groupby(['item_id']).groups.keys()
len(item_train)


# In[ ]:


shops_test=test.groupby(['shop_id']).groups.keys()
len(shops_test)


# In[ ]:


items_test=test.groupby(['item_id']).groups.keys()
len(items_test)


# In[ ]:


print('Train DS:',train.columns.values)
print('Test DS:',test.columns.values)
print('Item cats DS:',items_cats.columns.values)
print('Items DS:',items.columns.values)
print('Shops DS:',shops.columns.values)


# In[ ]:


train.head()


# In[ ]:


test.head()


# As our testing dataset has shop_id and item_id so in training dataset we will group month count based on these two columns and date block number column.

# In[ ]:


train_df=train.groupby(['shop_id','item_id','date_block_num']).sum().reset_index().sort_values(by=['item_id','shop_id'])#.sort_values(by='item_cnt_day',ascending=False)


# In[ ]:


train_df.head()


# In order to train a network using LSTM(Long Short Term Memory), we need to feed data from previous timestep into the network. In this case the timestep is month. As we know that we might not have data for consective months so we will use whatever previous value is available. We will call pandas's shift menthod on column item_cnt_day.

# In[ ]:


train_df['m1']=train_df.groupby(['shop_id','item_id']).item_cnt_day.shift()
train_df['m1'].fillna(0,inplace=True)
train_df


# Now we w

# In[ ]:


train_df['m2']=train_df.groupby(['shop_id','item_id']).m1.shift()
train_df['m2'].fillna(0,inplace=True)
train_df.head()


# Now we will rename column item_cnt_day to item_cnt_month

# In[ ]:


train_df.rename(columns={'item_cnt_day':'item_cnt_month'},inplace=True)
train_df.head()


# Now lets drop the index column:

# In[ ]:


finalDf=train_df[['shop_id','item_id','date_block_num','m1','m2','item_cnt_month']].reset_index()
finalDf.drop(['index'],axis=1,inplace=True)
finalDf.head()


# As training and testing data should have same shape so we will now join training and testing datasets on columns shop_id and item_id. For columns where no match is found, m1,m2 and item_cnt_month will be populated as 0.

# In[ ]:


newTest=pd.merge_asof(test, finalDf, left_index=True, right_index=True,on=['shop_id','item_id'])
newTest.head()


# Now we will create a deep neural network, there will be one layer for input. Its LSTM layer, we will feed it our sequence and see what predictions it will come up with. I am using adam optimizer here, its better because it computes individual learning rates for different parameters. 

# In[ ]:


model_lstm = Sequential()
model_lstm.add(LSTM(64, input_shape=(1,4)))
model_lstm.add(Dense(1))
model_lstm.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])


# Now i will split my data in x_train and y_train format

# In[ ]:


y_train=finalDf['item_cnt_month']
newTest.drop(['item_cnt_month'],axis=1,inplace=True)
x_train=finalDf[['shop_id','item_id','m1','m2']]


# Now we will reshape the data, as this is a timeseries problem so input must be in form of samples, timestamp and features. so we will reshape our training and testing data.

# In[ ]:


x_test=newTest[['shop_id','item_id','m1','m2']]
x_test.shape


# In[ ]:


x_test_reshaped=x_test.values.reshape((x_test.values.shape[0], 1, x_test.values.shape[1]))
x_test_reshaped.shape


# Compiling the model now

# In[ ]:


history = model_lstm.fit(x_train_reshaped, y_train, epochs=20, batch_size=100, shuffle=False)
#On my laptop i used 100 epochs and 10 batch size but it was taking too much time on Kaggle to run so i changed the parameters


# Now lets make a prediction on test data.

# In[ ]:


y_pre = model_lstm.predict(x_test_reshaped)
y_preF=np.round(y_pre,0)
submission = pd.DataFrame({'ID':test['ID'],'item_cnt_month':y_preF.ravel()})
#submission.to_csv('submission_Fsales.csv',index = False)

