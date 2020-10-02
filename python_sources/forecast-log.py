#!/usr/bin/env python
# coding: utf-8

# Submission and Description
# 
# forecast log(version 13/13)  1.29796 with item stats (price) 
# forecast log(version 9/13) 1.04838
# forecast log(version 8/13) 1.21527
# forecast log(version 7/13) 1.05547
# forecast log(version 6/13) 1.04162
# 

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))
items_df = pd.read_csv('../input/items.csv')
#shops_df = pd.read_csv('../input/shops.csv')

#icats_df = pd.read_csv('../input/item_categories.csv')
train_df = pd.read_csv('../input/sales_train.csv')
smpsb_df = pd.read_csv('../input/sample_submission.csv')
test_df  = pd.read_csv('../input/test.csv')
# Any results you write to the current directory are saved as output.
train_df['date'] = pd.to_datetime(train_df.date,format="%d.%m.%Y")

#create a logaritmic classification of the price
train_df['ln_price']= np.round(np.log(train_df['item_price']+1)*10).astype('str') 
train_df.head()


# In[19]:


items_df[items_df['item_id']>5310]


# In[2]:


#make some stats per item
itemmax=train_df.groupby(['item_id']).max()
itemmed=train_df.groupby(['item_id']).median()
itemstat=itemmax.merge(itemmed,how='inner',left_index=True,right_index=True).drop(['date','date_block_num_x','shop_id_x','shop_id_y'],axis=1).reset_index()
itemstat.head()


# In[3]:


## Pivot by monht to wide format
#p_df = train_df.pivot_table(index=['shop_id'], columns='date_block_num', values='item_cnt_day',aggfunc='sum').fillna(0.0)
p_df = train_df.pivot_table(index=['shop_id','item_id'], columns='date_block_num', values='item_cnt_day',aggfunc='sum').fillna(0.0).clip(-6,15000)


# In[4]:


p_df=np.log(p_df+7)


# In[5]:


p_df.head()


# In[6]:


#create Xtrain

Xtrain=pd.DataFrame([],p_df.index).reset_index()
#add stats
Xtrain=Xtrain.merge(itemstat,left_on='item_id',right_on='item_id')
#stringify 
for coli in Xtrain.columns:
    Xtrain[coli]=Xtrain[coli].astype('str')
Xtrain.head()
#Xtrain['shop_id']= Xtrain.shop_id.astype('str')
#Xtrain['item_id']= Xtrain.item_id.astype('str')
#prepare monthly sales log corrected
p_df=p_df.reset_index()
p_df['shop_id']= p_df.shop_id.astype('str')
p_df['item_id']= p_df.item_id.astype('str')


# In[7]:


Xtrain=Xtrain.merge(p_df,left_on=["shop_id", "item_id"],right_on=["shop_id", "item_id"])


# In[8]:


import xgboost as xgb
param = {'max_depth':15, 
         'subsample':1,
         'min_child_weight':0.5,
         'eta':1, 
         'num_round':1000, 
         'seed':1,
         'silent':0,
         'eval_metric':'rmse'}

progress = dict()



xgbtrain = xgb.DMatrix(Xtrain.iloc[:,:41].values, Xtrain.iloc[:,41].values)
watchlist  = [(xgbtrain,'train-rmse')]

bst = xgb.train(param, xgbtrain)
preds = bst.predict(xgb.DMatrix(Xtrain.iloc[:,:41].values))
from sklearn.metrics import mean_squared_error 
rmse = np.sqrt(mean_squared_error(preds,Xtrain.iloc[:,41].values))
print(rmse,np.exp(rmse))


# In[9]:


Xtest = pd.DataFrame([],test_df.index)
Xtest['shop_id']= test_df.shop_id
Xtest['item_id']= test_df.item_id

Xtest=Xtest.merge(itemstat,how='left',left_on='item_id',right_on='item_id')
print(Xtest.shape)
#stringify 
for coli in Xtest.columns:
    Xtest[coli]=Xtest[coli].astype('str')
Xtest.head()
Xtest=Xtest.merge(p_df,how='left',left_on=["shop_id", "item_id"],right_on=["shop_id", "item_id"]).fillna(np.log(7.0))
Xtest.head()


# **There are 'blanc' records**
# 
# now a CD with release date okt 15 has probably been sold... so thats impossible to forecast, except you have to learn from previous releases
# That's imho the trick

# In[10]:


# Move to one month front
#d = dict(zip(train_cleaned_df.columns[2:],list(np.array(list(train_cleaned_df.columns[2:])) - 1)))
#print(d)
#apply_df  = apply_df.rename(d, axis = 1).drop(['ID',-1],axis=1)
#print(apply_df.head())
#Xtest=apply_df.values

preds = bst.predict(xgb.DMatrix(Xtest.drop(0,axis=1).values) )


# In[11]:


np.exp(preds)-7


# In[12]:


preds=np.exp(preds)-7

# Normalize prediction to [0-20]
preds = list(map(lambda x: min(20,max(x,0)), list(preds)))
sub_df = pd.DataFrame({'ID':test_df.ID,'item_cnt_month': preds })
sub_df.describe()




# In[13]:




sub_df.to_csv('xg_boost4_cats.csv',index=False)

