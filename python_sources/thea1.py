#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import featuretools as ft
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import pickle
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# ## import dataset

# In[ ]:


new = pd.read_csv('../input/new_merchant_transactions.csv',parse_dates =["purchase_date"])
his = pd.read_csv('../input/historical_transactions.csv',parse_dates =["purchase_date"])
train = pd.read_csv( '../input/train.csv',parse_dates =["first_active_month"])
test = pd.read_csv( '../input/test.csv',parse_dates =["first_active_month"])
merchants = pd.read_csv( '../input/merchants.csv')


# ## check duplicates in merchants and train datasets, remove any duplicates

# In[ ]:


merchants = merchants.drop_duplicates(subset="merchant_id",keep="first")
print(train.shape[0]==train["card_id"].nunique())
print(merchants.shape[0]==merchants["merchant_id"].nunique())
#print(merchants[merchants["merchant_id"].duplicated()])


# ## drop non-important column for his, new and merchant

# In[ ]:


new_drop=new.drop(['purchase_date',"merchant_category_id",
                                                   "subsector_id","city_id",
                                                   "state_id"], axis=1)
his_drop=his.drop(['purchase_date',"merchant_category_id",
                                                   "subsector_id","city_id",
                                                   "state_id"], axis=1)


# In[ ]:


mer_drop = merchants.drop(['merchant_group_id',"merchant_category_id",
                                                   "subsector_id","most_recent_sales_range",
                                                   "most_recent_purchases_range",
                           'city_id','state_id'], axis=1)


# ## extract history transactions record for training data

# In[ ]:


train_his=train[["card_id"]].merge(his_drop, how='left', on="card_id")


# In[ ]:


print(train_his.shape)
train_his.head()


# ## **subsample his transactions**

# In[ ]:


#train_his_transactions=train_his_transactions.dropna()
train_his_sub=train_his.loc[np.random.choice(train_his.index, 1000000, replace=False)]


# In[ ]:


print(train_his_sub.shape)
train_his_sub.head()


# ## **one hot encode** for train_his_sub : authorized_flag; category_1; category_2; category_3

# In[ ]:


authorized_flag = pd.get_dummies(train_his_sub['authorized_flag'])
authorized_flag.columns = ['authorized_flag_N', 'authorized_flag_Y']
train_his_sub=train_his_sub.drop(['authorized_flag'], axis=1)
train_his_sub=pd.concat([train_his_sub, authorized_flag], axis=1)


# In[ ]:


train_his_sub.head()


# In[ ]:


category_1 = pd.get_dummies(train_his_sub['category_1'])
category_1.head()
category_1.columns = ['category_1_N', 'category_1_Y']
train_his_sub=train_his_sub.drop(['category_1'], axis=1)
train_his_sub=pd.concat([train_his_sub, category_1], axis=1)
train_his_sub.head()


# In[ ]:


category_2 = pd.get_dummies(train_his_sub['category_2'])
#category_2.head()
category_2.columns = ['category_2_1', 'category_2_2',"category_2_3","category_2_4","category_2_5"]
train_his_sub=train_his_sub.drop(['category_2'], axis=1)
train_his_sub=pd.concat([train_his_sub, category_2], axis=1)
train_his_sub.head()


# In[ ]:


category_3 = pd.get_dummies(train_his_sub['category_3'])
#category_3.head()
category_3.columns = ['category_3_A', 'category_3_B',"category_3_C"]
train_his_sub=train_his_sub.drop(['category_3'], axis=1)
train_his_sub=pd.concat([train_his_sub, category_3], axis=1)
train_his_sub.head()


# ## rename column for merging simplicity

# In[ ]:


train_his_sub.columns


# In[ ]:


train_his_sub.columns.values[1:]=["his_trans_" + str(col) for col in list(train_his_sub)[1:]]


# In[ ]:


train_his_sub.columns.values[2] = "merchant_id"


# In[ ]:


train_his_sub.columns.values


# In[ ]:


mer_drop.columns.values[1:]=["his_mer_" + str(col) for col in mer_drop.columns.values[1:]] 
mer_drop.columns


# ## merge with merchants (X : his transaction, Y : merchant)

# In[ ]:


train_his_sub_mer = train_his_sub.merge(mer_drop,how="left",on='merchant_id')
train_his_sub_mer.shape


# In[ ]:


train_his_sub_mer.head()


# ## **one hot encode** for train_his_sub_mer(merchants part) : category_1; category_2; category_3

# In[ ]:


category_1 = pd.get_dummies(train_his_sub_mer['his_mer_category_1'])
#category_1.head()
category_1.columns = ['his_mer_category_1_N', 'his_mer_category_1_Y']
train_his_sub_mer=train_his_sub_mer.drop(['his_mer_category_1'], axis=1)
train_his_sub_mer=pd.concat([train_his_sub_mer, category_1], axis=1)
train_his_sub_mer.head()


# In[ ]:


category_2 = pd.get_dummies(train_his_sub_mer['his_mer_category_2'])
#category_2.head()
category_2.columns = ['his_mer_category_2_1', 'his_mer_category_2_2',"his_mer_category_2_3","his_mer_category_2_4","his_mer_category_2_5"]
train_his_sub_mer=train_his_sub_mer.drop(['his_mer_category_2'], axis=1)
train_his_sub_mer=pd.concat([train_his_sub_mer, category_2], axis=1)
train_his_sub_mer.head()


# In[ ]:


category_4 = pd.get_dummies(train_his_sub_mer['his_mer_category_4'])
#category_1.head()
category_4.columns = ['his_mer_category_4_N', 'his_mer_category_4_Y']
train_his_sub_mer=train_his_sub_mer.drop(['his_mer_category_4'], axis=1)
train_his_sub_mer=pd.concat([train_his_sub_mer, category_4], axis=1)
train_his_sub_mer.head()


# In[ ]:


with open('train_his_sub_mer.pickle', 'wb') as f:
    pickle.dump(train_his_sub_mer, f)


# In[ ]:


with open('train_his_sub_mer.pickle', 'rb') as f:
    train_his_sub_mer = pickle.load(f)


# 

# ## same for new transaction

# In[ ]:


## extract history transactions record for training data

train_new=train[["card_id"]].merge(new_drop, how='left', on="card_id")

print(train_new.shape)
train_new.head()

## **one hot encode** for train_his_sub : authorized_flag; category_1; category_2; category_3
train_new['authorized_flag']=train_new['authorized_flag'].fillna("N")
authorized_flag = pd.get_dummies(train_new['authorized_flag'])
#print(train_new['authorized_flag'].value_counts())
#print(train_new['authorized_flag'].isna().sum())
authorized_flag.columns = ['authorized_flag_N', 'authorized_flag_Y']
train_new=train_new.drop(['authorized_flag'], axis=1)
train_new=pd.concat([train_new, authorized_flag], axis=1)

train_new.head()

category_1 = pd.get_dummies(train_new['category_1'])
category_1.head()
category_1.columns = ['category_1_N', 'category_1_Y']
train_new=train_new.drop(['category_1'], axis=1)
train_new=pd.concat([train_new, category_1], axis=1)
train_new.head()

category_2 = pd.get_dummies(train_new['category_2'])
#category_2.head()
category_2.columns = ['category_2_1', 'category_2_2',"category_2_3","category_2_4","category_2_5"]
train_new=train_new.drop(['category_2'], axis=1)
train_new=pd.concat([train_new, category_2], axis=1)
train_new.head()

category_3 = pd.get_dummies(train_new['category_3'])
#category_3.head()
category_3.columns = ['category_3_A', 'category_3_B',"category_3_C"]
train_new=train_new.drop(['category_3'], axis=1)
train_new=pd.concat([train_new, category_3], axis=1)
train_new.head()

## rename column for merging simplicity

print(train_new.columns)

train_new.columns.values[1:]=["new_trans_" + str(col) for col in list(train_new)[1:]]


train_new.columns.values[2] = "merchant_id"

print(train_new.columns.values)


# In[ ]:


mer_drop = merchants.drop(['merchant_group_id',"merchant_category_id",
                                                   "subsector_id","most_recent_sales_range",
                                                   "most_recent_purchases_range",
                           'city_id','state_id'], axis=1)
mer_drop.columns.values[1:]=["new_mer_" + str(col) for col in mer_drop.columns.values[1:]] 
print(mer_drop.columns)

## merge with merchants (X : his transaction, Y : merchant)

train_new_mer = train_new.merge(mer_drop,how="left",on='merchant_id')
train_new_mer.shape

train_new_mer.head()

## **one hot encode** for train_his_sub_mer(merchants part) : category_1; category_2; category_3

category_1 = pd.get_dummies(train_new_mer['new_mer_category_1'])
#category_1.head()
category_1.columns = ['new_mer_category_1_N', 'new_mer_category_1_Y']
train_new_mer=train_new_mer.drop(['new_mer_category_1'], axis=1)
train_new_mer=pd.concat([train_new_mer, category_1], axis=1)
train_new_mer.head()

category_2 = pd.get_dummies(train_new_mer['new_mer_category_2'])
#category_2.head()
category_2.columns = ['new_mer_category_2_1', 'new_mer_category_2_2',"new_mer_category_2_3","new_mer_category_2_4","new_mer_category_2_5"]
train_new_mer=train_new_mer.drop(['new_mer_category_2'], axis=1)
train_new_mer=pd.concat([train_new_mer, category_2], axis=1)
train_new_mer.head()

category_4 = pd.get_dummies(train_new_mer['new_mer_category_4'])
#category_1.head()
category_4.columns = ['new_mer_category_4_N', 'new_mer_category_4_Y']
train_new_mer=train_new_mer.drop(['new_mer_category_4'], axis=1)
train_new_mer=pd.concat([train_new_mer, category_4], axis=1)
train_new_mer.head()

with open('train_new_mer.pickle', 'wb') as f:
    pickle.dump(train_new_mer, f)

with open('train_new_mer.pickle', 'rb') as f:
    train_new_mer = pickle.load(f)

train_new_mer.head()


# # done

# In[ ]:


# #inner join (some training card id is not in new_transactions,delete these)
# train_new_transactions=train[["card_id"]].merge(new_transactions, how='inner', on="card_id")


# In[ ]:


# print(train_new_transactions.shape)
# train_new_transactions.head()


# In[ ]:


# train_new_transactions.columns=["new_trans_" + str(col) for col in train_new_transactions.columns]


# In[ ]:


# merchants.columns=["merchants_" + str(col) for col in merchants.columns] 


# In[ ]:


#drop merchants with same merchant id,return the first one
#merchants.drop_duplicates(subset="merchants_merchant_id",keep="first",inplace=True)


# In[ ]:


# merchants.head()


# In[ ]:


# merge_new_trans_merchants=pd.merge(train_new_transactions,merchants,how="left",left_on="new_trans_merchant_id",right_on="merchants_merchant_id")


# In[ ]:


# merge_new_trans_merchants['new_trans_authorized_flag'] = merge_new_trans_merchants['new_trans_authorized_flag'].apply(lambda x: 1 if x == 'Y' else 0)
# merge_new_trans_merchants['new_trans_category_1'] = merge_new_trans_merchants['new_trans_category_1'].apply(lambda x: 1 if x == 'Y' else 0)


# In[ ]:


# autorized_card_rate = merge_new_trans_merchants.groupby(['new_trans_card_id'])['new_trans_authorized_flag'].mean()


# In[ ]:


# new_trans_cate_1_rate_Y = merge_new_trans_merchants.groupby(['new_trans_card_id'])['new_trans_category_1'].mean()


# In[ ]:


# #create dummy variable for new_trans_category_2
# merge_new_trans_merchants['new_trans_category_2_1'] = merge_new_trans_merchants['new_trans_category_2'].apply(lambda x: 1 if x == 1 else 0)
# merge_new_trans_merchants['new_trans_category_2_2'] = merge_new_trans_merchants['new_trans_category_2'].apply(lambda x: 1 if x == 2 else 0)
# merge_new_trans_merchants['new_trans_category_2_3'] = merge_new_trans_merchants['new_trans_category_2'].apply(lambda x: 1 if x == 3 else 0)
# merge_new_trans_merchants['new_trans_category_2_4'] = merge_new_trans_merchants['new_trans_category_2'].apply(lambda x: 1 if x == 4 else 0)
# merge_new_trans_merchants['new_trans_category_2_5'] = merge_new_trans_merchants['new_trans_category_2'].apply(lambda x: 1 if x == 5 else 0)


# In[ ]:


# #calculate mean of each category in cate_2 group by card_id
# cate_2_1_rate = merge_new_trans_merchants.groupby(['new_trans_card_id'])['new_trans_category_2_1'].mean()
# cate_2_2_rate = merge_new_trans_merchants.groupby(['new_trans_card_id'])['new_trans_category_2_2'].mean()
# cate_2_3_rate = merge_new_trans_merchants.groupby(['new_trans_card_id'])['new_trans_category_2_3'].mean()
# cate_2_4_rate = merge_new_trans_merchants.groupby(['new_trans_card_id'])['new_trans_category_2_4'].mean()
# cate_2_5_rate = merge_new_trans_merchants.groupby(['new_trans_card_id'])['new_trans_category_2_5'].mean()


# In[ ]:


# #create dummy variable for new_trans_category_3
# merge_new_trans_merchants['new_trans_category_3_A'] = merge_new_trans_merchants['new_trans_category_3'].apply(lambda x: 1 if x == "A" else 0)
# merge_new_trans_merchants['new_trans_category_3_B'] = merge_new_trans_merchants['new_trans_category_3'].apply(lambda x: 1 if x == "B" else 0)
# merge_new_trans_merchants['new_trans_category_3_C'] = merge_new_trans_merchants['new_trans_category_3'].apply(lambda x: 1 if x == "C" else 0)


# In[ ]:


# #calculate mean of each category in cate_3 group by card_id
# cate_3_A_rate = merge_new_trans_merchants.groupby(['new_trans_card_id'])['new_trans_category_3_A'].mean()
# cate_3_B_rate = merge_new_trans_merchants.groupby(['new_trans_card_id'])['new_trans_category_3_B'].mean()
# cate_3_C_rate = merge_new_trans_merchants.groupby(['new_trans_card_id'])['new_trans_category_3_C'].mean()


# In[ ]:


# #create dummy variable for merchant_category_1
# merge_new_trans_merchants['merchants_category_1'] = merge_new_trans_merchants['merchants_category_1'].apply(lambda x: 1 if x == "Y" else 0)


# In[ ]:


# #calculate mean of each category in merchant_cate_1 group by card_id
# merchant_cate_1_Y_rate = merge_new_trans_merchants.groupby(['new_trans_card_id'])['merchants_category_1'].mean()


# In[ ]:


# #create dummy variable for merchants_category_2
# merge_new_trans_merchants['merchants_category_2_1'] = merge_new_trans_merchants['merchants_category_2'].apply(lambda x: 1 if x == 1 else 0)
# merge_new_trans_merchants['merchants_category_2_2'] = merge_new_trans_merchants['merchants_category_2'].apply(lambda x: 1 if x == 2 else 0)
# merge_new_trans_merchants['merchants_category_2_3'] = merge_new_trans_merchants['merchants_category_2'].apply(lambda x: 1 if x == 3 else 0)
# merge_new_trans_merchants['merchants_category_2_4'] = merge_new_trans_merchants['merchants_category_2'].apply(lambda x: 1 if x == 4 else 0)
# merge_new_trans_merchants['merchants_category_2_5'] = merge_new_trans_merchants['merchants_category_2'].apply(lambda x: 1 if x == 5 else 0)


# In[ ]:


# #calculate mean of each category in merchant cate_2 group by card_id
# merchants_cate_2_1_rate = merge_new_trans_merchants.groupby(['new_trans_card_id'])['merchants_category_2_1'].mean()
# merchants_cate_2_2_rate = merge_new_trans_merchants.groupby(['new_trans_card_id'])['merchants_category_2_2'].mean()
# merchants_cate_2_3_rate = merge_new_trans_merchants.groupby(['new_trans_card_id'])['merchants_category_2_3'].mean()
# merchants_cate_2_4_rate = merge_new_trans_merchants.groupby(['new_trans_card_id'])['merchants_category_2_4'].mean()
# merchants_cate_2_5_rate = merge_new_trans_merchants.groupby(['new_trans_card_id'])['merchants_category_2_5'].mean()


# In[ ]:


# merge_new_trans_merchants['merchants_category_4'] = merge_new_trans_merchants['merchants_category_4'].apply(lambda x: 1 if x == "Y" else 0)
# merchant_cate_4_Y_rate = merge_new_trans_merchants.groupby(['new_trans_card_id'])['merchants_category_4'].mean()


# In[ ]:


# new_trans_merchants_by_card_id={
#     "authorized_card_rate":autorized_card_rate,
#     "new_trans_category_1_rate_Y":new_trans_cate_1_rate_Y,
#     "new_trans_category_2_1_rate":cate_2_1_rate,
#     "new_trans_category_2_2_rate":cate_2_2_rate,
#     "new_trans_category_2_3_rate":cate_2_3_rate,
#     "new_trans_category_2_4_rate":cate_2_4_rate,
#     "new_trans_category_2_5_rate":cate_2_5_rate,
#     "new_trans_category_3_A_rate":cate_3_A_rate,
#     "new_trans_category_3_B_rate":cate_3_B_rate,
#     "new_trans_category_3_C_rate":cate_3_C_rate,
#     "merchants_category_1_Y_rate":merchant_cate_1_Y_rate,
#     "merchants_category_2_1_rate":merchants_cate_2_1_rate,
#     "merchants_category_2_2_rate":merchants_cate_2_2_rate,
#     "merchants_category_2_3_rate":merchants_cate_2_3_rate,
#     "merchants_category_2_4_rate":merchants_cate_2_4_rate,
#     "merchants_category_2_5_rate":merchants_cate_2_5_rate,
#     "merchants_category_4_Y_rate":merchant_cate_4_Y_rate
# }


# In[ ]:


# new_trans_merchants_by_card_id_df=pd.DataFrame(new_trans_merchants_by_card_id)


# In[ ]:


# new_trans_merchants_by_card_id_df=new_trans_merchants_by_card_id_df.reset_index()


# In[ ]:


# new_trans_merchants_by_card_id_df.head()


# In[ ]:


# with open('new_trans_merchants_by_card_id_df.pickle', 'wb') as f:
#     pickle.dump(new_trans_merchants_by_card_id_df, f)


# In[ ]:


# with open('new_trans_merchants_by_card_id_df.pickle', 'rb') as f:
#     new_trans_merchants_by_card_id_df = pickle.load(f)


# ## Automated Feature Engineering

# In[ ]:


train_his_sub_mer = reduce_mem_usage(train_his_sub_mer)
train_new_mer = reduce_mem_usage(train_new_mer)


# In[ ]:


es = ft.EntitySet(id = 'card')
es = es.entity_from_dataframe(entity_id = 'train', dataframe = train,index = 'card_id')


# In[ ]:


es = es.entity_from_dataframe(entity_id = 'train_his_sub_mer', 
                              dataframe = train_his_sub_mer,
                              make_index = True,
                              index = "train_his_sub_mer_id")


# In[ ]:


es = es.entity_from_dataframe(entity_id = 'train_new_mer', 
                              dataframe = train_new_mer,
                              make_index = True,
                              index = "train_new_mer_id")


# In[ ]:


r_train_his = ft.Relationship(es['train']['card_id'],
                                   es['train_his_sub_mer']['card_id'])
r_train_new = ft.Relationship(es['train']['card_id'],
                                    es['train_new_mer']['card_id'])

# Add the relationship to the entity set
es = es.add_relationship(r_train_his)
es = es.add_relationship(r_train_new)
es


# In[ ]:


features, feature_names = ft.dfs(entityset=es, target_entity='train', 
                                 max_depth = 2,
                                 agg_primitives = ['mean', 'max', 'percent_true', 'last'],
                                 trans_primitives = ['years', 'month', 'subtract', 'divide'])

with open('features.pickle', 'wb') as f:
    pickle.dump([features, feature_names], f)

features.head()


# In[ ]:


dic = pd.read_excel('../input/Data_Dictionary.xlsx', sheet_name='train')
dic


# In[ ]:


# with open('features.pickle', 'wb') as f:
#     pickle.dump([features, feature_names], f)


# In[ ]:


e = pd.read_excel('../input/Data_Dictionary.xlsx', sheet_name='history')
e


# In[ ]:




