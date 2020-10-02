#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd 

import os
print(os.listdir("../input"))
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import mode

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from imblearn.over_sampling import SMOTE


# In[ ]:


user = pd.read_csv("../input/historical_user_logs.csv",index_col=0)
train = pd.read_csv("../input/train.csv",index_col=0)
test = pd.read_csv("../input/test.csv",index_col=0)
a = pd.read_csv("../input/test.csv")


# In[ ]:


train.describe()


# In[ ]:


train.info()


# In[ ]:


user.head()
print(user.shape)


# In[ ]:


user.head()


# In[ ]:


train['user_id'].nunique()


# In[ ]:


for x in train.columns:
    print(" {} ............... {}".format(x,train[x].isna().sum()))


# In[ ]:


#print(mode(train['city_development_index']))
train['city_development_index']=train['city_development_index'].fillna(2)
train.drop('product_category_2',axis = 'columns',inplace=True)
#print(mode(train['user_group_id']))
train['user_group_id']=train['user_group_id'].fillna(3)
train['gender']=train['gender'].fillna('Male')
#print(mode(train['age_level']))
train['age_level']=train['age_level'].fillna(3)
#print(mode(train['user_depth']))
train['user_depth']=train['user_depth'].fillna(3)


# In[ ]:


sns.countplot(train['product'])
plt.show()


# In[ ]:


sns.countplot(train['gender'])
plt.show()


# In[ ]:


train[train.is_click == 0].shape[0]


# In[ ]:


sns.countplot(user['action'])
plt.show()


# In[ ]:


sns.countplot(train['age_level'])
plt.show()


# Age Level countplot Male and Female

# In[ ]:


female = train[train.gender == 'Female']
male = train[train.gender == 'Male']
sns.countplot(female.age_level)
plt.show()


# In[ ]:


sns.countplot(male.age_level)
plt.show()


# In[ ]:


sns.countplot(x = "product",hue = "gender",data =train)
plt.show()


# In[ ]:


train['DateTime'] = pd.to_datetime(train['DateTime'],errors = 'coerce')
train['day'] = train['DateTime'].dt.day.astype('uint8')
train['hour'] = train['DateTime'].dt.hour.astype('uint8')
train['minute'] = train['DateTime'].dt.minute.astype('uint8')
train['sec'] = train['DateTime'].dt.second.astype('uint8')


# In[ ]:


train.groupby(['user_id','campaign_id'])['age_level'].count()


# In[ ]:


GROUP_BY_NEXT_CLICKS = [
    {'groupby': ['webpage_id']},
    {'groupby': ['webpage_id', 'product_category_1']},
    {'groupby': ['webpage_id', 'product']},
    {'groupby': ['webpage_id', 'user_group_id']},
    {'groupby': ['user_id','campaign_id']},
    {'groupby': ['campaign_id', 'user_group_id','product']},
    {'groupby': ['user_id','product']}
]

for x in GROUP_BY_NEXT_CLICKS:
    name = '{}_nextclick'.format('_'.join(x['groupby']))
    feature = x['groupby']+['DateTime']
    train[name] = train[feature].groupby(x['groupby']).DateTime.transform(lambda y: y.diff().shift(-1)).dt.seconds
    


# In[ ]:


le = LabelEncoder()
train['product'] = le.fit_transform(train['product'])
train['gender'] = le.fit_transform(train['gender'])
train.head()


# In[ ]:


x


# In[ ]:


train=train.fillna(0)
y = train['is_click']
train = train.drop(['is_click','DateTime'],axis='columns')
x = train
x_train,x_test,y_train,y_test = train_test_split(x, y, test_size = 0.3)
sm = SMOTE(random_state=2)
x_train_res, y_train_res = sm.fit_sample(x_train, y_train.ravel())
x_test_res, y_test_res = sm.fit_sample(x_test,y_test.ravel())


# In[ ]:


model = XGBClassifier(colsample_bytree=0.2,
                       gamma=0.0,
                       learning_rate=0.05,
                       max_depth=6,
                       min_child_weight=1.5,
                       n_estimators=7200,
                       reg_alpha=0.9,
                       reg_lambda=0.6,
                       subsample=0.2,
                       seed=42,
                       silent=1)
model.fit(x_train_res,y_train_res)


# In[ ]:


from sklearn.metrics import roc_auc_score
predictions = model.predict(x_test_res)


# In[ ]:


roc_auc_score(y_test_res,predictions)


# In[ ]:


#print(mode(train['city_development_index']))
test['city_development_index']=test['city_development_index'].fillna(2)
test.drop('product_category_2',axis = 'columns',inplace=True)
#print(mode(train['user_group_id']))
test['user_group_id']=test['user_group_id'].fillna(3)
test['gender']=test['gender'].fillna('Male')
#print(mode(train['age_level']))
test['age_level']=test['age_level'].fillna(3)
#print(mode(train['user_depth']))
test['user_depth']=test['user_depth'].fillna(3)


# In[ ]:


test['DateTime'] = pd.to_datetime(test['DateTime'],errors = 'coerce')
test['day'] = test['DateTime'].dt.day.astype('uint8')
test['hour'] = test['DateTime'].dt.hour.astype('uint8')
test['minute'] = test['DateTime'].dt.minute.astype('uint8')
test['sec'] = test['DateTime'].dt.second.astype('uint8')


# In[ ]:


GROUP_BY_NEXT_CLICKS = [
    {'groupby': ['webpage_id']},
    {'groupby': ['webpage_id', 'product_category_1']},
    {'groupby': ['webpage_id', 'product']},
    {'groupby': ['webpage_id', 'user_group_id']},
    {'groupby': ['user_id','campaign_id']},
    {'groupby': ['campaign_id', 'user_group_id','product']},
    {'groupby': ['user_id','product']}
]

for x in GROUP_BY_NEXT_CLICKS:
    name = '{}_nextclick'.format('_'.join(x['groupby']))
    feature = x['groupby']+['DateTime']
    test[name] = test[feature].groupby(x['groupby']).DateTime.transform(lambda y: y.diff().shift(-1)).dt.seconds
    


# In[ ]:


test['product'] = le.fit_transform(test['product'])
test['gender'] = le.fit_transform(test['gender'])
test=test.drop('DateTime',axis = 'columns')


# In[ ]:


ans =model.predict(test)


# In[ ]:


pred = pd.DataFrame({'session_id':a['session_id'],'is_click':ans})
pred.to_csv('ans.csv',header = True,index =False)


# In[ ]:





# In[ ]:




