#!/usr/bin/env python
# coding: utf-8

# In[129]:


import numpy as np 
import pandas as pd 


import os
print(os.listdir("../input"))

import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import skew
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


# In[217]:


train = pd.read_csv('../input/train_foreclosure.csv')
agmt = pd.read_excel('../input/Customers_31JAN2019.xlsx')
cstmr = pd.read_excel('../input/LMS_31JAN2019.xlsx')
test = pd.read_csv('../input/test_foreclosure.csv')
test=test.fillna(0)
test['FORECLOSURE']=test['FORECLOSURE'].astype(int)


# In[131]:


train_f = pd.merge(train,cstmr,on = 'AGREEMENTID')
train_f = pd.merge(train_f,agmt,on = 'CUSTOMERID')
test_f = pd.merge(test,cstmr, on = 'AGREEMENTID')
test_f = pd.merge(test_f,agmt,on = 'CUSTOMERID')


# In[132]:


miss = train_f.isna().sum()/len(train_f)
miss_l = miss[miss > 0.75]
miss_s = miss[miss < 0.75]
train_f = train_f.drop(miss_l.index,axis = 'columns')
test_f = test_f.drop(miss_l.index,axis = 'columns')


# In[133]:


test_f.drop('FORECLOSURE',inplace = True,axis ='columns')


# In[134]:


miss_s = miss_s[miss_s > 0]
miss_s
train_f.shape


# In[135]:


train_f = train_f.dropna()
test_f = test_f.dropna()


# In[136]:


sns.countplot(train_f['FORECLOSURE'])
plt.show()


# In[137]:


number=train_f.select_dtypes(exclude = object)
number.head(5)


# In[138]:


sns.distplot(train_f['LOAN_AMT'])
plt.show()


# In[139]:


train_f['LOAN_AMT']= np.log(1+train_f['LOAN_AMT'])
test_f['LOAN_AMT']= np.log(1+test_f['LOAN_AMT'])
sns.distplot(train_f['LOAN_AMT'])
plt.show()


# In[140]:


number = number.drop(['AGREEMENTID','CUSTOMERID','FORECLOSURE','INTEREST_START_DATE','DUEDAY','AUTHORIZATIONDATE','LAST_RECEIPT_DATE','BRANCH_PINCODE'],axis ='columns')
skew = train_f[number.columns].apply(lambda x: skew(x.astype(float)))
skew = skew[skew > 0.75]
skew=skew.index
train_f[skew] = np.log1p(train_f[skew])
test_f[skew] = np.log1p(test_f[skew])


# In[141]:


obj = train_f.select_dtypes(include = object)
le = preprocessing.LabelEncoder()
for x in obj.columns:
    test_f[x] = le.fit_transform(test_f[x])
    train_f[x] = le.fit_transform(train_f[x])


# In[142]:


test_f.head(5)
test_f['AGREEMENTID'==]


# In[143]:


scaler = preprocessing.StandardScaler()
scaled = scaler.fit_transform(train_f[number.columns])
scaled_t = scaler.fit_transform(test_f[number.columns])

for i, col in enumerate(number.columns):
       train_f[col] = scaled[:,i]


# In[144]:


y = train_f['FORECLOSURE']
train_f.drop('FORECLOSURE',inplace = True,axis ='columns')
x = train_f
x_train,x_val,y_train,y_val = train_test_split(x,y,test_size =0.3,random_state =0)


# In[145]:


x_train.drop(['INTEREST_START_DATE', 'AUTHORIZATIONDATE', 'LAST_RECEIPT_DATE'],inplace = True
,axis = 'columns')
test_f.drop(['INTEREST_START_DATE', 'AUTHORIZATIONDATE', 'LAST_RECEIPT_DATE'],inplace = True
,axis = 'columns')
x_val.drop(['INTEREST_START_DATE', 'AUTHORIZATIONDATE', 'LAST_RECEIPT_DATE'],inplace = True,axis = 'columns')


# In[146]:


import xgboost as xgb
clr = xgb.XGBClassifier(colsample_bytree=0.2,
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
clr.fit(x_train,y_train)


# In[147]:


from sklearn.metrics import roc_auc_score
y_cal=clr.predict(x_val)
print(roc_auc_score(y_val,y_cal))


# In[148]:


y_test = clr.predict(test_f)
test_f['FORECLOSURE'] = y_test


# In[218]:


fin=test_f[test_f['FORECLOSURE']==1]['AGREEMENTID'].tolist()
for a in fin:
    test.loc[test.AGREEMENTID==a,'FORECLOSURE']=1
test.to_csv('ans.csv',index=False)

