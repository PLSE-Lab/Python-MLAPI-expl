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
df_train = pd.read_csv('../input/webclubrecruitment2019/TRAIN_DATA.csv');
df_test = pd.read_csv('../input/webclubrecruitment2019/TEST_DATA.csv');

df_train.head()
# Any results you write to the current directory are saved as output.


# In[ ]:


df_test.head()


# In[ ]:


df_train.isnull().sum()


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


(df_train.Class == 1).sum()


# In[ ]:


sns.countplot(df_train.Class)


# In[ ]:


plt.figure(figsize=(10,8))
sns.heatmap(df_train.corr())


# In[ ]:


sns.scatterplot(x = df_train.Class,y = df_train.V15,hue = df_train.V16)


# In[ ]:


df = pd.concat([df_train.iloc[:,1:-1],df_test.iloc[:,1:]])
df['V14_cat'] = (df_train.V14 == -1).astype(int)
df.drop('V14',inplace = True,axis = 1)


# In[ ]:


num_cols = ['V1','V6','V10','V12','V13','V15']
cat_cols = list(set(df.columns) - set(num_cols))


# In[ ]:


corr = df_train.corr()['Class'][num_cols].abs().sort_values(ascending = False)
corr


# In[ ]:


cat_cols_n = [col for col in cat_cols if df[col].nunique()>2]
cat_cols_2 = list(set(cat_cols) - set(cat_cols_n))


# In[ ]:


cat_cols_2


# fig,axs = plt.subplots(3,3,figsize = (14,10))
# for i in range(len(cat_cols)):
#     
#     sns.countplot(df_train[cat_cols[i]],hue = df_train.Class,ax = axs[i//3,i%3])

# In[ ]:


len(df_train)#v16,v5,v15


# cat_cols_n.remove('V16')
# cat_cols_2.remove('V5')
# num_cols.remove('V15')

# In[ ]:


#All features
X_1 = df_train.loc[:, 'V1':'V16']
y = df_train.Class;


# In[ ]:


#One hot encoded
df_n = pd.get_dummies(df,columns=cat_cols_n)
X_2 = df_n.iloc[:30000,:] 
x_test_2 = df_n.iloc[30000:,:]


# In[ ]:


X_2.shape


# In[ ]:


#feature selection
df_n = df[num_cols+cat_cols_2+cat_cols_n]
df_n = pd.get_dummies(df_n,columns=cat_cols_n)
X_3 = df_n.iloc[:30000,:] 
x_test_3 = df_n.iloc[30000:,:]


# In[ ]:


X_3.shape


# In[ ]:


from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


# In[ ]:


train_x,val_x,train_y,val_y = train_test_split(X_2,y,test_size = 0.2,random_state = 42)


# In[ ]:


train_y.shape


# In[ ]:


scores = {}


# for i in range(200,1201,200):
#     model = XGBClassifier(n_estimators=i,learning_rate=0.2)
#     model.fit(train_x,train_y)
#     preds = model.predict_proba(val_x)
#     scores[i] = roc_auc_score(val_y,preds[:,1])
#     print(scores[i])
#     

# In[ ]:


scores


# model2 = RandomForestClassifier(n_estimators=50, random_state=123)
# model2.fit(train_x,train_y)

# model2 = RandomForestClassifier(n_estimators=50, random_state=123)
# model2.fit(train_x,train_y)

# In[ ]:


model = XGBClassifier(n_estimators=300,learning_rate=0.1,early_stopping_rounds = 6,eval_set = [val_x,val_y])
model.fit(train_x,train_y)


# In[ ]:


preds = model.predict_proba(val_x)
score = roc_auc_score(val_y,preds[:,1])
score


# In[ ]:


model.fit(X_2,y)
test_x = df_test.loc[:, 'V1':'V16']
preds_y = model.predict_proba(x_test_2)


# In[ ]:


result=pd.DataFrame()
result['Id'] = df_test['Unnamed: 0']
result['PredictedValue'] = pd.DataFrame(preds_y[:,1])
result.head()


# In[ ]:


result.to_csv('output.csv', index=False)

