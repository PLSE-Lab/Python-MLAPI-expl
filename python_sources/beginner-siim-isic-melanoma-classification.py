#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd


# In[ ]:


train = pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/train.csv')
train


# In[ ]:


train.info()


# In[ ]:


train.isnull().sum()


# In[ ]:


train['sex'] = train['sex'].fillna('unknown')
train['age_approx'] = train['age_approx'].fillna(train['age_approx'].median())
train['anatom_site_general_challenge'] = train['anatom_site_general_challenge'].fillna('unknown')


# In[ ]:


train.isnull().sum()


# In[ ]:


test = pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/test.csv')


# In[ ]:


test


# In[ ]:


test.isnull().sum()


# In[ ]:


test['anatom_site_general_challenge'] = test['anatom_site_general_challenge'].fillna('unknown')


# In[ ]:


test.isnull().sum()


# In[ ]:


train.drop(['diagnosis', 'benign_malignant'],axis=True,inplace=True)


# In[ ]:


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()


# In[ ]:


train['image_name'] = le.fit_transform(train['image_name'])
train['patient_id'] = le.fit_transform(train['patient_id'])
train['sex'] = le.fit_transform(train['sex'])
train['anatom_site_general_challenge'] = le.fit_transform(train['anatom_site_general_challenge'])


# In[ ]:


train


# In[ ]:


X = train.drop(['target'], axis=1)
y = train.target


# In[ ]:


from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,BaggingClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier


# In[ ]:


import lightgbm as lgb


# In[ ]:


from lightgbm import LGBMClassifier


# In[ ]:





# In[ ]:


model = LGBMClassifier()


# In[ ]:


model.fit(X,y)


# In[ ]:


z = test


# In[ ]:


z


# In[ ]:


z['image_name'] = le.fit_transform(z['image_name'])
z['patient_id'] = le.fit_transform(z['patient_id'])
z['sex'] = le.fit_transform(z['sex'])
z['anatom_site_general_challenge'] = le.fit_transform(z['anatom_site_general_challenge'])


# In[ ]:


z


# In[ ]:





# In[ ]:


z


# In[ ]:


target = model.predict(z)


# In[ ]:


x = pd.DataFrame(target,columns=['target'])


# In[ ]:


q = pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/test.csv')


# In[ ]:


y = q.join(x)


# In[ ]:


y


# In[ ]:


y = y.drop(['patient_id','sex','age_approx','anatom_site_general_challenge'],axis=1)


# In[ ]:


y


# In[ ]:


y.to_csv('siim.csv', index=False)


# In[ ]:




