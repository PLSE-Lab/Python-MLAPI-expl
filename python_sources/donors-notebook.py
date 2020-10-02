#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import time
import gc
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
sns.set_style("dark")


# In[3]:


train_df = pd.read_csv("../input/train.csv", parse_dates=['project_submitted_datetime'])
resources_df = pd.read_csv("../input/resources.csv")
test_df = pd.read_csv("../input/test.csv", parse_dates=['project_submitted_datetime'])


# In[4]:


train_df.head(10)


# In[5]:


test_df.sample(10)


# In[6]:


resources_df.head(10)


# In[7]:


print(train_df.shape)
print(test_df.shape)


# In[8]:


train_y = train_df['project_is_approved']


# In[9]:


df = pd.concat([train_df,test_df],sort=True)
df.shape


# In[10]:


dropcols = ['project_essay_1','project_essay_2','project_resource_summary','project_essay_3','project_essay_4','project_title','teacher_id']
df = df.drop(dropcols, axis=1)
df.sample()


# In[11]:


df.info()


# In[12]:


df.describe(include=['O']).T


# In[ ]:


df.describe().T
df.fillna(method='ffill',inplace=True)
del train_df
gc.collect()


# In[ ]:


df['year'] = df['project_submitted_datetime'].dt.year
df['month'] = df['project_submitted_datetime'].dt.month
df['day'] = df['project_submitted_datetime'].dt.day
df['hour'] = df['project_submitted_datetime'].dt.hour
df = df.drop(['project_submitted_datetime',],axis=1)
answer = pd.DataFrame()
df = pd.get_dummies(df,columns=['id','project_is_approved','project_grade_category','project_subject_categories','project_subject_subcategories','teacher_prefix','school_state','year','month','day','hour'])


# In[ ]:


df.head()
df.shape


# In[ ]:


project_grade_categories = df['project_grade_category'].value_counts().reset_index()
plt.figure(figsize=(8,4))
sns.barplot(y='project_grade_category',x='index',data=project_grade_categories)


# In[ ]:


school_states = df['school_state'].value_counts().reset_index()
plt.figure(figsize=(18,12))
sns.barplot(y='school_state',x='index',data=school_states)


# In[ ]:


train_x = df.iloc[:182080,:]
test_x = df.iloc[-78035:,:]
answer = pd.DataFrame()
answer['id'] = test_x['id']
train_x = train_x.drop(['id'],axis=1)
test_x = test_x.drop(['id'],axis=1)
test_x.tail()


# In[ ]:


classifier = xgb.XGBClassifier(n_estimators=300,learning_rate=0.15,silent=False)
classifier.fit(train_x,train_y)
answer['project_is_approved'] = classifier.predict(test_x)


# In[ ]:


answer.to_csv('Submission_xgb.csv',index=False)


# In[ ]:


answer.head()


# In[ ]:




