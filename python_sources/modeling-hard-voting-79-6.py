#!/usr/bin/env python
# coding: utf-8

# In[124]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[125]:


df = pd.read_csv(r"../input/WA_Fn-UseC_-Telco-Customer-Churn.csv")


# In[126]:


df.head()


# In[127]:


df.info()


# In[128]:


df.isnull().sum()


# In[129]:


df.describe() 


# In[130]:


import matplotlib.pyplot as plt


# In[131]:


import seaborn as sns


# In[132]:


corr = df.corr()
corr.style.background_gradient(cmap='coolwarm')


# In[133]:


df['PaymentMethod'].value_counts()


# In[134]:


df_cat = df[['gender','SeniorCitizen','Partner','Dependents','PhoneService','MultipleLines','InternetService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies','Contract','PaperlessBilling','PaymentMethod','Churn']]


# In[135]:


df_cat.head()


# In[136]:


df.head()


# In[137]:


df = df.iloc[:,1:]


# In[138]:


df['TotalCharges'] = pd.to_numeric(df['TotalCharges'],errors = 'coerce')
df = df.dropna()


# In[139]:


df.head()
df['Churn'].replace(to_replace = 'No', value = 0, inplace = True)
df['Churn'].replace(to_replace = 'Yes', value = 1, inplace = True)


# In[140]:


df = pd.get_dummies(df)


# In[141]:



plt.figure(figsize=(20,8))
corr = df.corr()['Churn'].sort_values(ascending = False).plot(kind='bar')


# In[142]:


X = df.drop(['Churn'],1)
X.head()


# In[143]:


y = df['Churn']
y.head()


# In[144]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score


# In[145]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.30, random_state = 40)


# In[146]:


classifiers = [ ['LogisticRegression :', LogisticRegression()],
                 ['DecisionTree :',DecisionTreeClassifier()],
               ['RandomForest :',RandomForestClassifier()], 
               ['AdaBoostClassifier :', AdaBoostClassifier()],
               ['XGB :', XGBClassifier()]]


# In[147]:


predictions_df = pd.DataFrame()
predictions_df['actual_labels'] = y_test

for name,classifier in classifiers:
    classifier = classifier
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)
    predictions_df[name.strip(" :")] = predictions
    print(name, accuracy_score(y_test, predictions))


# In[148]:


from sklearn.ensemble import VotingClassifier
clf1 = AdaBoostClassifier()
clf2 = LogisticRegression()
clf3 = XGBClassifier()
clf4 = RandomForestClassifier()
vclf = VotingClassifier(estimators=[('adab', clf1), ('lr', clf2), ('xgb', clf3),('rf', clf4)], voting='hard')
vclf.fit(X_train, y_train)
predictions = vclf.predict(X_test)
print(accuracy_score(y_test, predictions))


# In[ ]:




