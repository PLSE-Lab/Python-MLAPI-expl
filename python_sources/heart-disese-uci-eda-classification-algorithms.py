#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import os
print(os.listdir("../input"))


# In[7]:


df=pd.read_csv('../input/heart.csv')


# In[8]:


df.head()


# In[9]:


df.describe()


# In[10]:


df.info()


# In[11]:


df.isnull().any()


# age vs disease

# In[12]:


df['age_cat']=pd.cut(df['age'], 5)


# In[13]:


sns.barplot(x=df.groupby(by="age_cat").target.sum().index,y=df.groupby(by="age_cat").target.sum())


# age of the people is normal in distribution

# most of the patients are in range 48-59

# In[14]:


sns.catplot(data = df, y ="age", x = "target", hue = 'sex', sharex=False)


# sex vs disease

# In[15]:


sns.countplot(x=df['target'],hue=df['sex'])


# there are more males in the data similarly there more males that have heart disease.
# 
# But if we see proportionately, females are more suffering from heart disease

# In[16]:


df['cp'].unique()


# In[17]:


df['cp'] = df['cp'].replace(0, 'typical angina')
df['cp'] = df['cp'].replace(1, 'atypical angina')
df['cp'] = df['cp'].replace(2, 'non-anginal pain')
df['cp'] = df['cp'].replace(3, 'asymptomatic')


# chestpain vs heart disease

# In[18]:


sns.barplot(x=df.groupby(by="cp").target.sum().index,y=df.groupby(by="cp").target.sum())


# disease more observed in patients with non-anginal pain

# In[19]:


df['trestbps'].unique()


# blood pressure at rest vs disease

# In[20]:


df['bps_cat']=pd.cut(df['trestbps'], 5)


# In[21]:


sns.barplot(x=df.groupby(by="bps_cat").target.sum().index,y=df.groupby(by="bps_cat").target.sum())


# diseased is occurance is more in patients with rest bp at 115 - 137

# In[22]:


sns.catplot(data = df, y ="trestbps", x = "target", hue='sex' ,sharex=False)


# person with hishest bp of 200 dont have heart disease

# In[23]:


df['chol_cat']=pd.cut(df['chol'], 5)


# cholestrol vs disease

# In[24]:


sns.barplot(x=df.groupby(by="chol_cat").target.sum().index,y=df.groupby(by="chol_cat").target.sum())


# In[25]:


sns.catplot(data = df, y ="chol", x = "target", hue='sex' ,sharex=False)


# there are very less people with high colestrols 
# 
# patients with cholestrol levels of 213-301 have more heart disease

# fbs vs heart disease

# In[26]:


sns.countplot(data = df, x = "target", hue='fbs')


# heart disease doesnot really depend upon fasting blood sugar

# restecg vs disease

# In[27]:


sns.countplot(data = df, x = "target", hue='restecg')


# In[28]:


sns.catplot(data = df, y ="restecg", x = "target" ,sharex=False)


# only 4-5 are showing probable or definite left ventricular hypertrophy by Estes' criteria from which only one person is suffering
# 
# normal ecg have few patients when compared to people with having ST-T wave abnormality

# In[29]:


df['thalach_cat']=pd.cut(df['thalach'], 5)


# highest heart rate reached vs disease

# In[30]:


sns.barplot(x=df.groupby(by="thalach_cat").target.sum().index,y=df.groupby(by="thalach_cat").target.sum())


# In[31]:


sns.catplot(data = df, y ="thalach", x = "target" ,sharex=False)


# persons with heart rate maximum at 149 - 176 are having heart disease

# exang vs disease

# In[32]:


sns.countplot(data = df, x = "target", hue='exang')


# patients with no enigma are more probably having heart disease

# In[33]:


df['oldpeak_cat']=pd.cut(df['oldpeak'], 5)


# old peak vs heart disease

# In[34]:


sns.barplot(x=df.groupby(by="oldpeak_cat").target.sum().index,y=df.groupby(by="oldpeak_cat").target.sum())


# In[35]:


sns.catplot(data = df, y ="oldpeak", x = "target" ,sharex=False)


# people with old peak less than 5 have suffered if thet have less than 1.25, the probability is high

# In[36]:


df['slope'] = df['slope'].replace(0,'upslope')
df['slope'] = df['slope'].replace(1,'flatslope')
df['slope'] = df['slope'].replace(2,'downslope')


# slope of ecg vs heart disease

# In[37]:


sns.countplot(data = df, x = "target", hue='slope')


# In[38]:


sns.catplot(data = df, x ="slope", y = "target" ,sharex=False)


# there are few people with 0(upslope) slope, the chances are almost equal
# 
# people with 1 flatslope suffer more
# 
# people with downslope have less probability

# ca vs heart disese

# In[39]:


sns.countplot(data = df, x = "target", hue='ca')


# person with 0 zeromajor vessels have high chances of heart disese

# thal vs heart disese

# In[40]:


df['thal'].unique()


# In[41]:


sns.countplot(data = df, x = "target", hue='thal')


# when thal = 0 , the person is more probobly having heart disease

# In[42]:


slope = pd.get_dummies(df['slope'])
slope.drop(slope.columns[[0]],axis=1,inplace=True)
df = pd.concat([df,slope],axis=1)


# In[43]:


thal = pd.get_dummies(df['thal'],prefix='thal')
thal.drop(thal.columns[[0]],axis=1,inplace=True)
df = pd.concat([df,thal],axis=1)


# In[44]:


del df['age_cat']
del df['bps_cat']
del df['chol_cat']
del df['thalach_cat']
del df['oldpeak_cat']


# In[45]:


df.head()


# In[46]:


plt.figure(figsize=(15,15))
sns.heatmap(df.corr(),annot=True,cmap='Blues')


# In[47]:


features = ['thal_3','thal_3','thal_1','flatslope','ca','oldpeak','exang','thalach','restecg','trestbps','sex','age']
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics


# In[48]:


def conflog(features):
    X = df[features]
    y = df.target
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=125)
    log = LogisticRegression(multi_class='auto')
    log.fit(X_train,y_train)
    y_train_pred = log.predict(X_train)   
    y_test_pred = log.predict(X_test)   
    print('for train')
    print(metrics.classification_report(y_train,y_train_pred))
    print(metrics.accuracy_score(y_train,y_train_pred))
    print('for test')
    print(metrics.classification_report(y_test,y_test_pred))
    print(metrics.accuracy_score(y_test,y_test_pred))


# In[49]:


conflog(features)


# In[50]:


def confNB(features):
    X = df[features]
    y = df.target
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=125)
    nb = GaussianNB()
    nb.fit(X_train,y_train)
    y_train_pred = nb.predict(X_train)   
    y_test_pred = nb.predict(X_test)   
    print('for train')
    print(metrics.classification_report(y_train,y_train_pred))
    print(metrics.accuracy_score(y_train,y_train_pred))
    print('for test')
    print(metrics.classification_report(y_test,y_test_pred))
    print(metrics.accuracy_score(y_test,y_test_pred))


# In[51]:


confNB(features)


# In[52]:


def confDT(features):
    X = df[features]
    y = df.target
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=125)
    dt = DecisionTreeClassifier()
    dt.fit(X_train,y_train)
    y_train_pred = dt.predict(X_train)   
    y_test_pred = dt.predict(X_test)   
    print('for train')
    print(metrics.classification_report(y_train,y_train_pred))
    print(metrics.accuracy_score(y_train,y_train_pred))
    print('for test')
    print(metrics.classification_report(y_test,y_test_pred))
    print(metrics.accuracy_score(y_test,y_test_pred))


# In[53]:


confDT(features)


# this model may not be appropriate with decision tree

# In[54]:


def confSVM(features):
    X = df[features]
    y = df.target
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=125)
    svm = SVC(gamma='scale')
    svm.fit(X_train,y_train)
    y_train_pred = svm.predict(X_train)   
    y_test_pred = svm.predict(X_test)   
    print('for train')
    print(metrics.classification_report(y_train,y_train_pred))
    print(metrics.accuracy_score(y_train,y_train_pred))
    print('for test')
    print(metrics.classification_report(y_test,y_test_pred))
    print(metrics.accuracy_score(y_test,y_test_pred))


# In[55]:


confSVM(features)


# In[56]:


def confRF(features):
    X = df[features]
    y = df.target
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=125)
    rf = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0)
    rf.fit(X_train,y_train)
    y_train_pred = rf.predict(X_train)   
    y_test_pred = rf.predict(X_test)   
    print('for train')
    print(metrics.classification_report(y_train,y_train_pred))
    print(metrics.accuracy_score(y_train,y_train_pred))
    print('for test')
    print(metrics.classification_report(y_test,y_test_pred))
    print(metrics.accuracy_score(y_test,y_test_pred))


# In[57]:


confRF(features)


# In[ ]:




