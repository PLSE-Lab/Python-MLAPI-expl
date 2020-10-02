#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import sklearn 
from sklearn.cluster import (KMeans,MiniBatchKMeans)
from sklearn.metrics import accuracy_score


# In[ ]:


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
sample = pd.read_csv('sample_submission.csv')
sub = pd.DataFrame({'custId':test.custId})


# In[ ]:


data = (train,test)


# In[ ]:


for col in train.columns:
    print(str(col) + "\t" + str(train[col].unique()))


# In[ ]:


def num(s):
    try:
        return float(s)
    except ValueError:
        return 0


# In[ ]:


for df in data:
    df.gender = df.gender.map({'Male':1,'Female':2})
    df.Married = df.Married.map({'Yes':1,'No':0})
    df.Children = df.Children.map({'Yes':1,'No':0})
    df.TVConnection = df.TVConnection.map({'No':0,'Cable':1,'DTH':1})
    df.Channel1 = df.Channel1.map({'Yes':1,'No':0,'No tv connection':0})
    df.Channel2 = df.Channel2.map({'Yes':1,'No':0,'No tv connection':0})
    df.Channel3 = df.Channel3.map({'Yes':1,'No':0,'No tv connection':0})
    df.Channel4 = df.Channel4.map({'Yes':1,'No':0,'No tv connection':0})
    df.Channel5 = df.Channel5.map({'Yes':1,'No':0,'No tv connection':0})
    df.Channel6 = df.Channel6.map({'Yes':1,'No':0,'No tv connection':0})
    df.Internet = df.Internet.map({'Yes':1,'No':0})
    df.HighSpeed = df.HighSpeed.map({'Yes':1,'No':0,'No internet':0})
    df.AddedServices = df.AddedServices.map({'Yes':1,'No':0})
    df.Subscription = df.Subscription.map({'Monthly':1, 'Biannually':2, 'Annually':3})
    df.PaymentMethod = df.PaymentMethod.map({'Cash':1, 'Bank transfer':2, 'Net Banking':3, 'Credit card':4})
    df.TotalCharges = df.TotalCharges.apply(num)


# In[ ]:


y=train.Satisfied
train.drop(['custId','Satisfied','tenure','AddedServices','PaymentMethod','TotalCharges','Internet'],axis=1,inplace=True)
test.drop(['custId','tenure','AddedServices','PaymentMethod','TotalCharges','Internet'],axis=1,inplace=True)


# In[ ]:


for df in data:
    for col in df.columns:
        df[col+'_freq'] = df[col].map(df[col].value_counts())


# In[ ]:


kmeans = KMeans(n_clusters=2, random_state=0).fit(train)


# In[ ]:


preds = kmeans.predict(train)


# In[ ]:


accuracy_score(y,preds)


# In[ ]:


minikmeans = MiniBatchKMeans(n_clusters=2,random_state=0,batch_size=6).partial_fit(train)


# In[ ]:


preds2 = minikmeans.predict(train)


# In[ ]:


accuracy_score(y,preds2)


# In[ ]:





# In[ ]:





# In[ ]:


pred_final = minikmeans.predict(test)


# In[ ]:


sub['Satisfied'] = pred_final


# In[ ]:


sub.to_csv('sub.csv',index=False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




