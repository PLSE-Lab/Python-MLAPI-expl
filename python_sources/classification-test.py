#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# The goal of this short notebook is to see if the discretised dataset will greatly improve the accuracy score of the random forest algorithm as well as the logistic regression algorithm.
# 
# In order to do so I created 2 functions to assist with the data preparation and cross validation.

# In[117]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder


# In[118]:


#reading data
df = pd.read_csv("../input/modified_census_data.csv")


# In[119]:


df.sample()


# In[120]:


#spliting data, I understood columns starting with LabelP are the modified ones.
df_original = pd.DataFrame(df,columns=[ 'LABEL', 'age', 'capital_gain', 'capital_loss',
       'education', 'education_num', 'fnlwgt', 'hours_per_week',
       'marital_status', 'native_country', 'occupation', 'race',
       'relationship', 'sex', 'workLABEL'])
df_discretised = pd.DataFrame(df,columns = ['LabelPage', 'LabelPcapital_gain',
       'LabelPcapital_loss', 'LabelPeducation', 'LabelPeducation_num',
       'LabelPhours_per_week', 'LabelPmarital_status', 'LabelPnative_country',
       'LabelPoccupation', 'LabelPrace', 'LabelPrelationship', 'LabelPsex',
       'LabelPworkLABEL','LABEL'])


# In[129]:


df.dtypes.unique()


# In[121]:


#Making all non numeric variables numeric.
def convert(data):
    col_to_transform = data.dtypes[data.dtypes!=int].index
    for i in col_to_transform:
        if data[i].unique().shape[0]>20:
            data[i] = LabelEncoder().fit_transform(data[i])
        else :
            data = pd.concat([data,pd.get_dummies(data[i])],axis = 1)
            del data[i]
    return data


# In[122]:


df_original = convert(df_original)
df_discretised = convert(df_discretised)


# In[123]:


#Cross validation
def cross_val(df,ratio = 0.5):
    train = df.sample(frac = ratio)
    test = df[~df.index.isin(train.index)].dropna()
    return train,test


# In[124]:


#original data benchmark
rf = RandomForestClassifier(n_estimators=100,max_depth=5,min_samples_split=10,n_jobs = -1)
m = LogisticRegression()
score_lr = []
score_rf = []
for i in range(5):
    train,test= cross_val(df_original)
    Ytrain=train['LABEL'];del train['LABEL']
    Ytest=test['LABEL'];del test['LABEL']
    
    m.fit(train,Ytrain)
    rf.fit(train,Ytrain)
    temp_lr = m.predict(test)
    temp_rf = rf.predict(test)
    score_rf.append(sum(temp_rf == Ytest)/len(Ytest))
    score_lr.append(sum(temp_lr == Ytest)/len(Ytest))


# In[125]:


print('Random forest : '+ str(np.mean(score_rf)))
print('Logistic Regression : '+ str(np.mean(score_lr)))


# In[126]:


#discretised data benchmark
rf = RandomForestClassifier(n_estimators=100,max_depth=5,min_samples_split=10,n_jobs = -1)
m = LogisticRegression()
score_lr = []
score_rf = []
for i in range(5):
    train,test= cross_val(df_discretised)
    Ytrain=train['LABEL'];del train['LABEL']
    Ytest=test['LABEL'];del test['LABEL']
    
    m.fit(train,Ytrain)
    rf.fit(train,Ytrain)
    temp_lr = m.predict(test)
    temp_rf = rf.predict(test)
    score_rf.append(sum(temp_rf == Ytest)/len(Ytest))
    score_lr.append(sum(temp_lr == Ytest)/len(Ytest))


# In[127]:


print('Random forest : '+ str(np.mean(score_rf)))
print('Logistic Regression : '+ str(np.mean(score_lr)))


# # Conclusion
# 
# We see that the modified dataset makes information more explicit to the logistic regression algorithm model. While the Random forest model trained on the original data slightly outperforms the model trained on the discretised data.
# 
# Would love to do a similar kernel on new datasets !
