#!/usr/bin/env python
# coding: utf-8

# In[27]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as ms
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split
from sklearn.cross_validation import KFold   
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics


# In[28]:


cancer_d=pd.read_csv("../input/data.csv")


# In[29]:


cancer_d.info()


# In[30]:


cancer_d.describe()


# In[31]:


cancer_d.corr()


# In[32]:


cancer_d.drop('id',axis=1,inplace=True)
cancer_d.drop('Unnamed: 32',axis=1,inplace=True)


# In[33]:


cancer_d.corr()


# In[34]:


cancer_d['diagnosis'] = cancer_d['diagnosis'].map({'M':1,'B':0})


# In[35]:


cancer_d.corr()


# In[10]:


ms.matrix(cancer_d)


# In[36]:


fts=list(cancer_d.columns[1:11])


# In[ ]:





# In[37]:


cancer_d.describe()


# In[38]:


from sklearn.model_selection import train_test_split

train,test = train_test_split(cancer_d, test_size=0.30, 
                                                    random_state=101)


# In[22]:


train.head()


# In[23]:


test.head()


# In[39]:


def classification_model(model, train,test, predictors, outcome):
  model.fit(train[predictors],train[outcome])
  predictions = model.predict(test[predictors])
  res= metrics.accuracy_score(predictions,test[outcome])
  print(res)
  model.fit(train[predictors],train[outcome]) 


# In[40]:



model = RandomForestClassifier()
outcome_var='diagnosis'
classification_model(model,train,test,fts,'diagnosis')


# In[ ]:




