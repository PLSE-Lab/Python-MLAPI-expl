#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


train=pd.read_csv("../input/train.csv")


# In[ ]:


test=pd.read_csv("../input/test.csv")


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


train.isnull().sum().sort_values(ascending=False)


# In[ ]:


train.rez_esc.fillna(train.rez_esc.mean(),inplace=True)
train.v18q1.fillna(train.v18q1.mean(),inplace=True)
train.v2a1.fillna(train.v2a1.mean(),inplace=True)
train.meaneduc.fillna(train.meaneduc.mean(),inplace=True)
train.SQBmeaned.fillna(train.SQBmeaned.mean(),inplace=True)


# In[ ]:


test.isnull().sum().sort_values(ascending=False)


# In[ ]:


test.rez_esc.fillna(test.rez_esc.mean(),inplace=True)
test.v18q1.fillna(test.v18q1.mean(),inplace=True)
test.v2a1.fillna(test.v2a1.mean(),inplace=True)
test.meaneduc.fillna(test.meaneduc.mean(),inplace=True)
test.SQBmeaned.fillna(test.SQBmeaned.mean(),inplace=True)


# In[ ]:


test.dtypes


# In[ ]:


intcols=train.select_dtypes(include=["int64"])
floatcols=train.select_dtypes(include=["float64"])
objectcols=train.select_dtypes(include=["object"])


# In[ ]:


from sklearn.preprocessing import LabelEncoder


# In[ ]:


le=LabelEncoder()


# In[ ]:


intcols1=intcols.apply(le.fit_transform)
objectcols1=objectcols.apply(le.fit_transform)


# In[ ]:


train1=pd.concat([intcols1,objectcols1,floatcols],axis=1)


# In[ ]:


intcols=test.select_dtypes(include=["int64"])
floatcols=test.select_dtypes(include=["float64"])
objectcols=test.select_dtypes(include=["object"])


# In[ ]:


intcols1=intcols.apply(le.fit_transform)
objectcols1=objectcols.apply(le.fit_transform)


# In[ ]:


test1=pd.concat([intcols1,objectcols1,floatcols],axis=1)


# In[ ]:


test1.dtypes


# In[ ]:


x=train1.drop(["Target"],axis=1)
y=train1.Target


# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


rfc=RandomForestClassifier()


# In[ ]:


rfcmodel=rfc.fit(x,y)


# In[ ]:


rfcmodel.score(x,y)


# In[ ]:


predict=rfcmodel.predict(test1)


# In[ ]:


predict


# In[ ]:


accuracy=round(rfcmodel.score(x,y)*100,2)


# In[ ]:


accuracy


# In[ ]:


sub=pd.read_csv('../input/sample_submission.csv')


# In[ ]:


sub.to_csv('sample_submission.csv',index=False)


# In[ ]:





# In[ ]:




