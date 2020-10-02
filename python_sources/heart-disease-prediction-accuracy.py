#!/usr/bin/env python
# coding: utf-8

# In[31]:


import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import os as os


# In[32]:


os.chdir("../input")


# In[33]:


os.listdir()


# In[34]:


data=pd.read_csv('../input/heart.csv')


# In[35]:


data.head(3)


# In[36]:


data.shape


# In[37]:


from sklearn.model_selection import train_test_split


# In[38]:


train,test=train_test_split(data,test_size=0.2)


# In[39]:


len(train)


# In[40]:


len(test)


# In[41]:


data.corr()


# In[42]:


sb.scatterplot(x=train['cp'], y= train['age'],hue=train['target'],data=train)


# In[43]:


sb.countplot(x=train['cp'],hue=train['target'],data=train)


# In[44]:


train.isnull().sum()


# In[45]:


sb.heatmap(train.isnull(),yticklabels=False,cbar='False')


# In[46]:


train.describe()


# In[47]:


test.isnull().sum()


# In[48]:


sb.boxplot(x='age',y='target',data=train,palette='Set1')


# In[49]:


train_out=train['target']


# In[50]:


train_in=train.drop(['target'],axis=1)


# In[51]:


train_in.shape


# In[52]:


from sklearn.linear_model import LogisticRegression


# In[53]:


log_reg=LogisticRegression()
log_reg.fit(train_in,train_out)


# In[54]:


test_out=test['target']


# In[55]:


test_in=test.drop(['target'],axis=1)


# In[56]:


prediction=log_reg.predict(test_in)


# In[57]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# In[58]:


accuracy_score(test_out,prediction)*100


# In[59]:


confusion_matrix(test_out,prediction)


# In[60]:


print(classification_report(test_out,prediction))


# In[ ]:




