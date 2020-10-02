#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[ ]:


test.shape


# In[ ]:


train.shape


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


train.columns.values


# In[ ]:


test.columns.values


# In[ ]:


train.describe()


# In[ ]:


test.describe()


# In[ ]:


#checking missing values
train.isnull().sum()


# In[ ]:


test.isnull().sum()


# In[ ]:


train.info()


# In[ ]:


test.info()


# In[ ]:


train['Outcome'].hist(bins = 20)
plt.show()


# In[ ]:


train.Outcome.unique()


# In[ ]:


train.Outcome.value_counts()


# # predicting

# In[ ]:


X = train.drop(['Outcome'], axis = 1)
y = train.Outcome


# In[ ]:


X.shape


# In[ ]:


y.shape


# In[ ]:


y.head()


# In[ ]:


X.head()


# In[ ]:


train.shape


# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier
clf = GradientBoostingClassifier(n_estimators = 105)
clf.fit(X,y)


# In[ ]:


predicted = clf.predict(test)


# In[ ]:


print(predicted)


# In[ ]:


test.shape


# In[ ]:


predicted.shape


# In[ ]:


output = pd.DataFrame(predicted,columns = ['Outcome'])
test = pd.read_csv('../input/test.csv')
output['Id'] = test['Id']
output[['Id','Outcome']].to_csv('submission_cloudy10.csv', index = False)
output.head()


# In[ ]:




