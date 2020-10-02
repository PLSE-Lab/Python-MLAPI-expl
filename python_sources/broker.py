#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:





# In[2]:


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


# In[3]:


x=pd.read_csv('../input/train.csv')


# In[4]:


y=x.drop(['Id'],axis=1)


# In[5]:


a=y.count()>800


# In[6]:


p=a[a==True]


# In[7]:


z=list(p.index)


# In[8]:


y=y[z]


# In[9]:


y.shape


# In[10]:


a=y.apply(pd.Series.nunique)<10


# In[11]:


p=a[a==True]
p=list(p.index)
p


# In[12]:


y=y.drop(['Street','LandContour','SaleCondition','SaleType','PoolArea','PavedDrive','GarageCond','GarageQual','Functional','KitchenAbvGr','BsmtHalfBath','Electrical','CentralAir','Heating','BsmtFinType2','BsmtCond','ExterCond','RoofMatl','RoofStyle','BldgType','Condition2','Condition1','LandSlope','Utilities'],axis=1)


# In[14]:


y.columns


# In[39]:


x=y.corr().iloc[:,-1]


# In[43]:


h=x[(x>0.4) | (x<-0.4)] 


# In[45]:


len(h)


# In[46]:


from sklearn.preprocessing import OneHotEncoder


# In[50]:


x=list(h.index)


# In[51]:


y=y[x]


# In[54]:


y=y.drop(['SalePrice'],axis=1)


# In[55]:


z=pd.read_csv('../input/test.csv')
z=z[y.columns]


# In[63]:


a=y.apply(lambda x:x.fillna(x.mean()))
b=z.apply(lambda x: x.fillna(x.mean()))


# In[65]:


o=pd.read_csv('../input/train.csv')


# In[67]:


import seaborn as sns
sns.distplot(o['SalePrice'])


# In[68]:


sns.distplot(np.log(o['SalePrice']))


# In[69]:


y


# In[70]:


a


# In[71]:


from sklearn.tree import DecisionTreeRegressor


# In[72]:


l=DecisionTreeRegressor()
from sklearn.model_selection import cross_val_score


# In[76]:


x=cross_val_score(l,a,np.log(o['SalePrice']),cv=5)


# In[81]:


l.fit(a,np.log(o['SalePrice']))


# In[91]:


x=l.predict(b)


# In[94]:


n=[np.exp(v) for v in x]


# In[98]:


x=pd.DataFrame(n)


# In[100]:


x.index=pd.read_csv('../input/test.csv')['Id']


# In[103]:


x.columns=['SalePrice']


# In[105]:


x.to_csv('result.csv')


# In[ ]:




