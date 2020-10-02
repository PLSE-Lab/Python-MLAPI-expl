#!/usr/bin/env python
# coding: utf-8

# In[52]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt  # Matlab-style plotting
import seaborn as sns
color = sns.color_palette()
sns.set_style('darkgrid')
import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)


from scipy import stats
from scipy.stats import norm, skew #for some statistics


# In[53]:


total = pd.read_csv("../input/weekly111/WeeklyActivationReport2.csv")


# In[54]:


normaltotal = total[1:]
normaltotal 


# In[55]:


corrmat = normaltotal.corr()
plt.subplots(figsize=(25,25))
sns.heatmap(corrmat, vmax=0.9, annot = True ,square=True)


# In[56]:


from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


# In[57]:


train = normaltotal[:12]
test = normaltotal[12:]
test["Marketing Spend"] = 3000
test["Traffic"] = 25500
test["Buy Now Events"] = 4150
test


# In[58]:



rforest = RandomForestClassifier()
LinearModel = LinearRegression()


# In[59]:


X_train = train.drop(["Activations"], axis = 1)
y_train = train["Activations"]
X_test = test.drop(["Activations"], axis = 1)


# In[60]:





# In[61]:


rforest.fit(X_train, y_train)
rforestpredict = rforest.predict(X_test)
rforestpredict


# In[62]:


LinearModel.fit(X_train, y_train)
LinearModelpredict = LinearModel.predict(X_test)
LinearModelpredict


# In[ ]:




