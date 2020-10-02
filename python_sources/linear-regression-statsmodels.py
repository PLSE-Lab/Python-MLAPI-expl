#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
sns.set()


# In[ ]:


data=pd.read_csv('../input/advertising-dataset/advertising.csv')
data.describe()


# In[ ]:


#dependent variable=Sales, independent variable=TV ad
s,t1,r,n=data['Sales'],data['TV'],data['Radio'],data['Newspaper']


# In[ ]:


plt.scatter(t1,s)
plt.xlabel('TV')
plt.ylabel('Sales')
plt.show()


# In[ ]:


plt.scatter(r,s)
plt.xlabel('Radio')
plt.ylabel('Sales')
plt.show()


# In[ ]:


plt.scatter(n,s)
plt.xlabel('Newspaper')
plt.ylabel('Sales')
plt.show()


# ### Looking at the scatter plots between Sales and Newspaper/TV/Radio  it is evident that Sales and TV has strong positive relationship.
# So we will now try to find the line of best fit for Sales and TV

# In[ ]:


t=sm.add_constant(t1)
results=sm.OLS(s,t).fit()
results.summary()


# In[ ]:


plt.scatter(t1,s)
yhat=6.9748+0.0555*t1
plt.plot(t1,yhat,c='orange',label='Regression')
plt.xlabel('TV')
plt.ylabel('Sales')
plt.show()


# ### Regression equation=0.0555*t1+6.9748
# From the summary we can see R-square=0.812 that means the Regression strongly explains the variability of data
# std. error=0.323 is the error (smaller means better)
