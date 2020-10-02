#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import re

import warnings
warnings.filterwarnings("ignore")

get_ipython().run_line_magic('matplotlib', 'inline')
import os
print(os.listdir("../input"))

df = pd.read_csv("../input/data.csv")

df.head()


# In[ ]:


df.info()


# In[ ]:


# Fixing Financials
financial = ['Value', 'Wage', 'Release Clause']
for f in financial:
    df[f] = df[f].apply(lambda x: str(x)[1:])
    
df.head()


# In[ ]:


# K=1.000 and M=1.000.000  
def convert(value):
    regex = r'K|M'
    m = re.search(regex, value)
    if m:
        value = re.sub(regex, "", value)
        
        if m.group() == "M":
            value = pd.to_numeric(value) * 1e6
            value = value / 1000
        else:
            value = pd.to_numeric(value) * 1e3
            value = value / 1000
            
    return value
            
for f in financial:
    df[f] = df[f].apply(convert)

df.head()


# In[ ]:


df['Value'] = df['Value'].astype(int)
df['Wage'] = df['Wage'].astype(int)
df.dtypes


# In[ ]:


df.head()


# In[ ]:


numerical = df.select_dtypes(include=['float64', 'int64']).keys().values

fig = plt.figure(figsize=(25,25))
ax = fig.gca()
df[numerical].hist(ax = ax)
plt.show()


# In[ ]:


sns.jointplot(x="Age", y="Wage", data=df.sample(100), kind='reg')


# In[ ]:


run_regression(df,'Wage ~ Age')


# ** Statistically grounded we may say that at every birthday players increase their wage by EUR 664.******

# > *But wait! As the histogram had shown there seems to be a great variance on Wage. Let's explore it!*

# In[ ]:


fig = plt.figure(figsize=(4,4))
ax = fig.gca()
df['Wage'].hist(ax = ax)
plt.show()


# In[ ]:


df['Wage'].describe()


# In[ ]:


df.groupby('Age')['Wage'].size().plot()


# In[ ]:


df.groupby('Age')['Wage'].mean().plot()


# In[ ]:


#Exploring it from a log point of view

df['log_wage'] = np.log1p(df['Wage'])

df.head()


# In[ ]:


fig = plt.figure(figsize=(4,4))
ax = fig.gca()
df['log_wage'].hist(ax = ax)
plt.show()


# In[ ]:


df['log_wage'].describe()


# In[ ]:


sns.jointplot(x="Age", y="log_wage", data=df.sample(100), kind='reg')


# In[ ]:


run_regression(df,'log_wage ~ Age')


# ** Turning it into a fixed growth rate though, we may say that, statistically, at every birthday players increase their wage by 5.72% **

# > *But wait! What about the others variables? Are they related to the relative impact that age has on wage? *

# In[ ]:


#checking the impact of dribbling

run_regression(df, 'log_wage ~ Age + Dribbling')


# **The analysis shows that Dribbling affects Age impact on Wage, meaning that with Dribbling skills CETERIS PARIBUS Age causes only a 5.66% of Wage variation effect (instead of 5.72%).**

# The dynamic proposed could keep going! Help yourself!
# Linear multivariate regression could answer the cause / effect ratio controlling the non analyzed variables.

# In[ ]:




