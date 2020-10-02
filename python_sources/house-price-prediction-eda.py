#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # **Life Cycle of Data Science Project******
# * Data Analysis
# * Feature Engineering
# * Feature Selection
# * Model Creation
# * Model Deployment

# # Data Analysis****
# * Missing Values
# * Numerical values
# * Distribution of numerical values
# * Categorical values
# * Cardinality of categorical values
# * Outliers
# * Relationship between variables

# In[ ]:


df=pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")


# In[ ]:


df.head()


# In[ ]:


len(df.columns)


# In[ ]:


#Features with NaN values
featurena=[fea for fea in df.columns if df[fea].isnull().sum()>1]


# In[ ]:


len(featurena)


# In[ ]:


for i in featurena:
    print(i,np.round(df[i].isnull().mean(),4),"% of missing values")


# In[ ]:


sns.heatmap(df.isnull(),xticklabels='auto',yticklabels=False)


# The above graph shows the distribution of null values of each variables

# We are goint to replace the null values with 1 and non null with 0.

# In[ ]:


data=df.copy()
for feature in featurena:
    data[feature]=np.where(data[feature].isnull(),1,0)
    data.groupby(feature)['SalePrice'].median().plot.bar()
    plt.title(feature)
    plt.show()


# **From the above figure we can say that the null values (1) have greater impact in some features.So we have to replace the null values with some useful treatment**

# In[ ]:


numfeature=[fea for fea in df.columns if df[fea].dtype!='O']
df[numfeature].head()


# ***Here we get the numerical features and there some of the year variables also included,We have to detect that*

# In[ ]:


yrfeature=[fea for fea in numfeature if 'Yr' in fea or 'Year' in fea]
print(yrfeature)


# In[ ]:


df.groupby('YrSold')['SalePrice'].median().plot()
plt.xlabel('YrSold')
plt.ylabel('Sales Price')


# In[ ]:


for fea in yrfeature:
    plt.scatter(df[fea],df['SalePrice'])
    plt.xlabel(fea)
    plt.ylabel('Sales Price')
    plt.show()


# **From the above figures we can say that the houses of recent years have higher prices**

# We are going to take only the discrete features

# In[ ]:


discretefea=[fea for fea in numfeature if len(df[fea].unique())<25 and fea not in yrfeature and ['Id']]
discretefea


# In[ ]:


print(len(discretefea))


# In[ ]:


for fea in discretefea:
    df.groupby(fea)['SalePrice'].median().plot.bar()
    plt.xlabel(fea)
    plt.ylabel('Sales Price')
    plt.show()


# From the above figure we can get the relationship of sales price with various variables.Here the overall quality gives the exponential growth with sales price

# # Continuous variables****

# In[ ]:


confea=[fea for fea in numfeature if fea not in discretefea+yrfeature+['Id']]
confea


# In[ ]:


print(len(confea))


# In[ ]:


for fea in confea:
    df[fea].hist(bins=25)
    plt.xlabel(fea)
    plt.ylabel('Count')
    plt.show()


# **From the above diagrams we can see that many of the features are not in gaussian form and some are in bell curve and some in skewed form.For regression problems we have to conver the features in gaussian form**

# In[ ]:


#dat=df.copy()
for fea in confea:
    dat=df.copy()
    if 0 in df[fea].unique():
        pass
    else:
        dat[fea]=np.log(dat[fea])
        dat['SalePrice']=np.log(dat['SalePrice'])
        plt.scatter(dat[fea],dat['SalePrice'])
        plt.xlabel(fea)
        plt.ylabel('Sales Price')
        plt.show()


# # Outliers****

# In[ ]:


for fea in confea:
    dat=df.copy()
    if 0 in dat[fea].unique():
        pass
    else:
        dat[fea]=np.log(dat[fea])
        dat['SalePrice']=np.log(dat['SalePrice'])
        dat.boxplot(column=fea)
        plt.ylabel(fea)
        plt.show()


# # Categorical Feature****

# In[ ]:


catfeature=[fea for fea in df.columns if df[fea].dtype=='O']
print(len(catfeature))


# In[ ]:


for fea in catfeature:
    print('The no of categories in {} feature is {} '.format(fea,len(df[fea].unique())))


# In[ ]:


for fea in catfeature:
    dat=df.copy()
    df.groupby(fea)['SalePrice'].median().plot.bar()
    plt.xlabel(fea)
    plt.ylabel('Sales Price')
    plt.show()


# In[ ]:




