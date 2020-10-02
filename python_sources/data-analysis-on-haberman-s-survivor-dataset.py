#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv('../input/habermans-survival-data-set/haberman.csv',names=['Age','OperationYear','No.of.AxillaryNodes','Class'],header=None)


# In[ ]:


df


# In[ ]:


df.columns


# It has **four features**

# # *Checking for Null values*

# In[ ]:


df.isna().sum()


# There are **No null values**, so no need of data imputation

# In[ ]:


df['Class'].nunique()


# In[ ]:


df['Class'].value_counts()


# Class has two values as 1 and 2. 1 represents the people who survived 5 years or more after the operation. 2 represents the people who didn't survive five years or people who died within five years after the operation

# In[ ]:


df['Class'] = df['Class'].map({1:'yes',2:'no'})


# In[ ]:


df.head()


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt


# # Univariate Analysis

# In[ ]:


sns.FacetGrid(df,hue='Class',height=5).map(sns.distplot,'Age').add_legend()


# In[ ]:


sns.FacetGrid(df,hue='Class',height=5).map(sns.distplot,'OperationYear').add_legend()


# In[ ]:


sns.FacetGrid(df,hue='Class',height=5).map(sns.distplot,'No.of.AxillaryNodes').add_legend()


# On the above univariate analysis of three columns **Age** and **OperationYear** doesn't show the significant relationship.
# 
# But the distplot with No.of.AxillaryNodes gives much information than other two.
# It shows people with axillarynodecount < 5 are more likely to survive. As the count increases, they are less likely to survive

# So we take No.of.AxillaryNodes and do pdf,cdf

# In[ ]:


count,bin_edges = np.histogram(df[df['Class']=='yes']['No.of.AxillaryNodes'],bins=10)
pdf=count/(sum(count))
cdf=np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:],cdf)
plt.grid()


# The above plot shows nearly 82% percent of survived people had **No.of.AxillaryNode** count <= 5 and gradually it decreases

# In[ ]:


count,bin_edges = np.histogram(df[df['Class']=='no']['No.of.AxillaryNodes'],bins=10)
pdf=count/(sum(count))
cdf=np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:],cdf)
plt.grid()


# The above plot shows that 57% people who had not survived also had **No.of.AxillaryNodes** count <=5 but it is low compared to survived one. And you can also see there is slight increase in probability between range 20-30, where there is no increased probability in survived plot

# In[ ]:


sns.boxplot(x='Class',y='Age',data=df)


# In[ ]:


sns.violinplot(x='Class',y='Age',data=df)


# The above plot shows that people aged 35-60(approx) are likely to survive. There is equal rate of survived and non-survived at the age range of 50-60

# In[ ]:


sns.boxplot(x='Class',y='OperationYear',data=df)


# In[ ]:


sns.violinplot(x='Class',y='OperationYear',data=df)


# The above plots shows that operation which had done from 1962-1965 tend to be failures

# In[ ]:


sns.boxplot(x='Class',y='No.of.AxillaryNodes',data=df)


# Here in survived, it shows 75% of people had node count < 5(approx). And you can see lots of outliers in the survived data

# In[ ]:


sns.violinplot(x='Class',y='No.of.AxillaryNodes',data=df)


# In the above plot, we come to know that more people in survived had nodecount zero(most) until 5. In non-survived people had nodecount from 3 to 20

# There isn't a proper conclusion that we can come with he use of any one variable

# # Bi-Variate Analysis

# In[ ]:


sns.FacetGrid(data=df,hue='Class',height=7).map(plt.scatter,'Age','No.of.AxillaryNodes')
plt.grid()


# Here people with zero node count are more likely to survive eventhough there are non-survived people having zero count. People of age range from 50-60 havingzero count are more likely to survive than of people who belong to age range of 40-50 of same case.
# 
# People who have not survived and had counts<=10 are of age range 40-60. And people who had counts>10 are less likely to survive

# In[ ]:


sns.pairplot(data=df,hue='Class')


# Considering the upper matrices,Age and No.of.AxillaryNodes have better information than other two

# # Multi-Variate Analysis

# In[ ]:


sns.jointplot(x='Age',y='No.of.AxillaryNodes',data=df[df['Class']=='yes'],kind='kde')


# Majority of people who had survived is from 45-65 and having nodes 0-2

# In[ ]:


sns.jointplot(x='Age',y='No.of.AxillaryNodes',data=df[df['Class']=='no'],kind='kde')


# This shows people who of age range 40-60 and having node counts from 0-5(most) aren't survived

# In[ ]:


sns.jointplot(x='Age',y='OperationYear',data=df[df['Class']=='yes'],kind='kde')


# In[ ]:


sns.jointplot(x='Age',y='OperationYear',data=df[df['Class']=='no'],kind='kde')


# # Conclusions

# Since more datapoints are overlapping, high accuracy of classification isn't possible. Yet we can consider **No.of.AxillaryNodes** for univariate and **age,No.of.AxillaryNodes** for bivariate and multi variate analysis to give better insights and better classification than others. The accuracy can be high if we had more features.

# In[ ]:




