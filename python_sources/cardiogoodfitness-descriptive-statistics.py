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


# Load the Cardio Dataset

mydata = pd.read_csv('/kaggle/input/cardiogoodfitness/CardioGoodFitness.csv')


# In[ ]:


#Display the first five rows of the data
mydata.head()


# In[ ]:


#Five point summary of the data
mydata.describe().T


# In[ ]:


#Info about the data
mydata.info()


# In[ ]:


#Shape of the data
mydata.shape


# In[ ]:


#Checking for the null values 

mydata.isna().any()


# **No Missing values in the dataset**

# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

mydata.hist(figsize=(20,30))


# In[ ]:


import seaborn as sns

sns.boxplot(x="Gender", y="Age", data=mydata)


# In[ ]:


pd.crosstab(mydata['Product'],mydata['Gender'] )


# In[ ]:


pd.crosstab(mydata['Product'],mydata['MaritalStatus'] )


# In[ ]:


sns.countplot(x="Product", hue="Gender", data=mydata)


# In[ ]:


pd.pivot_table(mydata, index=['Product', 'Gender'],
                     columns=[ 'MaritalStatus'], aggfunc=len)


# In[ ]:


pd.pivot_table(mydata,'Income', index=['Product', 'Gender'],
                     columns=[ 'MaritalStatus'])


# In[ ]:


pd.pivot_table(mydata,'Miles', index=['Product', 'Gender'],
                     columns=[ 'MaritalStatus'])


# In[ ]:


sns.pairplot(mydata)


# In[ ]:


mydata['Age'].std()


# In[ ]:


mydata['Age'].mean()


# In[ ]:


sns.distplot(mydata['Age'])


# In[ ]:


mydata.hist(by='Gender',column = 'Age')


# In[ ]:


mydata.hist(by='Gender',column = 'Income')


# In[ ]:


mydata.hist(by='Gender',column = 'Miles')


# In[ ]:


mydata.hist(by='Product',column = 'Miles', figsize=(20,30))


# In[ ]:


corr = mydata.corr()
corr


# In[ ]:


sns.heatmap(corr, annot=True)


# In[ ]:


# Simple Linear Regression


#Load function from sklearn
from sklearn import linear_model

# Create linear regression object
regr = linear_model.LinearRegression()

y = mydata['Miles']
x = mydata[['Usage','Fitness']]

# Train the model using the training sets
regr.fit(x,y)


# In[ ]:


regr.coef_


# In[ ]:


regr.intercept_


# In[ ]:


# MilesPredicted = -56.74 + 20.21*Usage + 27.20*Fitness

