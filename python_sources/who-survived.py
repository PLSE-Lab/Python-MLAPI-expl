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
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


train = pd.read_csv('/kaggle/input/titanic/train.csv')
test = pd.read_csv('/kaggle/input/titanic/test.csv')


# In[ ]:


test.head()


# In[ ]:


train.head()


# In[ ]:


print(train.shape)
print('\n')
print(test.shape)


# In[ ]:


train.columns


# In[ ]:


train.info()


# #### Data Description and Comments
# 
# 1) The <b> Survived </b> variable is our outcome or dependent variable. It is a binary nominal datatype of 1 for survived and 0 for did not survive. All other variables are potential predictor or independent variables. 
# 
# 2) <b> The PassengerID and Ticket </b> variables are assumed to be random unique identifiers, that have no impact on the outcome variable. Thus, they will be excluded from analysis.
# 
# 3) The <b> Pclass </b> variable is an ordinal datatype for the ticket class, a proxy for socio-economic status (SES), representing 1 = upper class, 2 = middle class, and 3 = lower class.
# 
# 4) <b> The Name </b> variable is a nominal datatype. It could be used in feature engineering to derive the gender from title, family size from surname, and SES from titles like doctor or master. Since these variables already exist, we can explore further if title, like master, makes a difference.
# 
# 
# 5) The <b> Sex and Embarked </b> variables are a nominal datatype. They will be converted to dummy variables for mathematical calculations.
# 
# 6) <b> The Age and Fare </b> variable are continuous quantitative datatypes.
# 
# 7) <b>The SibSp </b> represents number of related siblings/spouse aboard and Parch represents number of related parents/children aboard. Both are discrete quantitative datatypes. This can be used for feature engineering to create a family size and is alone variable.
# 
# 
# 8) The <b> Cabin </b> variable is a nominal datatype that can be used in feature engineering for approximate position on ship when the incident occurred and SES from deck levels. However, since there are many null values, it does not add value and thus is excluded from analysis.

# In[ ]:


sns.heatmap(train.isnull(),yticklabels=False,cmap='viridis',cbar=False)


# In[ ]:


sns.heatmap(test.isnull(),yticklabels=False,cmap='viridis',cbar=False)


# Roughly 20 percent of the Age data is missing. The proportion of Age missing is likely small enough for reasonable replacement with some form of imputation. Looking at the Cabin column, it looks like we are just missing too much of that data to do something useful with at a basic level. We'll probably drop this later, or change it to another feature like "Cabin Known: 1 or 0"
# 
# Let's continue on by visualizing some more of the data! 
# # EDA

# In[ ]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',data=train,palette='RdBu_r',)


# In[ ]:


sns.set_style('whitegrid')
sns.catplot(x='Sex',y='Survived',hue='Pclass',kind='bar',data=train)


# In[ ]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Pclass',data=train,palette='rainbow')

