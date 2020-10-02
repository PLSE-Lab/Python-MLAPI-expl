#!/usr/bin/env python
# coding: utf-8

# # Objective : 
# ## Analyze the potential pitfalls of using the popular Mean / Median NULL value replacement technique, i.e. Mean / Median Imputation

# This notebook has code which replaces NULLS in the "Age" column of the Titanic dataset with Mean / Median of the variable and explores the potential pitfalls of this approach in this specific situation. 

# ### Benefits of using Mean / Median imputation during data cleanup
# * Easy to implement
# * One of the fastest ways to rectify missing values

# ### Potential downsides of using Mean / Median imputation during data cleanup, which could negatively impact algorithms such as Linear Regression
# 
# * Changes variable distribution
# * Distorts the original relationship of a variable with other variables. i.e. Covariance
# * Creates unexpected outliers in the dataset
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


# load the Titanic Dataset , which could use some cleanup

data = pd.read_csv('/kaggle/input/titanic-dataset-from-kaggle/train.csv')
data.head(3)


# In[ ]:


# Look for attributes which has lot of nulls-cabin and age 
# are two potential candidate columns for our test

data.isnull().mean()


# ## Replace NULLS in the "Age" column with the Mean / Median of the variable

# In[ ]:


# Add column with median age

data['new_age_median'] = data['Age'].fillna(data.Age.median())

# Add column with mean age

data['new_age_mean'] = data['Age'].fillna(data.Age.mean())

# Display newly added columns, which shows two different approaches to 
# filling null values in the age column

data.loc[:, ['Name', 'Age','new_age_median', 'new_age_mean']].loc[data['Age'].isnull()].head(4)


# ## Analyze the distribution of variables after replacing NULLS in the "Age" column with Mean / Median of the variable

# In[ ]:


# we can see that the distribution has changed, as seen in the density values centered 
# around the middle of the graph below

fig_new = plt.figure()
x = fig_new.add_subplot(111)

# original distribution of variables (before replacing NULL values)

data['Age'].plot(kind='kde', x=x, color='red')

# replaced with mean
data['new_age_mean'].plot(kind='kde', x=x, color='purple')

# replaced with median
data['new_age_median'].plot(kind='kde', x=x, color='grey')

# Details
lines, labels = x.get_legend_handles_labels()
x.legend(lines, labels, loc='best')


# As seen above, the change in distribution of values in the 20-40 age range after replacing the NULLS with Mean / Median range, can impact some ML algorithms

# ## Analyze the covariance between age and other variables such as passenger fare, after the Mean / Median Imputation

# In[ ]:


# we also see that mean / median imputation may affect the relationship 
# with the other variables in the dataset; in otherwords, the Covariance is impacted

data[['Fare','Survived', 'Age', 'new_age_median', 'new_age_mean']].cov()


# As suspected, the relationship with other variables are also affected. This can cause distortions. E.g Covariance between original Age and Fare columns differs significantly from the Covariance between new_age_mean and Fare columns

# ## Analyze change in outliers

# In[ ]:


data[['Age', 'new_age_median', 'new_age_mean']].boxplot()


# As seen in the boxplot above, outliers have been significantly altered  between the original age column and the new columns (new_age_median & new_age_mean) where the NULLS were replaced by Mean / Median.

# # Conclusion

# While using mean/median imputation for data cleanup has many usefull applications, caution must be taken to ensure that the performance of ML algorithms are not negatively impacted; One can do this by first checking for the following aspects before replacing NULLS in a dataset using mean / median values.
# 
# * Change in distribution of variables
# * Change in variance
# * Change in number of outliers in the dataset
# 
