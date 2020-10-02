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


# ## Preliminary note ##
# 
# For some functions and reasoning of this notebook, I had consulted the following notebooks, already available at the Kernel:
# 
# https://www.kaggle.com/brandao/setting-a-limit-for-false-negatives-covid19
# 
# https://www.kaggle.com/ossamum/exploratory-data-analysis-and-feature-importance
# 
# https://www.kaggle.com/nazeboan/null-values-exploration-logreg-67-acc
# 
# https://www.kaggle.com/endelgs/98-accuracy-at-covid-19
# 
# https://www.kaggle.com/rodrigofragoso/exploring-nans-and-the-impact-of-unbalanced-data
# 
# https://www.kaggle.com/eduardosmorgan/quick-data-exploration-and-svm-cv
# 
# https://www.kaggle.com/andrewmvd/lightgbm-baseline
# 
# https://www.kaggle.com/dmvieira/overfitting-ward-semi-intensive-or-intensive-unit
# 
# https://www.kaggle.com/dmvieira/94-precision-on-covid-19
# 
# https://www.kaggle.com/rspadim/eda-first-try-python-lgb-shap

# In[ ]:


df = pd.read_excel('/kaggle/input/covid19/dataset.xlsx')


# In[ ]:


df.head(5)


# ## Cleaning Data for Task 1 ##

# In[ ]:


df['SARS-Cov-2 exam result'] = df['SARS-Cov-2 exam result'].map(lambda r: 1 if r == 'positive' else 0 )


# In[ ]:


df = df.rename(columns={'SARS-Cov-2 exam result': 'has_covid', 'Patient addmited to regular ward (1=yes, 0=no)': 'hospital',
                   'Patient addmited to semi-intensive unit (1=yes, 0=no)': 'semi_icu',
                   'Patient addmited to intensive care unit (1=yes, 0=no)': 'icu'})


# In[ ]:


corr = df.corr()
corr[['has_covid', 'hospital', 'semi_icu', 'icu']].sort_values('has_covid', ascending=False)


# ## Missing Data ##
# The main objective of this session is to establish patterns by NaN values, 'blocking' certain features together. We'll not use the age data yet, because there aren't people with NaN values in this feature. In another notebook, we'll try to group the data by age.
# This idea came from the graph in line [5] of https://www.kaggle.com/ossamum/exploratory-data-analysis-and-feature-importance, which shows the fraction of null-values in different features and we reproduce below.

# In[ ]:


features = df.iloc[:, 6:]
#For task 1, we won't use the hospital/semi-icu/icu data


# In[ ]:


import missingno as msno
# checking null values
msno.bar(features, figsize=(16, 4),log = True)


# In[ ]:


features.head(5)


# In[ ]:


#Dropping all NanColumns
all_nas = []
for f in features.columns:
    if features[f].isna().all():
        all_nas += [f]
features = features.drop(columns=all_nas) #Dropping all NanColumns


# In[ ]:


features.head(5)
#It's important to show the head of the dataframe because we know now that 5 features are completely needless


# In[ ]:


msno.bar(features, figsize=(16, 4), log = True)


# In[ ]:


msno.dendrogram(features)


# In[ ]:


nan_series = pd.Series(index = list(features.columns), data = np.zeros(len(features.columns)))
#Creating an empty nan_series. We'll add the nan_values below
nan_series.head(5)


# In[ ]:


for label in features.columns:
    for i in features[label].isnull().index:
        if features[label].isnull()[i] == True:
            nan_series[label]+=1
#This cell has a very high computational cost, by the way
nan_series.head(5)


# In[ ]:


nan_series.nunique()
#The value below shows that there are 40 categories of NaN values (41, if we count the all-NaN values)


# In[ ]:


np.unique(nan_series.values)
#These are the values of NaN for each group


# In[ ]:


#In this cell, we'll show the features in each category
unique_nan_series = np.unique(nan_series.values) 
for i in range(0, len(unique_nan_series)):
    print("Category ", i)
    print("Number of NaN values: ", unique_nan_series[i] )
    print("Features in category ", i, " :")
    for element in list(nan_series[nan_series == unique_nan_series[i]].index):
        print(element)
    print('\n')


# ## Preliminary Conclusion/Posterior Work ##
# 
# Except for the age feature, the features with minimum number of NaNs (Respiratory Syncytial Virus, Influenza A and Influenza B) have 4290 NaN (out of 5644), i.e.: 76% of NaN, making imputting techniques impracticable. One possible approach would be count the number of positives for each category and, if any category doesn't contribute at all (or little, according to a specified threshold) in prescribe whether or not a patient is covid-positive, we should drop this category.
# 
# A cause-effect relationship between groups, explained by a health professional, maybe would be useful. For example, if a patient who made a test in "Category 39" *necessarily* had make tests in Categories 7, 14, 20, etc... If this is true, we can drop some of the categories, since the objective of Task 1 is to determine probabilities of Sars-Cov-2 with *preliminary* tests.
# 
# Even without this specialized help, we can plot correlation *inside* the categories, to see if we can reduce the dimensionality of the problem.

# In[ ]:




