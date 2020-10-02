#!/usr/bin/env python
# coding: utf-8

# <a class="anchor" id="0"></a>
# # **TReNDS : EDA + Visualization + Models** 
# 
# 
# 
# ## **Introduction**
# 
# 
# [TReNDS Neuroimaging](https://www.kaggle.com/c/trends-assessment-prediction) competition has been recently launched at Kaggle. We can find the evaluation page [here](https://www.kaggle.com/c/trends-assessment-prediction/overview/evaluation)
# 
# Human brain research is among the most complex areas of study for scientists. We know that age and other factors can affect its function and structure, but more research is needed into what specifically occurs within the brain. With much of the research using MRI scans, data scientists are well positioned to support future insights. In particular, neuroimaging specialists look for measurable markers of behavior, health, or disorder to help identify relevant brain regions and their contribution to typical or symptomatic effects.
# 
# In this competition, we will predict multiple assessments plus age from multimodal brain MRI features. 
# 
# 
# 

# ![TReNDS](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F1537731%2Fa5fdbe17ca91e6713d2880887232c81a%2FScreen%20Shot%202019-12-09%20at%2011.25.31%20AM.png?generation=1575920121028151&alt=media)

# 
# **I hope you find this notebook useful and your <font color="red"><b>UPVOTES</b></font> keep me motivated**
# 
# 

# <a class="anchor" id="0.1"></a>
# # **Table of Contents** 
# 
# 1.	[Import libraries](#1)
# 2.	[Read datasets](#2)
# 3.  [Data Exploration](#3)
#    - 3.1 [Shape of the data](#3.1)
#    - 3.2 [Preview the data](#3.2)
#    - 3.3 [Check for missing values](#3.3)
# 4.  [Data Visualization](#4)
#    - 4.1 [Data Visualization of Features](#4.1)
#          - 4.1.1 [age](#4.1.1)
#          - 4.1.2 [domain1_var1](#4.1.2)
#          - 4.1.3 [domain1_var2](#4.1.3)
#          - 4.1.4 [domain2_var1](#4.1.4)
#          - 4.1.5 [domain2_var2](#4.1.5)
#    - 4.2 [Correlation Heatmap](#4.2)
#    - 4.3 [Pair Plot](#4.3)
# 5. [Modelling](#5)
# 6. [Submission](#6)     
# 
#    
# 

# # **1. Import libraries** <a class="anchor" id="1"></a>
# 
# [Table of Contents](#0.1)
# 

# In[ ]:


# ignore warnings 
import warnings
warnings.filterwarnings('ignore')

# import general packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns 
get_ipython().run_line_magic('matplotlib', 'inline')
color = sns.color_palette()


# algorithms
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor


# modeling helper functions
from sklearn.model_selection import GridSearchCV , KFold , cross_val_score


# to read / write access to some common neuroimaging file formats
## for more information, please visit : https://pypi.org/project/nibabel/
import nibabel


# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

# print all files in the data folder
base_url = '/kaggle/input/trends-assessment-prediction'
print(os.listdir(base_url))


# # **2. Read datasets** <a class="anchor" id="2"></a>
# 
# [Table of Contents](#0.1)
# 

# In[ ]:


get_ipython().run_line_magic('time', '')
loading_df = pd.read_csv(base_url +'/loading.csv')
sample_submission = pd.read_csv(base_url +'/sample_submission.csv')
train_df = pd.read_csv(base_url +'/train_scores.csv')


# # **3. Data Exploration** <a class="anchor" id="3"></a>
# 
# [Table of Contents](#0.1)

# ## **3.1 Shape of the data**  <a class="anchor" id="3.1"></a>
# 
# [Table of Contents](#0.1)
# 
# We will begin our data exploration by checking the shape of the data.

# In[ ]:


print(f'Size of loading_df : {loading_df.shape}')
print(f'Size of sample_submission : {sample_submission.shape}')
print(f'Size of train_df : {train_df.shape}')
print(f'Size of test set : {len(sample_submission)/5}')


# ## **3.2 Preview the data**  <a class="anchor" id="3.2"></a>
# 
# [Table of Contents](#0.1)
# 
# 
# Now, let's preview the data to gain more insights about the data.

# In[ ]:


def preview(df):
    print(df.head())


# In[ ]:


preview(loading_df)


# In[ ]:


preview(sample_submission)


# In[ ]:


preview(train_df)


# ## **3.3 Check for missing values**  <a class="anchor" id="3.3"></a>
# 
# [Table of Contents](#0.1)
# 
# 
# Now, let's check for missing values in training data.

# In[ ]:


missing_train_df = train_df.isnull().mean() * 100
missing_train_df.sort_values(ascending=False)


# In[ ]:


loading_df.isnull().sum().sum()


# In[ ]:


sample_submission.isnull().sum().sum()


# So, `loading_df` and `sample_submission` do not have any missing values. 

# # **4. Data Visualization** <a class="anchor" id="4"></a>
# 
# [Table of Contents](#0.1)

# ## **4.1 Data Visualization of features** <a class="anchor" id="4.1"></a>
# 
# [Table of Contents](#0.1)

# In[ ]:


target_labels = list(train_df.columns[1:])
target_labels


# We will check the distribution of target variables in training set. We will exclude `Id` from the training set.

# ### **4.1.1 age** <a class="anchor" id="4.1.1"></a>
# 
# [Table of Contents](#0.1)

# In[ ]:


x = train_df['age']
plt.figure(figsize=(8,6))
plt.hist(x, bins=25, color='g')
plt.xlabel('Age')
plt.ylabel('Number of patients')
plt.title('Age distribution of patients', fontsize = 16)
plt.show()


# Age does not contain any missing values.

# ### **4.1.2 domain1_var1** <a class="anchor" id="4.1.2"></a>
# 
# [Table of Contents](#0.1)

# In[ ]:


x = train_df['domain1_var1']
plt.figure(figsize=(8,6))
plt.hist(x, bins=25, color='c')
plt.xlabel('domain1_var1')
plt.ylabel('Number of patients')
plt.title('domain1_var1 distribution', fontsize = 16)
plt.show()


# We can see that `domain1_var1` distribution is approximately normal. So, we will fill the missing values with mean.

# In[ ]:


train_df['domain1_var1'].fillna(train_df['domain1_var1'].mean(), inplace=True)


# ### **4.1.3 domain1_var2** <a class="anchor" id="4.1.3"></a>
# 
# [Table of Contents](#0.1)

# In[ ]:


x = train_df['domain1_var2']
plt.figure(figsize=(8,6))
plt.hist(x, bins=25, color='pink')
plt.xlabel('domain1_var2')
plt.ylabel('Number of patients')
plt.title('domain1_var2 distribution', fontsize = 16)
plt.show()


# `domain1_var2` is skewed. So, we will fill missing values with median.

# In[ ]:


train_df['domain1_var2'].fillna(train_df['domain1_var2'].median(), inplace=True)


# ### **4.1.4 domain2_var1** <a class="anchor" id="4.1.4"></a>
# 
# [Table of Contents](#0.1)

# In[ ]:


x = train_df['domain2_var1']
plt.figure(figsize=(8,6))
plt.hist(x, bins=25, color='y')
plt.xlabel('domain2_var1')
plt.ylabel('Number of patients')
plt.title('domain2_var1 distribution', fontsize = 16)
plt.show()


# `domain2_var1` is approximately normal. So, we will fill missing values with mean.

# In[ ]:


train_df['domain2_var1'].fillna(train_df['domain2_var1'].mean(), inplace=True)


# ### **4.1.5 domain2_var2** <a class="anchor" id="4.1.5"></a>
# 
# [Table of Contents](#0.1)

# In[ ]:


x = train_df['domain2_var2']
plt.figure(figsize=(8,6))
plt.hist(x, bins=25, color='r')
plt.xlabel('domain2_var2')
plt.ylabel('Number of patients')
plt.title('domain2_var2 distribution', fontsize = 16)
plt.show()


# `domain2_var2` is approximately normal. So, we will fill missing values with mean.

# In[ ]:


train_df['domain2_var2'].fillna(train_df['domain2_var2'].mean(), inplace=True)


# In[ ]:


train_df.isnull().sum()


# There are no missing values in training set.

# ## **4.2 Correlation Heatmap**  <a class="anchor" id="4.2"></a> 
# 
# [Table of Contents](#0.1)
# 
# 
# Correlation Heatmap is used to visualize feature interactions in training set.

# In[ ]:


cols = train_df.columns[1:]
correlation = train_df[cols].corr()


# In[ ]:


mask = np.zeros_like(correlation)
mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"):
    f, ax = plt.subplots(figsize=(12,8))
    ax = sns.heatmap(correlation, mask=mask, vmax=.3, square=True, annot=True, cmap='YlGnBu', fmt='.2f', linecolor='white')
    ax.set_title('Correlation Heatmap of training dataset')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=30)           
    plt.show()


# ## **4.3 Pair Plot**  <a class="anchor" id="4.3"></a> 
# 
# [Table of Contents](#0.1)
# 
# 
# We will draw pairplot to visualize relationship between the target variables. 

# In[ ]:


sns.pairplot(train_df[cols], kind='scatter', diag_kind='hist', palette='Rainbow')
plt.show()


# ### **Observations** 
# 
# 
# From the above heatmap and pairplot, we can conclude that-
# 
# - `age` is positively correlated with `domain1_var1` and `domain2_var1`.
# 
# - `domain2_var1` and `domain2_var2` are positively correlated.

# # **5. Modelling** <a class="anchor" id="5"></a>
# 
# [Table of Contents](#0.1)
# 

# In[ ]:


train_ids = sorted(loading_df[loading_df['Id'].isin(train_df.Id)]['Id'].values)
test_ids = sorted(loading_df[~loading_df['Id'].isin(train_df.Id)]['Id'].values)
predictions = pd.DataFrame(test_ids, columns=['Id'], dtype=str)
features = ('age', 'domain1_var1', 'domain1_var2','domain2_var1','domain2_var2')


# In[ ]:


data = pd.merge(loading_df, train_df, on='Id')
X_train = data.drop(list(features), axis=1).drop('Id', axis=1)
y_train = data[list(features)]
X_test = loading_df[loading_df.Id.isin(test_ids)].drop('Id', axis=1)


# In[ ]:


names = ["Linear Regression", "Decision Tree", "Random Forest", "Neural Net" ]    


# In[ ]:


regressors = [
    LinearRegression(),
    DecisionTreeRegressor(max_depth=5),
    RandomForestRegressor(max_depth=5, n_estimators=10, max_features=1),
    MLPRegressor(alpha=1, max_iter=1000),
    ]


# In[ ]:


# iterate over classifiers and calculate cross-validation score
for name, reg in zip(names, regressors):
    scores = cross_val_score(reg, X_train, y_train, cv = 5, scoring='neg_mean_absolute_error')
    print(name , ':{:.4f}'.format(scores.mean()))


# We can see that Random Forest results in lowest negative MAE. So, we will use it for submission.

# # **6. Submission** <a class="anchor" id="6"></a>
# 
# [Table of Contents](#0.1)
# 

# ## To be continued. Please visit this space again.

# [Go to Top](#0)
