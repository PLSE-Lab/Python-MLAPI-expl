#!/usr/bin/env python
# coding: utf-8

# In[45]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# #  analysis of Bank Marketing

# # **Define Problem**
# 
# 
#      The data is related with direct marketing compaigns of a
#      Portuguese banking institution. The marketing compaigns are
#      based on phone calls.Often More one contact to the same clent  
#      were requred, in order to access if the product (bank term deposit) wou**ld be('yes') or not('no') subscribed.

# # **Specify Input And Output**
# 

# 
# **Categorical Variable :**
# 
# * Marital - (Married , Single , Divorced)",
# * Job - (Management,BlueCollar,Technician,entrepreneur,retired,admin)
# * Contact - (Telephone,Cellular,Unknown)
# * Education - (Primary,Secondary,Tertiary,Unknown)
# * Month - (Jan,Feb,Mar,Apr,May,Jun,Jul,Aug,Sep,Oct,Nov,Dec)
# * Poutcome - (Success,Failure,Other,Unknown)
# * Housing - (Yes/No)
# * Loan - (Yes/No)
# * [](http://)Default - (Yes/No)
# 

# **Numerical Variable:**
# 
# * Age
# * Balance
# * Day
# * Duration
# * Campaign
# * Pdays
# * Previous

# **class**
# 
# * deposit- (Yes/No)

# **Import Libraries**

# In[46]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[47]:


df = pd.read_csv('../input/bank.csv', delimiter= ";", header = 'infer')
df.head()


# # Exploratory Data Analysis:

# In[48]:


sns.heatmap(df.isnull(),yticklabels = False, cmap = 'viridis')


# **Finding Correlation between Features and class for Selection**
# 
# **1. Using Pairplot** 

# In[49]:


sns.pairplot(df)


# we can observe df is an not-Symmetric, so lets find out the correlation matrix to look into details.

# **2. Correlation matrix**

# In[50]:


df.corr()


# In[51]:


sns.heatmap(df.corr())


# **As Per the Pairplot, correlation matrix, and heatmap, observations as folllow:**
# 
# * Data is non-linearn asymetric
# * Hense selection of features will not depend upon correlaion factor.
# * also not a sinngle feature is correlated completely with class, hense requires combination of features.

# # Feature Selection techniques:
# 
# 1. Univariate Selection(non-negative fetures)
# 2. Recursive Feature Elimination(RFE)
# 3. Principle Component Analysis(PCA)(data reduction tech)
# 4. Feature importance(decision tree)
# 

# **Which feature selection technique should be used for our data?**
# 
# * Contains negative values, hense univariate Selection technique cannot be used.
# * PCA is data reduction technique. aim is to select best possible feature and not reduction and this is classified type of data.
# * PCA is an unsupervised method, used for dimensionality reduction.
# * Hense Decision tree technique and RFE can be used for Feature Selection.

# **Encoding Categorical and numerical data into digits form**

# In[52]:


df.dtypes


# # data preprocessing

# In[53]:


df_new = pd.get_dummies(df, columns=['job', 'marital', 'education','default',
                                    'housing','loan','contact','month','poutcome'] )


# In[54]:


df_new.y.replace(('yes','no'),(1,0),inplace = True)


# In[55]:


df_new.dtypes


# # Exploring the Feature

# In[56]:


df.education.unique()


# In[57]:


df.education.value_counts().plot(kind = 'barh')


# In[58]:


#Feature selection

y = pd.DataFrame(df_new['y'])
X = df_new.drop(['y'],axis=1)


# # Model Selection

# In[59]:


# data divide on training and testing sets

from sklearn.model_selection import train_test_split

X_train, X_test,y_train, y_test = train_test_split(X, y,test_size=0.3, random_state = 2)


# In[60]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# # Model Selection

# # 1)Logistic Regression

# In[61]:


from sklearn.linear_model import LogisticRegression


# In[62]:


logmodel = LogisticRegression()
logmodel.fit(X_train, y_train)


# In[63]:


y_pred = logmodel.predict(X_test)


# In[64]:


from sklearn.metrics import accuracy_score

acc_logmodel = round(accuracy_score(y_pred, y_test) * 100)
print(acc_logmodel)


# In[65]:


from sklearn.metrics import confusion_matrix, classification_report

print(confusion_matrix(y_test, y_pred))

print('\n')

print(classification_report(y_test, y_pred))


# In[ ]:




