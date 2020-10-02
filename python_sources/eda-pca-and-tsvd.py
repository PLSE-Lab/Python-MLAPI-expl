#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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


# In[ ]:


import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from collections import Counter


# In[ ]:


test = pd.read_csv("../input/test.csv")
train = pd.read_csv("../input/train.csv")


# In[ ]:


#remove the ID field from the train data
train = train.drop('ID', axis = 1)
test = test.drop('ID', axis=1)


# In[ ]:


#check columns of train data
train.columns


# Initial test and train data exploration

# In[ ]:


train.info()


# In[ ]:


#We have no categorical variables.
train.select_dtypes(include=['object']).dtypes


# In[ ]:


test.shape


# In[ ]:


train.shape


# In[ ]:


train.head(10)


# In[ ]:


train.isnull().sum().any()


# In[ ]:


test.isnull().sum().any()


# **Observations**
# 1. Same number of columns between the test and train data (train has 1 extra "target field" column)
# 2. Most importantly, the number of rows in the train data are **MUCH** lesser than the number of columns! We have around 3000 columns and around 4500 rows. Test data is better.. 
# 3. Columns names do not make sense, so we will have to perform feature extraction for this data to make sense.
# 4. **No missing data** in the train and test sets! 

# **UNIVARIATE ANALYSIS OF THE TARGET FIELD**
#    Let's pick the target field and try to analyse it.
# 

# In[ ]:


train['target'].describe()


# In[ ]:


#plot a distribution plot to see the distribution of the target field
plt.figure(figsize=(8,5))
sns.distplot(train['target'])


# This seems to be a highly skewed target variable. Let's take the log of it to check the distribution.

# In[ ]:


plt.figure(figsize=(8,5))
sns.distplot(np.log1p(train['target']), kde='False')


# Better distributed now! Let's check the train.info after taking the log

# In[ ]:


np.log1p(train['target']).describe()


# This is a LOT better! Helps us better understand the distribution of the 'target' column. Lets plot some scatter plots with a few other variabls to see its spread.
# Let's pick the first column '48df886f9'

# In[ ]:


plt.figure(figsize=(5,5))
plt.scatter(np.log(train['48df886f9']),np.log(train['target']))
plt.xlabel('48df886f9')
plt.ylabel('Target')


# Perform some basic checks on the target column.

# In[ ]:


train['target'].sort_values(ascending=False)


# *  We can see that "target" variable ranges from values of 10^5 to 10^9. 
# *  Next Check the count of the most common target value

# In[ ]:


Counter(train['target']).most_common()


# In[ ]:


#Plot a boxplot
plt.figure(figsize=(8,8))
sns.boxplot(train['target'], orient='v')


# In[ ]:


#separate the x and y variables for the train and test data
#taking the log of the target variable as it is not well distributed.
x_train = train.iloc[:,train.columns!='target']
y_train = np.log1p(train.iloc[:,train.columns=='target'])
x_test = test


# In[ ]:


#copy the x_train, y_train, and x_test datasets
x_train_copy= x_train.copy()
x_test_copy= x_test.copy()
y_train_copy= y_train.copy()


# In[ ]:


x_train.columns


# In[ ]:


print(y_train.head(10))


# In[ ]:


print(x_train.shape)


# In[ ]:


print(y_train.shape)


# In[ ]:


x_test.shape


# In[ ]:


train.columns


# **Remove the columns with standard deviation = 0 from test and train set.**
# - Standard Deviation = 0 means that **every data point in a column is equal to its mean**. Also means that all of a column's values are **identical**.
# - Such columns really do not help us in prediction. So we will drop them

# In[ ]:


drop_cols=[]
for cols in x_train.columns:
    if x_train[cols].std()==0:
        drop_cols.append(cols)
print("Number of constant columns to be dropped: ", len(drop_cols))
print(drop_cols)
x_train.drop(drop_cols,axis=1, inplace = True)


# Check for constant columns on the test data

# In[ ]:


drop_cols_test=[]
for cols in x_test.columns:
    if x_test[cols].std()==0:
        drop_cols_test.append(cols)
print("Number of constant columns to be dropped: ", len(drop_cols_test))
print(drop_cols_test)


# There are no constant columns from the test data. However, we still need to drop them as the shapes of the test and the train data need to be the same for modelling.

# In[ ]:


x_test.drop(drop_cols,axis=1, inplace = True)


# In[ ]:


x_train.shape


# In[ ]:


x_test.shape


# **DIMENSIONALITY REDUCTION**
# - One of the major problems with this dataset is that it has too many predictors (almost 4900+). To go through each of these predictors and see which ones are significant for the model is going to be a tedious task. Instead, we can use one of the all-time favourite dimensionality reduction technique - Principle Component Analysis.
# - Before we can use PCA, we need to **STANDARDISE** the data (Standardisation and Normalization are used inter-dependently. Standardisation is moulding the data to between -1 and +1 data points. Normalisation is normalising the data so that the data points lie along the mean.)

# In[ ]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()


# In[ ]:


x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# In[ ]:


print(x_train)


# Now that the data is scaled, we shall use PCA

# In[ ]:


from sklearn.decomposition import PCA


# In[ ]:


pca_x = PCA(0.95).fit(x_train)


# In[ ]:


print('%d components explain 95%% of the variation in data' % pca_x.n_components_)


# We can see that the first 1527 Principal Components attribute for about 95% variation in the data. We shall use these 1527 for our prediction 

# In[ ]:


pca = PCA(n_components=1527)
#fit with 1527 components on train data
pca.fit(x_train)
#transform on train data
x_train_pca = pca.transform(x_train)
#transform on test data
x_test_pca = pca.transform(x_test)


# **MODELLING AND PREDICTION**
# We shall use the following classifiers for our prediction 
# - Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
rf.fit(x_train_pca, y_train)


# In[ ]:


rf_pca_predict = rf.predict(x_test_pca)


# In[ ]:


rf_pca_predict = np.expm1(rf_pca_predict)
print(rf_pca_predict)


# In[ ]:


print(len(rf_pca_predict))


# In[ ]:


submission = pd.read_csv('../input/sample_submission.csv')
submission["target"] = rf_pca_predict


# In[ ]:


print(submission.head())
submission.to_csv('sub_PCA_LR.csv', index=False)


# In[ ]:


print(submission['target'])


# 
# **USING TSVD**
# - TSVD, which stands for Truncated Single Vector Decomposition is a dimensonality reduction methodology. Unlike PCA, we do not need to standardise the data before we pass it through a TSVD.
# - One of the main parameters is n_components which should be LESS THAN the number of dimensions.
# - For the sake of this problem, I shall randomly pick n_components as 1500, and then write a code to choose those components which attribute for 95% of variation in the data.
# - I am going to use the copies of the x_train, x_test, y_train datasets for TSVD.

# In[ ]:


from sklearn.decomposition import TruncatedSVD


# In[ ]:


svd_x = TruncatedSVD(n_components=1500,n_iter=20, random_state=42)
svd_x.fit(x_train_copy)


# In[ ]:


#code to select those components which attribute for 95% of variance in data
count = 0
for index, cumsum in enumerate(np.cumsum(svd_x.explained_variance_ratio_)):
    if cumsum <=0.95:
      count+=1  
    else:
        break
print(count)


# From the above result we can see that the first 601 components attrribte for 95% of the variation in data. We shall use these 601 components

# In[ ]:


for index, cumsum in enumerate(np.cumsum(svd_x.explained_variance_ratio_)):
    print(index, cumsum)


# In[ ]:


svd = TruncatedSVD(n_components=601, random_state=42)
#fit the TSVD on the train data
svd.fit(x_train_copy)
#transform on the x_train data
x_train_svd = svd.transform(x_train_copy)
#transform on the x_test data
x_test_svd = svd.transform(x_test_copy)


# Use a Random Forest Regressor for modelling

# In[ ]:


rf.fit(x_train_svd, y_train_copy)
rf_tsvd_predict = rf.predict(x_test_svd)
rf_tsvd_predict = np.expm1(rf_tsvd_predict)


# In[ ]:


print(rf_tsvd_predict)


# In[ ]:


submission = pd.read_csv('../input/sample_submission.csv')
submission["target"] = rf_tsvd_predict
print(submission.head())
submission.to_csv('sub_TSVD_LR.csv', index=False)


# In[ ]:




