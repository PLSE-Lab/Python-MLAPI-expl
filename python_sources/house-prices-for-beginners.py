#!/usr/bin/env python
# coding: utf-8

# # House Price prediction for beginners

# ![housesbanner.png](attachment:housesbanner.png)

# ### This kernel includes following topics
# *     Importing required libraries
# *     Handling Missing values
# *     Data Wrangling, convert categorical to numerical
# *     Applying the basic Regression models of sklearn
# *     Comparing the performance of the Regressors and choosing the best one

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# ### Importing the required libraries

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from scipy.stats import skew, kurtosis
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set()


# ### **Loading datasets**

# In[ ]:


train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')


# In[ ]:


# checking the shape of both datasetes
print(train.shape)
print(test.shape)


# **There's one more column in train dataset which is the target variable, we have to seperate it to make a whole dataframe of train and test dataset for the further analysis.**

# In[ ]:


target = train[['SalePrice']]
# dropping the target variable from train dataset
train.drop(columns=['SalePrice'],axis=1, inplace=True) 


# In[ ]:


# Now concating both datasets
df = pd.concat([train,test])
df.shape


# In[ ]:


df.info()


# **We see that we have columns with catogarical data and also numerical data but if we see that there's Id column which is nothing but representing the index so we will drop it becuase it has no relation with the target variable**
# 
# **In supervised learning like classification and regression problems we have two type of variables**
# * **Feature variables** 
# * **Target variables**
# 
# **A function we say y=f(x) here y is dependent on the input of x and x has the effect on y as x changes so does y. likely in machine learning we call x the features and y the target variable. With the help of features we predict the target variable.**

# ### Let's drop the Id column dataset

# In[ ]:


df.drop(columns=['Id'], axis=1, inplace=True)
df.shape


# ## Handling missing data
# **Handling missing data requires the great strategy and analysis to fit the model, **
# **Sometimes missing data is on purpose and sometimes it's not. So we will handle by looking at the number of missing data if there's missing data more than 30% then we will drop those columns and if less then we will replace the numerical columns with their mean and catogarical columns with their mode**
# 
# **Rather than handling the missing data  of train and test dataset individually it's better to concat them but before that we have to seperate the target column from train dataset to make them have equal number of features.**

# **Now let's visualize the missing values of whole dataset**

# In[ ]:


plt.figure(figsize=(16,12))
msno.bar(df, labels=True, fontsize=(10))


# **we see that there are a few columns having great number of missing values but let's check them percent wise**

# In[ ]:


#categorical variables having missing values more than 30%
for col in df:
    if df[col].dtype == 'object':
        if (df[col].isnull().sum()/df[col].isnull().count())*100 >=30:
            print(col)


# **In catogarical we have 5 columns with more than 30% of missing data so we're going to remove them**

# In[ ]:


df.drop(columns=['Alley','FireplaceQu','PoolQC','Fence','MiscFeature'], axis=1, inplace=True)


# **Now we will fill the rest of categorical columns with their most frequent value**

# In[ ]:


for col in df:
    if df[col].dtype == 'object':
        df[col] = df[col].fillna(df[col].mode()[0])


# **Now let's do it same for the Numerical data first let's see what percent missing values we have in Numerical columns** 

# In[ ]:


# Now for numericals
check = False
for col in df:
    if df[col].dtype != 'object':
        if (df[col].isnull().sum()/df[col].isnull().count())*100 >=30:
            print(col)
        else:
            check = True
if check:
    print('We do not have missing data more than 30%')


# In[ ]:


# let's see how much data we have missing in numerical data
for col in df:
    if df[col].dtype != 'object':
        print(df[col].isnull().sum(), end=' ')


# **As per trend to replace with their mean i don't think so it's good choice to do so becuase missing data has the importance becuase as I earlier mentioned above that sometimes data is missing on purpose so rather than filling it with mean I am going to fill it with 0, near to me zero seems meaningful in handling the data**

# In[ ]:


for col in df:
    if df[col].dtype != 'object':
        df[col] = df[col].fillna(0)


# **Lets see if we are done with missing values**

# In[ ]:


plt.figure(figsize=(16,12))
msno.bar(df, labels=True, fontsize=(10))


# **Hopefully we don't have any missing value left in the dataset**

# **there are object type columns having catogarical data As we know that ML models work on numeric data not on catogarical so we will perfom the one hot encoding to convert catogarical features into numeric and for that pandas has a function named as get_dummies() which itself creates the features from catogaries. Then check their dimensions.**

# In[ ]:


df = pd.get_dummies(df)
df.shape


# ### Normalizing the data
# **It's better to scale the data within the range of 0 and 1 before fitting to model**

# In[ ]:


from sklearn.preprocessing import MinMaxScaler
df_scaled = pd.DataFrame(MinMaxScaler().fit_transform(df))
df_scaled.columns = df.columns
df_scaled


# #### After EDA we have to split the datsaet back into two datasets train and dataset

# In[ ]:


train = df_scaled.iloc[:1460,:]
test = df_scaled.iloc[1460:,:]
print(train.shape)
print(test.shape)


# ### Now for training the model we need to split up the train dataset and target variable into X_train,X_val,y_train,y_val variables.

# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_val,y_train,y_val = train_test_split(train,target,test_size=0.2,random_state=1)


# ### We are going to apply few models and let's see which results best

# **Linear Regression**

# In[ ]:


from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train,y_train)
lr_preds = lr.predict(X_val)


# **Decission Tree**

# In[ ]:


from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor(random_state=1)
dt.fit(X_train,y_train)
dt_preds = dt.predict(X_val)


# **Random Forest**

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(random_state=1)
rf.fit(X_train,y_train)
rf_preds = rf.predict(X_val)


# **XGBoost**

# In[ ]:


from xgboost import XGBRegressor
xgb = XGBRegressor(random_state=1)
xgb.fit(X_train,y_train)
xgb_preds = xgb.predict(X_val)


# **Let's compare the models with their mean absolute error**

# In[ ]:


from sklearn.metrics import mean_absolute_error


# In[ ]:


compare = {'Models':['LinearRegression','DecissionTree','RandomForest','XGBoost'],
          'MeanAbsoluteError':[mean_absolute_error(lr_preds,y_val), mean_absolute_error(dt_preds,y_val), mean_absolute_error(rf_preds,y_val),
                              mean_absolute_error(xgb_preds,y_val)]}
pd.DataFrame(compare)


# **So we are going to test on xgboost model**

# In[ ]:


test_values = xgb.predict(test)
test_values


# ### loading the submission dataset to replace it's values of SalePrice column with our predicted values of test dataset

# In[ ]:


submission = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv')
submission['SalePrice'] = test_values
submission


# In[ ]:


submission.to_csv('Submission_3.csv', index=False)

