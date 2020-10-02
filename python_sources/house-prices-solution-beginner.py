#!/usr/bin/env python
# coding: utf-8

# # House Prices Solution 
# 
# This competition challenges you to predict the final price of each home with 79 explanatory variables describing (almost) every aspect of residential homes in Ames, Iowa.
# 
# In this notebook : <span style="color:ROYALBLUE">** Quick EDA **</span> -> <span style="color:BLUE">** Data cleaning **</span> -> <span style="color:MEDIUMBLUE">** Train machine learning regression algorithms to predict **</span> -> <span style="color:DARKBLUE">** Submission **</span>
# 
# 
# Let's Start!

# In[ ]:


#Import Required Libraries
import warnings
warnings.filterwarnings("ignore")

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))


# ## Quick EDA

# In[ ]:


train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv') 
test  = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')


# In[ ]:


train.shape


# In[ ]:


train.head()


# In[ ]:


import pandas_profiling
profile_report = pandas_profiling.ProfileReport(train)
profile_report


# ## Data Cleaning

# In[ ]:


# Dropping rows where the target is missing
Target = 'SalePrice'
train.dropna(axis=0, subset=[Target], inplace=True)


# In[ ]:


# Combine Test and Training sets to maintain consistancy.
data=pd.concat([train.iloc[:,:-1],test],axis=0)

print('train df has {} rows and {} features'.format(train.shape[0],train.shape[1]))
print('test df has {} rows and {} features'.format(test.shape[0],test.shape[1]))
print('Combined df has {} rows and {} features'.format(data.shape[0],data.shape[1]))


# In[ ]:


data.head()


# In[ ]:


# Dropping unwanted columns
data = data.drop(columns=['Id'],axis=1)


# In[ ]:


# Looking for Missing Values

def missingValuesInfo(df):
    total = df.isnull().sum().sort_values(ascending = False)
    percent = round(df.isnull().sum().sort_values(ascending = False)/len(df)*100, 2)
    temp = pd.concat([total, percent], axis = 1,keys= ['Total', 'Percent'])
    return temp.loc[(temp['Total'] > 0)]

missingValuesInfo(train)


# In[ ]:


# Missing Value Handling

def HandleMissingValues(df):
    # for Object columns fill using 'UNKOWN'
    # for Numeric columns fill using median
    num_cols = [cname for cname in df.columns if df[cname].dtype in ['int64', 'float64']]
    cat_cols = [cname for cname in df.columns if df[cname].dtype == "object"]
    values = {}
    for a in cat_cols:
        values[a] = 'UNKOWN'

    for a in num_cols:
        values[a] = df[a].median()
        
    df.fillna(value=values,inplace=True)
    
    
HandleMissingValues(data)
data.head()


# In[ ]:


# Check for any missing values
data.isnull().sum().sum()


# In[ ]:


#Categorical Feature Encoding

def getObjectColumnsList(df):
    return [cname for cname in df.columns if df[cname].dtype == "object"]

def PerformOneHotEncoding(df,columnsToEncode):
    return pd.get_dummies(df,columns = columnsToEncode)

cat_cols = getObjectColumnsList(data)
data = PerformOneHotEncoding(data,cat_cols)
data.head()


# In[ ]:


data.shape


# In[ ]:


#spliting the data into train and test datasets
train_data=data.iloc[:1460,:]
test_data=data.iloc[1460:,:]
print(train_data.shape)
test_data.shape


# In[ ]:


# Get X,y for modelling
X=train_data
y=train.loc[:,'SalePrice']


# ## Predictive Modeling

# In[ ]:


from sklearn.linear_model import RidgeCV

ridge_cv = RidgeCV(alphas=(0.01, 0.05, 0.1, 0.3, 1, 3, 5, 10))
ridge_cv.fit(X, y)
ridge_cv_preds=ridge_cv.predict(test_data)


# In[ ]:


import xgboost as xgb

model_xgb = xgb.XGBRegressor(n_estimators=340, max_depth=2, learning_rate=0.2)
model_xgb.fit(X, y)
xgb_preds=model_xgb.predict(test_data)


# In[ ]:


predictions = ( ridge_cv_preds + xgb_preds )/2


# ## Submission

# In[ ]:


#make the submission data frame
submission = {
    'Id': test.Id.values,
    'SalePrice': predictions
}
solution = pd.DataFrame(submission)
solution.head()


# In[ ]:


#make the submission file
solution.to_csv('submission.csv',index=False)


# ## Credits
# 

# * https://www.kaggle.com/redaabdou/house-prices-solution-data-cleaning-ml
# 
