#!/usr/bin/env python
# coding: utf-8

# # Intoduction - *Predicting the future sale price by doing EDA, Data Cleaning , Feature Encoding.*

# # Importing Necessary Libraries

# In[ ]:


import warnings
warnings.filterwarnings("ignore")

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# # Reading the train and test data

# In[ ]:


# Read train data
train = pd.read_csv('/kaggle/input/home-data-for-ml-course/train.csv')

# Read test data
test = pd.read_csv('/kaggle/input/home-data-for-ml-course/test.csv')


# In[ ]:


#looking at the train data
train.head()


# # EDA and Data Cleaning 

# In[ ]:


# Drop the  rows where the target is missing
Target = 'SalePrice'
train.dropna(axis=0, subset=[Target], inplace=True)


# In[ ]:


#combining train data and test data
df =pd.concat([train.iloc[:,:-1],test],axis=0)

print('train df has {} rows and {} features'.format(train.shape[0],train.shape[1]))
print('test df has {} rows and {} features'.format(test.shape[0],test.shape[1]))
print('Combined df has {} rows and {} features'.format(df.shape[0],df.shape[1]))


# In[ ]:


#look at the combined data
df.head()


# In[ ]:


#drop ID column
df = df.drop(columns=['Id'],axis=1)


# **Missing Values**

# In[ ]:


def missingValue(df1):
    total = df1.isnull().sum().sort_values(ascending = False)
    percent = round(df1.isnull().sum().sort_values(ascending = False)/len(df)*100, 2)
    temp = pd.concat([total, percent], axis = 1,keys= ['Total', 'Percent'])
    return temp.loc[(temp['Total'] > 0)]

missingValue(train)


# **Handling Missing Values**

# In[ ]:



def HandleMissingValues(df):
    # for Object columns fill using 'UNKNOWN'
    # for Numeric columns fill using median
    num_cols = [cname for cname in df.columns if df[cname].dtype in ['int64', 'float64']]
    cat_cols = [cname for cname in df.columns if df[cname].dtype == "object"]
    values = {}
    for a in cat_cols:
        values[a] = 'UNKNOWN'

    for a in num_cols:
        values[a] = df[a].median()
        
    df.fillna(value=values,inplace=True)
    
    
HandleMissingValues(df)
df.head()


# In[ ]:


#check if there are any missing values
df.isnull().sum().sum()


# # Feature Encoding

# In[ ]:



def getObjectColumnsList(df):
    return [cname for cname in df.columns if df[cname].dtype == "object"]

def PerformOneHotEncoding(df,columnsToEncode):
    return pd.get_dummies(df,columns = columnsToEncode)

cat_cols = getObjectColumnsList(df)
df = PerformOneHotEncoding(df,cat_cols)
df.head()


# In[ ]:


df.shape


# In[ ]:


#split into train and test
train_data=df.iloc[:1460,:]
test_data=df.iloc[1460:,:]
print(train_data.shape)
test_data.shape


# # Modelling

# In[ ]:


X=train_data
y=train.loc[:,'SalePrice']


# In[ ]:


from sklearn.linear_model import RidgeCV

ridge_cv = RidgeCV(alphas=(0.01, 0.05, 0.1, 0.3, 1, 3, 5, 10))
ridge_cv.fit(X, y)
ridge_cv_preds=ridge_cv.predict(test_data)


# In[ ]:



predictions = ( ridge_cv_preds )


# In[ ]:


import xgboost as xgb

model_xgb = xgb.XGBRegressor(n_estimators=340, max_depth=2, learning_rate=0.2)
model_xgb.fit(X, y)
xgb_preds=model_xgb.predict(test_data)


# In[ ]:


predictions2 = (xgb_preds)


# # Submission

# In[ ]:



submission = {
    'Id': test.Id.values,
    'SalePrice': predictions
}
solution = pd.DataFrame(submission)
solution.head()


# In[ ]:


#make the submission data frame
submission = {
    'Id': test.Id.values,
    'SalePrice': predictions2
}
solution = pd.DataFrame(submission)
solution.head()


# # Take the best predictions by checking the both on the leader board.
# # Thank you, Upvote the kernel if u like.

# 
