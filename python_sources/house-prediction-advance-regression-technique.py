#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as  sns


# ### Import train dataset

# In[ ]:


train_set = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")


# In[ ]:


train_set.describe()


# **Let;s Pinch this data little bit to know about it**

# In[ ]:


print("Train Data Features :: ", len(train_set.columns.tolist()))
print("Train Data Categorical Features::", len(train_set.select_dtypes(exclude=["number","bool_"]).columns.tolist()))
print("Train Data Continuous variable :: ",len(train_set.select_dtypes(exclude=["object_"]).columns.tolist()))
print("Overall Features Listing :: \n\n",train_set.columns.tolist())


# > > Great, so we exposed the inital insights of the data.
# * This data is having  so many Independent varibles, so we are gonna be jumping into Multivariate Analysis sooner :)
# * This data is having **37** Independent and **1** Dependent variable to be predicted. (Isn't it cool...!! )

# In[ ]:


test_set = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")


# In[ ]:


test_set.describe()


# In[ ]:


final_set = pd.concat([train_set,test_set],axis=0)


# In[ ]:


train_set = final_set.drop(columns=["SalePrice"])


# In[ ]:


len(train_set.columns)


# ### First insights of data

# In[ ]:


train_set.head()


# ### Data description

# In[ ]:


train_set.describe()


# ### Heat map to show the nan column wise

# In[ ]:


fig, ax = plt.subplots(figsize=(30,5))
sns.heatmap(train_set.isnull(),yticklabels=False,cbar=False,ax=ax)


# ### Get columns with NaN samples > 200

# In[ ]:


nanCols =  train_set.columns[train_set.isnull().sum() > 200].tolist()


# ### Features with Null>200 

# In[ ]:


nanCols


# ### Drop Columns according to logic [Remove features from data which have more null than 200]

# In[ ]:


train_set.drop(columns=nanCols,inplace = True)


# ### Look up for reduced Features

# In[ ]:


len(train_set.columns)


# ### Dataframe Lookup after initial feature removal

# In[ ]:


train_set.head()


# ### Data lookup after feature reductions

# In[ ]:


fig, ax = plt.subplots(figsize=(30,5))
sns.heatmap(train_set.isnull(),yticklabels=False,cbar=False,ax=ax)


# ### Still some of them are holding NaNs so time to fill the Nan with mean, median,mode accordingly
# 

# In[ ]:


def fill_nan(categorical_features,train_set):
    for i in categorical_features:
        train_set[i] =  train_set[i].fillna(train_set[i].mode()[0])
    
    return train_set


# ### Finding all the categorical features

# In[ ]:


categorical_features = train_set.select_dtypes(exclude=["number","bool_"]).columns.tolist()
len(categorical_features)


# ### Finding all the continuous features

# In[ ]:


continuous_features =  train_set.select_dtypes(exclude=["object_"]).columns.tolist()
len(continuous_features)


# ### Filling NaNs for Categorical Features

# In[ ]:


train_set = fill_nan(categorical_features,train_set)


# ### Initial lookup after filling Nan of categorical features

# In[ ]:


fig, ax = plt.subplots(figsize=(40,20))
sns.heatmap(train_set.isnull(),yticklabels=False,cbar=False,ax=ax)


# ### Filling NaNs For Continuous variable

# In[ ]:


def fill_nan_continuous(continuous_features,train_set):
    
    for i in continuous_features:
        train_set[i] =  train_set[i].fillna(train_set[i].mean())
    return train_set


# In[ ]:


train_set = fill_nan_continuous(continuous_features,train_set)


# In[ ]:


train_set.drop(columns=["Id"] , inplace=True)


# In[ ]:


sns.heatmap(train_set.isnull(),yticklabels=False,cbar=False)


# In[ ]:


len(train_set.columns)


# In[ ]:


train_set.head()


# In[ ]:


len(train_set.columns.tolist())


# ### OneHot Encoding Of Features

# In[ ]:


def one_hot_encoder(train_set):
    df  = train_set.copy(deep= True)
    dummies = pd.get_dummies(df,prefix="column_",drop_first=True)
    return dummies


# In[ ]:


train_set = one_hot_encoder(train_set)


# In[ ]:


len(train_set.columns)


# In[ ]:


train_set = train_set.loc[:,~train_set.columns.duplicated()]


# In[ ]:


train_df = train_set[0:1460]


# In[ ]:


train_df = pd.concat([train_df,final_set["SalePrice"][0:1460]],axis=1)


# In[ ]:


train_df.head()


# In[ ]:


test_df =  train_set[1461:]


# In[ ]:


test_df.head()


# In[ ]:


train_df.head()


# In[ ]:


import xgboost


# In[ ]:


Xtrain = train_df.drop(columns=["SalePrice"],axis=1)
yTrain = train_df["SalePrice"]
classifier = xgboost.XGBRFRegressor()
classifier.fit(Xtrain,yTrain)


# In[ ]:


# import pickle
# fileName = "../input/house-prices-advanced-regression-techniques/finalizedModel.pkl" 
# pickle.dump(classifier,open(fileName,"wb"))


# In[ ]:


y_pred = classifier.predict(test_df).tolist()


# In[ ]:


y_pred


# In[ ]:


predictions = pd.DataFrame(y_pred)
sf = pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv")
datasets = pd.concat([sf["Id"],predictions],axis=1)
datasets.columns = ["Id","SalePrice"]

# import os
# os.chdir('../output/kaggle/working')

datasets.to_csv("sample_submission.csv",index=False)
datasets.head()

