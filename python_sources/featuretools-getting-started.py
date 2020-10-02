#!/usr/bin/env python
# coding: utf-8

# # **A Hands-On Guide to Automated Feature Engineering using Featuretools in Python**

# In[ ]:


## DONE: Installed featuretools library in the environment
# !python -m pip install featuretools


# In[ ]:


## DONE: Key imports for any data science project
import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


## DONE: Create files dictionary for any file in the input directory
files = {}
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        files[filename] = os.path.join(dirname, filename)
        print(os.path.join(dirname, filename))


# In[ ]:


## DONE: Subsidiary import libraries for this project
import featuretools as ft


# In[ ]:


##DONE: Read in files from input directory
train = pd.read_csv(files['Train.csv'])
test = pd.read_csv(files['Test.csv'])
submission = pd.read_csv(files['Submission.csv'])


# In[ ]:


## DONE: Data preparation

# saving identifiers
test_Item_Identifier = test['Item_Identifier']
test_Outlet_Identifier = test['Outlet_Identifier']
sales = train['Item_Outlet_Sales']
train.drop(['Item_Outlet_Sales'], axis=1, inplace=True)


# In[ ]:


combi = train.append(test, ignore_index=True)


# In[ ]:


## DONE: Check the sum of the missing values in the dataframe
combi.isnull().sum()


# In[ ]:


# imputing missing data
combi['Item_Weight'].fillna(combi['Item_Weight'].mean(), inplace = True)
combi['Outlet_Size'].fillna("missing", inplace = True)


# In[ ]:


## DONE: Data preprocessing
combi['Item_Fat_Content'].value_counts()


# In[ ]:


## DONE: Dealing with categorical values i.e here we are doing label encoding
# dictionary to replace the categories
fat_content_dict = {'Low Fat':0, 'Regular':1, 'LF':0, 'reg':1, 'low fat':0}

combi['Item_Fat_Content'] = combi['Item_Fat_Content'].replace(fat_content_dict, regex=True)


# In[ ]:


# Created unique identifier
combi['id'] = combi['Item_Identifier'] + combi['Outlet_Identifier']
combi.drop(['Item_Identifier'], axis=1, inplace=True)


# In[ ]:


# creating and entity set 'es'
es = ft.EntitySet(id = 'sales')

# adding a dataframe 
es.entity_from_dataframe(entity_id = 'bigmart', dataframe = combi, index = 'id')


# In[ ]:


es.normalize_entity(base_entity_id='bigmart', new_entity_id='outlet', index = 'Outlet_Identifier', 
additional_variables = ['Outlet_Establishment_Year', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type'])


# In[ ]:


feature_matrix, feature_names = ft.dfs(entityset=es, 
target_entity = 'bigmart', 
max_depth = 2, 
verbose = 1, 
n_jobs = 3)


# In[ ]:


feature_matrix.columns


# In[ ]:


feature_matrix.head()


# In[ ]:


feature_matrix = feature_matrix.reindex(index=combi['id'])
feature_matrix = feature_matrix.reset_index()


# In[ ]:


from catboost import CatBoostRegressor


# In[ ]:


categorical_features = np.where(feature_matrix.dtypes == 'object')[0]

for i in categorical_features:
    feature_matrix.iloc[:,i] = feature_matrix.iloc[:,i].astype('str')


# In[ ]:


feature_matrix.drop(['id'], axis=1, inplace=True)
train = feature_matrix[:8523]
test = feature_matrix[8523:]


# In[ ]:


# removing uneccesary variables
train.drop(['Outlet_Identifier'], axis=1, inplace=True)
test.drop(['Outlet_Identifier'], axis=1, inplace=True)


# In[ ]:


# identifying categorical features
categorical_features = np.where(train.dtypes == 'object')[0]


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


# splitting train data into training and validation set
xtrain, xvalid, ytrain, yvalid = train_test_split(train, sales, test_size=0.25, random_state=11)


# In[ ]:


model_cat = CatBoostRegressor(iterations=100, learning_rate=0.3, depth=6, eval_metric='RMSE', random_seed=7)


# In[ ]:


model_cat.fit(xtrain, ytrain, cat_features=categorical_features, use_best_model=True)


# In[ ]:


model_cat.score(xvalid, yvalid)

