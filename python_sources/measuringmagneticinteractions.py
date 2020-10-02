#!/usr/bin/env python
# coding: utf-8

# * **General Information**
# 
# Develop an algorithm that can predict the magnetic interaction between two atoms in a molecule (i.e., the scalar coupling constant).

# **Importing Libraries**

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import gc
import os

import matplotlib.pyplot as plt
import seaborn as sns

import lightgbm as lgb
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics


# * **Accessing Data and Preperaing Data**

# In[ ]:


print(os.listdir("../input"))


# We will be using train.csv, test.csv and structures.csv files for our analysis and model preperation.
# 
# Lets load those files in dataframe.

# In[ ]:


train_original = pd.read_csv("../input/train.csv")
structures_original = pd.read_csv("../input/structures.csv")
test_original = pd.read_csv("../input/test.csv")


# Lets have a look at some records from each dataframe

# In[ ]:


train_original.head()


# In[ ]:


structures_original.head()


# In[ ]:


test_original.head()


# How many items are there in dsgdb9nsd_000015?

# In[ ]:


structures_original[structures_original['molecule_name'] == 'dsgdb9nsd_000015']


# OK.
# 
# As per above there are total 9 items in molecule dsgdb9nsd_000015.
# 
# Carbon - 2
# 
# Oxygen - 1
# 
# Hydrogen - 6
# 
# Total items in each molecule can be calculated by simply grouping structures_original dataframe by molecule_name ans atom with count as a aggregate function

# In[ ]:


moleculeCount = structures_original.groupby(by=['molecule_name','atom'])[['atom']].count()
moleculeCount.rename(columns={'atom':'count'},inplace = True)
moleculeCount = moleculeCount.unstack(fill_value=0)
moleculeCount = moleculeCount['count'].reset_index()

moleculeCount.head()


# In[ ]:


moleculeCount[moleculeCount['molecule_name'] == 'dsgdb9nsd_000015']


# Merge moleculeCount dataframe in original structures dataframe.So tha later we can use that for enriching train and test data.

# In[ ]:


structures = pd.DataFrame.merge(structures_original,moleculeCount
                               ,how='inner'
                               ,left_on = ['molecule_name'] 
                               ,right_on = ['molecule_name']
                              )

structures.head()


# Join structures dataframe with train and test data to include item counts in train and test data.

# In[ ]:


tmp_merge = pd.DataFrame.merge(train_original,structures
                               ,how='left'
                               ,left_on = ['molecule_name','atom_index_0'] 
                               ,right_on = ['molecule_name','atom_index']
                              )

tmp_merge = tmp_merge.merge(structures
                ,how='left'
                ,left_on = ['molecule_name','atom_index_1'] 
                ,right_on = ['molecule_name','atom_index']
               )

tmp_merge.drop(columns=['atom_index_x','atom_index_y','C_x','F_x','H_x','N_x','O_x'],inplace=True)
tmp_merge.columns = ['id' , 'molecule_name' , 'atom_0' , 'atom_1' , 'type' , 'scalar_coupling_constant' , 
                      'atom_nm_0' , 'x_0' , 'y_0' , 'z_0' , 'atom_nm_1' , 'x_1' , 'y_1' , 'z_1','C','F','H','N','O']

train = tmp_merge[['id' , 'molecule_name' , 'atom_0' , 'atom_1' , 'type'  , 'atom_nm_0' , 'x_0' ,
           'y_0' , 'z_0' , 'atom_nm_1' , 'x_1' , 'y_1' , 'z_1','C','F','H','N','O', 'scalar_coupling_constant']]
train.sort_values(by=['id','molecule_name'],inplace=True)
train.reset_index(inplace=True,drop=True)

tmp_merge = None

train.head()


# In[ ]:


tmp_merge = pd.DataFrame.merge(test_original,structures
                               ,how='inner'
                               ,left_on = ['molecule_name','atom_index_0'] 
                               ,right_on = ['molecule_name','atom_index']
                              )
tmp_merge = tmp_merge.merge(structures
                ,how='inner'
                ,left_on = ['molecule_name','atom_index_1'] 
                ,right_on = ['molecule_name','atom_index']
               )

tmp_merge.drop(columns=['atom_index_x','atom_index_y','C_x','F_x','H_x','N_x','O_x'],inplace=True)
tmp_merge.columns = ['id' , 'molecule_name' , 'atom_0' , 'atom_1' , 'type' ,  
                      'atom_nm_0' , 'x_0' , 'y_0' , 'z_0' , 'atom_nm_1' , 'x_1' , 'y_1' , 'z_1','C','F','H','N','O']


test = tmp_merge[['id' , 'molecule_name' , 'atom_0' , 'atom_1' , 'type'  , 'atom_nm_0' , 'x_0' ,
           'y_0' , 'z_0' , 'atom_nm_1' , 'x_1' , 'y_1' , 'z_1', 'C','F','H','N','O']]


test.sort_values(by=['id','molecule_name'],inplace=True)
test.reset_index(inplace=True,drop=True)

tmp_merge = None

test.head()


# In[ ]:



train_original = None
del train_original
structures_original = None
del structures_original
test_original = None
del test_original
structures = None
del structures
gc.collect()


# I am using below kernel to calculate distance between 2 items in a molecule.
# 
# https://www.kaggle.com/seriousran/just-speed-up-calculate-distance-from-benchmark
# 
# The Frobenius norm is given by [1]:
# 
# ||A||F = [\sum{i,j} abs(a_{i,j})^2]^{1/2}

# In[ ]:


train['dist'] = np.linalg.norm(train[['x_0', 'y_0', 'z_0']].values - train[['x_1', 'y_1', 'z_1']].values, axis=1)
test['dist'] = np.linalg.norm(test[['x_0', 'y_0', 'z_0']].values - test[['x_1', 'y_1', 'z_1']].values, axis=1)

train.drop(columns=['x_0', 'y_0', 'z_0','x_1', 'y_1', 'z_1'],inplace=True)
test.drop(columns=['x_0', 'y_0', 'z_0','x_1', 'y_1', 'z_1'],inplace=True)


# In[ ]:


train['type'] = pd.Categorical(train['type'])
train['atom_nm_1'] = pd.Categorical(train['atom_nm_1'])
test['type'] = pd.Categorical(test['type'])
test['atom_nm_1'] = pd.Categorical(test['atom_nm_1'])


# In[ ]:


train.head()


# In[ ]:


test.head()


# * **Model**

# In[ ]:


X = train[['atom_0' ,  'atom_1' , 'type', 'atom_nm_1', 'C' ,  'F' ,  'H' ,  'N' ,  'O' , 'dist' ]]

y = train['scalar_coupling_constant']


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=420)


# In[ ]:


lgb_train = lgb.Dataset(X_train,y_train,free_raw_data=True)
lgb_eval = lgb.Dataset(X_test,y_test,free_raw_data=True)


# In[ ]:


params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'learning_rate': 0.05,
    'num_leaves': 50, 
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0,
    'num_boost_round':5000,
    'reg_alpha': 0.1,
    'reg_lambda': 0.3,
'early_stopping_rounds':5
         }


# In[ ]:


gbm = lgb.train(
    params,
    lgb_train,

    valid_sets=lgb_eval
)


# In[ ]:


y_predict = gbm.predict(X_test)
mse = np.sqrt(metrics.mean_squared_error(y_predict,y_test))

print('Mean Squared Error is : '+str(mse))


# * **Sumbission:**

# In[ ]:


submission_df = pd.DataFrame(columns=['id', 'scalar_coupling_constant'])
submission_df['id'] = test['id']
submission_df['scalar_coupling_constant'] = gbm.predict(test[['atom_0' ,  'atom_1' , 'type', 'atom_nm_1', 'C' ,  'F' ,  'H' ,  'N' ,  'O' , 'dist' ]])
submission_df.to_csv('submissions.csv', header=True, index=False)
submission_df.head(10)

