#!/usr/bin/env python
# coding: utf-8

# The purpose of this contest is to predict the selling price of various homes. This notebook is a short explanation of the data and a simple function to deal with the categorical values. Any comments will be greatly appreciated. 
# 
# * Submission 1: No tuning GBM model with all provided data - MSE = 0.13576
# * Submission 2: Tuned GBM with cv and randomized search - MSE =  min_samples_leaf=3, max_features=0.13325
# * Submission 3: No tuning Lasso Model - MSE = 0.22636

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import os
print(os.listdir("../input"))


# In[2]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
samp = pd.read_csv('../input/sample_submission.csv')
train_ = train.drop('SalePrice', axis=1)
train_test = pd.concat([train_, test]) # easy way to deal with train and test data together 


# In[3]:


print(f"There are {test.shape[1]} colomns in the test set. The majority of them are possible inputes to a model. There are {train.shape[0]} rows in the training data set which can be used to train the a model. There are {len([x for x in train_test.columns if train_test[x].dtype == 'object'])} non-numerical columns, {len([x for x in train_test.columns if train_test[x].dtype == 'int64'])} int columns, and {len([x for x in train_test.columns if train_test[x].dtype == 'float64'])} float columns.")


# In[4]:


# The 'MSSubClass' value looks to be a catigorical value, so I will add it to the list of object columns.
# I will also remove 'Id' from the list of int columns
cat_cols = [x for x in train_test.columns if train_test[x].dtype == 'object']
int_cols = [x for x in train_test.columns if train_test[x].dtype == 'int64']
float_cols = [x for x in train_test.columns if train_test[x].dtype == 'float64']
cat_cols.append('MSSubClass')
remove_from_int = ['MSSubClass', 'Id']
int_cols = [x for x in int_cols if x not in remove_from_int]
num_cols = int_cols + float_cols


# In[5]:


# Take care of NAs
# I just used 0 for all NAs and converted to a string for the object columns in the function below.
train_test.fillna(0, inplace=True)


# In[6]:


from sklearn.preprocessing import CategoricalEncoder
def encode_cat(dat):   
    """ functon to return a labeled data frame with one hot encoding """
    cat_encoder = CategoricalEncoder(encoding="onehot-dense")
    dat = dat.astype('str')
    dat_reshaped = dat.values.reshape(-1, 1)
    dat_1hot = cat_encoder.fit_transform(dat_reshaped)
    col_names = [dat.name + "_" + str(x) for x in list(cat_encoder.categories_[0])]
    return pd.DataFrame(dat_1hot, columns=col_names)


# In[7]:


# Using the function 'encode_cat' I am able to keep the column names and go back to them when 
# I want.

cat_df = pd.DataFrame()

for x in cat_cols:
    cat_df = pd.concat([cat_df, encode_cat(train_test[x])], axis=1)
    
cat_df.index = train_test.index # use to add the two dfs together
full_df = pd.concat([train_test[num_cols], cat_df], axis=1)
full_df.columns


# In[11]:


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import ShuffleSplit


# In[12]:


from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
scaled_data = ss.fit_transform(full_df.values)

train_processed = scaled_data[:len(train),:]
test_processed = scaled_data[1460:,:]
log_sp = np.log1p(train['SalePrice'].values).ravel() # convert the target to natural log + 1 to take care of skew (see other kernals for graphs)


# In[14]:


## Lasso Model
from sklearn.linear_model import Lasso
l = Lasso(alpha=0.1, max_iter=50000)
l.fit(train_processed, log_sp)


# In[15]:


y_pred = l.predict(test_processed)


# In[16]:


"""# This is the easiest way to tune models that I have found so far.

# Dictionary with different hyperparameters to try out
param_grid={'n_estimators':[160, 170, 180],
           'min_samples_leaf':[2, 3, 4],
           'max_features':[1, 0.3, 0.1],
           'max_depth':[3,4,5]} 

cv = ShuffleSplit(n_splits=10)

gbr = GradientBoostingRegressor(min_samples_leaf=3, max_features=0.3, max_depth=4)
gs = GridSearchCV(estimator=gbr, cv=cv, param_grid=param_grid, n_jobs=5, verbose=1, scoring='neg_mean_squared_error')"""
gs.fit(train_processed, log_sp)


# In[ ]:


l.


# In[33]:


preds = np.expm1(l.predict(X=test_processed)) # predict using test and 

test['SalePrice'] = preds
submission = test[['Id', 'SalePrice']]
submission.to_csv('sub_3.csv', index=False) # I have been changing this as I go


# In[23]:


"""gbr = GradientBoostingRegressor(min_samples_leaf=3, max_features=0.3, max_depth=4, n_estimators=180) # the best parms I found from the CV
gbr.fit(y=log_sp, X=train_processed)
preds = np.expm1(gbr.predict(X=test_processed)) # predict using test and 

test['SalePrice'] = preds
submission = test[['Id', 'SalePrice']]
submission.to_csv('sub_2.csv', index=False) # I have been changing this as I go"""

