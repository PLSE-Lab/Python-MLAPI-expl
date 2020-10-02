#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_squared_error

import xgboost as xgb

import os
print(os.listdir("../input"))


# ## Data Preparation
# Importing the data and converting nominal variables to the type *category*. This is needed later in the StringIndexer class.

# In[ ]:


df = pd.read_csv('../input/train.csv', parse_dates=True, encoding='UTF-8')
# Change the nominal variables' dtype to categorical
df.vendor_id = df.vendor_id.astype('category')
df.store_and_fwd_flag = df.store_and_fwd_flag.astype('category')


# In[ ]:


X_submit = pd.read_csv('../input/test.csv', parse_dates=True, encoding='UTF-8')
X_submit.vendor_id = X_submit.vendor_id.astype('category')
X_submit.store_and_fwd_flag = X_submit.store_and_fwd_flag.astype('category')


# ### Transforming the target variable
# As trip_duration is not normally distributed, we apply a log transformation on it. 

# In[ ]:


vals = df.sample(10000)['trip_duration']  # sample to speed up the processing a bit

fig, ax = plt.subplots()
sns.distplot(vals, ax=ax)
ax.set(xlabel="Trip Duration",  title='Distribution of Trip Duration')
ax.legend()
plt.show()


# In[ ]:


df['log_trip_duration'] = np.log(df['trip_duration'].values + 1)


# In[ ]:


vals = df.sample(10000)['log_trip_duration']  # sample to speed up the processing a bit

fig, ax = plt.subplots()
sns.distplot(vals, ax=ax)
ax.set(xlabel="Log Trip Duration", xlim=[0,15], title='Distribution of Log Trip Duration')
ax.axvline(x=np.median(vals), color='m', label='Median', linestyle='--', linewidth=2)
ax.axvline(x=np.mean(vals), color='b', label='Mean', linestyle='--', linewidth=2)
ax.legend()
plt.show()


# In[ ]:


features_to_keep = ['passenger_count','pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude','vendor_id', 'store_and_fwd_flag']
X, y = df[features_to_keep], df.log_trip_duration


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=38)


# ## XGBoost

# ### Integrating Pandas and Sklearn
# Based on this [great article on Medium](https://medium.com/bigdatarepublic/integrating-pandas-and-scikit-learn-with-pipelines-f70eb6183696) I could use a Pandas DataFrame in a Pipeline without loosing track of the column names.

# In[ ]:


# Class to select Dataframe columns based on dtype
class TypeSelector(BaseEstimator, TransformerMixin):
    '''
    Returns a dataframe while keeping only the columns of the specified dtype
    '''
    def __init__(self, dtype):
        self.dtype = dtype
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        return X.select_dtypes(include=[self.dtype])


# In[ ]:


# Class to convert a categorical column into numeric values
class StringIndexer(BaseEstimator, TransformerMixin):
    '''
    Returns a dataframe with the categorical column values replaced with the codes
    Replaces missing value code -1 with a positive integer which is required by OneHotEncoder
    '''
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        return X.apply(lambda s: s.cat.codes.replace(
            {-1: len(s.cat.categories)}
        ))


# ### Constructing the Pipeline
# The Pipeline will apply different transformations depending on the dtype of the columns. Finally it will fit an XGBoost Regressor. The advantage now is that we can use this in a Grid Search.[](http://)

# In[ ]:


'''
pipeline = Pipeline([
    ('features', FeatureUnion(n_jobs=1, transformer_list=[
        # Part 1
        ('boolean', Pipeline([
            ('selector', TypeSelector('bool')),
        ])),  # booleans close
        
        ('numericals', Pipeline([
            ('selector', TypeSelector(np.number)),
            ('scaler', StandardScaler()),
        ])),  # numericals close
        
        # Part 2
        ('categoricals', Pipeline([
            ('selector', TypeSelector('category')),
            ('labeler', StringIndexer()),
            ('encoder', OneHotEncoder(handle_unknown='ignore')),
        ]))  # categoricals close
    ])),  # features close
    ("clf", xgb.XGBRegressor(objective="reg:linear", booster="gbtree", nthread=4))
])  # pipeline close
'''


# In[ ]:


'''
# 'clf__learning_rate': np.arange(0.05, 1.0, 0.05),
# 'clf__n_estimators': np.arange(50, 200, 50)
param_grid = {
    'clf__max_depth': np.arange(3, 10, 1)
}
'''


# ### Grid Search

# In[ ]:


'''
randomized_mse = RandomizedSearchCV(param_distributions=param_grid, estimator=pipeline, n_iter=2, scoring="neg_mean_squared_error", verbose=1, cv=3)

# Fit the estimator
randomized_mse.fit(X_train, y_train)
print(randomized_mse.best_score_)
print(randomized_mse.best_estimator_)
'''


# ### Applying the best model on the test set

# In[ ]:


'''
preds_test = randomized_mse.best_estimator_.predict(X_test)
mean_squared_error(y_test.values, preds_test)
'''


# ### Preparing for submission

# In[ ]:


'''
preds_submit = randomized_mse.best_estimator_.predict(X_submit)
X_submit['trip_duration'] = np.exp(preds_submit) - 1
X_submit[['id', 'trip_duration']].to_csv('bc_xgb_submission.csv', index=False)
'''


# ## CatBoost

# In[ ]:


from catboost import Pool, CatBoostRegressor


# In[ ]:


cat_features = [5,6]
# initialize Pool
train_pool = Pool(X_train, y_train, cat_features=cat_features)
test_pool = Pool(X_test, cat_features=cat_features) 


# In[ ]:


# specify the training parameters 
model = CatBoostRegressor(iterations=1000, loss_function='RMSE', random_seed=38, logging_level='Silent', learning_rate=0.1)
#train the model
model.fit(X_train, y_train, cat_features=cat_features)


# In[ ]:


# make the prediction using the resulting model
preds_test = model.predict(test_pool)
mean_squared_error(y_test.values, preds_test)


# In[ ]:


preds_submit = model.predict(X_submit[features_to_keep])
X_submit['trip_duration'] = np.exp(preds_submit) - 1
X_submit[['id', 'trip_duration']].to_csv('bc_catb_submission.csv', index=False)

