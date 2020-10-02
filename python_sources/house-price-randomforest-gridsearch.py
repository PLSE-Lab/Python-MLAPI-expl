#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # Load data 

# In[ ]:


import pandas as pd
from sklearn.model_selection import train_test_split

# Read the data
X_full = pd.read_csv('../input/home-data-for-ml-course/train.csv', index_col='Id')
X_test_full = pd.read_csv('../input/home-data-for-ml-course/test.csv', index_col='Id')

# Remove rows with missing target, separate target from predictors
X_full.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = X_full.SalePrice
X_full.drop(['SalePrice'], axis=1, inplace=True)

# To keep things simple, we'll use only numerical predictors
X = X_full.select_dtypes(exclude=['object'])
X_test = X_test_full.select_dtypes(exclude=['object'])

# Break off validation set from training data
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                      random_state=0)


# # Handle Missing Data 

# In[ ]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer()
X_imputed = pd.DataFrame(imputer.fit_transform(X))
X_imputed.columns = X.columns 
X_imputed


# # Setup hyper-parameters 

# In[ ]:


parameters = [{
    'n_estimators': list(range(100, 1001, 100)), 
    'max_leaf_nodes': list(range(2, 10, 1)), 
    'max_depth': list(range(6, 30, 1))
}]
print(parameters)


# # Define GridSearchCV for hyper-parameter tuning 

# In[ ]:


from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
gsearch = GridSearchCV(estimator=RandomForestRegressor(),
                       param_grid = parameters, 
                       scoring='neg_mean_absolute_error',
                       n_jobs=4,cv=5, verbose=7)


# # Fit the model

# In[ ]:


gsearch.fit(X_imputed, y)


# # Results

# In[ ]:


gsearch.best_score_


# In[ ]:


gsearch.cv_results_


# In[ ]:


gsearch.cv_results_.get('mean_test_score')


# In[ ]:


gsearch.cv_results_.get('mean_test_score')


# In[ ]:


gsearch.cv_results_.get('std_test_score')


# # Get the best parameters 

# In[ ]:


best_max_depth = gsearch.best_params_.get('max_depth')
best_max_depth


# In[ ]:


best_max_leaf_nodes = gsearch.best_params_.get('max_leaf_nodes')
best_max_leaf_nodes


# In[ ]:


best_n_estimators = gsearch.best_params_.get('n_estimators')
best_n_estimators


# # Build the final model with the best parameters 

# In[ ]:


final_model = RandomForestRegressor(n_estimators=best_n_estimators, 
                          max_depth=best_max_depth, 
                          max_leaf_nodes=best_max_depth)


# In[ ]:


final_model.fit(X_imputed, y)


# # Handle Missing values on test data 

# In[ ]:


X_test_imputed = pd.DataFrame(imputer.transform(X_test))
X_test_imputed.columns = X_test.columns 
X_test_imputed


# # Predict test data 

# In[ ]:


preds_test = final_model.predict(X_test_imputed)


# # Write predictions to submit 

# In[ ]:


# Save test predictions to file
output = pd.DataFrame({'Id': X_test.index,
                       'SalePrice': preds_test})
output.to_csv('submission.csv', index=False)

