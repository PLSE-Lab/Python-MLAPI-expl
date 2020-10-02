#!/usr/bin/env python
# coding: utf-8

# # GridSearchCV for hyper-parameters tuning XGBoost models using Pipelines 

# `GridSearchCV` is a cross validation technique for tuning a model. Similar to `cross_val_score`, it uses the cross validation process explained in the [Cross-Validation tutorial](https://www.kaggle.com/dansbecker/cross-validation). However, the main objective of `GridSearchCV` is to find the most optimal parameters. We pass on a parameters' dictionary to the function and the function compares the cross-validation score for each combination of parameters in the dictionary and returns the set of best parameters. The tutorial specifically covers `XGBBoost` since they are very sensitive to hyperparameters' tuning and here we demonstrate how to use early stopping rounds in `GridSearchCV`.
# 
# 
# We won't focus on the data loading. For now, you can imagine you are at a point where you already have train_X, val_X, train_y, and val_y. 

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import warnings 
warnings.filterwarnings('ignore')

# Read Data
data = pd.read_csv('../input/melb_data.csv')
cols_to_use = ['Rooms', 'Distance', 'Landsize', 'BuildingArea', 'YearBuilt']
X = np.array(data[cols_to_use])
y = data.Price
train_X, val_X, train_y, val_y = train_test_split(X, y)


# ### Building a pipeline

# We use `Imputer` to fill in missing values, followed by a `XGBRegressor` to make predictions.  These can be bundled together in a pipeline as shown below.

# In[ ]:


from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer

my_pipeline = Pipeline([('imputer', Imputer()), ('xgbrg', XGBRegressor())])


# The above code is similar to the [Pipeline tutorial](https://www.kaggle.com/dansbecker/pipelines) except that here we use `Pipeline` instead of `make_pipeline` because we want to have a name for every step in our pipeline so that we can call on a step and set parameters.

# ### GridSearchCV for tuning the model
# 
# Here we use `GridSearchCV` on our pipeline and train it on our training set. We are using 5-fold cross validation by passing the argument `cv=5` in the `GridSearchCV`. The `param_grid` is the dictionary for the values of parameters that we want to compare. Using `GridSearchCV` on a pipeline is very similar to use it on a regressor/classifier, except that we add the name of the regression (here `xgbrg__`) in front of the parameters' names in the `param_grid`. 

# In[ ]:


from sklearn.model_selection import GridSearchCV

param_grid = {
    "xgbrg__n_estimators": [10, 50, 100, 500],
    "xgbrg__learning_rate": [0.1, 0.5, 1],
}

fit_params = {"xgbrg__eval_set": [(val_X, val_y)], 
              "xgbrg__early_stopping_rounds": 10, 
              "xgbrg__verbose": False} 

searchCV = GridSearchCV(my_pipeline, cv=5, param_grid=param_grid, fit_params=fit_params)
searchCV.fit(train_X, train_y)  


# As explained in [Learning to Use XGBoost tutorial](https://www.kaggle.com/dansbecker/learning-to-use-xgboost), the number of trees in XGBoost models, that is `n_estimators`, are tuned by using `early_stopping_rounds`. The early stopping is decided by checking the prediction of the trained models on a validation set, and hence it is required that we pass an `eval_set` alongside the `early_stopping_rounds` in the `fit_params`.

# In[ ]:


searchCV.best_params_ 


# In[ ]:


searchCV.cv_results_['mean_train_score']


# In[ ]:


searchCV.cv_results_['mean_test_score']


# In[ ]:


searchCV.cv_results_['mean_train_score'].mean(), searchCV.cv_results_['mean_test_score'].mean()


# More to come!

# In[ ]:




