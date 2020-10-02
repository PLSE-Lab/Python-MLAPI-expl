#!/usr/bin/env python
# coding: utf-8

# # Introduction
# **This is the workspace for the [Machine Learning course](https://www.kaggle.com/learn/machine-learning).**
# 
# I've translated the concepts to work with the data in this notebook, the Iowa data. Each page in the Machine Learning course includes instructions for what code to write at that step in the course.
# 
# # Iowa Housing price prediction

# These are the modules you need.

# In[ ]:


import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import Imputer, OneHotEncoder
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.ensemble.partial_dependence import partial_dependence, plot_partial_dependence
from sklearn.pipeline import make_pipeline
from xgboost import XGBRegressor


# In[ ]:


data = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test_data = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
print(data.describe())
print(test_data.describe())


# In[ ]:


print(data.columns)
print(test_data.columns)


# Our target column is SalePrice.

# In[ ]:


y = data.SalePrice
print(y.head())


# In[ ]:


X = data.drop(['SalePrice','Id'], axis=1)
test_data_bak = test_data.copy()
print(test_data_bak.shape)
test_data = test_data.drop(['Id'], axis=1)
print(X.head())


# Check datatypes of all the columns.

# In[ ]:


print(X.dtypes)


# Let's handle for missing data.

# In[ ]:


missing_cols = [col for col in X.columns if X[col].isnull().any()]
X = X.drop(missing_cols, axis=1)
test_data = test_data.drop(missing_cols, axis=1)


# We should convert the categorical data, i.e., object datatypes into numerical data. It might contain important useful features. The number of categories in a categorical column shouldn't be too high, it may lead to overfitting. 

# In[ ]:


numeric_cols = [col for col in X.columns if X[col].dtype in ['int64','float64']]
low_cardinality_cols = [col for col in X.columns if X[col].nunique() < 10 and X[col].dtype == 'object']
train_X = X[numeric_cols + low_cardinality_cols]
test_X = test_data[numeric_cols + low_cardinality_cols] 


# Let's get dummies.

# In[ ]:


train_X_dummies = pd.get_dummies(train_X)
test_X_dummies = pd.get_dummies(test_X)
print(train_X_dummies.shape)
print(test_X_dummies.shape)


# We need to make the number of columns same for making them usable by the model.

# In[ ]:


final_train, final_test = train_X_dummies.align(test_X_dummies, join="inner", axis=1)
print(final_train.shape)
print(final_test.shape)
print(final_train.columns)
print(final_test.columns)


# Function for finding MAE through Cross Validation Scores.

# In[ ]:


def get_mae_by_cv(X, y):
    return (-1)*cross_val_score(RandomForestRegressor(50), X, y, scoring="neg_mean_absolute_error").mean()


# In[ ]:


print(get_mae_by_cv(final_train, y))


# We can use pipelines instead of the commented lines above for imputing.

# In[ ]:


xgb_pipeline = make_pipeline(Imputer(), XGBRegressor())
xgb_pipeline.fit(final_train, y)
cv_results_xgb = (-1)*cross_val_score(xgb_pipeline, final_train, y, scoring="neg_mean_absolute_error").mean()
print("XGB",cv_results_xgb)
xgb_pipeline_parameterized = make_pipeline(Imputer(), XGBRegressor(n_estimators=135, learning_rate=0.05))
xgb_pipeline_parameterized.fit(final_train, y)
cv_results_xgb_parameterized = (-1)*cross_val_score(xgb_pipeline_parameterized, final_train, y, scoring="neg_mean_absolute_error").mean()
print("XGB_parameterized",cv_results_xgb_parameterized)


# Let's predict.

# In[ ]:


predicted_values = xgb_pipeline_parameterized.predict(final_test)


# In[ ]:


print(*list(final_train.columns),sep='\n') #LotArea, OverallQual, YearBuilt
analyse_these_cols = ['LotArea','OverallQual','YearBuilt']


# We can only plot dependency graphs if we had already modelled and fitted our training data w.r.t. those features to be examined. And, let's use pipelines.

# In[ ]:


training_col_under_analysis = final_train[analyse_these_cols]
gbmodel = GradientBoostingRegressor()
gb_pipeline = make_pipeline(Imputer(), gbmodel)
gb_pipeline.fit(training_col_under_analysis, y)
plots = plot_partial_dependence(gbmodel, 
                                features=[0,1,2], 
                                X = training_col_under_analysis, 
                                feature_names = analyse_these_cols, 
                                grid_resolution=10)


# In[ ]:


my_submission = pd.DataFrame({'Id': test_data_bak.Id, 'SalePrice': predicted_values})


# In[ ]:


my_submission.to_csv('submission.csv', index=False)


# **Ask me in case if you needed any help. **
# 
# **Check the course contents. [ML Course Index](https://www.kaggle.com/learn/machine-learning)**

# 
