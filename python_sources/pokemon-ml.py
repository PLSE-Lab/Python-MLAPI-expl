#!/usr/bin/env python
# coding: utf-8

# ### POKEMON MACHINE LEARNING - Predict Pokemon HP stats
# 
# #### regression models: 1) Random Forest 2) XGBoost 
# 
# Steps:
# - understand datasets, choose features and target
# - ensure no data leakage in any features
# - study correlations between X & Y, if the relationship is weak, it is unlikely to contribute to the prediction
# 
# - data prep, casting, removing un-helpful columns
# - split train/test datasets
# - decide how to impute/encode numerical OR categorical data
# 
# - create pipelines for preprocessor and model
# - random forest model: experiment different parameters, compare MAE, use CV to get overall MAE, list importance
# - XGBoost model: experiment different parameters, compare MAE, use CV to get overall MAE, list importance
# 

# ## Understand Dataset
# **.columns** & **.describe()** - Min, Max, Average, etc

# In[ ]:


import pandas as pd
import numpy as np
data = pd.read_csv('../input/pokemon/Pokemon.csv')
data.columns
data.describe()


# ## Study Correlations to Identify Noise
# Feature vs Feature : High = redundant = noise. (Action = Drop)<br>
# Feature vs Target  : Very low = no contribution to prediction = noise.

# In[ ]:


data.corr()


# ## Choosing Target - HP (Health Power)

# In[ ]:


y = data.HP
X_full = data
X_sample = X_full.drop(['HP'], axis=1)
# X_sample['Type 2'].fillna(value='None',inplace=True) #enable if no combine
X_sample.head()


# ## Choosing Features
# #### 1. Base stats (Atk, Sp.Atk, Def, Sp.Def, Speed) 
# (Numerical) Use value as is. Impute missing values with MEAN value. 
# 
# #### 2. Is it Legendary? (Legendary) 
# (Bool) Convert boolean to Integer 0/1. Use value as is.
# 
# #### 3. Pokemon Types (Type 1 & Type 2)
# (Categorical) One-hot encode. Missing values = 0.

# In[ ]:


### Legendary
X_sample.Legendary = X_sample.Legendary.astype(int)

### Pokemon Type

# Get all unique types from column "type 1"
types = np.unique(data['Type 1'].values)

# Prepare columns for one-hot (combine)
# for t in types:
#     X_sample.insert(12, f'Type_{t}', 0)
    
# # One hot type 1 & type 2
# for i, r in data.iterrows():
#     type_1 = r['Type 1']
#     X_sample.at[i, f'Type_{type_1}'] = 1
    
#     type_2 = r['Type 2']
       
#     if not type_2 == 'nan':
#         X_sample.at[i, f'Type_{type_2}'] = 1
        
X_sample.columns


# ## Cleaning & Dropping Features
# 1. Unused values: **Index(#)**, **Name**
# 
# 2. Very low correlation: **Generation** - Does not correlate to any base stats
# 
# 3. Data leakage: **Total** - The sum of all base stats including HP. So it will be not available when model use for prediction.
# 
# 4. Leftovers after one-hot

# In[ ]:


# X_sample = X_sample.drop(['Name', '#', 'Generation', 'Total', 'Type 1', 'Type 2', 'Type_nan'], axis=1)
X_sample = X_sample.drop(['Name', '#', 'Generation', 'Total'], axis=1)
X_sample.head()


# ## Split Train/Test Data

# In[ ]:


from sklearn.model_selection import train_test_split

# One Hot Using pd.get_dummies()
X_sample = pd.get_dummies(X_sample)
print(X_sample.columns)

X_train, X_valid, y_train, y_valid = train_test_split(X_sample, y, train_size=0.8, test_size=0.2, random_state=0)
X_full_train, X_full_valid, y_full_train, y_full_valid = train_test_split(X_full, y, train_size=0.8, test_size=0.2, random_state=0)


# ## Identifying Features to Impute/Encode
# #### 1. Numerical columns
# impute missing values, some use MEAN, some use ZERO
# 
# #### 2. Categorical columns
# one hot encoding

# In[ ]:


# import math
# boolean_cols = [cname for cname in X_train.columns if X_train[cname].dtype == 'bool']
numerical_cols = [cname for cname in X_train.columns if X_train[cname].dtype in ['int64', 'float64'] 
                  and 'Type' not in cname ]
onehot_cols = [cname for cname in X_train.columns if 'Type' in cname ]
categorical_cols = [cname for cname in X_train.columns if X_train[cname].dtype == 'object']
cols_with_missing = [cols for cols in numerical_cols if X_train[cols].isnull().any()]

# print("Boolean columns: ", boolean_cols)
print("Numerical columns: ", numerical_cols)
print("OneHot columns: ", onehot_cols)
print("Categorical columns: ", categorical_cols)
print("Columns with missing values: ", cols_with_missing)


# Type 2 missing simply means the Pokemon does not have a secondary type

# ## Model 1 - Random Forest
# #### pipeline(preprocessor + model), train, predict, cross-validation

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error

# Preprocessor
oh_transformer = SimpleImputer(strategy='constant', fill_value=0)
num_transformer = SimpleImputer(strategy='mean')
cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value=None)),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
preprocessor = ColumnTransformer(
transformers=[
    ('oh', oh_transformer, onehot_cols),
    ('num', num_transformer, numerical_cols),
    ('cat', cat_transformer, categorical_cols)
])

# Model
rf_model = RandomForestRegressor(n_estimators=250, criterion='mae', random_state=0, n_jobs=-1)

# Pipeline
# rf_pipeline = Pipeline(steps=[('model', rf_model)])
rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', rf_model)])

# Fit model
rf_pipeline.fit(X_train, y_train)

# Predict
rf_preds = rf_pipeline.predict(X_valid)


# Need to decide how to impute categorical column "type 2". Currently I impute with empty string.

# In[ ]:


# Result
print("Model#1: RandomForest - Performance Summary\n----------")

rf_score = mean_absolute_error(y_valid, rf_preds)
print('MAE score (80/20 train-test split) :', rf_score)

from sklearn.model_selection import cross_val_score
rf_scores = -1 * cross_val_score(rf_pipeline, X_sample, y, cv=5, scoring='neg_mean_absolute_error')
# print("MAE scores (5CVs): ", rf_scores)
print("MAE score (Cross Validation)       :", rf_scores.mean())
print("Overall MAE =", round(rf_scores.mean(), 2))


# OneHotEncoder: 14.02 || pd.get_dummies: 14.04
# 
# ### Feature Importance (Random Forest)

# In[ ]:


rf_importances = pd.DataFrame({'feature': X_valid.columns, 'rf_importance': np.round(rf_model.feature_importances_,3)})
rf_importances = rf_importances.sort_values('rf_importance', ascending=False).set_index('feature')
rf_importances[:10]


# ## Model 2 - XG Boost
# #### pipeline(model), train, predict, cross-validation

# In[ ]:


from xgboost import XGBRegressor
# from sklearn.impute import SimpleImputer
# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error

# Temporarily remove preprocessor because it throws error otherwise

# Model
xgb_model = XGBRegressor(
            n_estimators=100,
            learning_rate=0.05, n_jobs=2, verbosity=0, random_state=0)

xgb_pipeline = Pipeline(steps=[('model', xgb_model)])
# xgb_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', xgb_model)])

# Fit model
xgb_pipeline.fit(X_train, y_train, 
                 model__early_stopping_rounds=3,
                 model__eval_set=[(X_valid, y_valid)],
                 model__verbose=0)

# Predict
xgb_preds = xgb_pipeline.predict(X_valid)


# In[ ]:


# Result
print("Model#2: XGBoost - Performance Summary\n----------")
xgb_score = mean_absolute_error(y_valid, xgb_preds)
print('MAE score (80/20 train-test split) :', xgb_score)

from sklearn.model_selection import cross_val_score
xgb_scores = -1 * cross_val_score(xgb_pipeline, X_sample, y, cv=5, scoring='neg_mean_absolute_error', verbose=0, error_score=0)
# print("MAE scores (5CVs): ", rf_scores)
print("MAE score (Cross Validation)       :", xgb_scores.mean())
print("Overall MAE =", round(xgb_scores.mean(), 2))


# OneHotEncoder: error || pd.get_dummies: 13.83
# 
# ### Feature Importance (XG Boost)

# In[ ]:


rf_importances = pd.DataFrame({'feature': X_valid.columns, 'xgb_importance': np.round(xgb_model.feature_importances_,3)})
rf_importances = rf_importances.sort_values('xgb_importance', ascending=False).set_index('feature')
rf_importances[:10]


# In[ ]:


# Display first 5 predictions & actual result
print("First 5 Predictions (Random Forest): ", rf_preds.tolist()[:5])
print("First 5 Predictions (XGBoost)      : ", xgb_preds.tolist()[:5])
print("First 5 Actual result              : ", y_valid.tolist()[:5])
X_full_valid.head()


# ## Conclusion
# * XG Boost slightly perform better than Random Forest
# * Both models are not accurate 
#     * Insufficient model tuning? Wrong params?
#     * Not enough features? -> explore other datasets
