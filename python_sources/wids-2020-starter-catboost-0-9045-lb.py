#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from catboost import Pool, cv, CatBoostClassifier, CatBoostRegressor

from sklearn.metrics import mean_squared_error, classification_report
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

# import xgboost
# import lightgbm as lgb
# from lightgbm import LGBMClassifier

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
import gc
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import metrics
pd.set_option('max_rows', 300)
import re

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

pd.set_option('display.max_columns', 300)
np.random.seed(566)
pd.set_option('display.max_rows', 200)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', '{:20,.2f}'.format)
pd.set_option('display.max_colwidth', -1)


# ## Load train/test data
# * If doing feature engineering, we could combine the 2 dataframes together then split them after. But we'll go for a naive approach in this kernel

# In[ ]:


TARGET_COL = "hospital_death"


# In[ ]:


df = pd.read_csv("/kaggle/input/widsdatathon2020/training_v2.csv")
print(df.shape)
display(df.nunique())
df.head()


# In[ ]:


df.isna().sum()


# In[ ]:


df.describe()


# In[ ]:


test = pd.read_csv("/kaggle/input/widsdatathon2020/unlabeled.csv")
print(test.shape)
display(test.nunique())
test.head()


# In[ ]:


test.isna().sum()


# * `encounter_id ,	patient_id	` are unique. We could probably drop, but there may be leaks from them so we'll keep (And we need them for the submission)
# * Their being unique means they are not candidates for generating historical features from them per patient. 
#     * We could try the hospital ID or surgery type or apache codes for that purpose

# ### if using cartboost or lgbm, we can define categorical variables
# 
# * catboost hyperparam tuning : https://colab.research.google.com/github/catboost/tutorials/blob/master/python_tutorial.ipynb#scrollTo=nSteluuu_mif
# 
# 
# * We see that many clearly continous numeric variables have relatively low cardinality (e.g. icu sensor readings) - making it tricky to define them automatically. 
# * Categorical columns are not necessarily string columns, could be numerical - e.g. hospital codes.. 
# 

# In[ ]:


print([c for c in df.columns if 7<df[c].nunique()<800])
## 
# categorical_cols = ['hospital_id','apache_3j_bodysystem', 'apache_2_bodysystem',
# "hospital_admit_source","icu_id","ethnicity"]


# In[ ]:


## print non numeric columns : We may need to
## define them as categorical / encode as numeric with label encoder, depending on ml model used
print([c for c in df.columns if (1<df[c].nunique()) & (df[c].dtype != np.number)& (df[c].dtype != int) ])


# In[ ]:


categorical_cols =  ['hospital_id',
 'ethnicity', 'gender', 'hospital_admit_source', 'icu_admit_source', 'icu_stay_type', 'icu_type', 'apache_3j_bodysystem', 'apache_2_bodysystem']

#['apache_3j_bodysystem', 'apache_2_bodysystem',
# "hospital_admit_source","icu_id","ethnicity"]


# ## Build & Tune catboost/GBM models
# * We could make a validation subset for early stopping - allowing us to more easily tune our models hyperparameters
# *  we can go with a defualt model for now - it gets good results ,as we'll see
# 
# * We can run on the GPU, giving a speed boost - to do this modify the kaggle kernel to use the GPU, and in the `fit` parameters set `task_type = "GPU"`
# 
# * Catboost and lgbm have `Pool`/`Dataset` objects, that can be used "internally" by them for some functions, e.g. to efficienctly CV
# 
# 

# In[ ]:


display(df[categorical_cols].dtypes)
display(df[categorical_cols].tail(3))
display(df[categorical_cols].isna().sum())


# * We Fill in empty string for missing  values in the string columns , otherwise catboost will give an error - "CatBoostError: Invalid type for cat_feature: cat_features must be integer or string, real number values and NaN values should be converted to string."
# 

# In[ ]:


df[categorical_cols] = df[categorical_cols].fillna("")

# same transformation for test data
test[categorical_cols] = test[categorical_cols].fillna("")

df[categorical_cols].isna().sum()


# ## Train model(s)

# In[ ]:


## useful "hidden" function - df._get_numeric_data()  - returns only numeric columns from a pandas dataframe. Useful for scikit learn models! 

X_train = df.drop([TARGET_COL],axis=1)
y_train = df[TARGET_COL]


# In[ ]:


## catBoost Pool object
train_pool = Pool(data=X_train,label = y_train,cat_features=categorical_cols,
#                   baseline= X_train[""], ## 
#                   group_id = X_train['hospital_id']
                 )

### OPT/TODO:  do train test split for early stopping then add that as an eval pool object : 


# ## Train a basic model

# In[ ]:


model_basic = CatBoostClassifier(verbose=False,iterations=50)#,learning_rate=0.1, task_type="GPU",)
model_basic.fit(train_pool, plot=True,silent=True)
print(model_basic.get_best_score())


# ### Hyperparameter search
# * We can do a gridsearch for best hyperparameters, such as learning rate, etc' 
#      * Another improvement: split evaluation set from train , and use it for early stopping + tun
#       * I leave this to the reader :)  

# In[ ]:


### hyperparameter tuning example grid for catboost : 
grid = {'learning_rate': [0.04, 0.1],
        'depth': [7, 11],
#         'l2_leaf_reg': [1, 3,9],
#        "iterations": [500],
       "custom_metric":['Logloss', 'AUC']}

model = CatBoostClassifier()

## can also do randomized search - more efficient typically, especially for large search space - `randomized_search`
grid_search_result = model.grid_search(grid, 
                                       train_pool,
                                       plot=True,
                                       refit = True, #  refit best model on all data
                                      partition_random_seed=42)

print(model.get_best_score())


# In[ ]:


print("best model params: \n",grid_search_result["params"])


# ## Features importances
# 
# * What are the most important features for predicting death? 
# 
# * We also look also at Shapley values : https://github.com/slundberg/shap
#     * Shap + Catboost tutorial :  https://github.com/slundberg/shap/blob/master/notebooks/tree_explainer/Catboost%20tutorial.ipynb
#       
# 

# In[ ]:


feature_importances = model.get_feature_importance(train_pool)
feature_names = X_train.columns
for score, name in sorted(zip(feature_importances, feature_names), reverse=True):
    if score > 0.05:
        print('{0}: {1:.2f}'.format(name, score))


# In[ ]:


import shap
shap.initjs()

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(train_pool)

# visualize the training set predictions
# SHAP plots for all the data is very slow, so we'll only do it for a sample. Taking the head instead of a random sample is dangerous! 
shap.force_plot(explainer.expected_value,shap_values[0,:400], X_train.iloc[0,:400])


# In[ ]:


# summarize the effects of all the features
shap.summary_plot(shap_values, X_train)


# * As we would expect, age is important. 
# * The top features are precalculated predictors of risk of death (which likely take age into account). 
# 
# * We see that there's a difference between hospitals , although it's not an especially clear or linear feature. One explanation may be differences in skill of departments/doctors inside each hospital, with these "latent"/hidden variables interacting with other factors in our dataset

# # Get predictions on test set and export for submission
# 
# * You can "ensemble"/average the predictions from the 2 catboost models as a quick improvement , even if there isn't much diversity added

# In[ ]:


test[TARGET_COL] = model.predict(test.drop([TARGET_COL],axis=1),prediction_type='Probability')[:,1]


# In[ ]:


test[["encounter_id","hospital_death"]].to_csv("submission.csv",index=False)

