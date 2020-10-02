#!/usr/bin/env python
# coding: utf-8

# This notebook is my try to use this tutorial 
# https://www.kaggle.com/sudalairajkumar/getting-started-with-h2o?utm_medium=email&utm_source=mailchimp&utm_campaign=datanotes-20180823
# for this dataset

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.simplefilter('ignore')

import os
print(os.listdir("../input"))


# In[ ]:


df = pd.read_csv('../input/WA_Fn-UseC_-Telco-Customer-Churn.csv', index_col='customerID')


# In[ ]:


df.columns


# In[ ]:


df.info()


# In[ ]:


for col in df.select_dtypes(include='object').columns:
    print(col)
    print(df[col].value_counts())


# Convert binary fields to digits

# In[ ]:


df.gender = df.gender.replace({'Male': 1, 'Female':0})
df.Partner = df.Partner.replace({'Yes': 1, 'No': 0})
df.Dependents = df.Dependents.replace({'Yes': 1, 'No': 0})
df.PhoneService = df.PhoneService.replace({'Yes': 1, 'No': 0})
df.PaperlessBilling = df.PaperlessBilling.replace({'Yes': 1, 'No': 0})
df.Churn = df.Churn.replace({'Yes': 1, 'No': 0})


# In[ ]:


yesno_cols = ['MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',  ]
for col in yesno_cols:
    df[col] = [1 if x  == 'Yes' else 0 for x in df[col]]


# In[ ]:


dum_columns = ['InternetService', 'Contract', 'PaymentMethod']
for col in dum_columns:
    dum = pd.get_dummies(df[col], prefix=col)
    df[dum.columns] = dum
df = df.drop(dum_columns, axis = 1)


# It was surprise, that TotalCharges contains spaces. Replace them and convert to float

# In[ ]:


df.TotalCharges = df.TotalCharges.str.replace(' ', '0').astype('float64')


# In[ ]:


df.head()


# **H2O begins!**

# In[ ]:


import h2o
import time
import itertools
import matplotlib.pyplot as plt
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.estimators.random_forest import H2ORandomForestEstimator


# In[ ]:


h2o.init()


# WE can convert to H2Oframe. Mmmmmm...

# In[ ]:


df = h2o.H2OFrame(df)


# **Data Exploration:**

# In[ ]:


df.describe()


# In[ ]:


for col in df.columns:
    df[col].hist()


# In[ ]:


plt.figure(figsize=(20,20))
corr = df.cor().as_data_frame()
corr.index = df.columns
sns.heatmap(corr, annot = True, cmap='RdYlGn', vmin=-1, vmax=1)
plt.title("Correlation Heatmap", fontsize=16)
plt.show()


# Split dataset to train. valid and test - 60\20\20

# In[ ]:


train, valid, test = df.split_frame(ratios=[0.6,0.2], seed=1234)


# In[ ]:


response = "Churn"
train[response] = train[response].asfactor()
valid[response] = valid[response].asfactor()
test[response] = test[response].asfactor()
print("Number of rows in train, valid and test set : ", train.shape[0], valid.shape[0], test.shape[0])


# **Model Tuning**

# In[ ]:


predictors = df.drop('Churn', axis=1).columns


# In[ ]:


predictors


# In[ ]:


gbm = H2OGradientBoostingEstimator()
gbm.train(x=predictors, y=response, training_frame=train)


# In[ ]:


print(gbm)


# In[ ]:


perf = gbm.model_performance(valid)
print(perf)


# In[ ]:


gbm_tune = H2OGradientBoostingEstimator(
    ntrees = 3000,
    learn_rate = 0.01,
    stopping_rounds = 20,
    stopping_metric = "AUC",
    col_sample_rate = 0.7,
    sample_rate = 0.7,
    seed = 42
)      
gbm_tune.train(x=predictors, y=response, training_frame=train, validation_frame=valid)


# In[ ]:


gbm_tune.model_performance(valid).auc()


# In[ ]:


gbm_tune.varimp_plot()


# **Grid Search**

# In[ ]:


from h2o.grid.grid_search import H2OGridSearch

gbm_grid = H2OGradientBoostingEstimator(
    ntrees = 3000,
    learn_rate = 0.01,
    stopping_rounds = 20,
    stopping_metric = "AUC",
    col_sample_rate = 0.7,
    sample_rate = 0.7,
    seed = 1234
) 

hyper_params = {'max_depth':[4,6,8,10,12]}
grid = H2OGridSearch(gbm_grid, hyper_params,
                         grid_id='depth_grid',
                         search_criteria={'strategy': "Cartesian"})
#Train grid search
grid.train(x=predictors, 
           y=response,
           training_frame=train,
           validation_frame=valid)


# In[ ]:


print(grid)


# In[ ]:


sorted_grid = grid.get_grid(sort_by='auc',decreasing=True)
print(sorted_grid)


# **K-Fold cross validation***

# In[ ]:


cv_gbm = H2OGradientBoostingEstimator(
    ntrees = 3000,
    learn_rate = 0.05,
    stopping_rounds = 20,
    stopping_metric = "AUC",
    nfolds=4, 
    seed=2018)
cv_gbm.train(x = predictors, y = response, training_frame = train, validation_frame=valid)
cv_summary = cv_gbm.cross_validation_metrics_summary().as_data_frame()
cv_summary


# In[ ]:


cv_gbm.model_performance(valid).auc()


# **XGBoost**

# In[ ]:


from h2o.estimators import H2OXGBoostEstimator

cv_xgb = H2OXGBoostEstimator(
    ntrees = 3000,
    learn_rate = 0.05,
    stopping_rounds = 20,
    stopping_metric = "AUC",
    nfolds=4, 
    seed=2018)
cv_xgb.train(x = predictors, y = response, training_frame = train, validation_frame=valid)
cv_xgb.model_performance(valid).auc()


# In[ ]:


cv_xgb.varimp_plot()


# In[ ]:


from h2o.automl import H2OAutoML

aml = H2OAutoML(max_models = 10, max_runtime_secs=600, seed = 1)
aml.train(x=predictors, y=response, training_frame=train, validation_frame=valid)


# In[ ]:


lb = aml.leaderboard
lb


# In[ ]:


aml.predict(test)


# In[ ]:




