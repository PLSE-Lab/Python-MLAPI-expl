#!/usr/bin/env python
# coding: utf-8

# # Load Libraries

# In[ ]:


import pandas as pd
import numpy as numpy
import h2o
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.grid.grid_search import H2OGridSearch
from sklearn import metrics
from sklearn.metrics import roc_auc_score


# In[ ]:


h2o.init()


# # Load Dataset 

# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


data = h2o.import_file("/kaggle/input/pima-indians-diabetes-database/diabetes.csv", destination_frame="data")


# In[ ]:


data.describe()


# # Get target and feature variables names

# In[ ]:


target = "Outcome"
features = [f for f in data.columns if f not in ['Outcome']]


# # Train Test Validation Split

# In[ ]:


train_df, valid_df, test_df = data.split_frame(ratios=[0.8, 0.19], seed=2018)


# In[ ]:


#Get Target data 

train_df[target] = train_df[target].asfactor()
valid_df[target] = valid_df[target].asfactor()
test_df[target] = test_df[target].asfactor()


# # Build Model

# In[ ]:


# define the predictor list - all the features analyzed before (all columns but 'default.payment.next.month')
predictors = features
# initialize the H2O GBM 
gbm = H2OGradientBoostingEstimator()
# train with the initialized model
gbm.train(x=predictors, y=target, training_frame=train_df)


# In[ ]:


gbm.summary()


# # Check Accuracy

# In[ ]:


#Train Accuracy
print(gbm.model_performance(train_df).auc())


# In[ ]:


#Test Accuracy
print(gbm.model_performance(valid_df).auc())


# # Model Tuning

# In[ ]:


tuned_gbm  = H2OGradientBoostingEstimator(
    ntrees = 2000,
    learn_rate = 0.02,
    stopping_rounds = 25,
    stopping_metric = "AUC",
    col_sample_rate = 0.65,
    sample_rate = 0.65,
    seed = 2018
)      
tuned_gbm.train(x=predictors, y=target, training_frame=train_df, validation_frame=valid_df)


# In[ ]:


tuned_gbm.model_performance(valid_df).auc()


# # Grid Search

# In[ ]:


grid_search_gbm = H2OGradientBoostingEstimator(
    stopping_rounds = 25,
    stopping_metric = "AUC",
    col_sample_rate = 0.65,
    sample_rate = 0.65,
    seed = 2018
) 

hyper_params = {
    'learn_rate':[0.01, 0.02, 0.03],
    'max_depth':[4,8,16,24],
    'ntrees':[50, 250, 1000]}

grid = H2OGridSearch(grid_search_gbm, hyper_params,
                         grid_id='depth_grid',
                         search_criteria={'strategy': "Cartesian"})
#Train grid search
grid.train(x=predictors, 
           y=target,
           training_frame=train_df,
           validation_frame=valid_df)


# # Show all models

# In[ ]:


grid_sorted = grid.get_grid(sort_by='auc',decreasing=True)
print(grid_sorted)


# In[ ]:


#Best Model
best_gbm = grid_sorted.models[0]
print(best_gbm)


# # Check Important Variables

# In[ ]:


best_gbm.varimp_plot()


# In[ ]:


pred_val = (best_gbm.predict(test_df[predictors])[0]).as_data_frame()
true_val = (test_df[target]).as_data_frame()
prediction_auc = roc_auc_score(pred_val, true_val)
prediction_auc


# # END
