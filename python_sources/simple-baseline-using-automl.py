#!/usr/bin/env python
# coding: utf-8

# # Auto Elo  
# 
# In this kernel I try to the power of [AutoML](http://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html from H2O) I stongly believe that these can serve as good starting points.  
# 

# I did not spend time in creating features but I borrowed it from several fantastic starter notebooks below, the top 2  notables ones are:
# [Notebook 1](https://www.kaggle.com/youhanlee/hello-elo-ensemble-will-help-you)
# [Notebook 2](https://www.kaggle.com/fabiendaniel/elo-world)

# In[ ]:


# Import all dependencies
import pandas as pd
import numpy as np
from sklearn import preprocessing
import warnings
import datetime

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import os
import gc

warnings.filterwarnings("ignore")
print(os.listdir("../input"))


# # All File imports

# In[ ]:


train = pd.read_csv("../input/train.csv", parse_dates=["first_active_month"])
test = pd.read_csv("../input/test.csv", parse_dates=["first_active_month"])
hist_trans = pd.read_csv("../input/historical_transactions.csv")
new_trans = pd.read_csv("../input/new_merchant_transactions.csv")


# # Feature Engineering

# In[ ]:


train["month"] = train["first_active_month"].dt.month
test["month"] = test["first_active_month"].dt.month
train["year"] = train["first_active_month"].dt.year
test["year"] = test["first_active_month"].dt.year
train['elapsed_time'] = (datetime.date(2018, 2, 1) - train['first_active_month'].dt.date).dt.days
test['elapsed_time'] = (datetime.date(2018, 2, 1) - test['first_active_month'].dt.date).dt.days
train = pd.get_dummies(train, columns=['feature_1', 'feature_2'])
test = pd.get_dummies(test, columns=['feature_1', 'feature_2'])
#train.head()


# In[ ]:


hist_trans = pd.get_dummies(hist_trans, columns=['category_2', 'category_3'])
hist_trans['authorized_flag'] = hist_trans['authorized_flag'].map({'Y': 1, 'N': 0})
hist_trans['category_1'] = hist_trans['category_1'].map({'Y': 1, 'N': 0})
#hist_trans.head()


# In[ ]:


def aggregate_transactions(trans, prefix):  
    trans.loc[:, 'purchase_date'] = pd.DatetimeIndex(trans['purchase_date']).                                      astype(np.int64) * 1e-9
    
    agg_func = {
        'authorized_flag': ['sum', 'mean'],
        'category_1': ['mean'],
        'category_2_1.0': ['mean'],
        'category_2_2.0': ['mean'],
        'category_2_3.0': ['mean'],
        'category_2_4.0': ['mean'],
        'category_2_5.0': ['mean'],
        'category_3_A': ['mean'],
        'category_3_B': ['mean'],
        'category_3_C': ['mean'],
        'merchant_id': ['nunique'],
        'purchase_amount': ['sum', 'mean', 'max', 'min', 'std'],
        'installments': ['sum', 'mean', 'max', 'min', 'std'],
        'purchase_date': [np.ptp],
        'month_lag': ['min', 'max']
    }
    agg_trans = trans.groupby(['card_id']).agg(agg_func)
    agg_trans.columns = [prefix + '_'.join(col).strip() 
                           for col in agg_trans.columns.values]
    agg_trans.reset_index(inplace=True)
    
    df = (trans.groupby('card_id')
          .size()
          .reset_index(name='{}transactions_count'.format(prefix)))
    
    agg_trans = pd.merge(df, agg_trans, on='card_id', how='left')
    
    return agg_trans


# In[ ]:


merch_hist = aggregate_transactions(hist_trans, prefix='hist_')
del hist_trans
gc.collect()
train = pd.merge(train, merch_hist, on='card_id',how='left')
test = pd.merge(test, merch_hist, on='card_id',how='left')
del merch_hist
gc.collect()

new_trans = pd.get_dummies(new_trans, columns=['category_2', 'category_3'])
new_trans['authorized_flag'] = new_trans['authorized_flag'].map({'Y': 1, 'N': 0})
new_trans['category_1'] = new_trans['category_1'].map({'Y': 1, 'N': 0})
merch_new = aggregate_transactions(new_trans, prefix='new_')
del new_trans
gc.collect()

train = pd.merge(train, merch_new, on='card_id',how='left')
test = pd.merge(test, merch_new, on='card_id',how='left')
del merch_new
gc.collect()


# In[ ]:


IDS = test['card_id']
target = train['target']
drops = ['card_id', 'first_active_month', 'target']
use_cols = [c for c in train.columns if c not in drops]
features = list(train[use_cols].columns)
train[features].head()


# In[ ]:


print(train[features].shape)
print(test[features].shape)


# In[ ]:


train = train[features+['target']]
test = test[features]
print('Final train data shape is:',train.shape)
print('Final test data shape is:', test.shape)


# # Let us bring in H2O now

# In[ ]:


import h2o
from h2o.automl import H2OAutoML


# In[ ]:


# intializing H2o
h2o.init()


# In[ ]:


# Convert pandas datarame into h2o dataframe
htrain = h2o.H2OFrame(train)
htest = h2o.H2OFrame(test)
del train, test
gc.collect()
print(htrain.shape, htest.shape)
print(htrain.head)


# In[ ]:


# Assign x as Independent and y as Dependent
x = htrain.columns
y = "target"
x.remove(y)


# In[ ]:


# Uncomment and run locally to understand the different Parameter
#?? H2OAutoML
"""
H2OAutoML(nfolds=5, balance_classes=False, class_sampling_factors=None, 
max_after_balance_size=5.0, max_runtime_secs=3600, max_models=None, stopping_metric='AUTO', 
stopping_tolerance=None, stopping_rounds=3, seed=None, project_name=None, 
exclude_algos=None, keep_cross_validation_predictions=False, 
keep_cross_validation_models=False, keep_cross_validation_fold_assignment=False, 
sort_metric='AUTO')
"""


# In[ ]:


# Since our goal is AutoML, I am specifying the least parameters
autoelo = H2OAutoML(max_runtime_secs=990000, seed=42)
autoelo.train(x=x, y=y, training_frame=htrain)


# In[ ]:


# Stack up and display the results of the top models, the metrics displayed will be relevant to regression problems
lb = autoelo.leaderboard
lb.head(rows=lb.nrows)


# In[ ]:


# Display the properties of the leader
autoelo.leader


# In[ ]:


# Let us get the predictions against the test data
preds = autoelo.leader.predict(htest)
preds = preds.as_data_frame()


# In[ ]:


# Let us create a submission
sub_df = pd.DataFrame({"card_id":IDS.values, "target": preds.predict.values})
print(sub_df.shape)
sub_df.head()
# Submit the prediction
sub_df.to_csv('AML_sub.csv', index=False)

