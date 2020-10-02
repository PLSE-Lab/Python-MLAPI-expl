#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import lightgbm as lightgbm

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import recall_score

mpl.style.use('seaborn')
np.set_printoptions(precision=4, suppress=True)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


# ## Introduction

# [LightGBM](https://github.com/Microsoft/LightGBM) is a fast, distributed, high performance gradient boosting framework based on decision tree algorithms open-sources by Microsoft. The library is used extensively in Kaggle competitions, and often forms part of the [winning solution](https://github.com/Microsoft/LightGBM/tree/master/examples).
# 
# This notebook gives an introduction to the using LightGBM, illustrating a few advance features and giving an overview of the parameters of the algorithm. For those interested in understanding gradient boosting a bit better, an overview of the technique is given here: [https://www.avanwyk.com/an-overview-of-lightgbm/](https://www.avanwyk.com/an-overview-of-lightgbm/).
# 
# The examples below will be at the hand of a classification task: we will attempt to detect credit card fraud. An overview of the dataset is given [here](https://www.kaggle.com/mlg-ulb/creditcardfraud). The dataset is highly imbalanced (there are very few positive examples, relative to the negative examples), an area within which GBDTs (Gradient Boosted Decision Trees) excel.

# ## Load and pre-process data
# 
# We normalize the `Amount` feature and also drop the `Time` feature as it is not useful for our analysis. Additionally, the dataset is very imbalanced, however, GBDTs are well suited for imbalanced datasets, provided class weights are given. Finally we split our data into training and validation datasets.

# In[ ]:


from sklearn.preprocessing import StandardScaler as StandardScaler
from sklearn.model_selection import train_test_split

data = pd.read_csv('../input/creditcard.csv')
data.head(10)
data['NormalizedAmount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1,1))
data = data.drop(['Time', 'Amount'], axis=1)

positive_percentage = data[data['Class'] == 1].shape[0]/data.shape[0] * 100
print("{:.2f}% of the data are positive examples (highly skewed).".format(positive_percentage))

X = data.drop('Class', axis=1)
y = data['Class']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.33)


# ## LightGBM

# In[ ]:


import lightgbm as lgb

# Wrap our training and validation sets in LightGBM Datasets.
lgb_train = lgb.Dataset(X_train, y_train, free_raw_data=False)
lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train, free_raw_data=False)


# ### Basic Usage

# Below is the core parameters driving the training of a gradient boosted machine, with a brief explanation of each. A full explanation of the core parameters are given [here](https://lightgbm.readthedocs.io/en/latest/Parameters.html#core-parameters).

# In[ ]:


core_params = {
    'boosting_type': 'gbdt', # GBM type: gradient boosted decision tree, rf (random forest), dart, goss.
    'objective': 'binary', # the optimization object: binary, regression, multiclass, xentropy.
    'learning_rate': 0.05, # the gradient descent learning or shrinkage rate, controls the step size.
    'num_leaves': 31, # the number of leaves in one tree.
    'nthread': 4, # number of threads to use for LightGBM, best set to number of actual cores.
    
    'metric': 'auc' # an additional metric to calculate during validation: area under curve (auc).
}


# Now we can train a Gradient Boosted Decision Tree using LightGBM.
# We wrap the training call in a function that trains the GBDT, plots the results of the training for us and returns the GBM and the validation results per iteration.

# In[ ]:


def train_gbm(params, training_set, validation_set, init_gbm=None, boost_rounds=100, early_stopping_rounds=0, metric='auc'):
    evals_result = {} 

    gbm = lgb.train(params, # parameter dict to use
                    training_set,
                    init_model=init_gbm, # initial model to use, for continuous training.
                    num_boost_round=boost_rounds, # the boosting rounds or number of iterations.
                    early_stopping_rounds=early_stopping_rounds, # early stopping iterations.
                    # stop training if *no* metric improves on *any* validation data.
                    valid_sets=validation_set,
                    evals_result=evals_result, # dict to store evaluation results in.
                    verbose_eval=False) # print evaluations during training.
    
    y_true = validation_set.label
    y_pred = gbm.predict(validation_set.data)
    fpr, tpr, threshold = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    
    plt.title("ROC Curve. Area under Curve: {:.3f}".format(roc_auc))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    _ = plt.plot(fpr, tpr, 'r')
    
    return gbm, evals_result


# In[ ]:


model, evals = train_gbm(core_params, lgb_train, lgb_val)


# Our initial model isn't doing very well. However, there are many parameters that we could tune to improve performance, speed up the training or reduce overfitting.

# ### Parameters

# Below we specify and explain a host of model parameters, with some being tuned to improve the model accuracy. A complete list of algorithm parameters is given [here](https://lightgbm.readthedocs.io/en/latest/Parameters.html).

# In[ ]:


advanced_params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'binary_logloss',
    
    'learning_rate': 0.01,
    'num_leaves': 41, # more leaves increases accuracy, but may lead to overfitting.
    
    'max_depth': 5, # the maximum tree depth. Shallower trees reduce overfitting.
    'min_split_gain': 0, # minimal loss gain to perform a split
    'min_child_samples': 21, # or min_data_in_leaf: specifies the minimum samples per leaf node.
    'min_child_weight': 5, # minimal sum hessian in one leaf. Controls overfitting.
    
    'lambda_l1': 0.5, # L1 regularization
    'lambda_l2': 0.5, # L2 regularization
    
    'feature_fraction': 0.5, # randomly select a fraction of the features before building each tree.
    # Speeds up training and controls overfitting.
    'bagging_fraction': 0.5, # allows for bagging or subsampling of data to speed up training.
    'bagging_freq': 0, # perform bagging on every Kth iteration, disabled if 0.
    
    'scale_pos_weight': 99, # add a weight to the positive class examples (compensates for imbalance).
    
    'subsample_for_bin': 200000, # amount of data to sample to determine histogram bins
    'max_bin': 1000, # the maximum number of bins to bucket feature values in.
    # LightGBM autocompresses memory based on this value. Larger bins improves accuracy.
    
    'nthread': 4, # number of threads to use for LightGBM, best set to number of actual cores.
}


# In[ ]:


get_ipython().run_cell_magic('time', '', 'model, evals = train_gbm(advanced_params, lgb_train, lgb_val, boost_rounds=500)')


# 

# Our parameter tuning helped and the model performs significantly better. The parameters used above improves the model's accuracy, but we could improve the model's speed by setting a `bagging_freq` and using a lower `max_bin`.

# ## Additional Features

# ### Continous Training
# 
# A model's training can be continued by passing an existing model as the `init_model` parameter to the training function.

# In[ ]:


model, evals = train_gbm(advanced_params, lgb_train, lgb_val, init_gbm=model, boost_rounds=500)


# ### Plotting
# 
# LightGBM has a number of useful plotting functions built in:

# In[ ]:


_ = lgb.plot_metric(evals) # training metrics


# In[ ]:


_ = lgb.plot_importance(model) # feature importance


# In[ ]:


_ = lgb.plot_tree(model, figsize=(20, 20)) # built trees


# ### Persistence
# LightGBM models can easily be saved and loaded to a file, or JSON:

# In[ ]:


model.save_model('cc_fraud_model.txt')

loaded_model = lgb.Booster(model_file='cc_fraud_model.txt')

# Output to JSON
model_json = model.dump_model()

