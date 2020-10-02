#!/usr/bin/env python
# coding: utf-8

# # Neptune tracking example
# 
# I will use the parametrers from the  ['Magic Parameters' kernel](https://www.kaggle.com/sandeepkumar121995/magic-parameters)
# 
# To get a better picture of what Neptune is, go to this [Medium blog post](http://bit.ly/2HtXtMH).
# 
# Let's start by importing the usual stuff.

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import lightgbm as lgb
import warnings
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from scikitplot.metrics import plot_confusion_matrix, plot_roc
from tqdm import tqdm

warnings.filterwarnings('ignore')


# and loading the data.

# In[ ]:


train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')


# ## Step 0
# Go to [neptune.ml](http://bit.ly/2FndEZO) and register.
# It is absolutely free, no card or anything required.
# 
# ## Step 1
# Initialize Neptune. Set the project name and authorization.
# 
# The recommended way is to create the`NEPTUNE_API_TOKEN` environment variable and pass your account token there.

# In[ ]:


import os
os.environ['NEPTUNE_API_TOKEN'] = 'your_long_api_token_goes_here'


# In[ ]:


import neptune

neptune.init(project_qualified_name='jakub-czakon/santander')


# ## Step 2
# 
# Define hyperparameters. Put everything you care about in one dictionary.

# In[ ]:


NAME = 'Magic Parameters'

N_SPLITS = 15
SEED = 1234

TRAIN_PARAMS = {
        'num_boosting_rounds': 1000000,
        'early_stopping_rounds' : 4000
        }

MODEL_PARAMS = {'bagging_freq': 5,
         'bagging_fraction': 0.335,
         'boost_from_average':'false',
         'boost': 'gbdt',
         'feature_fraction': 0.041,
         'learning_rate': 0.1,
         'max_depth': -1,
         'metric':'auc',
         'min_data_in_leaf': 80,
         'min_sum_hessian_in_leaf': 10.0,
         'num_leaves': 13,
         'num_threads': 8,
         'tree_learner': 'serial',
         'objective': 'binary',
         'verbosity': 1,
                     }

params = {**MODEL_PARAMS, **TRAIN_PARAMS}


# ## Step 3
# 
# Create an experiment and run training
# 
# In order to log stuff to neptune you need to create an experiment:
# 
#     with neptune.create_experiment():
# 
# and then simply log stuff like metrics or images to neptune:
# 
#         neptune.send_metric('roc_auc', roc_auc_oof)
#         ...
#         neptune.send_image('model_diagnostics', 'model_diagnostics.png')
# 
# **Optional (but cool)**
# 
# Prepare stuff for custom logging. 
#  1. **Lightgbm monitoring**:
#      I like to monitor my lightgbm training and compare the learning curves, so I want to create a `neptune_monitor` callback and  look at the charts as it trains.
#  1. **Model diagnoscs**:
#     I want to have a clear(er) picture of the situation so I log confusion matrix, ROC AUC curve and prediction distrubitions after every run. 

# In[ ]:


def neptune_monitor(prefix):
    def callback(env):
        for name, loss_name, loss_value, _ in env.evaluation_result_list:
            channel_name = '{}{}_{}'.format(prefix, name, loss_name)
            neptune.send_metric(channel_name, x=env.iteration, y=loss_value)
    return callback


def plot_prediction_distribution(y_true, y_pred, ax):
    df = pd.DataFrame({'prediction': y_pred, 'ground_truth': y_true})
    
    sns.distplot(df[df['ground_truth'] == 0]['prediction'], label='negative', ax=ax)
    sns.distplot(df[df['ground_truth'] == 1]['prediction'], label='positive', ax=ax)

    ax.legend(prop={'size': 16}, title = 'Labels')


# In[ ]:


with neptune.create_experiment(name=NAME,
                               params=params):

    folds = StratifiedKFold(n_splits=N_SPLITS, shuffle=False, random_state=SEED)
    
    features = [c for c in train_df.columns if c not in ['ID_code', 'target']]
    oof, predictions = np.zeros(len(train_df)), np.zeros(len(test_df))
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_df.values, train_df['target'].values)):
        print("Fold {}".format(fold_))
        trn_data = lgb.Dataset(train_df.iloc[trn_idx][features], 
                                         label=train_df['target'].iloc[trn_idx])
        val_data = lgb.Dataset(train_df.iloc[val_idx][features], 
                    label=train_df['target'].iloc[val_idx])

        monitor = neptune_monitor(prefix='fold{}_'.format(fold_))
        clf = lgb.train(MODEL_PARAMS, trn_data, 
                        TRAIN_PARAMS['num_boosting_rounds'], 
                        valid_sets = [trn_data, val_data], 
                        verbose_eval=5000, 
                        early_stopping_rounds = TRAIN_PARAMS
                        ['early_stopping_rounds'],
                        callbacks=[monitor])
        oof[val_idx] = clf.predict(train_df.iloc[val_idx][features], num_iteration=clf.best_iteration)
        predictions += clf.predict(test_df[features], num_iteration=clf.best_iteration) / folds.n_splits
    roc_auc_oof = roc_auc_score(train_df['target'], oof)
    print("CV score: {:<8.5f}".format(roc_auc_oof))
    neptune.send_metric('roc_auc', roc_auc_oof)

    preds = pd.DataFrame(oof, columns=['pos_preds'])
    preds['neg_preds'] = 1.0 - preds['pos_preds']
    fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(24, 6))
    plot_prediction_distribution(train_df['target'], preds['pos_preds'], ax=ax1);
    plot_roc(train_df['target'], preds[['neg_preds','pos_preds']], ax=ax2);
    plot_confusion_matrix(train_df['target'], oof>0.5, ax=ax3);
    fig.savefig('model_diagnostics.png') 
    neptune.send_image('model_diagnostics', 'model_diagnostics.png')

pd.DataFrame({"ID_code": train_df.ID_code.values, 'target':oof}).to_csv("oof_{}.csv".format(NAME), index=False)
pd.DataFrame({"ID_code": test_df.ID_code.values, 'target':predictions}).to_csv("submission_{}.csv".format(NAME), index=False)


# # Step 4
# 
# Go to the Experiment in Neptune -> https://ui.neptune.ml/jakub-czakon/santander/e/SAN1-59/charts.
# And see your training:
# 
# ![image](https://gist.githubusercontent.com/jakubczakon/f754769a39ea6b8fa9728ede49b9165c/raw/c8425bb2244200dcb86b8cf850db87696acc0322/kaggel_kernel1.png)
# 
# If you log more experiments you can compare them and stuff:
# 
# ![image](https://gist.githubusercontent.com/jakubczakon/f754769a39ea6b8fa9728ede49b9165c/raw/c8425bb2244200dcb86b8cf850db87696acc0322/kaggle_kernel2.png)
# 
# 
# 
# 
# 

# In[ ]:




