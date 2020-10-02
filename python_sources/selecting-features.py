#!/usr/bin/env python
# coding: utf-8

# # Feature selecture using target permutation
# 
# This notebook is a straightforward adaptation of [Olivier's kernel](https://www.kaggle.com/ogrellier/feature-selection-with-null-importances), where Olivier proposes a methodology to select the most relevant features of the model. As outlined by [Peter Hurford](https://www.kaggle.com/c/elo-merchant-category-recommendation/discussion/73937), just keeping the right features may help you model to score better. To obtain exhaustive details on the method's implementation, you should refer to the original kernel. **By the way, if you feel like upvoting this kernel, please upvote first the original one !**
# 
# ### Notebook  Content
# 1. [Creating a scoring function](#1)
# 1. [Build the benchmark for feature importance](#2)
# 1. [Display distribution examples](#3)
# 1. [Score features](#4)
# 1. [Save data](#5)

# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import lightgbm as lgb
from sklearn.model_selection import KFold
import warnings
import time
import sys
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
import gc
get_ipython().run_line_magic('matplotlib', 'inline')
warnings.simplefilter(action='ignore', category=FutureWarning)
gc.enable()


# First, we load the data, which has been pre-procesed in [another kernel](https://www.kaggle.com/fabiendaniel/elo-world):

# In[ ]:


train = pd.read_csv("../input/elo-world/train.csv", index_col=0)


# ## 1. Create a scoring function

# In[ ]:


def get_feature_importances(data, shuffle, seed=None):
    # Gather real features
    train_features = [f for f in data if f not in ['target', 'card_id', 'first_active_month']]
    categorical_feats = [c for c in train_features if 'feature_' in c]
    # Go over fold and keep track of CV score (train and valid) and feature importances
    
    # Shuffle target if required
    y = data['target'].copy()
    if shuffle:
        # Here you could as well use a binomial distribution
        y = data['target'].copy().sample(frac=1.0)
    
    # Fit LightGBM in RF mode, yes it's quicker than sklearn RandomForest
    dtrain = lgb.Dataset(data[train_features], y, free_raw_data=False, silent=True)
    lgb_params = {
        'num_leaves': 129,
        'min_data_in_leaf': 148, 
        'objective':'regression',
        'max_depth': 9,
        'learning_rate': 0.005,
        "min_child_samples": 24,
        "boosting": "gbdt",
        "feature_fraction": 0.7202,
        "bagging_freq": 1,
        "bagging_fraction": 0.8125 ,
        "bagging_seed": 11,
        "metric": 'rmse',
        "lambda_l1": 0.3468,
        "random_state": 133,
        "verbosity": -1
    }
    
    # Fit the model
    clf = lgb.train(params=lgb_params,
                    train_set=dtrain,
                    num_boost_round=850,
                   # categorical_feature=categorical_feats
                   )

    # Get feature importances
    imp_df = pd.DataFrame()
    imp_df["feature"] = list(train_features)
    imp_df["importance_gain"] = clf.feature_importance(importance_type='gain')
    imp_df["importance_split"] = clf.feature_importance(importance_type='split')
    imp_df['trn_score'] = mean_squared_error(clf.predict(data[train_features]), y)**0.5
    
    return imp_df


# ## 2. Build the benchmark for feature importance

# In[ ]:


# Seed the unexpected randomness of this world
np.random.seed(123)
# Get the actual importance, i.e. without shuffling
actual_imp_df = get_feature_importances(data=train, shuffle=False)


# In[ ]:


actual_imp_df.sort_values('importance_gain', ascending=False)[:10]


# In[ ]:


null_imp_df = pd.DataFrame()
nb_runs = 100
import time
start = time.time()
dsp = ''
for i in range(nb_runs):
    # Get current run importances
    imp_df = get_feature_importances(data=train, shuffle=True)
    imp_df['run'] = i + 1 
    # Concat the latest importances with the old ones
    null_imp_df = pd.concat([null_imp_df, imp_df], axis=0)
    # Erase previous message
    for l in range(len(dsp)):
        print('\b', end='', flush=True)
    # Display current run and time used
    spent = (time.time() - start) / 60
    dsp = 'Done with %4d of %4d (Spent %5.1f min)' % (i + 1, nb_runs, spent)
    print(dsp, end='', flush=True)


# ## 3. Display distribution examples
# 
# A few plots are better than any words

# In[ ]:


def display_distributions(actual_imp_df_, null_imp_df_, feature_):
    plt.figure(figsize=(13, 6))
    gs = gridspec.GridSpec(1, 2)
    # Plot Split importances
    ax = plt.subplot(gs[0, 0])
    a = ax.hist(null_imp_df_.loc[null_imp_df_['feature'] == feature_, 'importance_split'].values,
                label='Null importances')
    ax.vlines(x=actual_imp_df_.loc[actual_imp_df_['feature'] == feature_, 'importance_split'].mean(), 
               ymin=0, ymax=np.max(a[0]), color='r',linewidth=10, label='Real Target')
    ax.legend()
    ax.set_title('Split Importance of %s' % feature_.upper(), fontweight='bold')
    plt.xlabel('Null Importance (split) Distribution for %s ' % feature_.upper())
    # Plot Gain importances
    ax = plt.subplot(gs[0, 1])
    a = ax.hist(null_imp_df_.loc[null_imp_df_['feature'] == feature_, 'importance_gain'].values, label='Null importances')
    ax.vlines(x=actual_imp_df_.loc[actual_imp_df_['feature'] == feature_, 'importance_gain'].mean(), 
               ymin=0, ymax=np.max(a[0]), color='r',linewidth=10, label='Real Target')
    ax.legend()
    ax.set_title('Gain Importance of %s' % feature_.upper(), fontweight='bold')
    plt.xlabel('Null Importance (gain) Distribution for %s ' % feature_.upper())
        


# In[ ]:


actual_imp_df.sort_values('importance_gain', ascending=False)[:10]


# In[ ]:


display_distributions(actual_imp_df_=actual_imp_df, null_imp_df_=null_imp_df, feature_='new_installments_min')


# In[ ]:


display_distributions(actual_imp_df_=actual_imp_df, null_imp_df_=null_imp_df, feature_='new_month_lag_min')


# In[ ]:


display_distributions(actual_imp_df_=actual_imp_df, null_imp_df_=null_imp_df, feature_='auth_category_1_sum')


# In[ ]:


display_distributions(actual_imp_df_=actual_imp_df, null_imp_df_=null_imp_df, feature_='hist_month_diff_mean')


# From the above plot, **as stated by Olivier**, the power of the exposed feature selection method is demonstrated. In particular it is well known that :
#  - Any feature sufficient variance can be used and made sense of by tree models. You can always find splits that help scoring better
#  - Correlated features have decaying importances once one of them is used by the model. The chosen feature will have strong importance and its correlated suite will have decaying importances
#  
#  The current method allows to :
#   - Drop high variance features if they are not really related to the target
#   - Remove the decaying factor on correlated features, showing their real importance (or unbiased importance)

# ## 4. Score features
# 
# There are several ways to score features : 
#  - Compute the number of samples in the actual importances that are away from the null importances recorded distribution.
#  - Compute ratios like Actual / Null Max, Actual  / Null Mean,  Actual Mean / Null Max
#  
# Here, **following Olivier,** we use the log actual feature importance divided by the 75 percentile of null distribution.

# In[ ]:


feature_scores = []
max_features = 300
for _f in actual_imp_df['feature'].unique():
    f_null_imps_gain = null_imp_df.loc[null_imp_df['feature'] == _f, 'importance_gain'].values
    f_act_imps_gain = actual_imp_df.loc[actual_imp_df['feature'] == _f, 'importance_gain'].mean()
    gain_score = np.log(1e-10 + f_act_imps_gain / (1 + np.percentile(f_null_imps_gain, 75)))  
    f_null_imps_split = null_imp_df.loc[null_imp_df['feature'] == _f, 'importance_split'].values
    f_act_imps_split = actual_imp_df.loc[actual_imp_df['feature'] == _f, 'importance_split'].mean()
    split_score = np.log(1e-10 + f_act_imps_split / (1 + np.percentile(f_null_imps_split, 75)))  
    feature_scores.append((_f, split_score, gain_score))

scores_df = pd.DataFrame(feature_scores, columns=['feature', 'split_score', 'gain_score'])

plt.figure(figsize=(16, 25))
gs = gridspec.GridSpec(1, 2)
# Plot Split importances
ax = plt.subplot(gs[0, 0])
sns.barplot(x='split_score', y='feature', data=scores_df.sort_values('split_score', ascending=False).iloc[0:max_features], ax=ax)
ax.set_title('Feature scores wrt split importances', fontweight='bold', fontsize=14)
# Plot Gain importances
ax = plt.subplot(gs[0, 1])
sns.barplot(x='gain_score', y='feature', data=scores_df.sort_values('gain_score', ascending=False).iloc[0:max_features], ax=ax)
ax.set_title('Feature scores wrt gain importances', fontweight='bold', fontsize=14)
plt.tight_layout()


# ## 4. Save the data

# In[ ]:


null_imp_df.to_csv('null_importances_distribution_rf.csv')
actual_imp_df.to_csv('actual_importances_ditribution_rf.csv')


# In[ ]:




