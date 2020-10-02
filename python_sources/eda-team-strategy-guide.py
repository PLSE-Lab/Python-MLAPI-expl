#!/usr/bin/env python
# coding: utf-8

# # Visualise what features are important
# Why should you read through this kernel? The goal is to have a visual guide on which strategy leads to the win:
# 
# - the data will be read and memory footprint will be reduced;
# - aggregations of the data over teams are performed;
# - a baseline **LightGBM** model **on team level** will be trained (for detailed code on player level [see my other kernel](https://www.kaggle.com/mlisovyi/pubg-survivor-kit));
# - the training is implemented with a simple train/test split;
# - **use [SHAP package](https://github.com/slundberg/shap) for model explanation**;
# - **use [LIME package](https://github.com/marcotcr/lime) (described in the [paper](https://arxiv.org/abs/1602.04938)) for model explanation**

# We will use only a subset of games (=matches) to speed-up processing, as SHAP is very slow (LIME is somewhat faster, as it does linear models locally)

# In[ ]:


# The number of MATCHES to use in training. Whole training dataset is used anyway. Use it to have fast turn-around. Set to 50k for all entries
max_matches_trn=5000
# The number of entries from test to read in. Use it to have fast turn-around. Set to None for all entries
max_events_tst=None
# Number on CV folds
n_cv=3


# Define a function to reduce memory foorprint

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.simplefilter(action='ignore', category=Warning)

from sklearn.metrics import mean_squared_error, mean_absolute_error

import os
print(os.listdir("../input"))

def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object and col_type.name != 'category' and 'datetime' not in col_type.name:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        elif 'datetime' not in col_type.name:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df


# Read in the data

# In[ ]:


df_trn = pd.read_csv('../input/train.csv', nrows=None)
df_trn = reduce_mem_usage(df_trn)

df_trn = df_trn.query('matchId < @max_matches_trn')
print('Number of training entries after selecting a subset of matches: {}'.format(df_trn.shape[0]))
# we will NOT use in training
features_not2use = ['Id', 'groupId', 'matchId', 'numGroups']


# # Feature engineering: group by teams

# In[ ]:


agg_team = {c: ['mean', 'min', 'max', 'sum'] for c in [c for c in df_trn.columns if c not in features_not2use and c != 'winPlacePerc']}
agg_team['numGroups'] = ['size']
print(agg_team.keys())

def preprocess(df):    
    df_gb = df.groupby('groupId').agg(agg_team)
    df_gb.columns = pd.Index([e[0] + "_" + e[1].upper() for e in df_gb.columns])    
    return df_gb

df_trn_gb = preprocess(df_trn)
#this is needed, since for some teams sum of rideDistance is infinite. This is not swallowed by LIME
df_trn_gb = df_trn_gb.replace({np.inf: -1})

y = df_trn.groupby('groupId')['winPlacePerc'].median()

# since we train on the group and out final metric is on user level, we want to assign group size as the weight
w = df_trn_gb['numGroups_SIZE']


# # Simple train/test split

# In[ ]:


from sklearn.model_selection import train_test_split
X_trn, X_tst, y_trn, y_tst = train_test_split(df_trn_gb, y, test_size=0.33, random_state=42)


# # Train a model
# Start by defining handy helper functions...

# In[ ]:


get_ipython().run_cell_magic('time', '', '\nimport lightgbm as lgb\nfrom sklearn.base import clone\ndef train_single_model(clf_, X_, y_, random_state_=314, opt_parameters_={}, fit_params_={}):\n    \'\'\'\n    A wrapper to train a model with particular parameters\n    \'\'\'\n    c = clone(clf_)\n    c.set_params(**opt_parameters_)\n    c.set_params(random_state=random_state_)\n    return c.fit(X_, y_, **fit_params_)\n\nmdl_ = lgb.LGBMRegressor(max_depth=-1, min_child_samples=400, random_state=314, silent=True, metric=\'None\', \n                                        n_jobs=4, n_estimators=5000, learning_rate=0.1)\n\nfit_params_ = {"early_stopping_rounds":100, \n               "eval_metric" : \'mae\',\n               \'eval_names\': [\'train\', \'early_stop\'],\n               \'verbose\': 500,\n               \'eval_set\': [(X_trn,y_trn), (X_tst,y_tst)],\n               \'sample_weight\': y_trn.index.map(w).values,\n               \'eval_sample_weight\': [None, y_tst.index.map(w).values]\n              }\nopt_parameters_ = {\'objective\': \'mae\', \'colsample_bytree\': 0.75, \'min_child_weight\': 10.0, \'num_leaves\': 30, \'reg_alpha\': 1}\n                \nmdl = train_single_model(mdl_, X_trn, y_trn, \n                         fit_params_=fit_params_,\n                         opt_parameters_=opt_parameters_\n                        )')


# # Model interpretation with SHAP

# In[ ]:


import shap
shap.initjs()


# In[ ]:


get_ipython().run_cell_magic('time', '', 'explainer=shap.TreeExplainer(mdl.booster_)\nshap_values = explainer.shap_values(X_tst)')


# Visualise what effect features have on the final prediction. Quoting the SHAP github:
# 
# > The plot below sorts features by the sum of SHAP value magnitudes over all samples, and uses SHAP values to show the distribution of the impacts each feature has on the model output. The color represents the feature value (**red high, blue low**). 
# 
# This reveals for example that a high `walkDistance_MEAN` (average distance walked by team members) increases the predicted chance of winning (the `winPlacePerc`).

# In[ ]:


shap.summary_plot(shap_values, X_tst)


# Let's also look at the impact of various features on the predictions for each individual team. Quoting the documentation again:
# 
# > [The plot below]... shows features each contributing to push the model output from the base value (the average model output over the training dataset we passed) to the model output. Features pushing the prediction higher are shown in red, those pushing the prediction lower are in blue.
# 
# Note that the plot is actually interactive, so you can see names of variables, if you put cursor on individual components.

# In[ ]:


for i in range(5):
    display(shap.force_plot(explainer.expected_value, shap_values[i,:], X_trn.iloc[i,:]))


# And finally, let's look how different feature interactions affect the predicted placement. Quoting the documentation:
# 
# > To understand how a single feature effects the output of the model we can plot **the SHAP value of that feature vs. the value of the feature for all the examples in a dataset**. Since SHAP values represent a feature's responsibility for a change in the model output, the plot below represents **the change in predicted placement as either of `'killPlace_MAX', 'walkDistance_MEAN', 'weaponsAcquired_MIN'`changes**. 
# 
# Note that This plot also shows the strongest interaction on the feature with another feature in the dataset:
# 
# > Vertical dispersion at a single value of the X axis represents interaction effects with other features. To help reveal these interactions `dependence_plot` automatically selects another feature for coloring. 
# 
# In this case coloring by features on the Z axis highlights interactions.

# In[ ]:


for f in ['killPlace_MAX', 'walkDistance_MEAN', 'weaponsAcquired_MIN']:
    shap.dependence_plot(f, shap_values, X_tst)


# # Model interpretation with LIME

# In[ ]:


import lime
from  lime.lime_tabular import LimeTabularExplainer


# Note, that LIME seems to work with nupy arrays only and does to digest pandas objects. So we will use `pd.DataFrame.values`

# In[ ]:


explainer = LimeTabularExplainer(X_trn.values, 
                                 feature_names=X_trn.columns, 
                                 class_names=[], 
                                 verbose=True, 
                                 mode='regression')


# Build explanations for the first 5 examples

# In[ ]:


exp= []
for i in range(5):
    exp.append(explainer.explain_instance(X_tst.iloc[i,:].values, mdl.predict, num_features=10))


# Visualise which cuts were most important in the decision making for those 5 examples

# In[ ]:


for e in exp:
    _ = e.as_pyplot_figure()


# Visualise explanation for those 5 examples

# In[ ]:


for e in exp:
    _ = e.show_in_notebook()


# In[ ]:




