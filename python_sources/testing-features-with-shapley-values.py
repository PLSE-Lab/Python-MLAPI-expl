#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import gc 

import scipy
import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import FastICA
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.random_projection import GaussianRandomProjection
from sklearn.random_projection import SparseRandomProjection
from sklearn.manifold import TSNE
from sklearn.preprocessing import scale

import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# # 1. Feature Engineering
# ## 1.1. Data Loading & Pre-processing
# Summary:
# * Remove columns with zero variance
# * Remove duplicate columns and rows
# * Log-transform all columns
# * Mean-variance scale all columns excepting sparse entries

# In[ ]:


# Read train and test files
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')

# Find and drop duplicate rows
t = train_df.iloc[:,2:].duplicated(keep=False)
duplicated_indices = t[t].index.values
print("Removed {} duplicated rows: {}".format(len(duplicated_indices), duplicated_indices))
train_df.iat[duplicated_indices[0], 1] = np.expm1(np.log1p(train_df.target.loc[duplicated_indices]).mean()) # keep and update first with log mean
train_df.drop(duplicated_indices[1:], inplace=True) # drop remaining

# Get the combined data
total_df = pd.concat([train_df.drop('target', axis=1), test_df], axis=0).drop('ID', axis=1)

# Get the target
y = np.log1p(train_df.target)

# Train and test
train_idx = range(0, len(train_df))
test_idx = range(len(train_df), len(total_df))

# Columns to drop because there is no variation in training set
zero_std_cols = train_df.drop("ID", axis=1).columns[train_df.std() == 0]
total_df.drop(zero_std_cols, axis=1, inplace=True)
print("Removed {} constant columns".format(len(zero_std_cols)))

# Removing duplicate columns
_, unique_indices = np.unique(total_df.iloc[train_idx], axis=1, return_index=True)
colsToRemove = [i for i in range(total_df.shape[1]) if i not in unique_indices]
total_df = total_df.iloc[:, unique_indices]
print("Dropped {} duplicate columns: {}".format(len(colsToRemove), colsToRemove))

# Log-transform all column
total_df = np.log1p(total_df)

# Scale non-zero column values
for col in total_df.columns:    
    nonzero_rows = total_df[col] != 0
    total_df.loc[nonzero_rows, col] = scale(total_df.loc[nonzero_rows, col].values)


# ## 1.2. Aggregates
# Some of the suggested aggregation features...

# In[ ]:


aggregate_df = pd.DataFrame()

n0_df = total_df[total_df>0]

# V1 Features
aggregate_df['n0_count'] = total_df.astype(bool).sum(axis=1)
aggregate_df['n0_mean'] = n0_df.mean(axis=1)
aggregate_df['n0_median'] = n0_df.median(axis=1)
aggregate_df['n0_kurt'] = n0_df.kurt(axis=1)
aggregate_df['n0_min'] = n0_df.min(axis=1)
aggregate_df['n0_std'] = n0_df.std(axis=1)
aggregate_df['n0_skew'] = n0_df.skew(axis=1)
aggregate_df['mean'] = total_df.mean(axis=1)
aggregate_df['std'] = total_df.std(axis=1)
aggregate_df['max'] = total_df.max(axis=1)
aggregate_df['nunique'] = total_df.nunique(axis=1)
aggregate_df['sum_zeros'] = (total_df == 0).astype(int).sum(axis=1)
aggregate_df['geometric_mean'] = n0_df.apply(
    lambda x: np.exp(np.log(x).mean()), axis=1
)
del n0_df

aggregate_df.reset_index(drop=True, inplace=True)
print("Created features for: {}".format(aggregate_df.columns.values))


# ## 1.3. Unsupervised Feature Learning
# ### 1.3.1. Decomposition Methods
# Lots of people have been using decomposition methods to reduce the number of features. From my trials in [this notebook](https://www.kaggle.com/nanomathias/linear-regression-with-elastic-net), it seems like often it's only the first 10-20 components that are actually important for the modeling. Since we are testing features now, here I'll include 10 of each decomposition method.

# In[ ]:


COMPONENTS = 10

# Convert to sparse matrix
sparse_matrix = scipy.sparse.csr_matrix(total_df.values)

# V1 List of decomposition methods
methods = [
    {'method': TSNE(n_components=3, init='pca'), 'data': 'train'},
    {'method': TruncatedSVD(n_components=COMPONENTS), 'data': 'sparse'},
    {'method': PCA(n_components=COMPONENTS), 'data': 'total'},
    {'method': FastICA(n_components=COMPONENTS), 'data': 'total'},
    {'method': GaussianRandomProjection(n_components=COMPONENTS, eps=0.1), 'data': 'total'},
    {'method': SparseRandomProjection(n_components=COMPONENTS, dense_output=True), 'data': 'total'}
]

# Run all the methods
embeddings = []
for run in methods:
    name = run['method'].__class__.__name__
    
    # Run method on appropriate data
    if run['data'] == 'sparse':
        embedding = run['method'].fit_transform(sparse_matrix)
    elif run['data'] == 'train':
        embedding = run['method'].fit_transform(total_df.iloc[train_idx])
    else:
        embedding = run['method'].fit_transform(total_df)
        
    # Save in list of all embeddings
    embeddings.append(
        pd.DataFrame(embedding, columns=[f"{name}_{i}" for i in range(embedding.shape[1])])
    )
    print(f">> Ran {name}")
    
# Put all components into one dataframe
components_df = pd.concat(embeddings, axis=1).reset_index(drop=True)


# ### 1.3.2. Dense Autoencoder
# I saw a few people use autoencoders, but here I just implement a very simple one. From empirical tests it seems that the components I extract from this it doesn't make sense to have an embedded dimension higher than about 10, in terms of local CV score on the target at the end at least. 
# 
# These features do decently well, so I think it's worth investigating further in terms of hyperparameter tuning.

# In[ ]:


from keras.layers import *
from keras.optimizers import *
from keras.models import Model, Sequential

encoding_dim = 5

enc_input = Input((total_df.shape[1], ))
enc_output = Dense(128, activation='relu')(enc_input)
enc_output = Dropout(0.5)(enc_output)
enc_output = Dense(encoding_dim, activation='relu')(enc_output)

dec_input = Dense(128, activation='relu')(enc_output)
dec_output = Dropout(0.5)(dec_input)
dec_output = Dense(total_df.shape[1], activation='relu')(dec_output)

# This model maps an input to its reconstruction
vanilla_encoder = Model(enc_input, enc_output)
vanilla_autoencoder = Model(enc_input, dec_output)
vanilla_autoencoder.compile(optimizer=Adam(0.0001), loss='mean_squared_error')
vanilla_autoencoder.summary()

# Fit the autoencoder
vanilla_autoencoder.fit(
    total_df.values, total_df.values,
    epochs=6, batch_size=64,
    shuffle=True
)

# Put into dataframe
dense_ae_df = pd.DataFrame(
    vanilla_encoder.predict(total_df.values, batch_size=64), 
    columns=['dense_AE_{}'.format(i) for i in range(encoding_dim)]
).reset_index(drop=True)


# ## 1.4 Time series features
# from: https://www.kaggle.com/hmendonca/training-data-analyzes-time-series

# In[ ]:


cols = ['f190486d6', '58e2e02e6', 'eeb9cd3aa', '9fd594eec', '6eef030c1',
        '15ace8c9f', 'fb0f5dbfe', '58e056e12', '20aa07010', '024c577b9',
        'd6bb78916', 'b43a7cfd5', '58232a6fb', '1702b5bf0', '324921c7b',
        '62e59a501', '2ec5b290f', '241f0f867', 'fb49e4212', '66ace2992',
        'f74e8f13d', '5c6487af1', '963a49cdc', '26fc93eb7', '1931ccfdd',
        '703885424', '70feb1494', '491b9ee45', '23310aa6f', 'e176a204a',
        '6619d81fc', '1db387535', 'fc99f9426', '91f701ba2', '0572565c2',
        '190db8488', 'adb64ff71', 'c47340d97', 'c5a231d81', '0ff32eb98']


# In[ ]:


def moving_avg(df, prefix, win_size):
    print('Creating rolling average on {} columns'.format(win_size))
    ts = df.T.rolling(win_size*2, min_periods=1, center=True, win_type='gaussian').mean(std=win_size/2).iloc[ win_size//2 : : win_size ].T # rolling average
    ts.columns = [prefix+str(n) for n in np.arange(ts.columns.size)]
    ## also calculate moving deltas
    dts = pd.DataFrame(ts.iloc[:,:-1].values - ts.iloc[:,1:].values)
    dts.columns = ['d_'+prefix+str(n) for n in np.arange(dts.columns.size)]
    return ts.join(dts)

def moving_max(df, prefix, win_size):
    print('Creating rolling max on {} columns'.format(win_size))
    ts = df.T.rolling(win_size*2, min_periods=1, center=True).max().iloc[ win_size//2 : : win_size ].T # rolling max
    ts.columns = [prefix+str(n) for n in np.arange(ts.columns.size)]
    ## also calculate moving deltas
    dts = pd.DataFrame(ts.iloc[:,:-1].values - ts.iloc[:,1:].values)
    dts.columns = ['d_'+prefix+str(n) for n in np.arange(dts.columns.size)]
    return ts.join(dts)

rolling_df = pd.DataFrame()    
## yearly, quartally, monthly, fortnightly, weekly mean
# rolling_df = pd.concat([rolling_df, moving_avg(total_df, 'y_', 365)], axis=1)
# rolling_df = pd.concat([rolling_df, moving_avg(total_df, 'q_', 91)], axis=1)
rolling_df = pd.concat([rolling_df, moving_avg(total_df[cols], 'm_', 30)], axis=1)
rolling_df = pd.concat([rolling_df, moving_avg(total_df[cols], 'f_', 15)], axis=1)
rolling_df = pd.concat([rolling_df, moving_avg(total_df[cols], 'd_', 7)], axis=1)
##  ... and max
# rolling_df = pd.concat([rolling_df, moving_max(total_df, 'ym_', 365)], axis=1)
# rolling_df = pd.concat([rolling_df, moving_max(total_df, 'qm_', 91)], axis=1)
rolling_df = pd.concat([rolling_df, moving_max(total_df[cols], 'mm_', 30)], axis=1)
rolling_df = pd.concat([rolling_df, moving_max(total_df[cols], 'fm_', 15)], axis=1)
rolling_df = pd.concat([rolling_df, moving_max(total_df[cols], 'dm_', 7)], axis=1)

rolling_df.reset_index(inplace=True, drop=True)


# In[ ]:


# Put all features into one dataframe (i.e. aggregate, timeseries, components)
feature_df = pd.concat([components_df, aggregate_df, dense_ae_df, rolling_df], axis=1).fillna(0)


# ## 2. Lightgbm test + feature importance

# In[ ]:


from sklearn.model_selection import train_test_split
import lightgbm as lgb

dev_X, val_X, dev_y, val_y = train_test_split(feature_df.iloc[train_idx], y, test_size = 0.15, random_state = 42)

def run_lgb(train_X, train_y, val_X, val_y, test_X=None):
    params = {
        "objective" : "regression",
        "metric" : "rmse",
        "num_leaves" : 140,
        #"n_estimators" : 700,
        "max_depth" : 13,
        "max_bin" : 55,
        "learning_rate" : 0.005,
        "feature_fraction" : 0.9,
        "bagging_fraction" : 0.8,
        "bagging_frequency" : 5,
        "bagging_seed" : 42,
        "verbosity" : -1,
        "seed": 42
    }
    
    lgtrain = lgb.Dataset(train_X, label=train_y)
    lgval = lgb.Dataset(val_X, label=val_y)
    evals_result = {}
    model = lgb.train(params, lgtrain, 5000, 
                      valid_sets=[lgval], 
                      early_stopping_rounds=100, 
                      verbose_eval=200, 
                      evals_result=evals_result)
    
    if test_X is not None:
        pred_test_y = np.expm1(model.predict(test_X, num_iteration=model.best_iteration))
        return pred_test_y, model, evals_result
    else:
        return model, evals_result

#pred_test, model, evals_result = run_lgb(dev_X, dev_y, val_X, val_y, feature_df.iloc[test_idx])
model, evals_result = run_lgb(dev_X, dev_y, val_X, val_y)
print("LightGBM Training Completed...")


# In[ ]:


# feature importance
gain = model.feature_importance('gain')
featureimp = pd.DataFrame({
        'feature':model.feature_name(),
        'gain': gain / gain.sum()
    }).sort_values('gain', ascending=False)

plt.figure(figsize=(15,20))
sns.barplot(x="gain", y="feature", data=featureimp[:50])
plt.title('LightGBM Feature Importance (gain)')
plt.tight_layout()


# In[ ]:


# feature importance (split)
split = model.feature_importance('split')
featureimp = pd.DataFrame({
        'feature':model.feature_name(), 
        'split': split / split.sum()
    }).sort_values('split', ascending=False)

plt.figure(figsize=(15,20))
sns.barplot(x="split", y="feature", data=featureimp[:50])
plt.title('LightGBM Feature Importance (split)')
plt.tight_layout()


# ## Shapley values
# https://towardsdatascience.com/interpretable-machine-learning-with-xgboost-9ec80d148d27

# In[ ]:


import shap, xgboost

# train XGBoost model
# xgb_model = xgboost.train({"learning_rate": 0.01}, xgboost.DMatrix(dev_X, label=dev_y), 100)

# explain the model's predictions using SHAP values
# (same syntax works for LightGBM, CatBoost, and scikit-learn models)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(dev_X)

# feature importance
shap.summary_plot(shap_values, dev_X, plot_type="bar", max_display=50)


# In[ ]:


# load JS visualization code to notebook
shap.initjs()
# visualize the first prediction's explanation
shap.force_plot(shap_values[0,:], dev_X.iloc[0,:])


# In[ ]:


# summarize the effects of top features
shap.summary_plot(shap_values, dev_X, max_display=50)


# In[ ]:


from sklearn.feature_selection import SelectPercentile
percentile = 20
skb = SelectPercentile(score_func=lambda X,y: np.mean(np.abs(shap_values[:, :dev_X.shape[1]]), axis=0), percentile=percentile).fit(dev_X.values, dev_y.values)
print("Using {} features".format(skb.transform(val_X.values).shape[1]))
pred_test, s_model, s_evals_result = run_lgb(skb.transform(dev_X.values), dev_y,
                                         skb.transform(val_X.values), val_y,
                                         skb.transform(feature_df.iloc[test_idx].values))


# In[ ]:


sub = pd.read_csv('../input/sample_submission.csv')
sub["target"] = pred_test
print(sub.head())
sub.to_csv('feat_lgb{:.3f}.csv'.format(evals_result['valid_0']['rmse'][-1]), index=False)

