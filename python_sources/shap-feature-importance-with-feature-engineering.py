#!/usr/bin/env python
# coding: utf-8

# ## Explaining your model with SHAP values
# 
# Most of the models optimized for performance, like GBMs or Neural Networks are black-box models.
# For those models gaining an intuition of what is happening inside - how certain features influence its output, may be difficult.
# Even though for LightGBM or XGBoost feature importance can be given, it isn't always reliable. More details about why this is the case can be found in a great [article](https://towardsdatascience.com/interpretable-machine-learning-with-xgboost-9ec80d148d27) by one of SHAP authors.
# A proposed remedy is using [**SHAP**](https://github.com/slundberg/shap), which provides a unified approach for intepreting output of machine learning methods.
# Detailed information about how it works can be found in the above linked article and their [NIPS paper](http://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions).
# 
# Being able to explain model output is especially important in cases, when it is going to be deployed as a service or application. Knowing whether it learns useful features is crucial. 
# On tabular data, where domain knowledge is the key, one can check feature importance and deduce, if most important features according to the model can influence the predicted values in reality. Based on this, model reliability can be estimated. To give a basic example, when trying to predict price of a house, factors such as it's area or location should directly influence the price. If instead of those, a model will consider totally random feature important (like, let's say, mean temperature of a random month during last year), it is easy to arrive at a conclusion that something is definitely wrong.
# 
# In case of competitions, being able to interpret model output gives a direction of feature engineering, which is worth following.
# Here, we do not care about importance making sense this much but rather creating new features further improving model performance.
# When we know which of the features are considered important, there is a chance that features derived from those will provide model with additional information, enabling a boost in score.
# Another thing to consider is the differences between distribution of certain features in train and test data. If train set feature distribution is very different from its test set distribution and it is considered very important by a model, there may be a risk that model performance on the test set may be not very reliable.
# 
# Now,  let's check what kind of information SHAP will give us in case of the Two Sigma competition!
# Because the aim of the kernel is to showcase model importance output, basis for feature engineering is taken from a kernel, which scores well on LB. 
# 
# #### Data processing & feature engineering taken from: https://www.kaggle.com/qqgeogor/eda-script-67. Thanks!

# In[ ]:


import gc
import glob
import os
import random
import time
from datetime import date, datetime

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn import model_selection
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


pd.set_option("display.max_columns", 96)
pd.set_option("display.max_rows", 96)

plt.rcParams['figure.figsize'] = (12, 9)
plt.style.use('ggplot')

shap.initjs()


# In[ ]:


# Size of the dataset is limited to just n_debug_samples, because SHAP calculation is quite time-consuming.

debug = True
n_debug_samples = 10000


# In[ ]:


from multiprocessing import Pool


def create_lag(df_code, n_lag=[3, 7, 14, ], shift_size=1):
    code = df_code['assetCode'].unique()

    for col in return_features:
        for window in n_lag:
            rolled = df_code[col].shift(shift_size).rolling(window=window)
            lag_mean = rolled.mean()
            lag_max = rolled.max()
            lag_min = rolled.min()
            lag_std = rolled.std()
            df_code['%s_lag_%s_mean' % (col, window)] = lag_mean
            df_code['%s_lag_%s_max' % (col, window)] = lag_max
            df_code['%s_lag_%s_min' % (col, window)] = lag_min

    return df_code.fillna(-1)


def generate_lag_features(df, n_lag=[3, 7, 14]):
    
    
    features = ['time', 'assetCode', 'assetName', 'volume', 'close', 'open',
                'returnsClosePrevRaw1', 'returnsOpenPrevRaw1',
                'returnsClosePrevMktres1', 'returnsOpenPrevMktres1',
                'returnsClosePrevRaw10', 'returnsOpenPrevRaw10',
                'returnsClosePrevMktres10', 'returnsOpenPrevMktres10',
                'returnsOpenNextMktres10', 'universe']

    assetCodes = df['assetCode'].unique()
    print(assetCodes)
    all_df = []
    df_codes = df.groupby('assetCode')
    df_codes = [df_code[1][['time', 'assetCode'] + return_features]
                for df_code in df_codes]
    print('total %s df' % len(df_codes))

    pool = Pool(4)
    all_df = pool.map(create_lag, df_codes)

    new_df = pd.concat(all_df)
    new_df.drop(return_features, axis=1, inplace=True)
    pool.close()

    return new_df


def mis_impute(data):
    for i in data.columns:
        if data[i].dtype == "object":
            data[i] = data[i].fillna("other")
        elif (data[i].dtype == "int64" or data[i].dtype == "float64"):
            data[i] = data[i].fillna(data[i].mean())
        else:
            pass
    return data


def data_prep(market_train):
    lbl = {k: v for v, k in enumerate(market_train['assetCode'].unique())}
    market_train['assetCodeT'] = market_train['assetCode'].map(lbl)
    market_train = market_train.dropna(axis=0)
    return market_train


def exp_loss(p, y):
    y = y.get_label()
    grad = -y * (1.0 - 1.0 / (1.0 + np.exp(-y * p)))
    hess = -(np.exp(y * p) * (y * p - 1) - 1) / ((np.exp(y * p) + 1)**2)
    return grad, hess


# In[ ]:


from kaggle.competitions import twosigmanews
env = twosigmanews.make_env()
print('Done!')

(market_train_df, news_train_df) = env.get_training_data()


# In[ ]:


market_train_df['time'] = market_train_df['time'].dt.date
market_train_df = market_train_df.loc[market_train_df['time'] >= date(
    2010, 1, 1)]


return_features = ['returnsClosePrevMktres10',
                   'returnsClosePrevRaw10', 'open', 'close']
n_lag = [3, 7, 14]
new_df = generate_lag_features(market_train_df, n_lag=n_lag)
market_train_df = pd.merge(market_train_df, new_df,
                           how='left', on=['time', 'assetCode'])

market_train_df = mis_impute(market_train_df)
market_train_df = data_prep(market_train_df)


# In[ ]:


if debug:
    market_train_df = market_train_df.iloc[:n_debug_samples, :]


up = market_train_df['returnsOpenNextMktres10'] >= 0
universe = market_train_df['universe'].values
d = market_train_df['time']


fcol = [c for c in market_train_df if c not in [
    'assetCode', 'assetCodes', 'assetCodesLen', 'assetName', 'audiences',
    'firstCreated', 'headline', 'headlineTag', 'marketCommentary', 'provider',
    'returnsOpenNextMktres10', 'sourceId', 'subjects', 'time', 'time_x',
    'universe', 'sourceTimestamp']]


X = market_train_df[fcol]
up = up.values
r = market_train_df.returnsOpenNextMktres10.values

mins = np.min(X, axis=0)
maxs = np.max(X, axis=0)
rng = maxs - mins
X = 1 - ((maxs - X) / rng)
assert X.shape[0] == up.shape[0] == r.shape[0]


# In[ ]:


X_train, X_test, up_train, up_test, r_train, r_test, u_train, u_test, d_train, d_test = model_selection.train_test_split(
    X, up, r, universe, d, test_size=0.25, random_state=99)


train_cols = X_train.columns.tolist()

train_data = lgb.Dataset(X_train, label=up_train.astype(int), 
                         feature_name=train_cols)
test_data = lgb.Dataset(X_test, label=up_test.astype(int), 
                        feature_name=train_cols, reference=train_data)


# In[ ]:


# LGB parameters:
params = {'learning_rate': 0.05,
          'boosting': 'gbdt', 
          'objective': 'binary',
          'num_leaves': 2000,
          'min_data_in_leaf': 200,
          'max_bin': 200,
          'max_depth': 16,
          'seed': 2018,
          'nthread': 10,}


# LGB training:
lgb_model = lgb.train(params, train_data, 
                      num_boost_round=1000, 
                      valid_sets=(test_data,), 
                      valid_names=('valid',), 
                      verbose_eval=25, 
                      early_stopping_rounds=20)


# ## SHAP importance:

# In[ ]:


# DF, based on which importance is checked
X_importance = X_test

# Explain model predictions using shap library:
explainer = shap.TreeExplainer(lgb_model)
shap_values = explainer.shap_values(X_importance)


# ### Summary plot - most important features:
# 
# We will begin analysis of importance with most important features for a model based on validation set.
# Here, we will use `summary_plot`. This type of plot aggregates SHAP values for all the features and all samples in the selected set.
# Then SHAP values are sorted, so the first one shown is the most important feature. In addition to that, we are provided with information of how each feature affects the model output.
# 
# Two most important features are `returnsClosePrevRaw10_lag_3_mean` and `_max`. But here an interesting pattern can be noticed - low values of both of these features are clustered in a very dense region (blue blob), whereas features such as `returnsOpenPrevRaw10` and `returnsClosePrevRaw10` have much more uniform distribution, where for the first one (`returnsOpenPrevRaw10`) its high values push the prediction to 1 and for the second (`returnsClosePrevRaw10`) its low values push the prediction to 1.

# In[ ]:


# Plot summary_plot
shap.summary_plot(shap_values, X_importance)


# In[ ]:


# Plot summary_plot as barplot:
shap.summary_plot(shap_values, X_importance, plot_type='bar')


# ![](http://)Let's take a closer look at the feature:

# In[ ]:


X_importance.returnsClosePrevRaw10_lag_3_mean.value_counts()


# In[ ]:


plt.hist(X_importance.returnsClosePrevRaw10_lag_3_mean, bins=100)


# A lot of values are simply 0, probably this is the blue cluster on SHAP plot.

# ### Dependence plot:
# 
# Intepretation of an influence of a single feature on the model output is given by dependence plots.
# Plot below represents the influence of `returnsClosePrevRaw10_lag_3_mean`.  
# Vertical dispersion at a single value of `returnsClosePrevRaw10_lag_3_mean` represents interaction with other features from the data.
# Dependence plot automatically selected `returnsOpenPrevMktres10` as a feature for coloring.
# We can see that high values of returnsClosePrevRaw10_lag_3_mean influences the model output more significantly for observations, where returnsOpenPrevMktres10 has also high values.
# Worth noting is again the cluster containing values of returnsClosePrevRaw10_lag_3_mean equal to 0.

# In[ ]:


shap.dependence_plot("returnsClosePrevRaw10_lag_3_mean", shap_values, X_importance)


# ### Dependence plot - `volume`
# 
# Let's take a look at how `volume` will affect the model output.
# This time, our selected feature is plotted against `returnsClosePrevRaw10_lag_3_mean`.
# A direct relationship between volume and returnsClosePrevRaw10_lag_3_mean is difficult to notice.
# There are some values of volume significantly higher than the most, but their influence on model output isn't very big and their values of returnsClosePrevRaw10_lag_3_mean are both low and high.

# In[ ]:


shap.dependence_plot("volume", shap_values, X_importance)


# ### SHAP Interaction Values
# 
# As cited after [shap notebook](https://github.com/slundberg/shap/blob/master/notebooks/tree_explainer/NHANES%20I%20Survival%20Model.ipynb):
# 
# "See the Tree SHAP paper for more details, but briefly, SHAP interaction values are a generalization of SHAP values to higher order interactions. Fast exact computation of pairwise interactions are implemented in the latest version of XGBoost with the pred_interactions flag. With this flag XGBoost returns a matrix for every prediction, where the main effects are on the diagonal and the interaction effects are off-diagonal. The main effects are similar to the SHAP values you would get for a linear model, and the interaction effects captures all the higher-order interactions are divide them up among the pairwise interaction terms. Note that the sum of the entire interaction matrix is the difference between the model's current output and expected output, and so the interaction effects on the off-diagonal are split in half (since there are two of each). When plotting interaction effects the SHAP package automatically multiplies the off-diagonal values by two to get the full interaction effect."
# 
# 
# For this one, we will further limit the dataset size to speed up the computation.

# In[ ]:


X_interaction = X_importance.iloc[:500,:]

shap_interaction_values = shap.TreeExplainer(lgb_model).shap_interaction_values(X_interaction)


# ### Interaction - a summary:

# In[ ]:


shap.summary_plot(shap_interaction_values, X_interaction)


# ### Interaction - dependence plots:
# 
# Using dependence plots with interaction values enables assessment of main and interaction effects between features.
# First, let's take a look at raw `returnsClosePrevRaw10_lag_3_mean` dependence plot again and then compare it to interaction plots capturing both main and interaction effects.

# In[ ]:


# Raw dependence plot:

shap.dependence_plot(
    "returnsClosePrevRaw10_lag_3_mean",
    shap_values, X_importance)


# In[ ]:


# Interaction values dependence plot capturing main effects:

shap.dependence_plot(
    ("returnsClosePrevRaw10_lag_3_mean", "returnsClosePrevRaw10_lag_3_mean"),
    shap_interaction_values, X_interaction)


# Vertical disperion for interaction plot, which captures main effects, is lower than in the original plot, which was to be expected.

# In[ ]:


# Interaction values dependence plot capturing interaction effects:

shap.dependence_plot(
    ("returnsClosePrevRaw10_lag_3_mean", "returnsOpenPrevMktres10"),
    shap_interaction_values, X_interaction)


# When interaction between `returnsClosePrevRaw10_lag_3_mean` and `returnsOpenPrevMktres10` is checked, one interesting result is that for returnsClosePrevRaw10_lag_3_mean equal to 0, higher values of returnsOpenPrevMktres10 seem to push the model output lower, whereas higher values of returnsOpenPrevMktres10 for values of returnsClosePrevRaw10_lag_3_mean above 0.6 push the model output higher.

# ### Get important features according to SHAP:

# In[ ]:


shap_sum = np.abs(shap_values).mean(axis=0)
importance_df = pd.DataFrame([X_importance.columns.tolist(), shap_sum.tolist()]).T
importance_df.columns = ['column_name', 'shap_importance']
importance_df = importance_df.sort_values('shap_importance', ascending=False)
importance_df


# In[ ]:




