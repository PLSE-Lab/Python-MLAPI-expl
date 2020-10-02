#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import pandas as pd
import shap
from sklearn import preprocessing


# In[ ]:


plt.style.use('seaborn')
plt.rcParams['figure.figsize'] = 12, 7


# # The fallacy of encoding assetCode
# By @marketneutral
# 
# There have been a few kernel shares of gradient-boosted decision trees ("GBDT"; e.g., `lightgbm`) applied directly to the data "as is" in this competition. The results of this show that `assetCode` (and `assetName`), as a categorical variable, is a substantially important feature. Does this make sense? If you simply know the ticker symbol should that add predictive power to the model? On the face of it, it seems implausible and that the use of `assetCode` rather is simply leaking future information and producing an overfit model. I investigate this idea in this kernel.
# 
# ## Minimal Reproduction
# First, let's do a minimal reproduction. There is no effort here to do parameter tuning, or to create a great model per se. Here we just want to fit the bare minimum GBDT model and see what features the model thinks are important. We just want to reproduce in a minimal way that the model will find `assetCode` and `assetName` to be very important.

# In[ ]:


# Make environment and get data
from kaggle.competitions import twosigmanews
env = twosigmanews.make_env()
(market_train_df, news_train_df) = env.get_training_data()


# In[ ]:


# Dropping assetName just to focus exclusively on one categorical variable
market_train_df.drop('assetName', axis=1, inplace=True)


# In[ ]:


def make_test_train(df, split=0.80):
    # Label encode the assetCode feature
    X = df[df.universe==1]
    le = preprocessing.LabelEncoder()
    X = X.assign(assetCode = le.fit_transform(X.assetCode))
    
    # split test and train
    train_ct = int(X.shape[0]*split)
    y_train, y_test = X['returnsOpenNextMktres10'][:train_ct], X['returnsOpenNextMktres10'][train_ct:]
    X = X.drop(['time', 'returnsOpenNextMktres10'], axis=1)
    X_train, X_test = X.iloc[:train_ct,], X.iloc[train_ct:,]
    return X, X_train, X_test, y_train, y_test


# In[ ]:


# Make the encoding and split
X, X_train, X_test, y_train, y_test = make_test_train(market_train_df)


# In[ ]:


def make_lgb(X_train, X_test, y_train, y_test, categorical_cols = ['assetCode']):
    # Set up LightGBM data structures
    train_cols = X_train.columns.tolist()
    dtrain = lgb.Dataset(X_train.values, y_train, feature_name=train_cols, categorical_feature=categorical_cols)
    dvalid = lgb.Dataset(X_test.values, y_test, feature_name=train_cols, categorical_feature=categorical_cols)
    return dtrain, dvalid


# In[ ]:


# Set up the LightGBM data structures
dtrain, dvalid = make_lgb(X_train, X_test, y_train, y_test)


# In[ ]:


# Set up the LightGBM params
lgb_params = dict(
    objective='regression_l1', learning_rate=0.1, num_leaves=127, max_depth=-1, bagging_fraction=0.75,
    bagging_freq=2, feature_fraction=0.5, lambda_l1=1.0, seed=1015
)


# In[ ]:


# Fit and predict
evals_result = {}
m = lgb.train(
    lgb_params, dtrain, num_boost_round=1000, valid_sets=(dvalid,), valid_names=('valid',), 
    verbose_eval=25, early_stopping_rounds=20, evals_result=evals_result
)


# In[ ]:


# Plot reported feature importance
lgb.plot_importance(m);


# In[ ]:


lgb.plot_importance(m, importance_type='gain');


# Let's see what SHAP thinks. Per the [GitHub repo](https://github.com/slundberg/shap), **SHAP** (SHapley Additive exPlanations) is a unified approach to explain the output of any machine learning model. SHAP connects game theory with local explanations, uniting several previous methods [1-7] and representing the only possible consistent and locally accurate additive feature attribution method based on expectations (see the [SHAP NIPS paper](http://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions) for details).

# In[ ]:


shap_explainer = shap.TreeExplainer(m)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'sample = X.sample(frac=0.50, random_state=100)\nshap_values = shap_explainer.shap_values(sample)')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'shap.summary_plot(shap_values, sample)')


# So indeed,  consistent with other public kernels, the model believes `assetCode` is a valuable feature. 
# 
# However, this result is from leaked information. Let's try this model on **completely random data**.
# 
# ## Spoof Completely Random Dataset
# 
# Below I create a random DataFrame in the shape of the `market_data_df` DataFrame. Then I fit the same LightGBM model as above. Since this is random data, there should be no material predictive features, right?

# In[ ]:


# Create some random assetCodes (this is a nice snippet fro McKinney, Wes. Python for Data Analysis: Data Wrangling with Pandas, NumPy, and IPython (p. 340). O'Reilly Media. Kindle Edition.)
import random; random.seed(0)
import string
num_stocks = 1250
def rands(n):
    choices = string.ascii_uppercase
    return ''.join([random.choice(choices) for _ in range(n)])
assetCodes = np.array([rands(5) for _ in range(num_stocks)])


# In[ ]:


# Spoof intraday and overnight returns
days_in_year = 260
total_days = days_in_year*7
on_vol_frac = 0.2  # overnight volatility fraction

annualized_vol = 0.20
open_to_close_returns = np.random.normal(0.0, scale=annualized_vol*(1-on_vol_frac)/np.sqrt(days_in_year), size=(total_days, num_stocks))
close_to_open_returns = np.random.normal(0, scale=annualized_vol*(on_vol_frac)/np.sqrt(days_in_year), size=(total_days, num_stocks))
open_to_open_returns = close_to_open_returns + open_to_close_returns 
close_to_close_returns = close_to_open_returns + np.roll(open_to_close_returns, -1)

# Make price series
prices_close = 100*np.cumprod(1+close_to_close_returns, axis=0)
prices_open = prices_close*(1+close_to_open_returns)


# In[ ]:


import itertools

# Make into a DataFrame
dates = pd.date_range(end=pd.Timestamp('2017-12-31'), periods=total_days)
spoofed_df = pd.DataFrame(
    data={'close': prices_close.flatten('F'), 'open': prices_open.flatten('F')},
    index = pd.MultiIndex.from_tuples(
        list(itertools.product(assetCodes, dates)), names=('assetCode', 'time')
    )
)
spoofed_df['universe'] = 1.0


# In[ ]:


spoofed_df.head()

# Looks good!


# In[ ]:


spoofed_df = spoofed_df.reset_index().sort_values(['assetCode','time']).set_index(['assetCode', 'time'])


# In[ ]:


# make sure we did the open/close transform properly. Looks good.
spoofed_df.loc['MYNBI', ['open', 'close']]['1Q2013'].plot();


# In[ ]:


# # Make the "return" based features

spoofed_df = spoofed_df.assign(
     returnsClosePrevRaw1 = spoofed_df.groupby(level='assetCode').
     apply(lambda x: x.close/x.close.shift(1) -1)
     .reset_index(0, drop=True)
)

spoofed_df = spoofed_df.assign(
     returnsOpenPrevRaw1 = spoofed_df.groupby(level='assetCode').
     apply(lambda x: x.open/x.open.shift(1) -1)
     .reset_index(0, drop=True)
)

spoofed_df = spoofed_df.assign(
     returnsOpenPrevRaw10 = spoofed_df.groupby(level='assetCode').
     apply(lambda x: (x.open/x.open.shift(10)) - 1)
     .reset_index(0, drop=True)
)

spoofed_df = spoofed_df.assign(
     returnsClosePrevRaw10 = spoofed_df.groupby(level='assetCode').
     apply(lambda x: x.close/x.close.shift(10)-1)
     .reset_index(0, drop=True)
)


# In[ ]:


# Make the target variable
spoofed_df = spoofed_df.assign(
    returnsOpenNextMktres10 = spoofed_df.groupby(level='assetCode').
    apply(lambda x: (x.open.shift(-10)/x.open)-1)
    .reset_index(0, drop=True)
)


# In[ ]:


# Drop the edges where we don't have data to make returns
spoofed_df = spoofed_df.reset_index().dropna()


# In[ ]:


# Split the data
X, X_train, X_test, y_train, y_test = make_test_train(spoofed_df)

# Set up LightGBM data structures
dtrain, dvalid = make_lgb(X_train, X_test, y_train, y_test)


# In[ ]:


# Fit and predict
evals_result = {}
m = lgb.train(
    lgb_params, dtrain, num_boost_round=1000, valid_sets=(dvalid,), valid_names=('valid',), 
    verbose_eval=25, early_stopping_rounds=20, evals_result=evals_result
)


# In[ ]:


lgb.plot_importance(m);


# In[ ]:


lgb.plot_importance(m, importance_type='gain');


# In[ ]:


shap_explainer = shap.TreeExplainer(m)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'sample = X.sample(frac=0.50, random_state=100)\nshap_values = shap_explainer.shap_values(sample)')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'shap.summary_plot(shap_values, sample)')


# So **in random data** we find that `assetCode` is predictive. In training, this category encodes something like the average of the target variable across the entire training set. It's like if I told you the total return for AAPL and NFLX were +50% and -25% over the entire training set (in-sample). Thus to make a good prediction in-sample, I would use the knowledge of the ticker to predict a positive target for each day for AAPL and a negative target for each day for NFLX. These would be decent predictions becuase we knew the sign and relative magnitudes of the total period. Of course out of sample, we would not expect this to hold. **So, bottom line, don't encode `assetCode` or `assetName`.**

# In[ ]:




