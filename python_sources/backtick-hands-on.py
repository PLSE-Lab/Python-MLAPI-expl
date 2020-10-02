#!/usr/bin/env python
# coding: utf-8

# # Backtick hands on kernel
# After some initial setup a few sectionsfollows, that upon completetion will allow you to implement a simple stock market price prediction pipeline.
# 
# The pandas API documenation is a useful source to solve some of the problems: https://pandas.pydata.org/pandas-docs/stable/reference/index.html
# 
# ## 1 Data investigation
# ## 2 Preprocessing
# ## 3 Feature Creation
# ## 4 Binary Classification in LightGBM
# ## 5 Confidence Interval & Scoring
# ## 6 Model Evaluation
# ## 7 Voting

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import lightgbm as lgb
from kaggle.competitions import twosigmanews
from sklearn.metrics import confusion_matrix, accuracy_score


# # Load the data from the competition environment
# You can only load the data from the env once - you'll need to completely restart your kernel if you lose it (second right button at the very bottom)

# In[ ]:


env = twosigmanews.make_env()

market_orig, news_orig = env.get_training_data()


# In[ ]:


# We'll use data from 2013 onwards to speed things up a bit
market_orig = market_orig[market_orig['time'] >= "2013-01-01"]
market_orig.head()


# # Section 1, Data Investigation
# 

# ##  1.1 How many unique assetNames are there?

# In[ ]:


def unique_asset_names(df):
    return "TODO"

unique_asset_names(market_orig)


# ## 1.2 Find the most common assetName.
# Hint:
# 
# 'noitagergga tnuoc htiw denibmoc ybpuorg noitcnuf sadnap eht esu nac uoy'[::-1]

# In[ ]:


def most_common_asset_name(df):
    return "TODO"

most_common_asset_name(market_orig)


# ## 1.3  What is the maximum value of returnsOpenNextMktres10?

# In[ ]:


def max_next10(df):
    return "TODO"

max_next10(market_orig)


# ## 1.4 Plot the close price of Facebook (assetCode="FB.O") over time
# 
# Hint:
# 
# 'sixa x sa nmuloc emit eht ssap ,biltolptam fo noitcnuf tolp eht tuo kcehc'[::-1]

# In[ ]:


# Matplotlib is available as "plt"
plt.figure(figsize=(12,7))

def plot_fb(df):
    return "TODO"

plot_fb(market_orig)


# ## 1.5 What is the min, max and mean amount of days an asset is present in the dataset? (use "assetCode")

# In[ ]:


def asset_presence(df):
    return "TODO"

asset_presence(market_orig)


# # Section 2. Preprocessing

# ## 2.1 Remove all rows where assetName is "Unknown"
# *Hint:*
# 
# 'dnammoc nisi eht htiw emarfatad eht no gniksam naeloob esu ot si noitulos elbissop a'[::-1]

# In[ ]:


def remove_unknown(df):
    return df

market = remove_unknown(market_orig)


# ## 2.2 Limit the returnsOpenNextMktres10 column to values between -1 and 1
# 
# *Hint:*
# 
# 'noitcnuf pilc sadnap eht yrt'[::-1]

# In[ ]:


def clip_next10(df):
    return df

market = clip_next10(market)


# ## 2.3 Remove instances where the asset is not present in the data for more than 30 days (use "assetCode")

# In[ ]:


def remove_short_lived(df):
    return df

market = remove_short_lived(market)


# # Section 3. Feature Creation

# ## 3.1 Add the (RSI) indicator for a few different periods (default is 14)

# In[ ]:


# https://github.com/bukosabino/ta/blob/master/ta/momentum.py
def ema(series, periods, fillna=False):
    if fillna:
        return series.ewm(span=periods, min_periods=0).mean()
    return series.ewm(span=periods, min_periods=periods).mean()

def rsi(close, n=14, fillna=False):
    """Relative Strength Index (RSI)
    Compares the magnitude of recent gains and losses over a specified time
    period to measure speed and change of price movements of a security. It is
    primarily used to attempt to identify overbought or oversold conditions in
    the trading of an asset.
    https://www.investopedia.com/terms/r/rsi.asp
    Args:
        close(pandas.Series): dataset 'Close' column.
        n(int): n period.
        fillna(bool): if True, fill nan values.
    Returns:
        pandas.Series: New feature generated.
    """
    diff = close.diff()
    which_dn = diff < 0

    up, dn = diff, diff*0
    up[which_dn], dn[which_dn] = 0, -up[which_dn]

    emaup = ema(up, n, fillna)
    emadn = ema(dn, n, fillna)

    rsi = 100 * emaup / (emaup + emadn)
    if fillna:
        rsi = rsi.replace([np.inf, -np.inf], np.nan).fillna(50)
    return pd.Series(rsi, name='rsi')


# In[ ]:


# TODO: Add the RSI indicator to the market df


# ## 3.2 Add features which measures the average volume of the last [5, 10, 20] days
# 
# *Hint:*
# 
# 'emarfatad eht revo wodniw gnillor a etaerc nac uoy'[::-1]

# In[ ]:


def add_volume_avg(df):
    return df

add_volume_avg(market)


# ## 3.3 (Optional) - Add additional lag features for some relevant columns. Compute min, max and mean for different window sizes
# 
# *Hint:*
# 
# '76-tpircs-ade/rogoegqq/moc.elggak.www//:sptth :lenrek siht ni noitaripsni dna spit ecnamrofrep doog emos ereht'[::-1]
# 
# 
# 

# In[ ]:


def add_lag_features(df):
    return df
    
market = add_lag_features(market)


# 
# # Section 4 - Binary Classification in LightGBM 

# ## 4.1 Separate the data into a train and test set based on reasonable dates
# 
# It is wise to have small gap between your train set and test set.

# In[ ]:


# Returns:
#    X: numpy matrix with relevant features only
#    y: numpy array of class values, 0 if returnsOpenNextMktres10 is negative, else 1

TRAIN_END_DATE  = ""
TEST_START_DATE = ""

def get_data(df):
    pass

X_train, y_train = get_data()
X_test, y_test   = get_data()


# ## 4.2 Familiarize yourself with the LightGBM API

# In[ ]:


def train_clf(X_train, y_train):
    # https://lightgbm.readthedocs.io/en/latest/Parameters.html
    params = {
        'objective': 'binary',
        'num_threads': 4
    }

    train_set = lgb.Dataset(X_train, y_train)

    # https://lightgbm.readthedocs.io/en/latest/Python-API.html#training-api
    lgb_clf = lgb.train(params, train_set)
    
    return lgb_clf

clf = train_clf(X_train, y_train)


# ## 4.3 Report the test accuracy score of your classifier
# 

# In[ ]:


def accuracy(clf, X_test, y_test):
    return 0

accuracy(clf, X_test, y_test)


# ## 4.4 Plot the feature importances of your trained model

# In[ ]:


# You can use this helper, if you want.
def plot_feature_importances(clf, feature_columns):
    features_imp = pd.DataFrame()
    features_imp['features'] = list(feature_columns)[:]
    features_imp['importance'] = clf.feature_importance()
    features_imp = features_imp.sort_values(by='importance', ascending=False).reset_index()
    shape = features_imp.shape[0]
    
    y_plot = -np.arange(shape)
    plt.figure(figsize=(10,7))
    plt.barh(y_plot, features_imp.loc[:shape,'importance'].values)
    plt.yticks(y_plot,(features_imp.loc[:shape,'features']))
    plt.xlabel('Feature importance')
    plt.title('Features importance')
    plt.tight_layout()


# # 5 Confidence Interval & Scoring

# ## 5.1 Construct a confidence value for each sample in the test set

# In[ ]:


# Returns a series of confidence values
def get_confidence(clf, X_test):
    y_pred = clf.predict(X_test)
    
    return "TODO"

confidence = get_confidence(clf, X_test)


# ## 5.2 Compute the score of your strategy
# 
# Make sure you group the predictions by day, and use the formula in the competition description.

# In[ ]:


# You can use this helper:
def get_scoring_data(market):
    test_df         = market[[market['time'] > TEST_START_DATE]
    test_df['date'] = df['time'].dt.date
                     
    actual_returns  = test_df['returnsOpenNextMktres10'].values.clip(-1, 1)
    universe        = test_df['universe']
    dates           = test_df['date']

    return actual_returns, universe, dates

actual_returns, universe, dates = get_scoring_data(market)


# In[ ]:


def score(confidence, actual_returns, universe, dates):
    return 0

score(confidence, actual_returns, universe, dates)


# ## 5.3 Plot the daily returns of your strategy

# In[ ]:


# Modify the "score" function above to plot your strategy's daily returns


# # 6 Model Evaluation

# ## 6.1 Cross-validate your model using an appropriate approach for time series data
# Lets see if the model performs similarly on data from different time periods.
# 
# Hopyfully, you can reuse some of the functions you've previously created.

# In[ ]:


def cross_validate():
    # 1. Create folds based on date from the data
    # 2. Train a classifier for each fold
    # 3. Test against related test set
    # 4. Evaluate the results
    
    pass

cross_validate()


# # 7. Voting
# 

# ## 7.1 Democratize your solution. 
# Implement a way for multiple classifiers to have a say about the confidence interval

# In[ ]:


# Implement a voting strategy, you can reuse a lot of the code from the cross_validation step


# # 8 Revise your pipeline
# Now you hopefully have a good sense of the problem, and you can back and try to improve each step a long the way. 
# 
# Here's a few things you can try for a better score:
# 
# * Better preprocessing
# * Use the news data set
# * Multiclass classification
# * Probability Selection
# * Further post-preprocessing strategies
# * Any ideas that you might have

# In[ ]:





# In[ ]:




