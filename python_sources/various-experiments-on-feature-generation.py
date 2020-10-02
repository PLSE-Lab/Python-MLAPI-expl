#!/usr/bin/env python
# coding: utf-8

# In this competition, I tried to come up with a lot of useful features to improve my model. Unfortunately, I didn't know how to create features quickly so I could not use many features in the end because the more feature I have, the more it will take to create them. Vadim kindly shared a [kernel](https://www.kaggle.com/nareyko/fast-lags-calculation-concept-using-numpy-arrays/comments#454588) that shows how to create features very fast. Using this code, it is possible to make use of many features. So, I wanted to experiment with a lot of features that might be useful.
# Credits for efficient feature creation algorithms goes to [Vadim](https://www.kaggle.com/nareyko). Other kernels that inspired me for ideas include [simple quant features kernel](https://www.kaggle.com/youhanlee/simple-quant-features-using-python), [eda kernel](https://www.kaggle.com/qqgeogor/eda-script-67) and [another eda with nice visualization](https://www.kaggle.com/artgor/eda-feature-engineering-and-everything)

# In[ ]:


from kaggle.competitions import twosigmanews
import catboost as catb
from catboost import CatBoostClassifier
from datetime import datetime, date
import gc
import lightgbm as lgb
import multiprocessing
from multiprocessing import Pool
import numpy as np
import pandas as pd
from resource import getrusage, RUSAGE_SELF
from sklearn import model_selection
from xgboost import XGBClassifier


# In[ ]:


np.random.seed(1) # I don't know if using numpy random seeding helps in reproducing results but I do it anyways to be safe


# In[ ]:


global STARTED_TIME
STARTED_TIME = datetime.now()

# It's better to use cpu_count from the system - who knows what happens during test phase
global N_THREADS
N_THREADS = multiprocessing.cpu_count() * 2 

print(f'N_THREADS: {N_THREADS}')


# In[ ]:


# FILTERDATE - start date for the train data
FILTERDATE = date(2007, 1, 1)

# SAMPLEDATE - use it for sampling and fast sanity check of scripts. Since I am generating too many features limiting data amount will help in not crashing
# In production, it would be better to cherry pick useful features and discard non important features. And then use more data again.
SAMPLEDATE = None
SAMPLEDATE = date(2008, 1, 30)


# In[ ]:


global N_WINDOW, BASE_FEATURES

N_WINDOW = np.sort([5, 10, 20, 252])

# Features for lags calculation
BASE_FEATURES = [
    'returnsOpenPrevMktres10',
    'returnsOpenPrevRaw10',
    'open',
    'close']


# In[ ]:


# Tracking time and memory usage
global MAXRSS
MAXRSS = getrusage(RUSAGE_SELF).ru_maxrss
def using(point=""):
    global MAXRSS, STARTED_TIME
    print(str(datetime.now()-STARTED_TIME).split('.')[0], point, end=' ')
    max_rss = getrusage(RUSAGE_SELF).ru_maxrss
    if max_rss > MAXRSS:
        MAXRSS = max_rss
    print(f'max RSS {MAXRSS/1024/1024:.1f}Gib')
    gc.collect()


# Generate features with the usual window statics (mean, median, max, min, exponentially weighted mean).

# In[ ]:


global FILLNA
FILLNA = -99999

ewm = pd.Series.ewm

def generate_features_for_df_by_assetCode(df_by_code):
    prevlag = 1
    for window in N_WINDOW:
        rolled = df_by_code[BASE_FEATURES].shift(prevlag).rolling(window=window)
        df_by_code = df_by_code.join(rolled.mean().add_suffix(f'_window_{window}_mean'))
        df_by_code = df_by_code.join(rolled.median().add_suffix(f'_window_{window}_median'))
        df_by_code = df_by_code.join(rolled.max().add_suffix(f'_window_{window}_max'))
        df_by_code = df_by_code.join(rolled.min().add_suffix(f'_window_{window}_min'))
        for col in BASE_FEATURES: # not sure if this can be optimized without using for loop but I only know how to calculate exponentially moving averages like this
            df_by_code[col + f'_window_{window}_ewm'] = ewm(df_by_code[col], span=window).mean().add_suffix(f'_window_{window}_ewm')
    return df_by_code.fillna(FILLNA)

def generate_features(df):
    global BASE_FEATURES, N_THREADS
    all_df = []
    df_codes = df.groupby('assetCode')
    df_codes = [df_code[1][['time','assetCode'] + BASE_FEATURES] for df_code in df_codes]
    pool = Pool(N_THREADS)
    all_df = pool.map(generate_features_for_df_by_assetCode, df_codes)
    new_df = pd.concat(all_df)
    new_df.drop(BASE_FEATURES,axis=1,inplace=True)
    pool.close()
    return new_df


# In[ ]:


# The following functions are used for initialization and expanding of numpy arrays
# for storing historical data of all assets.

# It helps to have very fast lags creation.

# Initialization of history array
def initialize_values(items=5000, features=4, history=15):
    return np.ones((items, features, history))*np.nan

# Expanding of history array for new assets
def expand_history_array(history_array, items=100):
    return np.concatenate([history_array, initialize_values(items, history_array.shape[1], history_array.shape[2])])

# codes dictionary maps assetCode to the index in the history array
# if we found new assetCode - we have to store it and expand history
def get_index_by_assetCode(assetCode):
    global code2array_idx, history
    try: 
        return code2array_idx[assetCode]
    except KeyError:
        code2array_idx[assetCode] = len(code2array_idx)
        if len(code2array_idx) > history.shape[0]:
            history = expand_history_array(history, 100)
        return code2array_idx[assetCode]

# list2codes returns numpy array of indices of assetCodes in history array(for each day)
def codes_list2idx_array(codes_list):
    return np.array([get_index_by_assetCode(assetCode) for assetCode in codes_list])


# In[ ]:


env = twosigmanews.make_env()


# In[ ]:


(market_train_df, news_train_df) = env.get_training_data()


# In[ ]:


# Memory limit 16GB leaves me no choice but to drop some columns early on.
# Kernel crashes when all memory is used up.
market_train_df = market_train_df.drop(['universe'], axis = 1)
news_train_df = news_train_df.drop(['sourceTimestamp', 'sourceId', 'headline', 'takeSequence', 
                                    'provider', 'subjects', 'audiences', 'bodySize', 
                                    'companyCount', 'headlineTag', 'marketCommentary', 'sentenceCount', 'wordCount',
                                    'relevance', 'sentimentWordCount', 'noveltyCount12H', 'noveltyCount24H',
                                    'noveltyCount3D', 'noveltyCount5D', 'noveltyCount7D', 'volumeCounts12H',
                                    'volumeCounts24H', 'volumeCounts3D', 'volumeCounts5D',
                                    'volumeCounts7D'], axis = 1)
# uncomment if you are not using news data
# del news_train_df
using('Data loaded')


# In[ ]:


def process_time(df):
    df['time'] = df['time'].dt.date
    return df

market_train_df = process_time(market_train_df)
news_train_df = process_time(news_train_df)


# In[ ]:


# Dataframe filtering
print('DF Filtering')
market_train_df = market_train_df.loc[market_train_df['time']>=FILTERDATE]
news_train_df = news_train_df.loc[news_train_df['time']>=FILTERDATE]

if SAMPLEDATE is not None:
    market_train_df = market_train_df.loc[market_train_df['time']<=SAMPLEDATE]  
    news_train_df = news_train_df.loc[news_train_df['time']<=SAMPLEDATE]  
using('Done')


# The first feature I think is interesting is ratio of a single asset's 'volume', 'close' to mean value of 'market'. Because all assets have different average volumes and closing price (I think in this regard, either one of open or close is usually enough). If single asset's 'volume'/'close''s ratio changes in relation to 'market_mean' it would be a possible indicator that implies an asset is oversold / overbought. Thus, we will include these ratios to base features.

# In[ ]:


def add_market_mean_col(market_df):
    daily_market_mean_df = market_df.groupby('time').mean()
    daily_market_mean_df = daily_market_mean_df[['volume', 'close']]
    merged_df = market_df.merge(daily_market_mean_df, left_on='time',
                                right_index=True, suffixes=("",'_market_mean'))
    merged_df['volume/volume_market_mean'] = merged_df['volume'] / merged_df['volume_market_mean']
    merged_df['close/close_market_mean'] = merged_df['close'] / merged_df['close_market_mean']
    return merged_df.reset_index(drop = True)

BASE_FEATURES = BASE_FEATURES + ['volume', 'volume/volume_market_mean', 'close/close_market_mean']

market_train_df = add_market_mean_col(market_train_df)
market_train_df.head(3)


# In[ ]:


market_train_df.tail(3) # check different days have different market mean


# Most day traders would consider how much an asset price appreciated / depreciated on a single trading day is an important factor in trading.

# In[ ]:


def generate_open_close_ratio(df):
    df['open/close'] = df['open'] / df['close']
    
BASE_FEATURES = BASE_FEATURES + ['open/close']

generate_open_close_ratio(market_train_df)
using('Done')


# Raw return features themselves are not that meaningful (unlike market residual features) because 1 dollar appreciation of an asset which has price of 5 and that of 100 have very different implications. Raw return features ratio to 'open', 'close' might be more useful.

# In[ ]:


open_raw_cols = ['returnsOpenPrevRaw1', 'returnsOpenPrevRaw10']
close_raw_cols = ['returnsClosePrevRaw1', 'returnsClosePrevRaw10']

def raw_features_to_ratio_features(df):
    for col in open_raw_cols:
        df[col + '/open' ] = df[col] / df['open']
    for col in close_raw_cols:
        df[col + '/close'] = df[col] / df['close']

BASE_FEATURES = BASE_FEATURES + ['returnsClosePrevRaw1/close', 'returnsClosePrevRaw10/close', 'returnsOpenPrevRaw1/open', 'returnsOpenPrevRaw10/open']

raw_features_to_ratio_features(market_train_df)
market_train_df.head(3)


# Not related to feature generations but I want to get rid of outliers (too much variations of price in one day). Among the deleted rows, opening price 999.99 seems that it is a dummy value that was filled.

# In[ ]:


origlen = len(market_train_df)
print(market_train_df.loc[market_train_df['open/close'] >= 3][['open', 'close']])
print(market_train_df.loc[market_train_df['open/close'] <= 0.3][['open', 'close']])
market_train_df = market_train_df.loc[market_train_df['open/close'] < 3]
market_train_df = market_train_df.loc[market_train_df['open/close'] > 0.3]
print(origlen - len(market_train_df), "row deleted")
using('Done')


# We are going to be end up generating too many features. `generated_non_feature_cols` will be used to filter out unnecessary columns before training starts. (filter by `non_feature_cols` but adding columns to `generated_non_feature_cols` works because `non_feature_cols` includes `generated_non_feature_cols`.)

# In[ ]:


target = 'returnsOpenNextMktres10'
generated_non_feature_cols = [] # Use it to filter out generated cols that turns out not so useful after analyzing feature importances
non_feature_cols = ['assetCode', 'assetName', target, 'time', 'time_x', 'volume_y',] + generated_non_feature_cols


# In[ ]:


new_df = generate_features(market_train_df)
using('Done')


# In[ ]:


market_train_df = pd.merge(market_train_df, new_df, how = 'left', on = ['time', 'assetCode'])
del new_df
using('Done')


# I am not sure if it is a good thing to use assetCode as a feature, because it is like an id column and we are putting our model in a risk of overfitting.
# If model memorize too much about assetCode and the characteristics of each asset changes a lot in testing phase, it wouldn't be good. 
# (e.g think about case a company changes strategy/ceo, reduces outstanding shares etc, it will be very different from training data even if the code is the same)
# So it might be good thing to not use 'assetCodeT' as a feature to avoid overfitting.

# In[ ]:


# label encoding of assetCode
def encode_assetCode(market_train):
    global code2array_idx
    market_train['assetCodeT'] = market_train['assetCode'].map(code2array_idx)
    market_train = market_train.dropna(axis=0)
    return market_train

code2array_idx = dict(
    zip(market_train_df.assetCode.unique(), np.arange(market_train_df.assetCode.nunique()))
)

market_train_df = encode_assetCode(market_train_df)


# A lot of news data columns are deleted but let's make use of what's left. I think sentiment related columns are most important anyways but who knows.

# In[ ]:


def merge_with_news_data(market_df, news_df):
    news_df['firstCreated'] = news_df.firstCreated.dt.hour
    news_df['assetCodesLen'] = news_df['assetCodes'].map(lambda x: len(eval(x)))
    news_df['asset_sentiment_count'] = news_df.groupby(['assetName', 'sentimentClass'])['firstCreated'].transform('count')
    # I don't use assetCode for joining key, but use assetName. One news row has multiple assetCode so they don't match nicely with market data
    # Also remember assetCodes with same assetName are related assets (normal stock and preferred stock from same company e.g.)
    kcol = ['time', 'assetName']
    news_df = news_df.groupby(kcol, as_index=False).mean()
    market_df = pd.merge(market_df, news_df, how='left', on=kcol, suffixes=("", "_news"))
    return market_df

market_train_df = merge_with_news_data(market_train_df, news_train_df)
del news_train_df
using("Merged news data")
market_train_df.head(3)


# In[ ]:


# this cell only need it for preparing to predict, uncomment to prepare to predict.
# history stores information for all assets, required features

# history = initialize_values(len(code2array_idx), len(BASE_FEATURES), np.max(N_WINDOW)+1)

# # Get the latest information for assets
# latest_events = market_train_df.groupby('assetCode').tail(np.max(N_WINDOW)+1)
# # but we may have different informations size for different assets
# latest_events_size = latest_events.groupby('assetCode').size()


# In[ ]:


# Filling the history array, this cell only need it for preparing to predict, uncomment to prepare to predict.

# for s in latest_events_size.unique():
#     for i in range(len(BASE_FEATURES)):
#         # l is a Dataframe with assets with the same history size for each asset
#         l = latest_events[
#             latest_events.assetCode.isin(latest_events_size[latest_events_size==s].index.values)
#         ].groupby('assetCode')[BASE_FEATURES[i]].apply(list)

#         # v is a 2D array contains history information of feature RETURN_FEATURES[i] 
#         v = np.array([k for k in l.values])

#         # r contains indexes (in the history array) of all assets
#         r = codes_list2idx_array(l.index.values)

#         # Finally, filling history array
#         history[r, i, -s:] = v
#         del l, v, r

# del latest_events, latest_events_size
# using('Done')


# Split training and validation data set.

# In[ ]:


y = market_train_df[target] >= 0
y = y.values.astype(int)
fcol = [c for c in market_train_df if c not in non_feature_cols]

X = market_train_df[fcol]
print("len:", len(X.columns))
for col in X.columns:
    print(col, end = ', ')
X = X.values

# Scaling of X values, Tree based models shouldn't need scaling tho?
maxs = np.max(X, axis=0)
rng = maxs - np.min(X, axis=0)
X = 1 - ((maxs - X) / rng)

# Sanity check
assert X.shape[0] == y.shape[0]

X_train, X_val, y_train, y_val = model_selection.train_test_split(X, y, test_size=0.25, random_state=99)

del market_train_df, X, y
using('done')


# Start training of 3 different models. By no means this can be used as benchmark result because only lgbm is tuned and the others mostly use default paramters. Still it's interesting that XGB stops training so early on. With current parameters, lgbm works the best in terms of low loss.

# In[ ]:


# Train lgbm

lgb_train_data = lgb.Dataset(X_train, label = y_train)
lgb_test_data = lgb.Dataset(X_val, label = y_val)

params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'learning_rate': 0.19016805202090095,
        'num_leaves': 2583,
        'min_data_in_leaf': 213,
        'num_iteration': 172,
        'max_bin': 220,
        'seed': 42,
    }

gbm = lgb.train(params,
                lgb_train_data,
                valid_sets=lgb_test_data,
                early_stopping_rounds=5,
                verbose_eval=30,
            )

del lgb_train_data, lgb_test_data
using('Training done')


# In[ ]:


# Train catboost

train_pool = catb.Pool(X_train, y_train)
validate_pool = catb.Pool(X_val, y_val)

catb_model = CatBoostClassifier(
    iterations = 300,
    random_seed=42,
)

catb_model.fit(
    X_train, y_train.astype(int),
    eval_set=(X_val, y_val),
    verbose=50,
    plot=False
)


# In[ ]:


# Train XGB
xgb = XGBClassifier(random_state=42) 
eval_set = [(X_val, y_val)] 
xgb.fit(X_train, y_train, eval_metric="logloss", early_stopping_rounds=5, eval_set=eval_set, verbose=True)


# Now that training is finished, let's check how many features we have and what percentage of contribution each feature makes on average. If some features are contributing below average you might consider getting rid of those features.

# In[ ]:


print("total features:", len(fcol), ", average:", 100/len(fcol))


# Now let's check the feature importance. When we have too many features, I find it it's not so useful to draw graph because feature names are so crammed and hard to read. So I print features in the order of importance and in terms of percentage of total feature importance. It is intereting to see each model views feature importances slightly differently.

# In[ ]:


def show_feature_importances(feature_importances):
    total_feature_importances = sum(feature_importances)
    assert len(feature_importances) == len(fcol) # sanity check
    for score, feature_name in sorted(zip(feature_importances, fcol), reverse=True):
        print('{}: {}'.format(feature_name, score/total_feature_importances * 100))


# In[ ]:


# lgbm importances split
show_feature_importances(gbm.feature_importance(importance_type='split'))


# In[ ]:


# Cat boost feature importance
show_feature_importances(catb_model.get_feature_importance(train_pool))


# In[ ]:


# XGB feature importance
show_feature_importances(xgb.feature_importances_)


# In[ ]:


# Another standard for deciding feature importance
# lgbm importances gain
# show_feature_importances(gbm.feature_importance(importance_type='gain'))


# Having too many features don't always help. It can make it harder for models to train or sometimes lead them to overfit. Let's get some list of features whose importance is way below average(0.37%). Use percentage threshold to set the bar. Whopping 123 features have less than 0.1% importance. Remember you can add these colums to `generated_non_feature_cols` list and run the kernel again to use only important enough columns.

# In[ ]:


def get_non_important_features(feature_importances, threshold):
    total_feature_importances = sum(feature_importances)
    assert len(feature_importances) == len(fcol) # sanity check
    return [feature_name for score, feature_name in sorted(zip(feature_importances, fcol), reverse=True) if ((score * 100) / total_feature_importances)  < threshold]

non_features = get_non_important_features(gbm.feature_importance(importance_type='split'), threshold = 0.1)
print(len(non_features))
non_features


# In[ ]:


# Other standards for deciding feature importance
# print(get_non_important_features(gbm.feature_importance(importance_type='gain'), threshold = 0.1))
# print(get_non_important_features(catb_model.get_feature_importance(train_pool), threshold = 0.1))
# print(get_non_important_features(xgb.feature_importances_, threshold = 0.1))


# This kernel only focuses on various experimental features generation, checking their importances, how to filter out non-important features. Not all the features generated are useful. However, it is easy to exclude features using `generated_non_feature_cols` variable. I can imagine there can be many more interesting features than I experimented here. What were some of your favorite features that you created? Let me know.
# Thank you for reading.

# In[ ]:


using('finished!')


# In[ ]:




