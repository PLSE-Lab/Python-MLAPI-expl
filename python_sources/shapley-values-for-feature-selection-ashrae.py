#!/usr/bin/env python
# coding: utf-8

# This kernel shows how to use [Shapley values](https://www.google.com/search?q=Shapley+values) for feature importance scoring. This is not a new thing and has been used in several competitions in the past e.g. [a old kernel.](https://www.kaggle.com/hmendonca/lightgbm-predictions-explained-with-shap-0-796). Shapley values are not the default metric from LGBM but can easily be implemented with the [SHAP library](https://github.com/slundberg/shap), which is also very useful for model explainability.   
# Once you start experimenting and adding many features to your model, these techniques became crucial to select only the best features and avoid over-fitting.

# ### ASHRAE - Great Energy Predictor III
# 
# I'll summarise below the insights from my brief data analysis and whatever is shared in public discussions.
# This is still highly under progress and will be updated when possible.
# 
# Quoting Chris Balbach (our Competition Host):
# > Consider the scenario laid out in this diagram. This competition simulates the modelling challenge presented at the end of the timeline when measured energy and weather conditions are known and the adjusted baseline energy must be calculated. 
# ![Energy Consumption/Demand vs Time](https://www.mdpi.com/make/make-01-00056/article_deploy/html/images/make-01-00056-g001-550.jpg)
# > For further details, I recommend reading [this paper](https://www.mdpi.com/2504-4990/1/3/56) by Clayton Miller.
# 
# We are actually using data from 2016 to predict the demand for both 2017 and 2018, which is a hard task. However, other public kernels have showed that you can give a relatively good estimate even with a very simple linear model.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os, gc
import random
import datetime

from tqdm import tqdm_notebook as tqdm

# matplotlib and seaborn for plotting
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# import plotly.offline as py
# py.init_notebook_mode(connected=True)
# from plotly.offline import init_notebook_mode, iplot
# init_notebook_mode(connected=True)
# import plotly.graph_objs as go
# import plotly.offline as offline
# offline.init_notebook_mode()

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, mean_squared_log_error

import lightgbm as lgb
import shap


# In[ ]:


path = '../input/ashrae-energy-prediction'
# Input data files are available in the "../input/" directory.
for dirname, _, filenames in os.walk(path):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# # Load data

# In[ ]:


get_ipython().run_cell_magic('time', '', "# unimportant features (see importance below)\nunimportant_cols = ['wind_direction', 'wind_speed', 'sea_level_pressure']\ntarget = 'meter_reading'\n\ndef load_data(source='train', path=path):\n    ''' load and merge all tables '''\n    assert source in ['train', 'test']\n    \n    building = pd.read_csv(f'{path}/building_metadata.csv', dtype={'building_id':np.uint16, 'site_id':np.uint8})\n    weather  = pd.read_csv(f'{path}/weather_{source}.csv', parse_dates=['timestamp'],\n                                                           dtype={'site_id':np.uint8, 'air_temperature':np.float16,\n                                                                  'cloud_coverage':np.float16, 'dew_temperature':np.float16,\n                                                                  'precip_depth_1_hr':np.float16},\n                                                           usecols=lambda c: c not in unimportant_cols)\n    df = pd.read_csv(f'{path}/{source}.csv', dtype={'building_id':np.uint16, 'meter':np.uint8}, parse_dates=['timestamp'])\n    df = df.merge(building, on='building_id', how='left')\n    df = df.merge(weather, on=['site_id', 'timestamp'], how='left')\n    return df\n\n# load and display some samples\ntrain = load_data('train')\ntrain.sample(5)")


# ### Test data

# In[ ]:


test = load_data('test')
test.sample(5)


# # EDA and sample statistics

# In[ ]:


print(f'Training from {train.timestamp.min()} to {train.timestamp.max()}, and predicting from {test.timestamp.min()} to {test.timestamp.max()}')


# ### Building types on each site location

# In[ ]:


building_counts = train.groupby(['primary_use', 'site_id']).building_id.nunique().to_frame('counts')
building_counts = building_counts.reset_index().pivot(index='primary_use', columns='site_id', values='counts')

fig, ax = plt.subplots(figsize=(16,8))
_ = sns.heatmap(building_counts, annot=True, cmap='Reds',
                xticklabels=building_counts.columns.values,
                yticklabels=building_counts.index.values)


# ### Distribution of Meter readings

# In[ ]:


# target's log-log histogram:
ax = np.log1p(train.meter_reading).hist()
ax.set_yscale('log')

# describe raw values first
train.meter_reading.describe()


# ### distribution in the types of meters

# In[ ]:


# check the distribution in the types of meters
meters = train.groupby('building_id').meter.nunique().to_frame()
ax = meters.hist()
_ = ax[0][0].set_title('Distribution of types of meters\n{0:electricity, 1:water, 2:steam, 3:hotwater}') # from the official starter kernel
# from the graphs it looks like steam and hotwater are reversed (e.g.: 3:steam, 2:hotwater) but that shouldn't make any difference to the model


# ### display a single time series (notice measurement errors and discontinuities)

# In[ ]:


building_id = 1258  # a building with all 4 meters:  meters[meters.meter == 4]
meters = train[train['building_id'] == building_id].meter.nunique()

for meter in range(meters):
    fig, ax = plt.subplots()
    plt.title(f'Building {building_id} Meter {meter}')
    ax2 = ax.twinx()
    # plot meter_reading
    idx = (train['building_id'] == building_id) & (train['meter'] == meter)
    dates = matplotlib.dates.date2num(train.loc[idx, 'timestamp'])
    ax2.plot_date(dates, train.loc[idx, 'meter_reading'], '-', label='meter_reading', alpha=0.8)
    # plot air_temperature
    dates = matplotlib.dates.date2num(train.loc[train['building_id'] == building_id, 'timestamp'])
    ax.plot_date(dates, train.loc[train['building_id'] == building_id, 'air_temperature'], '.', color='tab:cyan', label='air_temperature')
    ax.set_ylabel('air_temperature'); ax2.set_ylabel('meter_reading')
    ax.legend(loc='upper left'); ax2.legend(loc='upper right')


# ### now let's see what's the expected prediction in the test set for the same building

# In[ ]:


meter = 1 # pick a meter

train_sample = train[(train['building_id'] == building_id) & (train['meter'] == meter)]  # same train sample as above

test['meter_reading'] = 0.0
test_sample = test[(test['building_id'] == building_id) & (test['meter'] == meter)]  # and the same meter in the test set

fig, ax = plt.subplots(figsize=(16,4))
plt.title(f'Building {building_id} Meter {meter}')
ax.xaxis.set_tick_params(rotation=30, labelsize=10)
ax2 = ax.twinx()

# plot training sample
dates = matplotlib.dates.date2num(train_sample['timestamp'])
ax2.plot_date(dates, train_sample['meter_reading'], '-', label='train', alpha=0.8)
ax.plot_date(dates, train_sample['air_temperature'], '.', color='tab:cyan', label='air_temperature')

# plot test sample
dates = matplotlib.dates.date2num(test_sample['timestamp'])
ax2.plot_date(dates, test_sample['meter_reading'], '*', label='test', alpha=0.8)
ax.plot_date(dates, test_sample['air_temperature'], '.', color='tab:cyan', label='air_temperature')

ax.set_ylabel('air_temperature'); ax2.set_ylabel('meter_reading')
ax.legend(loc='upper left'); ax2.legend(loc='upper right')

del train_sample; del test_sample; del dates


# ### Check for missing data

# In[ ]:


# some feature stats
train.describe()


# In[ ]:


# the counts above expose the missing data (Should we drop or refill the missing data?)
print("Ratio of available data (not NAN's):")
data_ratios = train.count()/len(train)
data_ratios


# In[ ]:


# Is the same happening in the test set? Yes
print("Ratio of available data (not NAN's):")
test.count()/len(test)


# In[ ]:


# we can refill with averages
train.loc[:, data_ratios < 1.0].mean()


# # Preprocess data

# In[ ]:


class ASHRAE3Preprocessor(object):
    @classmethod
    def fit(cls, df, data_ratios=data_ratios):
        cls.avgs = df.loc[:,data_ratios < 1.0].mean()
        cls.pu_le = LabelEncoder()
        cls.pu_le.fit(df["primary_use"])

    @classmethod
    def transform(cls, df):
        df = df.fillna(cls.avgs) # refill NAN with averages
        df['primary_use'] = np.uint8(cls.pu_le.transform(df['primary_use']))  # encode labels

        # expand datetime into its components
        df['hour'] = np.uint8(df['timestamp'].dt.hour)
        df['day'] = np.uint8(df['timestamp'].dt.day)
        df['weekday'] = np.uint8(df['timestamp'].dt.weekday)
        df['month'] = np.uint8(df['timestamp'].dt.month)
        df['year'] = np.uint8(df['timestamp'].dt.year-2000)
        
        # parse and cast columns to a smaller type
        df.rename(columns={"square_feet": "log_square_feet"}, inplace=True)
        df['log_square_feet'] = np.float16(np.log(df['log_square_feet']))
        df['year_built'] = np.uint8(df['year_built']-1900)
        df['floor_count'] = np.uint8(df['floor_count'])
        
        # remove redundant columns
        for col in df.columns:
            if col in ['timestamp', 'row_id']:
                del df[col]
    
        # extract target column
        if 'meter_reading' in df.columns:
            df['meter_reading'] = np.log1p(df['meter_reading']).astype(np.float32) # comp metric uses log errors

        return df
        
ASHRAE3Preprocessor.fit(train)


# In[ ]:


train = ASHRAE3Preprocessor.transform(train)
train.sample(7)


# In[ ]:


train.dtypes


# # Feature ranked correlation

# In[ ]:


fig, ax = plt.subplots(figsize=(16,8))
# use a ranked correlation to catch nonlinearities (linear correlation is not important for boosted  trees e.g. LightGBM)
corr = train[[col for col in train.columns if col != 'year']].sample(100100).corr(method='spearman')
_ = sns.heatmap(corr, annot=True,
                xticklabels=corr.columns.values,
                yticklabels=corr.columns.values)


# # Train K folds

# In[ ]:


# force the model to use the weather data instead of dates, to avoid overfitting to the past history
features = [col for col in train.columns if col not in [target, 'year', 'month', 'day']]
# sample features
train[features].sample(5)


# In[ ]:


folds = 4
seed = 42
kf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
models = []
shap_values = np.zeros(train[features].shape)
shap_sampling = 125000  # reduce compute cost
oof_pred = np.zeros(train.shape[0])  # out of fold predictions

## stratify data by building_id
for i, (tr_idx, val_idx) in tqdm(enumerate(kf.split(train, train['building_id'])), total=folds):
    def fit_regressor(tr_idx, val_idx): # memory closure
        tr_x, tr_y = train[features].iloc[tr_idx],  train[target].iloc[tr_idx]
        vl_x, vl_y = train[features].iloc[val_idx], train[target].iloc[val_idx]
        print({'fold':i, 'train size':len(tr_x), 'eval size':len(vl_x)})

        tr_data = lgb.Dataset(tr_x, label=tr_y)
        vl_data = lgb.Dataset(vl_x, label=vl_y)  
        clf = lgb.LGBMRegressor(n_estimators=900,
                                learning_rate=0.33,
                                feature_fraction=0.9,
                                subsample=0.25,  # batches of 25% of the data
                                subsample_freq=1,
                                num_leaves=20,
                                lambda_l1=0.5,  # regularisation
                                lambda_l2=0.5,
                                seed=i,   # seed diversification
                                metric='rmse')
        clf.fit(tr_x, tr_y,
                eval_set=[(vl_x, vl_y)],
#                 early_stopping_rounds=50,
                verbose=150)
        # sample shapley values
        fold_importance = shap.TreeExplainer(clf).shap_values(vl_x[:shap_sampling])
        # out of fold predictions
        valid_prediticion = clf.predict(vl_x, num_iteration=clf.best_iteration_)
        oof_loss = np.sqrt(mean_squared_error(vl_y, valid_prediticion)) # target is already in log scale
        print(f'Fold:{i} RMSLE: {oof_loss:.4f}')
        return clf, fold_importance, valid_prediticion

    clf, shap_values[val_idx[:shap_sampling]], oof_pred[val_idx] = fit_regressor(tr_idx, val_idx)
    models.append(clf)
    
gc.collect()


# In[ ]:


oof_loss = np.sqrt(mean_squared_error(train[target], oof_pred)) # target is already in log scale
print(f'OOF RMSLE: {oof_loss:.4f}')

# save out of fold predictions
pd.DataFrame(oof_pred).to_csv(f'oof_preds{oof_loss:.4f}.csv.bz')


# # Feature importance

# ### Classical LGBM feature importance (gain on first fold)

# In[ ]:


_ = lgb.plot_importance(models[0], importance_type='gain')


# ### Shapley Values feature importance across all folds

# In[ ]:


shap.summary_plot(shap_values, train[features], plot_type="bar")


# In[ ]:


ma_shap = pd.DataFrame(sorted(zip(abs(shap_values).mean(axis=0), features), reverse=True),
                       columns=['Mean Abs Shapley', 'Feature']).set_index('Feature')
# fig, ax = plt.subplots(figsize=(2,6))
# _ = sns.heatmap(ma_shap, annot=True, cmap='Blues', fmt='.06f')
ma_shap


# ### Model explainability with SHAP

# In[ ]:


# visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)
shap.force_plot(shap.TreeExplainer(models[0]).expected_value, shap_values[0,:], train[features].iloc[0,:], matplotlib=True)


# # Check prediction

# In[ ]:


# load and pre-process test data
test = ASHRAE3Preprocessor.transform(test)
test.sample(5)


# ### now let's revisit the same meter we had initially looked at, and check the predictions for each fold

# In[ ]:


def recover_timestamp(x):
    ''' reassemble timestamp using date components '''
    return datetime.datetime.strptime(f'{x.year}-{x.month}-{x.day} {x.hour}', '%y-%m-%d %H')

fig, ax = plt.subplots(figsize=(16,4))
plt.title(f'Building {building_id} Meter {meter} on all {folds} prediction folds')
ax.xaxis.set_tick_params(rotation=30, labelsize=10)
ax2 = ax.twinx()

train_sample = train[(train['building_id'] == building_id) & (train['meter'] == meter)]  # same training sample as before
test_sample = test[(test['building_id'] == building_id) & (test['meter'] == meter)]   # and the same meter in the test set

# plot training sample
dates = matplotlib.dates.date2num(train_sample[['year', 'month', 'day', 'hour']].apply(recover_timestamp, axis=1))
ax.plot_date(dates, train_sample['air_temperature'], '.', color='tab:cyan', label='air_temperature')
ax2.plot_date(dates, np.expm1(train_sample['meter_reading']), '-', color='tab:blue', label='train')

# plot prediction sample
dates = matplotlib.dates.date2num(test_sample[['year', 'month', 'day', 'hour']].apply(recover_timestamp, axis=1))
ax.plot_date(dates, test_sample['air_temperature'], '.', color='tab:cyan', label='air_temperature')
for i,model in enumerate(models):
    ax2.plot_date(dates, np.expm1(model.predict(test_sample[features])), '-', label=f'prediction{i}', alpha=0.4)

ax.set_ylabel('air_temperature'); ax2.set_ylabel('meter_reading (+prediction)')
ax.legend(loc='upper left'); ax2.legend(loc='upper right')
_ = plt.show()

del test_sample; del train_sample
_ = gc.collect()


# In[ ]:


# # Check if all test buildings and meters are the same as in the training data
# train_buildings = np.unique(train[['building_id', 'meter']].values, axis=0)
# # del train; gc.collect()

# test_buildings  = np.unique(test[['building_id', 'meter']].values, axis=0)

# print(len(train_buildings), len(test_buildings))
# [b for b in test_buildings if b not in train_buildings]


# # Test Inference and Submission

# In[ ]:


# split test data into batches
set_size = len(test)
iterations = 50
batch_size = set_size // iterations

print(set_size, iterations, batch_size)
assert set_size == iterations * batch_size


# In[ ]:


meter_reading = []
for i in tqdm(range(iterations)):
    pos = i*batch_size
    fold_preds = [np.expm1(model.predict(test[features].iloc[pos : pos+batch_size])) for model in models]
    meter_reading.extend(np.mean(fold_preds, axis=0))

print(len(meter_reading))
assert len(meter_reading) == set_size


# ### Save submission

# In[ ]:


submission = pd.read_csv(f'{path}/sample_submission.csv')
submission['meter_reading'] = np.clip(meter_reading, a_min=0, a_max=None) # clip min at zero


# In[ ]:


submission.to_csv('submission.csv', index=False)
submission.head(9)


# In[ ]:


# prediction's log-log histogram:
plt.yscale('log')
_ = np.log1p(submission['meter_reading']).hist()

# describe raw submission
submission['meter_reading'].describe()

