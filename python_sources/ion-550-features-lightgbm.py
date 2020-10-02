#!/usr/bin/env python
# coding: utf-8

# # Ion 550 Features (LightGBM)
# 
# ### Credit for the Feature Engineering code goes to [artgor](https://www.kaggle.com/artgor). I took his FE template and implemented a simple Light GBM model to see how these aggregated features perform. Note that I had to reduce the number of Batch Size and Window Size options in order to avoid memory issues, so it isn't the full 550 feature set. I plan on trying out different Batch and Window Sizes to see if I can't land on ones that lead to best performance.
# 
# * ### For access to the original 550 feature code, see the dataset in this link: https://www.kaggle.com/artgor/ion-features

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import h5py
import gc
from sklearn.model_selection import KFold, train_test_split


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
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
    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose:
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (
                start_mem - end_mem) / start_mem))
    return df


def generate_features(data: pd.DataFrame,
                      batch_sizes: list,
                      window_sizes: list) -> pd.DataFrame:
    """
    Generate features for https://www.kaggle.com/c/liverpool-ion-switching

    Generate various aggregations over the data.

    Args:
        window_sizes: window sizes for rolling features
        batch_sizes: batch sizes for which features are aggregated
        data: original dataframe

    Returns:
        dataframe with generated features
    """
    for batch_size in batch_sizes:
        data['batch'] = ((data['time'] * 10_000) - 1) // batch_size
        data['batch_index'] = ((data['time'] * 10_000) - 1) - (data['batch'] * batch_size)
        data['batch_slices'] = data['batch_index'] // (batch_size / 10)
        data['batch_slices2'] = data['batch'].astype(str).str.zfill(3) + '_' + data['batch_slices'].astype(
            str).str.zfill(3)
        data['batch_slices3'] = data['batch_index'] // (batch_size / 5)
        data['batch_slices4'] = data['batch'].astype(str).str.zfill(3) + '_' + data['batch_slices3'].astype(
            str).str.zfill(3)
        data['batch_slices5'] = data['batch_index'] // (batch_size / 2)
        data['batch_slices6'] = data['batch'].astype(str).str.zfill(3) + '_' + data['batch_slices5'].astype(
            str).str.zfill(3)

        for agg_feature in ['batch', 'batch_slices2', 'batch_slices4', 'batch_slices6']:
            data[f"min_{agg_feature}_{batch_size}"] = data.groupby(agg_feature)['signal'].transform('min')
            data[f"max_{agg_feature}_{batch_size}"] = data.groupby(agg_feature)['signal'].transform('max')
            data[f"std_{agg_feature}_{batch_size}"] = data.groupby(agg_feature)['signal'].transform('std')
            data[f"mean_{agg_feature}_{batch_size}"] = data.groupby(agg_feature)['signal'].transform('mean')
            data[f"median_{agg_feature}_{batch_size}"] = data.groupby(agg_feature)['signal'].transform('median')

            data[f"mean_abs_chg_{agg_feature}_{batch_size}"] = data.groupby(agg_feature)['signal'].apply(
                lambda x: np.mean(np.abs(np.diff(x))))
            data[f"abs_max_{agg_feature}_{batch_size}"] = data.groupby(agg_feature)['signal'].apply(
                lambda x: np.max(np.abs(x)))
            data[f"abs_min_{agg_feature}_{batch_size}"] = data.groupby(agg_feature)['signal'].apply(
                lambda x: np.min(np.abs(x)))

            data[f"min_{agg_feature}_{batch_size}_diff"] = data[f"min_{agg_feature}_{batch_size}"] - data['signal']
            data[f"max_{agg_feature}_{batch_size}_diff"] = data[f"max_{agg_feature}_{batch_size}"] - data['signal']
            data[f"std_{agg_feature}_{batch_size}_diff"] = data[f"std_{agg_feature}_{batch_size}"] - data['signal']
            data[f"mean_{agg_feature}_{batch_size}_diff"] = data[f"mean_{agg_feature}_{batch_size}"] - data['signal']
            data[f"median_{agg_feature}_{batch_size}_diff"] = data[f"median_{agg_feature}_{batch_size}"] - data[
                'signal']

            data[f"range_{agg_feature}_{batch_size}"] = data[f"max_{agg_feature}_{batch_size}"] - data[
                f"min_{agg_feature}_{batch_size}"]
            data[f"maxtomin_{agg_feature}_{batch_size}"] = data[f"max_{agg_feature}_{batch_size}"] / data[
                f"min_{agg_feature}_{batch_size}"]
            data[f"abs_avg_{agg_feature}_{batch_size}"] = (data[f"abs_min_{agg_feature}_{batch_size}"] + data[
                f"abs_max_{agg_feature}_{batch_size}"]) / 2

            data[f'signal_shift+1_{agg_feature}_{batch_size}'] = data.groupby([agg_feature]).shift(1)['signal']
            data[f'signal_shift-1_{agg_feature}_{batch_size}'] = data.groupby([agg_feature]).shift(-1)['signal']
            data[f'signal_shift+2_{agg_feature}_{batch_size}'] = data.groupby([agg_feature]).shift(2)['signal']
            data[f'signal_shift-2_{agg_feature}_{batch_size}'] = data.groupby([agg_feature]).shift(-2)['signal']

            data[f"signal_shift+1_{agg_feature}_{batch_size}_diff"] = data[f"signal_shift+1_{agg_feature}_{batch_size}"] - data['signal']
            data[f"signal_shift-1_{agg_feature}_{batch_size}_diff"] = data[f"signal_shift-1_{agg_feature}_{batch_size}"] - data['signal']
            data[f"signal_shift+2_{agg_feature}_{batch_size}_diff"] = data[f"signal_shift+2_{agg_feature}_{batch_size}"] - data['signal']
            data[f"signal_shift-2_{agg_feature}_{batch_size}_diff"] = data[f"signal_shift-2_{agg_feature}_{batch_size}"] - data['signal']

        for window in window_sizes:
            window = min(batch_size, window)

            data["rolling_mean_" + str(window) + '_batch_' + str(batch_size)] =                 data.groupby('batch')['signal'].rolling(window=window).mean().reset_index()['signal']
            data["rolling_std_" + str(window) + '_batch_' + str(batch_size)] =                 data.groupby('batch')['signal'].rolling(window=window).std().reset_index()['signal']
            data["rolling_min_" + str(window) + '_batch_' + str(batch_size)] =                 data.groupby('batch')['signal'].rolling(window=window).min().reset_index()['signal']
            data["rolling_max_" + str(window) + '_batch_' + str(batch_size)] =                 data.groupby('batch')['signal'].rolling(window=window).max().reset_index()['signal']

            data[f'exp_Moving__{window}_{batch_size}'] = data.groupby('batch')['signal'].apply(
                lambda x: x.ewm(alpha=0.5, adjust=False).mean())
        data = reduce_mem_usage(data)
    data.fillna(0, inplace=True)

    return data


def read_data(path: str = ''):
    """
    Read train, test data

    Args:
        path: path to the data

    Returns:
        two dataframes
    """
    train_df = pd.read_csv(f'{path}/train.csv')
    test_df = pd.read_csv(f'{path}/test.csv')
    return train_df, test_df


# In[ ]:


directory = '/kaggle/input/liverpool-ion-switching/'
train,test = read_data(directory)
sample_submission = pd.read_csv(f'{directory}/sample_submission.csv') 


# In[ ]:


batch_sizes = [25000]
window_sizes = [10, 25, 50, 5000, 10000]

generated_train = generate_features(train, batch_sizes, window_sizes)

del train
gc.collect()


# In[ ]:


feats = generated_train.columns
print(feats)
feats = np.delete(feats,[2,3,4,5,6,7,8,9,10]) # Delete Target from features
target = ['open_channels']


# In[ ]:


generated_test = generate_features(test, batch_sizes, window_sizes)

del test
gc.collect()


# In[ ]:


import lightgbm as lgb
params = {'learning_rate': 0.05, 'max_depth': -1, 'num_leaves':200, 'metric': 'rmse', 'random_state': 42, 'n_jobs':-1, 'sample_fraction':0.33} 


# In[ ]:


# Thanks to https://www.kaggle.com/siavrez/simple-eda-model
from sklearn.metrics import confusion_matrix, f1_score, mean_absolute_error, make_scorer
def MacroF1Metric(preds, dtrain):
    labels = dtrain.get_label()
    preds = np.round(np.clip(preds, 0, 10)).astype(int)
    score = f1_score(labels, preds, average = 'macro')
    return ('MacroF1Metric', score, True)


# In[ ]:


x1, x2, y1, y2 = train_test_split(generated_train[feats], generated_train[target], test_size=0.3, random_state=42)
model = lgb.train(params, lgb.Dataset(x1, y1), 2000,  lgb.Dataset(x2, y2), verbose_eval=100, early_stopping_rounds=100, feval=MacroF1Metric)
del x1, x2, y1, y2
gc.collect()


# In[ ]:


preds_ = model.predict(generated_test[feats], num_iteration=model.best_iteration)


# In[ ]:


sample_submission['open_channels'] = np.round(np.clip(preds_, 0, 10)).astype(int)
sample_submission.to_csv('submission.csv', index=False, float_format='%.4f')
display(sample_submission.head())


# In[ ]:


fig =  plt.figure(figsize = (25,25))
axes = fig.add_subplot(111)
lgb.plot_importance(model,ax = axes,height = 0.5)
plt.show();plt.close()

