#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from tqdm import tqdm_notebook as tqdm
import numpy as np
import pandas as pd

raw = pd.read_csv('../input/train.csv', dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64})


# In[ ]:


from tsfresh.feature_extraction import feature_calculators
import librosa
import pywt


np.random.seed(1337)
noise = np.random.normal(0, 0.5, 150_000)


def denoise_signal_simple(x, wavelet='db4', level=1):
    coeff = pywt.wavedec(x, wavelet, mode="per")
    #univeral threshold
    uthresh = 10
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeff[1:])
    # Reconstruct the signal using the thresholded coefficients
    return pywt.waverec(coeff, wavelet, mode='per')


def feature_gen(z):
    X = pd.DataFrame(index=[0], dtype=np.float64)
    
    z = z + noise
    z = z - np.median(z)

    den_sample_simple = denoise_signal_simple(z)
    mfcc = librosa.feature.mfcc(z)
    mfcc_mean = mfcc.mean(axis=1)
    percentile_roll50_std_20 = np.percentile(pd.Series(z).rolling(50).std().dropna().values, 20)
    
    X['var_num_peaks_2_denoise_simple'] = feature_calculators.number_peaks(den_sample_simple, 2)
    X['var_percentile_roll50_std_20'] = percentile_roll50_std_20
    X['var_mfcc_mean18'] = mfcc_mean[18]
    X['var_mfcc_mean4'] = mfcc_mean[4]
    
    return X


# In[ ]:


from joblib import Parallel, delayed
import scipy as sp
import itertools
import gc

def parse_sample(sample, start):
    delta = feature_gen(sample['acoustic_data'].values)
    delta['start'] = start
    delta['target'] = sample['time_to_failure'].values[-1]
    return delta
    
def sample_train_gen(df, segment_size=150_000, indices_to_calculate=[0]):
    result = Parallel(n_jobs=4, temp_folder="/tmp", max_nbytes=None, backend="multiprocessing")(delayed(parse_sample)(df[int(i) : int(i) + segment_size], int(i)) 
                                                                                                for i in tqdm(indices_to_calculate))
    data = [r.values for r in result]
    data = np.vstack(data)
    X = pd.DataFrame(data, columns=result[0].columns)
    X = X.sort_values("start")
    return X

def parse_sample_test(seg_id):
    sample = pd.read_csv('../input/test/' + seg_id + '.csv', dtype={'acoustic_data': np.int32})
    delta = feature_gen(sample['acoustic_data'].values)
    delta['seg_id'] = seg_id
    return delta

def sample_test_gen():
    X = pd.DataFrame()
    submission = pd.read_csv('../input/sample_submission.csv', index_col='seg_id')
    result = Parallel(n_jobs=4, temp_folder="/tmp", max_nbytes=None, backend="multiprocessing")(delayed(parse_sample_test)(seg_id) for seg_id in tqdm(submission.index))
    data = [r.values for r in result]
    data = np.vstack(data)
    X = pd.DataFrame(data, columns=result[0].columns)
    return X

indices_to_calculate = raw.index.values[::150_000][:-1]

train = sample_train_gen(raw, indices_to_calculate=indices_to_calculate)
gc.collect()
test = sample_test_gen()


# In[ ]:


etq_meta = [
{"start":0,         "end":5656574},
{"start":5656574,   "end":50085878},
{"start":50085878,  "end":104677356},
{"start":104677356, "end":138772453},
{"start":138772453, "end":187641820},
{"start":187641820, "end":218652630},
{"start":218652630, "end":245829585},
{"start":245829585, "end":307838917},
{"start":307838917, "end":338276287},
{"start":338276287, "end":375377848},
{"start":375377848, "end":419368880},
{"start":419368880, "end":461811623},
{"start":461811623, "end":495800225},
{"start":495800225, "end":528777115},
{"start":528777115, "end":585568144},
{"start":585568144, "end":621985673},
{"start":621985673, "end":629145480},
]

for i, etq in enumerate(etq_meta):
    train.loc[(train['start'] + 150_000 >= etq["start"]) & (train['start'] <= etq["end"] - 150_000), "eq"] = i

train_sample = train[train["eq"].isin([2, 7, 0, 4, 11, 13, 9, 1, 14, 10])]
    


# In[ ]:


print(f"Mean:   {train_sample['target'].mean():.4}")
print(f"Median: {train_sample['target'].median():.4}")


# In[ ]:


from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from numpy import random
import lightgbm as lgb

random.seed(1234)

features = ['var_num_peaks_2_denoise_simple','var_percentile_roll50_std_20','var_mfcc_mean4',  'var_mfcc_mean18']
target = train_sample["target"].values

train_X = train_sample[features].values
test_X = test[features].values

submission = pd.read_csv('../input/sample_submission.csv', index_col='seg_id')
oof = np.zeros(len(train_X))
prediction = np.zeros(len(submission))

n_fold = 3

kf = KFold(n_splits=n_fold, shuffle=True, random_state=1337)
kf = list(kf.split(np.arange(len(train_sample))))

for fold_n, (train_index, valid_index) in enumerate(kf):
    print('Fold', fold_n)

    trn_data = lgb.Dataset(train_X[train_index], label=target[train_index])
    val_data = lgb.Dataset(train_X[valid_index], label=target[valid_index])
    
    params = {'num_leaves': 4,
      'min_data_in_leaf': 5,
      'objective':'fair',
      'max_depth': -1,
      'learning_rate': 0.02,
      "boosting": "gbdt",
      'boost_from_average': True,
      "feature_fraction": 0.9,
      "bagging_freq": 1,
      "bagging_fraction": 0.5,
      "bagging_seed": 0,
      "metric": 'mae',
      "verbosity": -1,
      'max_bin': 500,
      'reg_alpha': 0,
      'reg_lambda': 0,
      'seed': 0,
      'n_jobs': 1
      }

    clf = lgb.train(params, trn_data, 1000000, valid_sets = [trn_data, val_data], verbose_eval=1000, early_stopping_rounds = 1000)

    oof[valid_index] += clf.predict(train_X[valid_index], num_iteration=clf.best_iteration)
    prediction += clf.predict(test_X, num_iteration=clf.best_iteration)

prediction /= n_fold

print('\nMAE: ', mean_absolute_error(target, oof))


# In[ ]:


submission['time_to_failure'] = prediction 
print(submission.head())
submission.to_csv('submission.csv')

