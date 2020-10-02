#!/usr/bin/env python
# coding: utf-8

# <h1><center><font size="6">Feature selection simplified</font></center></h1>
# 
# 

# # <a id='1'>Introduction</a>  
# 
# ## Simulated earthquake experiment
# The data are from an experiment conducted on rock in a double direct shear geometry subjected to bi-axial loading, a classic laboratory earthquake model.
# 
# Two fault gouge layers are sheared simultaneously while subjected to a constant normal load and a prescribed shear velocity. The laboratory faults fail in repetitive cycles of stick and slip that is meant to mimic the cycle of loading and failure on tectonic faults. While the experiment is considerably simpler than a fault in Earth, it shares many physical characteristics.
# 
# Los Alamos' initial work showed that the prediction of laboratory earthquakes from continuous seismic data is possible in the case of quasi-periodic laboratory seismic cycles.
# 
# ## Competition
# In this competition, the team has provided a much more challenging dataset with considerably more aperiodic earthquake failures.
# Objective of the competition is to predict the failures for each test set.

# # <a id='2'>Prepare the data analysis</a>  
# 
# ## Load packages

# In[ ]:


import gc
import os
import time
import logging
import datetime
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
import lightgbm as lgb
from scipy import stats
from tqdm import tqdm_notebook
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')


# ## Load data

# In[ ]:


PATH="../input/"
os.listdir(PATH)


# In[ ]:


print("There are {} files in test folder".format(len(os.listdir(os.path.join(PATH, 'test' )))))


# In[ ]:


get_ipython().run_cell_magic('time', '', "train_df = pd.read_csv(os.path.join(PATH,'train.csv'), dtype={'acoustic_data': np.int16, 'time_to_failure': np.float32})")


# In[ ]:


print("Train: rows:{} cols:{}".format(train_df.shape[0], train_df.shape[1]))


# # <a id='3'>Calculate aggregated features</a>  

# In[ ]:


rows = 150000
segments = int(np.floor(train_df.shape[0] / rows))
print("Number of segments: ", segments)


# In[ ]:


train_X = pd.DataFrame(index=range(segments), dtype=np.float64)
train_y = pd.DataFrame(index=range(segments), dtype=np.float64, columns=['time_to_failure'])


# In[ ]:


def create_features(seg_id, seg, X):
    xc = pd.Series(seg['acoustic_data'].values)   
    zc = np.fft.fft(xc)
    realFFT = np.real(zc)
    imagFFT = np.imag(zc)
    X.loc[seg_id, 'mean'] = xc.mean()
    X.loc[seg_id, 'std'] = xc.std()
    X.loc[seg_id, 'max'] = xc.max()
    X.loc[seg_id, 'min'] = xc.min()
    X.loc[seg_id, 'sum'] = xc.sum()
    X.loc[seg_id, 'mad'] = xc.mad()
    X.loc[seg_id, 'kurt'] = xc.kurtosis()
    X.loc[seg_id, 'skew'] = xc.skew()
    X.loc[seg_id, 'med'] = xc.median()
    X.loc[seg_id, 'abs_mean'] = np.abs(xc).mean()
    X.loc[seg_id, 'q95'] = np.quantile(xc, 0.95)
    X.loc[seg_id, 'q99'] = np.quantile(xc, 0.99)
    X.loc[seg_id, 'q05'] = np.quantile(xc, 0.05)
    X.loc[seg_id, 'q01'] = np.quantile(xc, 0.01)
    X.loc[seg_id, 'Rmean'] = realFFT.mean()
    X.loc[seg_id, 'Rstd'] = realFFT.std()
    X.loc[seg_id, 'Rmax'] = realFFT.max()
    X.loc[seg_id, 'Rmin'] = realFFT.min()
    X.loc[seg_id, 'Imean'] = imagFFT.mean()
    X.loc[seg_id, 'Istd'] = imagFFT.std()
    X.loc[seg_id, 'Imax'] = imagFFT.max()
    X.loc[seg_id, 'Imin'] = imagFFT.min()
    X.loc[seg_id, 'std_first_50000'] = xc[:50000].std()
    X.loc[seg_id, 'std_last_50000'] = xc[-50000:].std()
    X.loc[seg_id, 'std_first_25000'] = xc[:25000].std()
    X.loc[seg_id, 'std_last_25000'] = xc[-25000:].std()
    X.loc[seg_id, 'std_first_10000'] = xc[:10000].std()
    X.loc[seg_id, 'std_last_10000'] = xc[-10000:].std()


# In[ ]:


# iterate over all segments
for seg_id in tqdm_notebook(range(segments)):
    seg = train_df.iloc[seg_id*rows:seg_id*rows+rows]
    create_features(seg_id, seg, train_X)
    train_y.loc[seg_id, 'time_to_failure'] = seg['time_to_failure'].values[-1]


# In[ ]:


submission = pd.read_csv('../input/sample_submission.csv', index_col='seg_id')
test_X = pd.DataFrame(columns=train_X.columns, dtype=np.float64, index=submission.index)


# In[ ]:


for seg_id in tqdm_notebook(test_X.index):
    seg = pd.read_csv('../input/test/' + seg_id + '.csv')
    create_features(seg_id, seg, test_X)


# In[ ]:


print("Train X: {} y: {} Test X: {}".format(train_X.shape, train_y.shape, test_X.shape))


# In[ ]:


train_X.head()


# In[ ]:


test_X.head()


# In[ ]:


scaler = StandardScaler()
scaler.fit(pd.concat([train_X, test_X]))
scaled_train_X = pd.DataFrame(scaler.transform(train_X), columns=train_X.columns)
scaled_test_X = pd.DataFrame(scaler.transform(test_X), columns=test_X.columns)


# In[ ]:


# feature lists 
features = ['mean', 'std', 'max', 'min', 'sum', 'mad', 'kurt', 'skew', 'med','abs_mean', 'q95', 'q99', 'q05', 'q01', 'Rmean', 'Rstd', 'Rmax','Rmin', 'Imean', 'Istd', 'Imax', 'Imin', 'std_first_50000', 'std_last_50000', 'std_first_25000','std_last_25000', 'std_first_10000','std_last_10000']


# ## Aggregated features and time to failure
# 
# showing aggregated features and time to failure on the same graph. 

# In[ ]:


def plot_acc_agg_ttf_data(features, title="Averaged accoustic data and ttf"):
    for feature in features:
        fig, ax1 = plt.subplots(figsize=(16, 8))
        plt.title('Averaged accoustic data ({}) and time to failure'.format(feature))
        plt.plot(train_X[feature], color='r')
        ax1.set_xlabel('training samples')
        ax1.set_ylabel('acoustic data ({})'.format(feature), color='r')
        plt.legend(['acoustic data ({})'.format(feature)], loc=(0.01, 0.95))
        ax2 = ax1.twinx()
        plt.plot(train_y, color='b')
        ax2.set_ylabel('time to failure', color='b')
        plt.legend(['time to failure'], loc=(0.01, 0.9))
        plt.grid(True)


# In[ ]:


plot_acc_agg_ttf_data(features)


# <br>
# I hope this kernel will help you to sore high.
# <br>

# # <a id='6'>References</a>  
# 
# [1] LANL Earthquake Prediction, https://www.kaggle.com/c/LANL-Earthquake-Prediction  
# [2] Shaking Earth, https://www.kaggle.com/allunia/shaking-earth  
# [3] Earthquake FE - more features and samles, https://www.kaggle.com/artgor/earthquakes-fe-more-features-and-samples  
# [4] LANL Earthquake New Approach EDA
#  https://www.kaggle.com/gpreda/lanl-earthquake-new-approach-eda/notebook   
# 
# 
