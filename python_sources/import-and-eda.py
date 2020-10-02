#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


train = pd.read_csv('../input/train.csv', nrows = 100000, dtype={'acoustic_data':np.int16, 'time_to_failure': np.float64})
train.head(10)


# In[ ]:


fig, ax = plt.subplots(1,4,figsize=(13,4))
ax[0].plot(train.time_to_failure.values)
ax[0].set_xlabel('Index'); ax[0].set_ylabel('time to failure')
ax[1].plot(train.acoustic_data.values)
ax[1].set_xlabel('Index'); ax[1].set_ylabel('acoustic data')
ax[2].plot(np.diff(train.time_to_failure.values))
ax[2].set_xlabel('Index'); ax[2].set_ylabel('step of time_to_failure')
ax[3].plot(train.acoustic_data.values, train.time_to_failure.values, 'o', alpha=0.1)
ax[3].set_xlabel('acoustic data'); ax[3].set_ylabel('time to failure')
plt.tight_layout(pad=2)


# The time_to_failure does not appear to change from the table. It is in fact changing but very slowly by roughly 1 ns, then every 4096 point it jumps by 1 ms.
# 
# The 4096 segments could be related to the way the instrument takes the measurements. We can plot 3 segments to see what's going on.[](http://)

# In[ ]:


fig, ax = plt.subplots(1,4,figsize=(13,4))
n = 4096*3
ax[0].plot(train.time_to_failure.values[:n])
ax[0].set_xlabel('Index'); ax[0].set_ylabel('time to failure')
ax[1].plot(train.acoustic_data.values)
ax[1].set_xlabel('Index'); ax[1].set_ylabel('acoustic data')
ax[2].plot(np.diff(train.time_to_failure.values))
ax[2].set_xlabel('Index'); ax[2].set_ylabel('step of time_to_failure')
ax[3].plot(train.acoustic_data.values, train.time_to_failure.values, 'o', alpha=0.1)
ax[3].set_xlabel('acoustic data'); ax[3].set_ylabel('time to failure')
plt.tight_layout(pad=2)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'train = pd.read_csv(\'../input/train.csv\' , dtype={\'acoustic_data\': np.int16, \'time_to_failure\': np.float32})\n\n#visualize 1% of samples data, first 100 datapoints\ntrain_ad_sample_df = train[\'acoustic_data\'].values[::100]\ntrain_ttf_sample_df = train[\'time_to_failure\'].values[::100]\n\n#function for plotting based on both features\ndef plot_acc_ttf_data(train_ad_sample_df, train_ttf_sample_df, title="Acoustic data and time to failure: 1% sampled data"):\n    fig, ax1 = plt.subplots(figsize=(12, 8))\n    plt.title(title)\n    plt.plot(train_ad_sample_df, color=\'r\')\n    ax1.set_ylabel(\'acoustic data\', color=\'r\')\n    plt.legend([\'acoustic data\'], loc=(0.01, 0.95))\n    ax2 = ax1.twinx()\n    plt.plot(train_ttf_sample_df, color=\'b\')\n    ax2.set_ylabel(\'time to failure\', color=\'b\')\n    plt.legend([\'time to failure\'], loc=(0.01, 0.9))\n    plt.grid(True)\n\nplot_acc_ttf_data(train_ad_sample_df, train_ttf_sample_df)\n\ndel train_ad_sample_df\ndel train_ttf_sample_df')


# In[ ]:


get_ipython().run_cell_magic('time', '', "# Generate Features for the data set\ndef gen_features(X):\n    strain = []\n    strain.append(X.mean())\n    strain.append(X.std())\n    strain.append(X.min())\n    strain.append(X.max())\n    strain.append(X.kurtosis())\n    strain.append(X.skew())\n    strain.append(np.quantile(X,0.01))\n    strain.append(np.quantile(X,0.05))\n    strain.append(np.quantile(X,0.95))\n    strain.append(np.quantile(X,0.99))\n    strain.append(np.abs(X).max())\n    strain.append(np.abs(X).mean())\n    strain.append(np.abs(X).std())\n    return pd.Series(strain)\n\ntrain = pd.read_csv('../input/train.csv', iterator=True, chunksize=150_000, dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64})\n\nX_train = pd.DataFrame()\ny_train = pd.Series()\nfor df in train:\n    ch = gen_features(df['acoustic_data'])\n    X_train = X_train.append(ch, ignore_index=True)\n    y_train = y_train.append(pd.Series(df['time_to_failure'].values[-1]))")


# In[ ]:


get_ipython().run_cell_magic('time', '', "submission = pd.read_csv('../input/sample_submission.csv', index_col='seg_id')\nX_test = pd.DataFrame()\n\nfor seg_id in submission.index:\n    seg = pd.read_csv('../input/test/' + seg_id + '.csv')\n    ch = gen_features(seg['acoustic_data'])\n    X_test = X_test.append(ch, ignore_index=True)")


# In[ ]:


from catboost import CatBoostRegressor, Pool
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR


# In[ ]:


scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)


# In[ ]:


X_test_scaled = scaler.transform(X_test)


# # Catboost

# In[ ]:


get_ipython().run_cell_magic('time', '', "train_pool = Pool(X_train, y_train)\ncat_model = CatBoostRegressor(iterations=10000, loss_function='MAE', boosting_type='Ordered')\ncat_model.fit(X_train, y_train, silent=True)\ny_pred_cat = cat_model.predict(X_test)\ny_train_cat = cat_model.predict(X_train)\n\nsubmission['time_to_failure'] = y_pred_cat\nsubmission.to_csv('submission_cat.csv')")


# In[ ]:


from sklearn.metrics import mean_absolute_error

mean_absolute_error(y_train,y_train_cat)


# # SVR

# In[ ]:


get_ipython().run_cell_magic('time', '', '\nparameters = [{\'gamma\': [0.001, 0.005, 0.01, 0.02, 0.05, 0.1],\n               \'C\': [0.1, 0.2, 0.25, 0.5, 1, 1.5, 2]}]\n\nreg1 = GridSearchCV(SVR(kernel=\'rbf\', tol=0.01), parameters, cv=5, scoring=\'neg_mean_absolute_error\')\nreg1.fit(X_train_scaled, y_train.values.flatten())\ny_pred1 = reg1.predict(X_train_scaled)\n\nprint("Best CV score: {:.4f}".format(reg1.best_score_))\nprint(reg1.best_params_)\n\ny_pred_SVR = reg1.predict(X_test_scaled)\nsubmission[\'time_to_failure\'] = y_pred_SVR\nsubmission.to_csv(\'submission_SVR.csv\')')


# In[ ]:


from sklearn.metrics import mean_absolute_error

mean_absolute_error(y_train,y_pred1)


# In[ ]:



