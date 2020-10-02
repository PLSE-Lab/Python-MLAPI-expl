#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install fastai==0.7.0')


# In[ ]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


from fastai.imports import *
from fastai.structured import *

from pandas_summary import DataFrameSummary
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from IPython.display import display

from sklearn import metrics


# In[ ]:





# In[ ]:


get_ipython().run_cell_magic('time', '', 'train = pd.read_csv("../input/train.csv",\n                     dtype={"acoustic_data": np.int16, "time_to_failure": np.float64},\n                     nrows=100_000_000)')


# In[ ]:


rows = 10000
segments = int(np.floor(train.shape[0] / rows))

df_train = pd.DataFrame(index=range(segments), dtype=np.float64,
                       columns=['ave', 'std', 'max', 'min', 'sum', 'range', 'time_to_failure'])

for segment in tqdm(range(segments)):
    seg = train.iloc[segment*rows:segment*rows+rows]
    x = seg['acoustic_data'].values
    y = seg['time_to_failure'].values[-1]
    
    df_train.loc[segment, 'time_to_failure'] = y
    
    df_train.loc[segment, 'ave'] = x.mean()
    df_train.loc[segment, 'std'] = x.std()
    df_train.loc[segment, 'max'] = x.max()
    df_train.loc[segment, 'min'] = x.min()
    df_train.loc[segment, 'sum'] = x.sum()
    df_train.loc[segment, 'range'] = x.max()-x.min()


# In[ ]:


df_train = df_train.sample(frac=1, axis=1).reset_index(drop=True)


# In[ ]:


df_train.time_to_failure = np.log(df_train.time_to_failure)


# In[ ]:


df_trn, y_trn, nas = proc_df(df_train, 'time_to_failure')


# In[ ]:


def split_vals(a,n): 
    return a[:n].copy(), a[n:].copy()
    
train_required_ratio = 0.80
n_trn = int(len(df_trn) * train_required_ratio)

X_train, X_valid = split_vals(df_trn, n_trn) 


# In[ ]:


y_train, y_valid = split_vals(y_trn, n_trn)
X_train.shape, X_valid.shape


# In[ ]:


def print_score(m):
    res = [metrics.mean_absolute_error(m.predict(X_train), y_train), metrics.mean_absolute_error(m.predict(X_valid), y_valid),
                m.score(X_train, y_train), m.score(X_valid, y_valid)]
    if hasattr(m, 'oob_score_'): res.append(m.oob_score_)
    print(res)


# In[ ]:


m = RandomForestRegressor(n_estimators=100, n_jobs=-1)
m.fit(X_train, y_train)
print_score(m)


# In[ ]:


preds = np.stack([t.predict(X_valid) for t in m.estimators_])
preds[:,0], np.mean(preds[:,0]), y_valid[0]


# In[ ]:


submission = pd.read_csv('../input/sample_submission.csv', index_col='seg_id')
X_test = pd.DataFrame(columns=X_train.columns, dtype=np.float64, index=submission.index)
for seg_id in X_test.index:
    seg = pd.read_csv('../input/test/' + seg_id + '.csv')
    
    x = seg['acoustic_data'].values
    
    X_test.loc[seg_id, 'ave'] = x.mean()
    X_test.loc[seg_id, 'std'] = x.std()
    X_test.loc[seg_id, 'max'] = x.max()
    X_test.loc[seg_id, 'min'] = x.min()
    X_test.loc[seg_id, 'sum'] = x.sum()
    X_test.loc[seg_id, 'range'] = x.max()-x.min()


# In[ ]:





# In[ ]:


preds = np.stack([t.predict(X_test) for t in m.estimators_])
preds2 = np.mean(preds, axis=0)
preds2


# In[ ]:


X_test2 = X_test
X_test2['time_to_failure'] = preds2


# In[ ]:


X_test2.info()


# In[ ]:


pred_df = X_test2.drop(columns=['min','range','ave','sum','std','max'])


# In[ ]:


pred_df.rename_axis('seg_id')


# In[ ]:


pred_df.info()


# In[ ]:


pred_df.to_csv('datapreds.csv', sep=',', index=True, quotechar=' ')


# In[ ]:




