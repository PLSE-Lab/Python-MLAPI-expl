#!/usr/bin/env python
# coding: utf-8

# # Acknowledgements
# 
# Original Kernel: https://www.kaggle.com/yamsam/ashrae-leak-validation-and-more/notebook#Leak-Validation-for-public-kernels(not-used-leak-data),
# 
# https://www.kaggle.com/khoongweihao/ashrae-leak-validation-bruteforce-heuristic-search
# 
# Additions: Added a search method based on gradient update

# # All we need is Leak Validation(LV) ?
# 
# * **if you like this kernel, please upvote original kernels.**
# * update site-4 and site-15
# * Turn GPU on for better performance

# this kernel is still work in progress, but i hope you can find something usefull from this.

# In[ ]:


import gc
import os
from pathlib import Path
import random
import sys

from tqdm import tqdm_notebook as tqdm
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns

from IPython.core.display import display, HTML

# --- plotly ---
from plotly import tools, subplots
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.express as px
import plotly.figure_factory as ff

# --- models ---
from sklearn import preprocessing
from sklearn.model_selection import KFold
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

from sklearn.metrics import mean_squared_error


# In[ ]:



# Original code from https://www.kaggle.com/gemartin/load-data-reduce-memory-usage by @gemartin
# Modified to support timestamp type, categorical type
# Modified to add option to use float16 or not. feather format does not support float16.
from pandas.api.types import is_datetime64_any_dtype as is_datetime
from pandas.api.types import is_categorical_dtype

def reduce_mem_usage(df, use_float16=False):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        if is_datetime(df[col]) or is_categorical_dtype(df[col]):
            # skip datetime type or categorical type
            continue
        col_type = df[col].dtype
        
        if col_type != object:
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
                if use_float16 and c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df


# In[ ]:


get_ipython().run_cell_magic('time', '', "root = Path('../input/ashrae-feather-format-for-fast-loading')\n\ntrain_df = pd.read_feather(root/'train.feather')\ntest_df = pd.read_feather(root/'test.feather')\n#weather_train_df = pd.read_feather(root/'weather_train.feather')\n#weather_test_df = pd.read_feather(root/'weather_test.feather')\nbuilding_meta_df = pd.read_feather(root/'building_metadata.feather')")


# In[ ]:


# i'm now using my leak data station kernel to shortcut.
leak_df = pd.read_feather('../input/ashrae-leak-data-station/leak.feather')

leak_df.fillna(0, inplace=True)
leak_df = leak_df[(leak_df.timestamp.dt.year > 2016) & (leak_df.timestamp.dt.year < 2019)]
leak_df.loc[leak_df.meter_reading < 0, 'meter_reading'] = 0 # remove large negative values
leak_df = leak_df[leak_df.building_id!=245]


# In[ ]:


leak_df.meter.value_counts()


# In[ ]:


print (leak_df.duplicated().sum())


# In[ ]:


print (len(leak_df) / len(train_df))


# In[ ]:


get_ipython().system(' ls ../input')


# In[ ]:


del train_df
gc.collect()


# # Leak Validation for public kernels(not used leak data)

# In[ ]:


sample_submission1 = pd.read_csv('../input/ashrae-kfold-lightgbm-without-leak-1-08/submission.csv', index_col=0)
sample_submission2 = pd.read_csv('../input/ashrae-half-and-half/submission.csv', index_col=0)
sample_submission3 = pd.read_csv('../input/ashrae-highway-kernel-route4/submission.csv', index_col=0)


# In[ ]:


test_df['pred1'] = sample_submission1.meter_reading
test_df['pred2'] = sample_submission2.meter_reading
test_df['pred3'] = sample_submission3.meter_reading

test_df.loc[test_df.pred3<0, 'pred3'] = 0 

del  sample_submission1,  sample_submission2,  sample_submission3
gc.collect()

test_df = reduce_mem_usage(test_df)
leak_df = reduce_mem_usage(leak_df)


# In[ ]:


leak_df = leak_df.merge(test_df[['building_id', 'meter', 'timestamp', 'pred1', 'pred2', 'pred3', 'row_id']], left_on = ['building_id', 'meter', 'timestamp'], right_on = ['building_id', 'meter', 'timestamp'], how = "left")
leak_df = leak_df.merge(building_meta_df[['building_id', 'site_id']], on='building_id', how='left')


# In[ ]:


leak_df['pred1_l1p'] = np.log1p(leak_df.pred1)
leak_df['pred2_l1p'] = np.log1p(leak_df.pred2)
leak_df['pred3_l1p'] = np.log1p(leak_df.pred3)
leak_df['meter_reading_l1p'] = np.log1p(leak_df.meter_reading)


# In[ ]:


leak_df.head()


# In[ ]:


leak_df[leak_df.pred1_l1p.isnull()]


# In[ ]:


#ashrae-kfold-lightgbm-without-leak-1-08
sns.distplot(leak_df.pred1_l1p)
sns.distplot(leak_df.meter_reading_l1p)

leak_score = np.sqrt(mean_squared_error(leak_df.pred1_l1p, leak_df.meter_reading_l1p))
print ('score1=', leak_score)


# In[ ]:


#ashrae-half-and-half
sns.distplot(leak_df.pred2_l1p)
sns.distplot(leak_df.meter_reading_l1p)

leak_score = np.sqrt(mean_squared_error(leak_df.pred2_l1p, leak_df.meter_reading_l1p))
print ('score2=', leak_score)


# In[ ]:


# meter split based
sns.distplot(leak_df.pred3_l1p)
sns.distplot(leak_df.meter_reading_l1p)

leak_score = np.sqrt(mean_squared_error(leak_df.pred3_l1p, leak_df.meter_reading_l1p))
print ('score3=', leak_score)


# In[ ]:


# ashrae-kfold-lightgbm-without-leak-1-08 looks best


# # Leak Validation for Blending

# A one idea how we can use LV usefull is blending. We probably can find best blending method without LB probing and it's means we can save our submission.

# In[ ]:


leak_df['mean_pred'] = np.mean(leak_df[['pred1', 'pred2', 'pred3']].values, axis=1)
leak_df['mean_pred_l1p'] = np.log1p(leak_df.mean_pred)
leak_score = np.sqrt(mean_squared_error(leak_df.mean_pred_l1p, leak_df.meter_reading_l1p))


sns.distplot(leak_df.mean_pred_l1p)
sns.distplot(leak_df.meter_reading_l1p)

print ('mean score=', leak_score)


# In[ ]:


leak_df['median_pred'] = np.median(leak_df[['pred1', 'pred2', 'pred3']].values, axis=1)
leak_df['median_pred_l1p'] = np.log1p(leak_df.median_pred)
leak_score = np.sqrt(mean_squared_error(leak_df.median_pred_l1p, leak_df.meter_reading_l1p))

sns.distplot(leak_df.median_pred_l1p)
sns.distplot(leak_df.meter_reading_l1p)

print ('meadian score=', leak_score)


# Ummm... it looks mean blending is beter than median blending

# # Find Best Weight

# In[ ]:


N = 10
scores = np.zeros(N,)
for i in range(N):
    p = i * 1./N
    v = p * leak_df['pred1'].values + (1.-p) * leak_df ['pred3'].values
    vl1p = np.log1p(v)
    scores[i] = np.sqrt(mean_squared_error(vl1p, leak_df.meter_reading_l1p))


# In[ ]:


plt.plot(scores)


# In[ ]:


best_weight = np.argmin(scores) *  1./N
print (scores.min(), best_weight)


# In[ ]:


# and more
scores = np.zeros(N,)
for i in range(N):
    p = i * 1./N
    v =  p * (best_weight * leak_df['pred1'].values + (1.-best_weight) * leak_df ['pred3'].values) + (1.-p) * leak_df ['pred2'].values
    vl1p = np.log1p(v)
    scores[i] = np.sqrt(mean_squared_error(vl1p, leak_df.meter_reading_l1p))


# In[ ]:


plt.plot(scores)


# In[ ]:


best_weight2 = np.argmin(scores) *  1./N
print (scores.min(), best_weight2)
# its seams better than simple mean 0.92079717


# # Combination Search by using gradient descent

# $x$ \- input 
# 
# $y$ \- target
# 
# $w$ \- weights
# 
# Let $f(x)=w^\top x$, we want to minimize
# 
# $$L(x,y)=(\log(f(x)+1)-\log(y+1))^2$$

# In[ ]:


# Prepare data

X_train = np.array([leak_df['pred1'].values,leak_df['pred2'].values, leak_df['pred3'].values]).T
y_train = leak_df.meter_reading_l1p


# In[ ]:


import tensorflow as tf

weights = tf.Variable([[0.3],[0.3],[0.3]])
steps = 1000

lr = 0.1

opt = tf.optimizers.SGD(lr)

#Speed up the train step by precompiling
@tf.function()
def train_step(opt):
    with tf.GradientTape() as tape:
        y  = tf.matmul(X_train, weights)[:,0]
        loss = tf.reduce_mean((tf.math.log1p(y) - y_train) ** 2)
    grads = tape.gradient(loss, weights)
    opt.apply_gradients([(grads, weights)])
    
    return loss

prev_loss = 9999
for i in range(steps):
    loss = train_step(opt)
    if loss > prev_loss:
        lr /= 2
        opt.lr = lr
        
    prev_loss = loss
    print(f'step: {i} {loss.numpy()}')   
    


# In[ ]:


# MSE
np.sqrt(loss)


# # Submit

# In[ ]:


sample_submission = pd.read_feather(os.path.join(root, 'sample_submission.feather'))

ws = weights.numpy()

w1 = ws[0,0]
w2 = ws[1,0]
w3 = ws[2,0]
print("The weights are: w1=" + str(w1) + ", w2=" + str(w2) + ", w3=" + str(w3))

sample_submission['meter_reading'] = w1 * test_df.pred1 +  w2 * test_df.pred2  + w3 * test_df.pred3
sample_submission.loc[sample_submission.meter_reading < 0, 'meter_reading'] = 0


# In[ ]:


leak_df.head()


# In[ ]:


sns.distplot(np.log1p(sample_submission.meter_reading))


# In[ ]:


leak_df = leak_df[['meter_reading', 'row_id']].set_index('row_id').dropna()
sample_submission.loc[leak_df.index, 'meter_reading'] = leak_df['meter_reading']


# In[ ]:


sns.distplot(np.log1p(sample_submission.meter_reading))


# In[ ]:


sample_submission.head()


# In[ ]:


sample_submission.to_csv('submission.csv', index=False, float_format='%.4f')


# # Future Work
# 
# - Do cross-validation on leak data?
