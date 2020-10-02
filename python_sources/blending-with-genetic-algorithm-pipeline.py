#!/usr/bin/env python
# coding: utf-8

# ### Based on Kernels: 
# 
# https://www.kaggle.com/khoongweihao/ashrae-leak-validation-bruteforce-heuristic-search
# 
# https://www.kaggle.com/roydatascience/ashrae-stratified-kfold-lightgbm
# 

# In[ ]:


import gc
import os
from pathlib import Path
import random
import sys

from tqdm import tqdm_notebook as tqdm
import numpy as np 
import pandas as pd 

import matplotlib.pyplot as plt
import seaborn as sns

from plotly import tools, subplots
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.express as px
import plotly.figure_factory as ff

from sklearn import preprocessing
from sklearn.model_selection import KFold
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


get_ipython().run_cell_magic('time', '', "root = Path('../input/ashrae-feather-format-for-fast-loading')\ntrain_df = pd.read_feather(root/'train.feather')\ntest_df = pd.read_feather(root/'test.feather')\nbuilding_meta_df = pd.read_feather(root/'building_metadata.feather')")


# In[ ]:


leak_df = pd.read_feather('../input/ashrae-leak-data-station/leak.feather')

leak_df.fillna(0, inplace=True)
leak_df = leak_df[(leak_df.timestamp.dt.year > 2016) & (leak_df.timestamp.dt.year < 2019)]
leak_df.loc[leak_df.meter_reading < 0, 'meter_reading'] = 0 
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


# # Leak Validation for public kernels without leaks

# In[ ]:


sample_submission1 = pd.read_csv('../input/ashrae-kfold-lightgbm-without-leak-1-08/submission.csv', index_col=0)
sample_submission2 = pd.read_csv('../input/ashrae-half-and-half/submission.csv', index_col=0)
sample_submission3 = pd.read_csv('../input/ashrae-highway-kernel-route4/submission.csv', index_col=0)
sample_submission4 = pd.read_csv('../input/stratifiedkfoldlgbxopy/submission.csv', index_col=0)


# In[ ]:


test_df['pred1'] = sample_submission1.meter_reading
test_df['pred2'] = sample_submission2.meter_reading
test_df['pred3'] = sample_submission3.meter_reading
test_df['pred4'] = sample_submission4.meter_reading

test_df.loc[test_df.pred3<0, 'pred3'] = 0 
test_df.loc[test_df.pred3<0, 'pred4'] = 0 

del  sample_submission1,  sample_submission2,  sample_submission3, sample_submission4
gc.collect()

test_df = reduce_mem_usage(test_df)
leak_df = reduce_mem_usage(leak_df)


# In[ ]:


leak_df = leak_df.merge(test_df[['building_id', 'meter', 'timestamp', 'pred1', 'pred2', 'pred3', 'pred4', 'row_id']], left_on = ['building_id', 'meter', 'timestamp'], right_on = ['building_id', 'meter', 'timestamp'], how = "left")
leak_df = leak_df.merge(building_meta_df[['building_id', 'site_id']], on='building_id', how='left')


# In[ ]:


leak_df['pred1_l1p'] = np.log1p(leak_df.pred1)
leak_df['pred2_l1p'] = np.log1p(leak_df.pred2)
leak_df['pred3_l1p'] = np.log1p(leak_df.pred3)
leak_df['pred4_l1p'] = np.log1p(leak_df.pred4)
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


# kfold lgbm
sns.distplot(leak_df.pred4_l1p)
sns.distplot(leak_df.meter_reading_l1p)

leak_score = np.sqrt(mean_squared_error(leak_df.pred4_l1p, leak_df.meter_reading_l1p))
print ('score4=', leak_score)


# # Leak Validation for Blending

# In[ ]:


leak_df['mean_pred'] = np.mean(leak_df[['pred1', 'pred2', 'pred3', 'pred4']].values, axis=1)
leak_df['mean_pred_l1p'] = np.log1p(leak_df.mean_pred)
leak_score = np.sqrt(mean_squared_error(leak_df.mean_pred_l1p, leak_df.meter_reading_l1p))

sns.distplot(leak_df.mean_pred_l1p)
sns.distplot(leak_df.meter_reading_l1p)

print ('mean score=', leak_score)


# # Genetic Algorithm

# In[ ]:


class GAOptimizer:
    def __init__(self, function, min_value=0.2, max_value=0.8, population_size=50, dimention=4):
        self.function = function
        self.population = np.random.uniform(min_value, max_value, (population_size, dimention))
        self.population_size = population_size
        self.dimention = dimention
        half_dim1 = int(dimention/2)
        half_dim2 = int(dimention/2) + 1 if dimention%2 else int(dimention/2)
        self.co_weights = np.hstack([np.ones((population_size, half_dim1)), np.zeros((population_size, half_dim2))])
        
    def crossover(self):
        old_population = np.copy(self.population)
        population = np.copy(self.population)
        np.random.shuffle(population)
        new_population = old_population*self.co_weights + population*(1 - self.co_weights) + np.random.normal(0, 0.1, (self.population_size, self.dimention))
        
        return np.vstack([old_population, new_population])
    
    def selector(self, n=None):
        f_values = self.function(self.population)
        self.population = self.population[np.argsort(f_values)[:self.population_size]]
        if n:
            return self.population[np.argsort(f_values)[:n]], self.function(self.population[np.argsort(f_values)[:n]])
        

    def fit(self, iters):
        for i in range(iters):
            self.population = self.crossover()
            self.selector()
        return self.selector(1)


# In[ ]:


def func_to_opt(scores):
    score = []
    for x in scores:
        v = x[0] * leak_df['pred1'].values + x[1] * leak_df['pred2'].values +         x[2] * leak_df['pred3'].values + x[3] * leak_df['pred4'].values
        val  = (v > 0).astype(int)*v
        vl1p = np.log1p(val)
        curr_score = np.sqrt(mean_squared_error(vl1p, leak_df.meter_reading_l1p)) 
        score.append(curr_score)
    return np.array(score)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'ga_holder = GAOptimizer(function=lambda l: func_to_opt(l))\nresult = ga_holder.fit(40)  \nprint(result)')


# In[ ]:


final = result[0].flatten()/result[0].flatten().sum()
print(final)


# In[ ]:


func_to_opt(np.array([final]))


# # Submit

# In[ ]:


sample_submission = pd.read_feather(os.path.join(root, 'sample_submission.feather'))

w1 = final[0]
w2 = final[1]
w3 = final[2]
w4 = final[3]

sample_submission['meter_reading'] = w1 * test_df.pred1 +  w2 * test_df.pred2  + w3 * test_df.pred3 +w4 * test_df.pred4
sample_submission.loc[sample_submission.meter_reading < 0, 'meter_reading'] = 0


# In[ ]:


sample_submission.head()


# In[ ]:


sns.distplot(np.log1p(sample_submission.meter_reading))


# In[ ]:


leak_df = leak_df[['meter_reading', 'row_id']].set_index('row_id').dropna()
sample_submission.loc[leak_df.index, 'meter_reading'] = leak_df['meter_reading']


# In[ ]:


sns.distplot(np.log1p(sample_submission.meter_reading))


# In[ ]:


sample_submission.isna().sum()


# In[ ]:


sample_submission.to_csv('submission.csv', index=False, float_format='%.5f')


# In[ ]:




