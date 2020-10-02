#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from tqdm import tqdm
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
pd.set_option('display.float_format', '{:.2f}'.format)
print(os.listdir("../input/eyn-original"))
x_min, x_max = -1., 1.
y_min, y_max = -.3, .3


# In[2]:


df_train = pd.read_pickle("../input/eyn-original-df/df_train.pickle")
df_test = pd.read_pickle("../input/eyn-original-df/df_test.pickle")
print(df_train.shape, df_test.shape)
df_train.head()


# In[3]:


def is_inside(arr_x, arr_y):
    inside = ((arr_x > x_min) & 
             (arr_x < x_max) & 
             (arr_y > y_min) & 
             (arr_y < y_max)).astype(float)
    inside[np.isnan(arr_x)] = np.nan  # replace nan position with nan judgement
    return inside

df_train['entry_in'] = is_inside(df_train['x_entry'], df_train['y_entry'])
df_train['exit_in'] = is_inside(df_train['x_exit'], df_train['y_exit'])
df_test['entry_in'] = is_inside(df_test['x_entry'], df_test['y_entry'])
df_test['exit_in'] = is_inside(df_test['x_exit'], df_test['y_exit'])
df_train.head()


# In[4]:


def calc_diff_alt(arr_entry, arr_exit):
    return np.append(np.array(arr_entry)[:-1] - np.array(arr_exit)[1:], np.nan)

df_train['dur'] = df_train['t_exit'] - df_train['t_entry']
df_test['dur'] = df_test['t_exit'] - df_test['t_entry']
df_train['dur_a'] = calc_diff_alt(df_train['t_entry'], df_train['t_exit'])
df_test['dur_a'] = calc_diff_alt(df_test['t_entry'], df_test['t_exit'])
df_test.head()


# In[5]:


def calc_dist(x1, x2, y1, y2):
    x1, x2, y1, y2 = [np.array(arr) for arr in [x1, x2, y1, y2]]
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    
df_train['dist'] = calc_dist(df_train['x_entry'], df_train['x_exit'], df_train['y_entry'], df_train['y_exit'])
df_test['dist'] = calc_dist(df_test['x_entry'], df_test['x_exit'], df_test['y_entry'], df_test['y_exit'])
df_train['dist_a'] = np.append(calc_dist(df_train['x_entry'][:-1], df_train['x_exit'][1:], 
                                         df_train['y_entry'][:-1], df_train['y_exit'][1:]),np.nan)
df_test['dist_a'] = np.append(calc_dist(df_test['x_entry'][:-1], df_test['x_exit'][1:], 
                                        df_test['y_entry'][:-1], df_test['y_exit'][1:]),np.nan)
df_train.head()


# In[6]:


def nan_divide(numerator, denominator):
    result = np.divide(numerator, denominator, 
                       out=np.zeros_like(denominator), where=denominator!=0)
    result[np.isnan(numerator)] = np.nan
    return result

df_train['speed'] = nan_divide(df_train['dist'], df_train['dur'])
df_test['speed']= nan_divide(df_test['dist'], df_test['dur'])
df_train['speed_a'] = nan_divide(df_train['dist_a'], df_train['dur_a'])
df_test['speed_a'] = nan_divide(df_test['dist_a'], df_test['dur_a'])
df_train.head()


# In[7]:


df_train['dir_x'] = nan_divide(df_train['x_exit'] - df_train['x_entry'], df_train['dist'])
df_train['dir_y'] = nan_divide(df_train['y_exit'] - df_train['y_entry'], df_train['dist'])
df_test['dir_x'] = nan_divide(df_test['x_exit'] - df_test['x_entry'], df_test['dist'])
df_test['dir_y'] = nan_divide(df_test['y_exit'] - df_test['y_entry'], df_test['dist'])
df_train['dir_x_a'] = nan_divide(calc_diff_alt(df_train['x_entry'], df_train['x_exit']), df_train['dist_a'])
df_train['dir_y_a'] = nan_divide(calc_diff_alt(df_train['y_entry'], df_train['y_exit']), df_train['dist_a'])
df_test['dir_x_a'] = nan_divide(calc_diff_alt(df_test['x_entry'], df_test['x_exit']), df_test['dist_a'])
df_test['dir_y_a'] = nan_divide(calc_diff_alt(df_test['y_entry'], df_test['y_exit']), df_test['dist_a'])
df_train.head()


# In[8]:


df_train.to_pickle("df_train.pickle")
df_test.to_pickle("df_test.pickle")


# In[9]:


# to load
df_train = pd.read_pickle("df_train.pickle")
df_test = pd.read_pickle("df_test.pickle")
print(df_train.shape, df_test.shape)


# In[ ]:




