#!/usr/bin/env python
# coding: utf-8

# # Summary
# ![img](https://i.imgur.com/d0wC7Qm.png)
# ### Samples in test & test supplement(short as 'sup' in the figure) are overlapping and totally time sorted
# ### BUT some of them are in slightly different order within the same second
# ### Overlapping samples can be replaced by those in test set by following indexes and offsets
# - offsets: [21290876, 29475763, 35793790]
# - test head indexes: [0, 6202933, 12316147]
# - test tail indexes: [6202932, 12316146, 18790468]
# 
# ## NOTICE: 
# ## The calculation for the offsets is simply based on observation of several head & tail samples of test and supplement set and if there' s anything wrong please comment!
# 
# ## DOWNLOAD
# 
# - https://www.kaggle.com/johnfarrell/talkingdata-adtracking-test-order-supplement/data

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import os
import gc
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
dtype = {
        'ip'               : 'uint32',
        'app'              : 'uint16',
        'device'           : 'uint16',
        'os'               : 'uint16',
        'channel'          : 'uint16',
        'is_attributed'    : 'uint8',
        'click_id'         : 'uint32',
        'click_hour'       : 'uint8',
        'click_second'     : 'uint8',
        'click_minute'     : 'uint8'
        }
target_col = 'is_attributed'
key_li = ['ip', 'app', 'device', 'os', 'channel', 
          'click_hour', 'click_minute', 'click_second']
from functools import lru_cache
get_datetime     = (lru_cache())(lambda x:pd.to_datetime(x))
get_click_day    = (lru_cache())(lambda x:int(x[ 8:10]))
get_click_hour   = (lru_cache())(lambda x:int(x[11:13]))
get_click_minute = (lru_cache())(lambda x:int(x[14:16]))
get_click_second = (lru_cache())(lambda x:int(x[17:19]))
get_ac_delta = (lru_cache())(lambda x:int(x.total_seconds()))

def preprocess(df):
    df['click_hour']   = df['click_time'].apply(get_click_hour)
    df['click_minute'] = df['click_time'].apply(get_click_minute)
    df['click_second'] = df['click_time'].apply(get_click_second)
    return df


# In[2]:


df_sup = pd.read_csv('../input/talkingdata-adtracking-fraud-detection/test_supplement.csv', dtype=dtype)
print('supplement dataset loaded, shape', df_sup.shape)
df_test = pd.read_csv('../input/talkingdata-adtracking-fraud-detection/test.csv', dtype=dtype)
print('test dataset loaded, shape', df_test.shape)
df_sup['click_time'] = df_sup['click_time'].apply(get_datetime)
df_test['click_time'] = df_test['click_time'].apply(get_datetime)


# In[3]:


df_sup['click_time'].min(), df_sup['click_time'].max()


# In[4]:


test_indexes = [0, 6202933, 12316147]
offsets = [21290876, 29475763, 35793790]


# ## Check head part in test & responding samples in test supplements
# ### Head part: 
# - 2017-11-10 04:00:00
# - 2017-11-10 09:00:00
# - 2017-11-10 13:00:00

# In[5]:


i = 0
df_test.iloc[test_indexes[i]:].head()


# In[6]:


df_sup.iloc[test_indexes[i]+offsets[i]:].head()


# In[7]:


i = 1
df_test.iloc[test_indexes[i]:].head()


# In[8]:


df_sup.iloc[test_indexes[i]+offsets[i]:].head()


# In[9]:


i = 2
df_test.iloc[test_indexes[i]:].head()


# In[10]:


df_sup.iloc[test_indexes[i]+offsets[i]:].head()


# ## Check tail part in test & responding samples in test supplements
# ### Tail part: 
# - 2017-11-10 06:00:00
# - 2017-11-10 11:00:00
# - 2017-11-10 15:00:00

# In[11]:


df_test[df_test['click_time']==pd.to_datetime('2017-11-10 06:00:00')].tail(1)


# In[12]:


df_test[df_test['click_time']==pd.to_datetime('2017-11-10 11:00:00')].tail(1)


# In[13]:


df_test[df_test['click_time']==pd.to_datetime('2017-11-10 15:00:00')].tail(1)


# In[14]:


test_tail_indexes = [6202932, 12316146, 18790468]


# In[15]:


i = 0
df_test.iloc[:test_tail_indexes[i]+1].tail()


# In[16]:


df_sup.iloc[:test_tail_indexes[i]+offsets[i]+1].tail()


# In[17]:


i = 1
df_test.iloc[:test_tail_indexes[i]+1].tail()


# In[18]:


df_sup.iloc[:test_tail_indexes[i]+offsets[i]+1].tail()


# In[19]:


i = 2
df_test.iloc[:test_tail_indexes[i]+1].tail()


# In[20]:


df_sup.iloc[:test_tail_indexes[i]+offsets[i]+1].tail()


# ## Replacement

# In[21]:


df_sup['eval_set'] = 0
del df_sup['click_id']
df_sup.head()


# In[ ]:


df_test['eval_set'] = 1
del df_test['click_id']
gc.collect()


# In[ ]:


for i, (head_idx, tail_idx, offset) in enumerate(zip(test_indexes, test_tail_indexes, offsets)):
    print(i)
    print('index:', head_idx, tail_idx, 'offset', offset)
    df_sup.iloc[head_idx+offset:tail_idx+1+offset] = df_test.iloc[head_idx:tail_idx+1].values
df_sup = df_sup.reset_index(drop=True)
dtype['eval_set'] = 'uint8'
for c in ['ip', 'app', 'device', 'os', 'channel', 'eval_set']:
    df_sup[c] = df_sup[c].astype(dtype[c])


# In[ ]:


df_test.head()


# In[ ]:


df_sup[df_sup['eval_set']==1].head()


# ## Here the 'click_id' is not included, please change the code if this is unexpected
# ## And for the 1Gb limit of kernel, please run on local machine...
# ## Or download here:
# 
# - https://www.kaggle.com/johnfarrell/talkingdata-adtracking-test-order-supplement/data

# In[ ]:


print('OSError: [Errno 28] No space left on device')
print('Please run locally...')
#df_sup.to_csv('test_ordered_supplement.csv', index=False)
#error...
#OSError: [Errno 28] No space left on device

