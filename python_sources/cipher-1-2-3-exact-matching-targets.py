#!/usr/bin/env python
# coding: utf-8

# This kernel provides targets for the competition test set using exact matchings with plain-text chunks, for cipher #1, #2 & #3.
# 
# It outputs two dataframes as pickles:
# * test_123.pkl which contains as target for cipher #1, #2 & #3 the list of all possible targets from decrypted ciphertext exact matchin (see https://www.kaggle.com/leflal/you-cannot-avoid-multiple-targets)
# * test_sub.pkl which contains one target for cipher #1, #2 & #3 (chosen among the above list of possible targets), ready for submission
# 
# If you use the output pickle in your work, be it to cross-check your model or enhance your submission,  or if you simply appreciate this contribution, ** please upvote this kernel, thanks and have a nice 2019**.

# In[ ]:


import numpy as np 
import pandas as pd 

import os

from IPython.core.display import display
from sklearn.datasets import fetch_20newsgroups


# In[ ]:


competition_path = '20-newsgroups-ciphertext-challenge'


# In[ ]:


test = pd.read_csv('../input/' + competition_path + '/test.csv').rename(columns={'ciphertext' : 'text'})


# In[ ]:


train_p = fetch_20newsgroups(subset='train')
test_p = fetch_20newsgroups(subset='test')


# In[ ]:


df_p = pd.concat([pd.DataFrame(data = np.c_[train_p['data'], train_p['target']],
                                   columns= ['text','target']),
                      pd.DataFrame(data = np.c_[test_p['data'], test_p['target']],
                                   columns= ['text','target'])],
                     axis=0).reset_index(drop=True)


# In[ ]:


df_p['target'] = df_p['target'].astype(np.int8)


# In[ ]:


def find_targets(p_indexes_set):
    return np.sort(df_p.loc[p_indexes_set]['target'].unique())


# In[ ]:


pickle_1_path = '../input/test-1/'

df_p_indexes_1 = pd.read_pickle(pickle_1_path + 'df_p_indexes-1.pkl')
df_p_indexes_1['target'] = df_p_indexes_1['p_indexes'].map(find_targets)
display(df_p_indexes_1[df_p_indexes_1['target'].map(len) > 1 ])


# In[ ]:


test = test.join(df_p_indexes_1[['target']])


# In[ ]:


pickle_2_path = '../input/test-2/'

df_p_indexes_2 = pd.read_pickle(pickle_2_path + 'df_p_indexes-2.pkl')
df_p_indexes_2['target'] = df_p_indexes_2['p_indexes'].map(find_targets)
display(df_p_indexes_2[df_p_indexes_2['target'].map(len) > 1 ])


# In[ ]:


test.loc[df_p_indexes_2.index,'target'] = df_p_indexes_2['target']


# In[ ]:


pickle_3_path = '../input/cipher-3-solution/'

df_p_indexes_3 = pd.read_pickle(pickle_3_path + 'test_3.pkl')
df_p_indexes_3['target'] = df_p_indexes_3['p_indexes'].map(find_targets)
display(df_p_indexes_3[df_p_indexes_3['target'].map(len) > 1 ])


# In[ ]:


test.loc[df_p_indexes_3.index,'target'] = df_p_indexes_3['target']


# In[ ]:


test.head()


# In[ ]:


test[test['target'].isnull()]['difficulty'].unique()


# In[ ]:


test.to_pickle('test_123.pkl')


# In[ ]:


test.loc[test['difficulty'] < 4,'target'] = test.loc[test['difficulty'] < 4,'target'].map(lambda x: x[0])
#You can implement the target choice you want within the possible targets here


# In[ ]:


test.to_pickle('test_sub.pkl')

