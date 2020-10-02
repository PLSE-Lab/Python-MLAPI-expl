#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import gc
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestRegressor


# In[ ]:


print('Loading datasets...')
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

y = np.log1p( train['target'].values )
IDtest  = test['ID'].values

print('Merging all...')
test['target'] = np.nan
train = train.append(test).reset_index() # merge train and test
del test
gc.collect()

print("Create Model...")
train = train[train.columns.drop(['index','ID','target'])] # only get "X" vector
gc.collect()


# In[ ]:


get_ipython().run_cell_magic('time', '', 'print("rounding...")\nfor i in train.columns:\n    train[i] = np.round( np.log1p(train[i]) , decimals=3 )\n\ngc.collect()\nprint(train.head(5))')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'print("To String...")\n\ntmp = train[train.columns[0]].apply(str)\ntmp[ tmp==\'0.0\' ] = \'\'\nCV = pd.DataFrame()\nCV[\'features\'] = tmp\n\nfor feat in train.columns[1:]:\n    tmp = train[feat].apply(str)\n    tmp[ tmp==\'0.0\' ] = \'\'\n    CV[\'features\'] = CV[\'features\'] + tmp + \' \'\n\ndel train\ngc.collect()\nprint( CV )')


# In[ ]:


get_ipython().run_cell_magic('time', '', "rd = CountVectorizer( lowercase=True, ngram_range=(1, 1), max_df=0.99, min_df=2)\ntrain = rd.fit_transform( CV['features'] )\ndel rd, CV\ngc.collect()\nprint(train.shape)")


# In[ ]:


get_ipython().run_cell_magic('time', '', "rd = RandomForestRegressor(n_estimators=2222, criterion='mse', max_depth=10, max_features=0.51, n_jobs=-1)\nrd.fit( train[:4459,:], y )\ngc.collect()")


# In[ ]:


get_ipython().run_cell_magic('time', '', "sub = pd.DataFrame( {'ID':IDtest} )\nsub['target'] = np.expm1( rd.predict( train[4459:,:] ) ).astype(np.int)\nsub.to_csv( 'giba-rf-1.csv', index=False )\nprint( sub.head(20) )")

