#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd

import warnings
warnings.simplefilter('ignore')

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


submission = pd.read_csv('../input/cat-in-the-dat/sample_submission.csv', index_col='id')


# In[ ]:


kernels = pd.read_csv('../input/cat-in-dat-kernels/kernels.csv', index_col='id')


# In[ ]:


kernels.head()


# In[ ]:


import glob

def make_filename(idx):
    return glob.glob('../input/cat-in-dat-kernels/' + str(idx) + '__submission__*.csv')[0]

def read_predictions(idx):
    temp = pd.read_csv(make_filename(idx), index_col='id')
    temp.columns = [str(idx)]
    return temp


predictions = pd.concat([read_predictions(idx) for idx in kernels.index], axis=1)
predictions.shape


# In[ ]:


predictions.head()


# ## Correlation matrix

# In[ ]:


# From https://seaborn.pydata.org/examples/many_pairwise_correlations.html

import seaborn as sns
import matplotlib.pyplot as plt

corr = predictions.corr()

mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(12, 12))

sns.heatmap(corr, mask=mask, cmap='Blues', vmin=0.95, center=0, linewidths=1, annot=True, fmt='.4f')


# ## Stack

# In[ ]:


submission['target'] = predictions.mean(axis=1)
submission.to_csv('stack-mean.csv')


# In[ ]:


submission.head()


# ## Weighted sum

# In[ ]:


scores = kernels['score']

sum_scores = sum(scores)

weights = [x / sum_scores for x in scores]


# In[ ]:


sum_predictions = predictions.dot(pd.Series(weights, index=predictions.columns))


# In[ ]:


sum_predictions.head()


# In[ ]:


submission['target'] = sum_predictions
submission.to_csv('stack-weighted-sum.csv')


# ## Filter

# In[ ]:


N = 3

selected = kernels.sort_values('score', ascending=False).head(N)


# In[ ]:


print('Max selected score =', selected['score'].max())
print('Min selected score =', selected['score'].min())


# In[ ]:


filter_predictions = predictions.loc[:,selected.index.values.astype(str)]


# In[ ]:


filter_predictions.head()


# In[ ]:


submission['target'] = filter_predictions.mean(axis=1)
submission.to_csv('stack-filtered.csv')

