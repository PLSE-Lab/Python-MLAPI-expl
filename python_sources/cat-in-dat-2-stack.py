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


submission = pd.read_csv('../input/cat-in-the-dat-ii/sample_submission.csv', index_col='id')


# In[ ]:


kernels = pd.read_csv('../input/cat-in-dat-2-public-kernels/kernels.csv', index_col='id')


# In[ ]:


kernels.head(10)


# In[ ]:


import glob

def make_filename(idx):
    return glob.glob('../input/cat-in-dat-2-public-kernels/' + str(idx) + '__submission.csv')[0]

def read_predictions(idx):
    temp = pd.read_csv(make_filename(idx), index_col='id')
    temp.columns = [str(idx)]
    return temp


predictions = pd.concat([read_predictions(idx) for idx in kernels.index], axis=1)
predictions.shape


# In[ ]:


predictions.head()


# ## Distribution

# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt


plt.figure(figsize=(10,10))

for column in predictions.columns:
    sns.kdeplot(predictions[column], label=column)

plt.show()


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

plt.show()


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


weighted_sum_prediction = predictions.dot(pd.Series(weights, index=predictions.columns))


# In[ ]:


weighted_sum_prediction.head()


# In[ ]:


submission['target'] = weighted_sum_prediction
submission.to_csv('stack-weighted-sum.csv')


# ## Blend by ranking

# In[ ]:


scores = kernels['score']

sum_scores = sum(scores)

weights = [x / sum_scores for x in scores]


# In[ ]:


from scipy.stats import rankdata


def blend_by_ranking(data, weights):
    out = np.zeros(data.shape[0])
    for idx,column in enumerate(data.columns):
        out += weights[idx] * rankdata(data[column].values)
    out /= np.max(out)
    return out


# In[ ]:


blend_by_ranking_prediction = blend_by_ranking(predictions, weights)


# In[ ]:


blend_by_ranking_prediction


# In[ ]:


submission['target'] = blend_by_ranking_prediction
submission.to_csv('stack-blend-by-ranking.csv')

