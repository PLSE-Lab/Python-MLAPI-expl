#!/usr/bin/env python
# coding: utf-8

# ### Motivation
# This submission is based on a simple statistical idea: the probability distribution for yards gained on a play follows the distribution of all plays previously measured.

# In[ ]:


from tqdm import tqdm_notebook as tqdm
import numpy as np
import pandas as pd
pd.options.display.max_columns = 200

from kaggle.competitions import nflrush

import holoviews as hv
from holoviews import opts
hv.extension('bokeh')


# ### Probabilities
# Here are graphs of the Probability Density Function (PDF) and Cumulative Distribution Function (CDF) for yards gained in the 2017 and 2018 seasons.

# In[ ]:


train_plays = pd.read_csv('../input/nfl-big-data-bowl-2020/train.csv', 
                          usecols=['PlayId', 'Yards'])\
              .drop_duplicates('PlayId').reset_index(drop=True)

train_yards_array = train_plays.Yards.values


# In[ ]:


pdf, edges = np.histogram(train_yards_array, bins=199, range=(-99, 99), density=True)
hist = hv.Histogram((pdf, edges)).opts(width=600)

cdf = pdf.cumsum().clip(0, 1)
curve = hv.Curve(dict(zip(np.arange(-99,99), cdf))).opts(width=600)
(hist+curve).cols(1)


# ### Score and visualization
# Let's see how this scores on the training data using K-fold.

# In[ ]:


from sklearn.model_selection import KFold

vsize = len(train_plays)

def calc_crps(preds_cume, actuals):
    stops = np.arange(-99, 100)
    unit_steps = stops >= actuals.reshape(-1, 1)
    crps = ((preds_cume - unit_steps)**2).mean().mean()
    return crps

kf_params = {'n_splits': 5,
             'random_state': 3438
             }

kf = KFold(**kf_params)
scores = np.zeros((kf.get_n_splits()))
pdfs_all = np.zeros((vsize,199))
for i, (train_idx, val_idx) in enumerate(kf.split(train_yards_array)):
    pdf, edges = np.histogram(train_yards_array[train_idx], bins=199, 
                              range=(-99, 99), density=True)
    cdf = pdf.cumsum().clip(0, 1)
    preds_cume = np.array([cdf]*len(val_idx))
    score = calc_crps(preds_cume, train_yards_array[val_idx])
    scores[i] = score
    pdfs_all[val_idx] = pdf
    print(score)

print(f'mean: {scores.mean():.5f}',
      f'stdp: {scores.std():.5f}', sep='\n')


# Not bad for such a simple benchmark. Here's a visualization of what it looks like. The predicted distributions for each play are shown by the background spectrum (slightly different for each fold). Actual gains are sorted and appear as green dots.

# In[ ]:


import matplotlib.pyplot as plt

plays_sorted = train_plays.sort_values('Yards', ascending=False)
pdfs_all = pdfs_all[plays_sorted.index]

fig = plt.figure(figsize=(15,8))
predpic = plt.imshow(pdfs_all,aspect='auto', cmap='hot', extent = (-99.5, 99.5, vsize+.5, -0.5))
plt.scatter(x=plays_sorted.Yards, y=np.arange(vsize), s=0.1, c='green')
plt.show()


# ### Submission
# All that's left is to submit the distribution for the test set plays.The LB score should be similar to the CV score.

# In[ ]:


def get_cdf_df(yards_array):
    pdf, edges = np.histogram(yards_array, bins=199,
                 range=(-99,99), density=True)
    cdf = pdf.cumsum().clip(0, 1)
    cdf_df = pd.DataFrame(data=cdf.reshape(-1, 1).T, 
                            columns=['Yards'+str(i) for i in range(-99,100)])
    return cdf_df


# In[ ]:


env = nflrush.make_env()
iter_test = env.iter_test()

for (test_play, _) in tqdm(iter_test, total=3438):
    test_play_cdf = get_cdf_df(train_yards_array)
    env.predict(test_play_cdf)

env.write_submission_file()

