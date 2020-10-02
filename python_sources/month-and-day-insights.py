#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=0.9)


# In[ ]:


data = pd.read_csv('/kaggle/input/cat-in-the-dat/train.csv', index_col='id')
hmap_options = dict(annot=True, linewidths=1, square=True)
prior = data.target.mean()


# `month` and `day` data are suposed to follow a cyclic pattern.

# In[ ]:


plt.figure(figsize=(12, 4))

plt.subplot(121)
(data.groupby('day').target.mean() - prior).plot.bar()
plt.ylabel('cat prob. relative to prior')

plt.subplot(122)
(data.groupby('month').target.mean() - prior).plot.bar()
plt.ylabel('cat prob. relative to prior')

plt.tight_layout()


# `day` has a clear cyclic pattern but that can't be said about `month` data.

# In[ ]:


def fix_ylim():
    # seaborn+matplotlib bug workaround
    ymin, ymax = plt.ylim()
    plt.ylim(math.ceil(ymin), math.floor(ymax))
    
def cyclic(series, start, size):
    angle = 2 * np.pi * (series - start) / size
    sin = pd.Series(np.sin(angle), index=angle.index, name=f'sin {series.name}')
    cos = pd.Series(np.cos(angle), index=angle.index, name=f'cos {series.name}')
    return pd.concat([sin, cos], axis=1)


# Let's analyze `target` distribution looking at `day` and `month`. First, checking the sample size grouped by `month` and `day` we observe a low amount of observations for both day and month 6.

# In[ ]:


h = data.groupby(['month', 'day']).target.count().unstack('month')
plt.figure(figsize=(8, 5.5))
sns.heatmap(h, cmap='Blues', fmt='d', cbar=False, **hmap_options)
plt.suptitle('Sample size')
fix_ylim()
plt.tight_layout()


# Column(Row) 6 will be replaced by average values of column(row) 5 and 7.

# In[ ]:


m = data.groupby(['month', 'day']).target.mean() - prior
m = m.unstack('month')
m[6] = (m[5] + m[7]) / 2
m = m.stack().unstack('day')
m[6] = (m[5] + m[7]) / 2
m = m.stack().unstack('month')


# In[ ]:


plt.figure(figsize=(12, 6))
sns.heatmap(m, cmap='Spectral_r', fmt='.1%', **hmap_options)
fix_ylim()


# `day`'s cyclic pattern is still present. It changes with each month as expected from the independence of variables.

# In[ ]:


df = m.stack().reset_index().rename({0: 'target'}, axis=1)

df2 = cyclic(df.month, 1, 12)
df3 = cyclic(df.day, 1, 7)

df2['sin month'] += 0.14 * df3['sin day']
df2['cos month'] += 0.14 * df3['cos day']

plt.figure(figsize=(7, 7))
sp_options = dict(palette='Spectral_r', vmin=-0.1, vmax=0.1, legend=False, s=400)
sns.scatterplot(x='sin month', y='cos month', hue=df['target'], data=df2, **sp_options)
plt.axis('off')
plt.tight_layout()

months = pd.Series(np.arange(1, 13), name='month')
month_xy = cyclic(months, 1, 12).set_index(months)

for month in months:
    xy = tuple( month_xy.loc[month] - [0.04, 0.03] )
    plt.annotate(f'{month:2d}', xy=xy)


# By replacing the month data with twelve well-defined cycles we can compensate for the lack of information about month and day 6.
