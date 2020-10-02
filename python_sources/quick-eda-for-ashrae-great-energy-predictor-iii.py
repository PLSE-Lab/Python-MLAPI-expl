#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from pathlib import Path

from IPython.display import display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import norm, probplot


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


INPUT_PATH = Path('/kaggle', 'input', 'ashrae-energy-prediction')
TRAIN_PATH = INPUT_PATH / 'train.csv'


# In[ ]:


df_train = pd.read_csv(TRAIN_PATH, parse_dates=['timestamp'])
df_train.head()


# In[ ]:


df_train['timestamp'].min(), df_train['timestamp'].max()


# In[ ]:


df_train.info()


# In[ ]:


display(df_train.describe().T)
display(pd.DataFrame([df_train[col].nunique() for col in df_train.columns],
             index=df_train.columns, columns=['nunique']))


# In[ ]:


df_train['timestamp'].min(), df_train['timestamp'].max()


# The training set covers 1 year of data (test set might covers the 2 other years), from 01/01/2016 to 12/31/2016.

# In[ ]:


ax = sns.distplot(df_train['building_id'].value_counts(),
                  kde=False, hist_kws=dict(density=True));
ax.set_title('Histogram of the frequencies of building_id');


# Seems there is a pattern in the number of occurences of `building_ids`. We see 3 modes, around 10000, 18000 and 27000. That means, the building_ids have not been sampled randomly.

# In[ ]:


METER_MAPPER = {0: 'electricity', 1: 'chilledwater', 2: 'steam', 3: 'hotwater'}


# In[ ]:


_, ax = plt.subplots()

df_train['meter'].value_counts().plot.bar(ax=ax)
ax.set_xlabel("Meter")
ax.set_ylabel('Count')
ax.set_xticklabels(METER_MAPPER.values());


# Electricity is the most common energy type, by far.

# In[ ]:


ax = sns.countplot(df_train.groupby('building_id')['meter'].nunique())
ax.set_title("Frequencies of count of meter per building_id");


# Most `building_id` have only one energy source, some have 2 or 3 while very few have 4.  
# Is there some temporality, e.g. buildings have changed their energy source over the years?

# In[ ]:


sns.distplot(df_train['meter_reading']);


# In[ ]:


df_train['meter_reading'].describe()


# This is looking like a power-law, let's see the impact of a log transform.

# In[ ]:


_, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
sns.distplot(np.log1p(df_train['meter_reading']), fit=norm, ax=ax1)
_ = probplot(np.log1p(df_train['meter_reading']), plot=ax2);


# Far from lognormal, we have a spike around 0 (e.g log1p(0)), that means we have a lot of reading_meters close or equals to 0.

# In[ ]:


df_train.loc[df_train['meter_reading'] == 0].shape[0]


# Indeed, close to 2 millions of meter readings are zero-valued, that's almost 10%.

# In[ ]:


non_zero_meter_readings = df_train.loc[df_train['meter_reading'] != 0, 'meter_reading']

_, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
sns.distplot(np.log1p(non_zero_meter_readings), fit=norm, ax=ax1)
_ = probplot(np.log1p(non_zero_meter_readings), plot=ax2);


# Looking a bit better but still not close to a normal probplot, cutting the purely zero-valued meter is not enough, there seems to be a huge amount of non-zero small values.

# In[ ]:


df_train['log1p_meter_reading'] = np.log1p(df_train['meter_reading'])
df_train.head()


# In[ ]:


_, axes = plt.subplots(1, 2, figsize=(14, 4))

sns.barplot(x='meter', y='log1p_meter_reading', data=df_train, ax=axes[0])
sns.violinplot(x='meter', y='log1p_meter_reading', data=df_train, ax=axes[1])

for ax in axes:
    ax.set_xticklabels(METER_MAPPER.values());


# ### Variations over time

# In[ ]:


frequencies_grouper = [
    ('H', 'hour'),
    ('D', 'day'),
    ('D', 'dayofweek'),
    ('W', 'week'),
    ('MS', 'month'),
]


# In[ ]:


for _, g in frequencies_grouper:
    df_train[g] = getattr(df_train['timestamp'].dt, g)
df_train.head()


# In[ ]:


fig = plt.figure(figsize=(18, 8), tight_layout=True)

gs = fig.add_gridspec(2, 6)

axes = np.array([
    fig.add_subplot(gs[0, :3]),
    fig.add_subplot(gs[0, 3:]),
    fig.add_subplot(gs[1, :2]),
    fig.add_subplot(gs[1, 2:4]),
    fig.add_subplot(gs[1, 4:]),
])

for (f, g), ax in zip(frequencies_grouper, axes):
    grouper = (df_train.groupby(pd.Grouper(freq=f, key='timestamp'))               .agg(average=('meter_reading', 'mean')))
    grouper[g] = getattr(grouper.index, g)
    sns.barplot(x=g, y='average', data=grouper, ax=ax)
    ax.set_xlabel(g)
    ax.set_ylabel(f"Average meter reading")


# Hard to see small periods trends like this, but we can see that for the year 2016, average meter readings for months 3, 4, 5 and 6 were particularly high.

# ### To be continued...
