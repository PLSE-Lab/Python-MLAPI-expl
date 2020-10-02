#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

get_ipython().run_line_magic('matplotlib', 'inline')
mpl.style.use('seaborn')
sns.set(rc={'figure.figsize':(12,5)});
gfig = plt.figure(figsize=(12,5));


# In[ ]:


df = pd.read_csv('../input/train.csv', parse_dates=['click_time', 'attributed_time'], nrows=1000000)
categorical = ['ip', 'app', 'device', 'os', 'channel', 'is_attributed']
for c in categorical:
    df[c] = df[c].astype('category')


# ## Basic Overview

# We can start with a basic overview of the data.

# In[ ]:


df.sample(10)


# In[ ]:


df.describe()


# The data contains very few `attributed_time` entries, creating an extreme class imbalance for prediction:

# In[ ]:


df['is_attributed'].value_counts()


# We can calculate potential class weights for the classes as follows:

# In[ ]:


from sklearn.utils import class_weight
class_weights = class_weight.compute_class_weight('balanced', df['is_attributed'].unique(), df['is_attributed'].values)
print("is_attributed==0 weight: {},\nis_attributed==1 weight: {}".format(class_weights[0], class_weights[1]))


# As seen in the class weighting, the class imbalance is truly massive. The class weights might be useful when looking into a deep learning approach.

# ### Top 20 by IP, Device, OS, App, Channel: attributed vs non-attributed

# Next we explore the top features comparing attributed vs non-attributed entries.

# In[ ]:


fig, axes = plt.subplots(2, 5)
fig.set_figheight(8)
fig.set_figwidth(20)
fig.tight_layout()
attributed = [0, 1]
attributes = ['ip', 'device', 'os', 'app', 'channel']
for attributed in attributed:
    for idx, attr in enumerate(attributes):
        values = df[df.is_attributed == attributed][attr].value_counts().head(20)
        ax = values.plot.bar(ax=axes[attributed][idx])
        ax.set_title(attr)
        if idx == 0:
            if attributed == 0:
                h = ax.set_ylabel('not-attributed', rotation='vertical', size='large')
            else:
                h= ax.set_ylabel('attributed', rotation='vertical', size='large')
plt.subplots_adjust(hspace=0.3)


# There are a number of notable things:
# * there are a small number of IPs that have many more attributtions than other IPs.
# * attributed clicks are spread across more devices
# * similarly, attributed clicks are more evenly spread across OSs
# * there a a small number of apps that have many more attributions than other apps.
# * specific channels seem to have more attributions than others

# ## Frequency of clicked time vs attributed time

# An interesting pattern to consider is to look at both `click_time` and `attributed_time` time of day (by hour) and see whether there are any noticable patterns.

# In[ ]:


df['click_h'] = df['click_time'].dt.hour + df['click_time'].dt.minute / 60
df['attributed_h'] = df['attributed_time'].dt.hour + df['attributed_time'].dt.minute / 60


# In[ ]:


fig, axes = plt.subplots(1, 2)
fig.set_figwidth(20)
ax = df['click_h'].plot.hist(bins=24, ax=axes[0])
xl = ax.set_xlabel('hour')
title = ax.set_title('click_time')

ax = df['attributed_h'].plot.hist(bins=24, ax=axes[1])
xl = ax.set_xlabel('hour')
title = ax.set_title('attributed_time')


# The amount of data dwarves the click_time frequences, but it is clear that many more clicks take place after business hours. Attributed time is spread more evenly, but is also skewed toward the evening.

# 
# ### Clicks aggregated by time

# We can also look at clicks aggregate by time: clicks per month, day and hour and correlate these with IP and other features.

# #### Time based aggegation

# In[ ]:


df['click_month'] = df['click_time'].dt.month
df['click_day'] = df['click_time'].dt.day
df['click_hour'] = df['click_time'].dt.hour


# In[ ]:


# convert back to object to avoid an issue with merging with categorical data: https://github.com/pandas-dev/pandas/issues/18646
for c in categorical:
    df[c] = df[c].astype('object')


# In[ ]:


def create_click_aggregate(frame, name, idxs):
    aggregate = frame.groupby(by=idxs, as_index=False).click_time.count()
    aggregate = aggregate.rename(columns={'click_time': name})
    return frame.merge(aggregate, on=idxs)


# In[ ]:


def create_attributed_aggregate(frame, name, idxs):
    aggregate = frame[frame['is_attributed'] == 1].groupby(by=idxs, as_index=False).is_attributed.count()
    aggregate = aggregate.rename(columns={'is_attributed': name})
    return frame.merge(aggregate, on=idxs)


# In[ ]:


df = create_click_aggregate(df, 'total_clicks', ['ip'])
df = create_click_aggregate(df, 'clicks_in_day', ['ip', 'click_month', 'click_day'])
df = create_click_aggregate(df, 'clicks_in_hour', ['ip', 'click_month', 'click_day', 'click_hour'])


# In[ ]:


df = create_attributed_aggregate(df, 'total_attributions', ['ip'])
df = create_attributed_aggregate(df, 'attributed_in_day', ['ip', 'click_month', 'click_day'])
df = create_attributed_aggregate(df, 'attributed_in_hour', ['ip', 'click_month', 'click_day', 'click_hour'])


# In[ ]:


fig, axes = plt.subplots(3, 1)
fig.set_figheight(8)
fig.set_figwidth(20)
fig.tight_layout()

time_aggregates = [('total_clicks', 'total_attributions'), ('clicks_in_day', 'attributed_in_day'), ('clicks_in_hour', 'attributed_in_hour')]
row = 0
for time_aggregate in time_aggregates:
    ax = df[['ip', time_aggregate[0], time_aggregate[1]]].drop_duplicates().sort_values(time_aggregate[0], ascending=False).head(20).set_index('ip').plot.bar(ax=axes[row], secondary_y=time_aggregate[1])
    if row == 0:
        ax.set_title('Non-attributed')
    row+=1


# Notable patterns we can see here:
# * total_attributions increase with total_clicks, but is somewhat noisy.
# * attributions per day increases with clicks per day, but is also noisy.
# * there appears to be a slight, negative correlation between clicks in an hour and the attributions in the hour

# ## Unique feature value correlation by IP

# Finally look at potential feature correlations, calculating unique values by IP.

# In[ ]:


def unique_values_by_ip(frame, value):
    n_values_by_ip = frame.groupby(by='ip')[value].nunique()
    frame.set_index('ip', inplace=True)
    frame['n_' + value] = n_values_by_ip
    frame.reset_index(inplace=True)
    return frame


# In[ ]:


df = unique_values_by_ip(df, 'os')
df = unique_values_by_ip(df, 'app')
df = unique_values_by_ip(df, 'device')
df = unique_values_by_ip(df, 'channel')


# In[ ]:


facets = ['n_os', 'n_app', 'n_channel', 'n_device', 'total_clicks', 'total_attributions']

combinations = [c for c in itertools.combinations(facets, 2)]
rows = 5
cols = int(len(combinations) / rows)

fig, axes = plt.subplots(rows, cols)
fig.set_figheight(20)
fig.set_figwidth(20)
fig.tight_layout()

idx = 0
for row in range(0, rows):
    for col in range(0, cols):
        combo = combinations[idx]
        ax = df.plot.hexbin(combo[0], combo[1], ax=axes[row, col], gridsize=22)
        idx+=1


# There are a few interesting obsevations here:
# * exponential increase in number of channels relative to total clicks by IP
# * a similar increase in number of channels relative to total attributions by IP
# * a number of expected linear correlations. E.g. n_os vs n_app, n_app vs n_device

# ## [Update] References

# Other great EDAs that explore similar aspects of the data:
# * https://www.kaggle.com/kailex/talkingdata-eda-and-class-imbalance
# * https://www.kaggle.com/yuliagm/talkingdata-eda-plus-time-patterns

# In[ ]:




