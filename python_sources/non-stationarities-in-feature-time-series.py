#!/usr/bin/env python
# coding: utf-8

# # Summary of Non-stationarities and Anomalies in Feature Time Series

# In[ ]:


import numpy as np
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')

from bokeh.io import output_notebook, show
from bokeh.models import ColumnDataSource, HoverTool, Range1d, Span
from bokeh.plotting import figure
from bokeh.palettes import Set1_9
from bokeh.transform import factor_cmap

import datetime

output_notebook()


# After trying a few generic model-building techniques, I started to reflect on my data exploration and processing procedure to find out why I was not able to build better models that not only score higher but also produce predictions that make sense. I soon realised that I missed a very important step in the analysis: compare different parts of the data as well as compare the training set and the test set to see if the data samples are distributed identically and independently, if the training and test samples are distributed similarly, and if there are any inconsistencies preventing us from generalise from one part of the data to another. For any data that can be treated as a time series, if we are to perform any predictive analysis on it, it is important to look out for the stationarity and trend consistency of the series.
# 
# Non-stationary time series is problematic for prediction. If the features next year will look completely different from this year, how can we be sure that we can generalise what we learned this year to the next year? Most machine learning algorithms are better at interpolation than extrapolation, so we always have to be careful when the range of the training features and features used to predict do not overlap well. Sudden changes in the time series may also indicate the occurrence of major events that should be taken into account in the analysis.
# 
# For our problem, we first look at the time ranges of the training and test dataset to see whether they are random samples from the same population or non-overlapping samples from different populations:

# In[ ]:


train = pd.read_pickle('../input/gstore-revenue-data-preprocessing/train.pkl')
test = pd.read_pickle('../input/gstore-revenue-data-preprocessing/test.pkl')


# As we see below, the training and testing data are split according to time. There is no overlap between them, and apparently they cannot possibly have been drawn from the same sample, and it is likely that something has changed in the behaviour of the features over the transition from the training period to the testing period, and we have to watch out for it.

# In[ ]:


train['visitStartTime'].describe()


# In[ ]:


test['visitStartTime'].describe()


# Let us record the starting time of the testing samples:

# In[ ]:


test_start_time = np.min(test['visitStartTime'])


# In[ ]:


test_start_time


# ### Sessions over time

# Now let us first look at the total sessions over time, including both training and testing data:

# In[ ]:


train_ts = train.set_index('visitStartTime')
test_ts = test.set_index('visitStartTime')


# In[ ]:


train_daily_sess_count = pd.DataFrame({'count': train_ts.groupby(pd.Grouper(freq='D')).size(), 'set': 'train'})
test_daily_sess_count = pd.DataFrame({'count': test_ts.groupby(pd.Grouper(freq='D')).size(), 'set': 'test'})
daily_sess_count = pd.concat([train_daily_sess_count, test_daily_sess_count], axis=0)


# In[ ]:


p = figure(x_axis_type='datetime', width=700, height=400, title='daily session count over time')
p.line(
    source=daily_sess_count[daily_sess_count['set'] == 'train'],
    x='visitStartTime',
    y='count',
    color=Set1_9[0])
p.line(
    source=daily_sess_count[daily_sess_count['set'] == 'test'],
    x='visitStartTime',
    y='count',
    color=Set1_9[1])
p.legend.location = 'top_left'
test_start = Span(location=test_start_time.timestamp() * 1000, dimension='height', line_dash='dashed')
p.add_layout(test_start)
show(p)


# Apart from one obvious outlier, we see that daily sessions fluctuates quite a bit over time. Trend-wise, there is a steep decline around the end of 2016 and a slow increasing trend afterwards. Overall, the a period from 2017 to the end of the training data appears to have more in common with the test portion of the data. I would like to find the reason for the decrease in daily sessions around the end of 2016.

# ### Pageviews and hits over time:

# In[ ]:


train_daily_pageviews = pd.DataFrame({
    'count':
    train_ts.groupby(pd.Grouper(freq='D'))['totals.pageviews'].sum(),
    'set':
    'train'
})
test_daily_pageviews = pd.DataFrame({
    'count':
    test_ts.groupby(pd.Grouper(freq='D'))['totals.pageviews'].sum(),
    'set':
    'test'
})
daily_pageviews = pd.concat(
    [train_daily_pageviews, test_daily_pageviews], axis=0)


# In[ ]:


train_daily_hit_count = pd.DataFrame({
    'count':
    train_ts.groupby(pd.Grouper(freq='D'))['totals.hits'].sum(),
    'set':
    'train'
})
test_daily_hit_count = pd.DataFrame({
    'count':
    test_ts.groupby(pd.Grouper(freq='D'))['totals.hits'].sum(),
    'set':
    'test'
})
daily_hit_count = pd.concat(
    [train_daily_hit_count, test_daily_hit_count], axis=0)


# In[ ]:


p = figure(x_axis_type='datetime', width=700, height=400, title='pageviews over time')
p.line(
    source=daily_pageviews[daily_pageviews['set'] == 'train'],
    x='visitStartTime',
    y='count',
    color=Set1_9[0])
p.line(
    source=daily_pageviews[daily_pageviews['set'] == 'test'],
    x='visitStartTime',
    y='count',
    color=Set1_9[1])
p.legend.location = 'top_left'
test_start = Span(location=test_start_time.timestamp() * 1000, dimension='height', line_dash='dashed')
p.add_layout(test_start)
show(p)

p = figure(x_axis_type='datetime', width=700, height=400, title='hits over time')
p.line(
    source=daily_hit_count[daily_hit_count['set'] == 'train'],
    x='visitStartTime',
    y='count',
    color=Set1_9[0])
p.line(
    source=daily_hit_count[daily_hit_count['set'] == 'test'],
    x='visitStartTime',
    y='count',
    color=Set1_9[1])
p.legend.location = 'top_left'
test_start = Span(
    location=test_start_time.timestamp() * 1000,
    dimension='height',
    line_dash='dashed')
p.add_layout(test_start)
show(p)


# We do not see a clear difference in the overall trend between pageviews and hits, but we can see that both have a yearly pattern that starts low at the beginning of the year, climbs throughout the year and drops at the end. This trend appears to be consistent in the slice of data we can see, however unfortunately we only have one cycle within the training set, so the cycle-to-cycle pattern and how that relates to overall revenue cannot be easily discovered.

# ### Hits/pageviews over time:

# In[ ]:


train_daily_hitrate = pd.DataFrame({
    'count':
    train_ts.groupby(pd.Grouper(freq='D')).apply(
        lambda x: (x['totals.hits'].sum() + 1) / (x['totals.pageviews'].sum() + 1)
    ),
    'set':
    'train'
})
test_daily_hitrate = pd.DataFrame({
    'count':
    test_ts.groupby(pd.Grouper(freq='D')).apply(
        lambda x: (x['totals.hits'].sum() + 1) / (x['totals.pageviews'].sum() + 1)
    ),
    'set':
    'test'
})
daily_hitrate = pd.concat([train_daily_hitrate, test_daily_hitrate], axis=0)


# In[ ]:


p = figure(x_axis_type='datetime', width=700, height=400, title='hits/pageviews ratio over time')
p.line(
    source=daily_hitrate[daily_hitrate['set'] == 'train'],
    x='visitStartTime',
    y='count',
    color=Set1_9[0])
p.line(
    source=daily_hitrate[daily_hitrate['set'] == 'test'],
    x='visitStartTime',
    y='count',
    color=Set1_9[1])
p.legend.location = 'top_left'
test_start = Span(location=test_start_time.timestamp() * 1000, dimension='height', line_dash='dashed')
p.add_layout(test_start)
show(p)


# ### Daily mobile percentage over time:

# In[ ]:


train_daily_mobile_rate = pd.DataFrame({
    'count':
    train_ts.groupby(pd.Grouper(freq='D'))['device.isMobile'].mean(),
    'set':
    'train'
})
test_daily_mobile_rate = pd.DataFrame({
    'count':
    test_ts.groupby(pd.Grouper(freq='D'))['device.isMobile'].mean(),
    'set':
    'test'
})
daily_mobile_rate = pd.concat([train_daily_mobile_rate, test_daily_mobile_rate], axis=0)


# In[ ]:


p = figure(x_axis_type='datetime', width=700, height=400, title='daily mobile percentage over time')
p.line(
    source=daily_mobile_rate[daily_mobile_rate['set'] == 'train'],
    x='visitStartTime',
    y='count',
    color=Set1_9[0])
p.line(
    source=daily_mobile_rate[daily_mobile_rate['set'] == 'test'],
    x='visitStartTime',
    y='count',
    color=Set1_9[1])
p.legend.location = 'top_left'
test_start = Span(location=test_start_time.timestamp() * 1000, dimension='height', line_dash='dashed')
p.add_layout(test_start)
show(p)


# We might have expected the percentage of mobile users to gradually increase over time, but we also observed a rather peculiar jump around the end of 2016. Does it have more to do with the overall mobile market, or is it just because Google launched or revamped an online store for mobile? Is it related to the dip in daily sessions we see earlier? We have yet to answer these questions.
# 
# Anyway, the observation here indicates that the condition for before the end of 2016 and after might be different enough that the usefulness of this feature at predicting revenue might not carry over to afterwards, including the test portion of the data, which is more similar to what happened after 2016. Now I will not be surprised if in some models, removing this feature from data points prior to 2017 actually increases its performance.

# ### ChannelGrouping categories weight over time:

# In[ ]:


train_daily_channel_grouping = train_ts.groupby([pd.Grouper(freq='D'), 'channelGrouping']).size().reset_index()
train_daily_channel_grouping['set'] = 'train'
test_daily_channel_grouping = test_ts.groupby([pd.Grouper(freq='D'), 'channelGrouping']).size().reset_index()
test_daily_channel_grouping['set'] = 'test'
daily_channel_grouping = pd.concat([train_daily_channel_grouping, test_daily_channel_grouping], axis=0)
daily_channel_grouping.rename(columns={0: 'count'}, inplace=True)
daily_channel_grouping['perc'] = daily_channel_grouping['count']


# In[ ]:


daily_channel_grouping_merged = pd.merge(
    daily_channel_grouping,
    daily_sess_count.reset_index()[['visitStartTime', 'count'
                                    ]].rename(columns={'count': 'sess_count'}),
    on='visitStartTime')
daily_channel_grouping_merged[
    'perc'] = daily_channel_grouping_merged['count'] / daily_channel_grouping_merged['sess_count']


# In[ ]:


p = figure(
    x_axis_type='datetime',
    width=700,
    height=600,
    title='channelGrouping percentage shares over time')
channels = list(daily_channel_grouping['channelGrouping'].unique())
for idx, channel in enumerate(channels):
    p.line(
        source=daily_channel_grouping_merged[
            (daily_channel_grouping_merged['channelGrouping'] == channel)
            & (daily_channel_grouping_merged['perc'] < 1)],
        x='visitStartTime',
        y='perc',
        color=Set1_9[idx],
        legend=channel)
p.y_range = Range1d(0, 1)
p.legend.click_policy = 'hide'
p.add_tools(
    HoverTool(tooltips=[('category',
                         '@channelGrouping'), ('percentage',
                                               '@perc'), ('count', '@count')]))
test_start = Span(
    location=test_start_time.timestamp() * 1000,
    dimension='height',
    line_dash='dashed')
p.add_layout(test_start)
show(p)


# **(Click on the legend to hide one or more lines in the plot)**
# 
# Apparently a lot was going on at the end of 2016. Not only did 'Social' channel suffered a massive drop in percentage share, 'Organic Search' recovered from a dip and the 'Direct' channel enjoyed a moderate increase. This is also around the time that the overall session count decreased, so the drop in the absolute number of 'Social' channel is likely to driving force for the series of changes. 'Social' channel further suffered a drop around May of 2017. We also noticed that the 'Display' channel enjoyed a massive share spike in the test period, which had not happened at all throughout the training peroid. Does this mean we have to treat each of the channels differently and assign them different 'usefulness rating' for predicting user revenue? It appears to be the case here. Treating this attribute as a single column looks like a less appealing option after seeing this.

# ### Top browser weight over time:

# In[ ]:


train_daily_browser = train_ts.groupby([pd.Grouper(freq='D'), 'device.browser']).size().reset_index()
train_daily_browser['set'] = 'train'
test_daily_browser = test_ts.groupby([pd.Grouper(freq='D'), 'device.browser']).size().reset_index()
test_daily_browser['set'] = 'test'
daily_browser = pd.concat([train_daily_browser, test_daily_browser], axis=0)
daily_browser.rename(columns={0: 'count'}, inplace=True)
daily_browser['perc'] = daily_browser['count']

daily_browser_merged = pd.merge(
    daily_browser,
    daily_sess_count.reset_index()[['visitStartTime', 'count'
                                    ]].rename(columns={'count': 'sess_count'}),
    on='visitStartTime')
daily_browser_merged[
    'perc'] = daily_browser_merged['count'] / daily_browser_merged['sess_count']


# In[ ]:


top_browsers = set(train['device.browser'].value_counts().index[:5])
top_browsers


# In[ ]:


p = figure(
    x_axis_type='datetime',
    width=700,
    height=600,
    title='browser percentage shares over time')
for idx, browser in enumerate(top_browsers):
    p.line(
        source=daily_browser_merged[
            (daily_browser_merged['device.browser'] == browser)
            & (daily_browser_merged['perc'] < 1)],
        x='visitStartTime',
        y='perc',
        color=Set1_9[idx],
        legend=browser)
p.y_range = Range1d(0, 1)
p.legend.click_policy = 'hide'
p.add_tools(
    HoverTool(tooltips=[('category',
                         '@{device.browser}'), ('percentage',
                                                '@perc'), ('count',
                                                           '@count')]))
test_start = Span(
    location=test_start_time.timestamp() * 1000,
    dimension='height',
    line_dash='dashed')
p.add_layout(test_start)
show(p)


# We can see from this plot that the browser percentage shares are not behaving normally around a short period of time, but otherwise appear to be someewhat stationary over time.

# ### Top OS weight over time:

# In[ ]:


train_daily_os = train_ts.groupby([pd.Grouper(freq='D'), 'device.operatingSystem']).size().reset_index()
train_daily_os['set'] = 'train'
test_daily_os = test_ts.groupby([pd.Grouper(freq='D'), 'device.operatingSystem']).size().reset_index()
test_daily_os['set'] = 'test'
daily_os = pd.concat([train_daily_os, test_daily_os], axis=0)
daily_os.rename(columns={0: 'count'}, inplace=True)
daily_os['perc'] = daily_os['count']

daily_os_merged = pd.merge(
    daily_os,
    daily_sess_count.reset_index()[['visitStartTime', 'count'
                                    ]].rename(columns={'count': 'sess_count'}),
    on='visitStartTime')
daily_os_merged[
    'perc'] = daily_os_merged['count'] / daily_os_merged['sess_count']


# In[ ]:


top_os = set(train['device.operatingSystem'].value_counts().index[:6])
top_os


# In[ ]:


p = figure(
    x_axis_type='datetime',
    width=700,
    height=600,
    title='OS percentage shares over time')
for idx, os in enumerate(top_os):
    p.line(
        source=daily_os_merged[
            (daily_os_merged['device.operatingSystem'] == os)
            & (daily_os_merged['perc'] < 1)],
        x='visitStartTime',
        y='perc',
        color=Set1_9[idx],
        legend=os)
p.y_range = Range1d(0, 0.7)
p.legend.click_policy = 'hide'
p.add_tools(
    HoverTool(tooltips=[('category',
                         '@{device.operatingSystem}'), ('percentage',
                                                        '@perc'), ('count',
                                                                   '@count')]))
test_start = Span(
    location=test_start_time.timestamp() * 1000,
    dimension='height',
    line_dash='dashed')
p.add_layout(test_start)
show(p)


# Here we see a large jump of mobile OS share around the same time we saw the overall mobile percentage numbers jump earlier. There are also a few spikes for Android in the test portion of the data, which is quite unlike what it looked like before. There is also a spike followed by continue decline for Mac users.

# ### Revenue over time?

# In[ ]:


train_daily_revenue = pd.DataFrame({
    'sum':
    train_ts.groupby(pd.Grouper(freq='D'))['totals.transactionRevenue'].sum(),
    'set':
    'train'
})
test_daily_revenue = pd.DataFrame({'dummy': test_ts.groupby(pd.Grouper(freq='D')).size(), 'set': 'test'})
test_daily_revenue['sum'] = train_daily_revenue['sum'].mean()
test_daily_revenue.drop('dummy', axis=1, inplace=True)
daily_revenue = pd.concat([train_daily_revenue, test_daily_revenue], axis=0, sort=True)

p = figure(
    x_axis_type='datetime',
    width=700,
    height=400,
    title='revenue over time?')
p.line(
    source=daily_revenue[daily_revenue['set'] == 'train'],
    x='visitStartTime',
    y='sum',
    color=Set1_9[0])
p.line(
    source=daily_revenue[daily_revenue['set'] == 'test'],
    x='visitStartTime',
    y='sum',
    color=Set1_9[1])
p.legend.location = 'top_left'
test_start = Span(
    location=test_start_time.timestamp() * 1000,
    dimension='height',
    line_dash='dashed')
p.add_layout(test_start)
show(p)


# As we see above, many of the attributes are either non-stationary, have peculiar trends during a certain period of time, or behaves quite differently in the training and test set. Surely, some of these non-stationarities and train/test distinctions will have to be taken into account for making predictions about future.
# 
# (Many questions to answer still)

# In[ ]:




