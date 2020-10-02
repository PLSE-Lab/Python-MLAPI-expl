#!/usr/bin/env python
# coding: utf-8

# ## Simple Significance Testing of Calendar Events on Sale of Each Item
# ### Summary:
#    * For each item I compute a t-test statistic and pvalue, evaluating whether the the sale of the item is "significantly" affected by the approaching of an event.
# 
# ### Observations and caveats:
#    * Note that some items can be negatively affected by an event.
#    * While the event days are consecutive ranges (see method below), the non-event days are not. If we were to select non-event days as consecutive too, the results could possibly change a bit.
#    * "Correlation doesn't mean causation": sale of a food item can increase by Valentine's day appraoching but that might be because it's the cold February driving people to buy ingredients for their chili. But this would show as a yearly trend in the data too.
# 
#    
# ### Method:
#    * Extract all events from the calendar.
#    * For each event, extract the days of the event.
#    * Given a range (for example a week before) extract all days within that range for all events. (call these "event days"
#    * For each event, generate a random non-event set of days. 
#        ** number of these days are set to be equal to the number of "event days".
#        ** specify a range of days around the event that cannot include non-event days (here my non-event days are at least one month away from the event itself)
#    * For each item and for each event, perform Welch's t-test and get the pvalue associated with it

# **module imports:** 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
import random
from scipy.stats import ttest_ind

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# **reading calendar:**

# In[ ]:


calendar = pd.read_csv(os.path.join(dirname, "calendar.csv"))
calendar.tail(5)


# There are only a few days where two events coincide:

# In[ ]:


calendar[~pd.isna(calendar['event_name_1']) & ~pd.isna(calendar['event_name_2'])]


# Quick look at the list of all events in the calendar:

# In[ ]:


calendar['event_name_1'].unique()


# **Reading sales data:** There are ~30K rows in this dataset, but number of unique items are ~3K, that's because each item is repeated ten times (once for each store; 10 stores in total).

# In[ ]:


sales = pd.read_csv(os.path.join(dirname, "sales_train_validation.csv"))
print(sales.shape)
print(sales['item_id'].unique().shape)
print(sales['store_id'].unique().shape)


# Quick look at sales table:

# In[ ]:


sales.head()


# This function extracts a dictionary of event names mapped to their days of observation in the calendar:

# In[ ]:


def get_event_days(calendar):
    events = {i:[] for i in np.concatenate((calendar['event_name_1'].unique(), calendar['event_name_2'].unique())) if not pd.isna(i)}
    for event in events.keys():
        event_days = calendar.iloc[np.where((calendar['event_name_1'] == event) | (calendar['event_name_2'] == event))]['d'].tolist()
        events[event] = event_days
    return events


# In[ ]:


events = get_event_days(calendar)
print(events['SuperBowl'])


# This function extends the list of days of observance for each event to a range. For example for Valentines day and an *offset* parameter of 7 it generates list of all days in the data that fall between Feb 14 to Feb21. Since some of these days will fall on the validation data for which we don't have sale data [yet], it will cause problems later. So the *possible_days* parameter allows inputting a list of days for which we have the sale data (we will use sales.columns).

# In[ ]:


def get_event_range(events, offset = -14, possible_days = None):
    events_range = {}
    for event, days in events.items():
        events_range[event] = []
        days = [int(float(day[2:])) for day in days]
        if offset < 0:
            for day in days:
                e_range = ["d_" + str(i) for i in list(range(max(1, day + offset), day + 1))]
                events_range[event].extend(e_range)
        else:
            for day in days:
                e_range = ["d_" + str(i) for i in list(range(day, day + offset + 1))]
                events_range[event].extend(e_range)
        if possible_days is not None:
            events_range[event] = [i for i in events_range[event] if i in possible_days]
    return events_range   


# In[ ]:


seven_day_after = get_event_range(events, 7, sales.columns)
seven_day_before = get_event_range(events, -7, sales.columns)
one_month_after = get_event_range(events, 30, sales.columns)
one_month_before = get_event_range(events, -30, sales.columns)
one_month_twoway = {event: sorted(list(set(one_month_before[event] + one_month_after[event]))) for event in events.keys()}


# This is what the following function does: for each event generates a list of days that do not overlap with the days of the event. For example for V day we selected all days from Feb 14 to Feb 21, now we want a control group that does not overlap this set of days, so we select from the rest of available days. 
# 
# Since we want to reduce the effect of the event on our control group, we will potentially want to exclude a bigger range of dates than only the seven days, that can be done by the second parameter *non_avail_days*. I will use 1 week before the event as event days, and exclude one month before and after the event from non-event days.
# 
# How many non-event days for each event do we want to sample? This can be specified by *count* parameter. If it is not provided, the same number of days that are in the "event days" that are provided in the first argument (for our case seven days).

# In[ ]:


def get_nonevent_days(events, non_avail_days, all_days, count = None):
    #events: is a dictionary of event names to list of observance of each event
    #non_avail_days: list of days for each event that we don't want to consider as non-event (dictionary formatted again)
    #all_days: list of all possible days in the dataset
    #count: how many non-event days per event should be sampled. if count is none, len of days in "events" is used.
    all_days = set(all_days)
    non_events = {}
    for event in events.keys():
        all_nonevent = all_days - set(non_avail_days[event])
        if count is None:
            count = len(events[event])
        non_events[event] = list(random.sample(all_nonevent, count))
    return non_events
        


# In[ ]:


nonevents = get_nonevent_days(seven_day_before, one_month_twoway, list(sales.columns)[6:])


# In[ ]:


print(len(nonevents['SuperBowl']))
print(len(events['SuperBowl']))
print(len(one_month_before['SuperBowl']))
print(len(one_month_twoway['SuperBowl']))


# Now we iterate over all events, and run the t-test, put the output in a dataframe.
# 
# *comparison_dfs* is a dictionary mapping event names to dataframes, where each dataframes has one row per item and one column per "event" or "non-event" day. (plus a column for sum of event/nonevent days and pvalues). We will later use this to plot a few examples.
# 
# *event_item_significance* is a dataframe that includes one row per item and one column per event. Each element is the pvalue associated with the effect of the event on sale of the item for the corresponding row and column.

# In[ ]:


event_range = seven_day_before
nonevents = get_nonevent_days(event_range, one_month_twoway, list(sales.columns)[6:])
event_item_significance = pd.DataFrame({'item_id':sales['item_id'].unique()})
comparison_dfs = {}
for event in events.keys():
    sales_event = sales[['item_id'] + event_range[event]]
    sales_nonevent = sales[['item_id'] + nonevents[event]]
    sum_event= sales_event.groupby('item_id').sum()
    sum_event['event_sum'] = sum_event.sum(axis = 1)
    sum_nonevent = sales_nonevent.groupby('item_id').sum()#
    sum_nonevent['nonevent_sum'] = sum_nonevent.sum(axis = 1)
    sum_event.reset_index(inplace = True)
    sum_nonevent.reset_index(inplace = True)
    sales_comp = sum_event.merge(sum_nonevent)
    #sum_event.head()
    test_stat, pvals = ttest_ind(sales_comp[event_range[event]], sales_comp[nonevents[event]], axis = 1, equal_var = False)
    sales_comp['test_stat'] = test_stat
    sales_comp['pval_' + event] = pvals
    comparison_dfs[event] = sales_comp
    event_item_significance = event_item_significance.merge(sales_comp[['item_id', 'pval_' + event]], on = "item_id")


# In[ ]:


event_item_significance.head()


# Now let's look at a few examples:

# In[ ]:


def plot_item(event, item_index):
    item_id = comparison_dfs[event].iloc[item_index]['item_id']
    item = pd.DataFrame({'event_sale': comparison_dfs[event].iloc[item_index][event_range[event]].values,
                        'nonevent_sale': comparison_dfs[event].iloc[item_index][nonevents[event]].values})
    plt.boxplot(item.T, labels = ['event', 'nonevent'])
    plt.title("sale of " + item_id + " wrt " + event + " pval:" + str(event_item_significance[event_item_significance['item_id'] == item_id]['pval_' + event].values))
    #item.head()


# In[ ]:


event = 'SuperBowl'
item_index = 5
plot_item(event, item_index)


# In[ ]:


event = 'SuperBowl'
item_index = 1000
plot_item(event, item_index)


# In[ ]:


event = 'SuperBowl'
item_index = 3000
plot_item(event, item_index)


# In[ ]:


event = 'Easter'
item_index = 1000
plot_item(event, item_index)


# In[ ]:




