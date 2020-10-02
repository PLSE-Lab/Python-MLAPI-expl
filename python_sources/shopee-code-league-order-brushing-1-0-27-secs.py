#!/usr/bin/env python
# coding: utf-8

# # Result Summary 

# **All 3 links have 315 brushing shops and ~ 30 seconds runtime on full data**
# 
# - [Score: 1](#1)  (do not double count orders no matter how many brushing windows they appear in)
# - [Score 0.99508](#2) (max user after aggregating all brushing windows found (same score [with updating](#2.1) or [without updating](#2) within a `start_idx`))
# - [Score 0.99263](#3) (pick out max user within each brushing window)

# ![window.jpg](attachment:window.jpg)

# # Interpretation challenges
# 
# **Questions**
# - The task contains many issues to be clear of:
#     1. How to create 1hour windows (event_time being start of 1 hr vs "at any instance")
#     2. Aggregate highest proportion user within each window or after all windows are found for a shop
#     3. Recount same orderids/userids that appeared in different brushing windows vs count once
#     
#   
# **Reasons**
# 1. If windows only created with hour beginning at timestamp, we are losing the opportunity to exclude users (by moving the 1hour window back until just behind previous event_time) to reduce the denominator of unique users and increase chances of increasing concentration ratio. The examples provided could have been misleading because it shows multiple 1hour windows with the start being an `event_time` rather than an imaginary time. There were subtle hints _"at any instance" (Overview-->Description)_ and _"Please consider all possible '1 hour' time interval" (Examples--> Case 2b)_ that a fast reader could miss.
# 
# 2. If the highest proportion was calculated within each window, there would be no need for Case 4 to show 2 brushing windows. However, no matter getting max users within each or all windows, the denominator does not need to be calculated. Also, the highest proportion users from Case 4 example is the same for both ways of calculating. Case 3 used _User 201343856 had the highest proportion of orders during the order brushing period_. This hints at the method of calculating within each window.
# 
# 3. Because time can be shifted infinitesimally small, there are infinitely many possible brushing windows. To deal with this problem, we can think of only working with "relevant windows" (each unique set of orders to be enumerated). Because a user can appear in the tail of one window and the head of another window when other users transact before or after him (which creates a "relevant window"), generating brushing users for a shop based on calculating window statistics (both within each window (0.99263) and all windows(0.99508)) would cause the same user/orderid to be counted multiple times for a user. Such a phenomenon of other user events increasing the brushing suspicion level of a certain user does not make sense. Thus, relevant windows should be seen as only useful for labeling each order as brushed vs not brushed. The 0.99508 solution could have more brushing users than 1.0 solution for shopid 155143347, 156883302 because of overcounting a user for the same orderid when this user appears in multiple windows. 

# # Solution strategy (same for all top 3 scores except step 4)
# 
# 1. Time continuous, impossible to enumerate all time windows --> just find all the sets of orders that can appear within 1hour (like enumerating combinations but within a time bound of 1hr from the first order) 
#     - If orders in brushing window land exactly 1 hour apart, 1hr assumed exclusive on right side
# 2. Start from every timestamp of sorted timestamps for shop and find the smallest and largest set of rows that can make a 1hr window for current timestamp, then calculate through all the windows from smallest to largest 
# 3. For concentrate rate to be >= 3, the window has to be at least size 3, so can start growing window from this size as minimum, but this minimum could be pulled up by the previous timestamp which controls how far back the 1hour window for current timestamp can be pushed back
# 4. Save order and user information if is brushing window to accumulate later
# 
# 
# - Using Evaluation page formula, 
# ```
# number of 0 = x
# 0.005*x + (18770-x) = 407.275
# x = number of 0 = 18445 --> number of brushing shops = 18770 - 18455 = 315
# ```
# - There are 315 brushing shops expected 

# In[ ]:


from collections import Counter
from functools import reduce
import operator
from bisect import bisect_left

import numpy as np
import pandas as pd

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

pd.options.display.max_rows = 500
pd.set_option('display.max_colwidth', None)


# In[ ]:


orders = pd.read_csv('order_brush_order.csv')


# In[ ]:


orders.head()


# In[ ]:


orders.info()


# In[ ]:


for col in orders:
    print(f'{col} nunique: ',orders[col].nunique())
    
for col in orders:
    print(f'{col} duplicated: ',sum(orders[col].duplicated()))


# In[ ]:


orders.head()


# In[ ]:


orders[orders.groupby('shopid').event_time.apply(pd.Series.duplicated,keep=False)].sort_values('event_time')


# In[ ]:


orders['event_time'] = pd.to_datetime(orders['event_time']) # to ensure proper sorting, not necessary but to be safe


# In[ ]:


orders.dtypes


# In[ ]:


# sort for easy debugging when comparing against kaggle examples 
orders_sorted = orders.sort_values(['shopid','event_time'])
orders_sorted.head(100)


# # Getting 1 shop for easy debugging using examples

# In[ ]:


# good for preventing repeated time spent on groupby, but cannot slice groupby object to estimate time during full run
shop_gb = orders_sorted.groupby(['shopid'])  


# <a id='2.1'></a>
# ###  max user after aggregating all windows
# - updates Counter for each start_idx so counts are based on largest window for each start_idx
# - Makes no sense on hindsight, but gets 0.99508
# 

# In[ ]:


test_cases = {1:8996761,
              2:27121667,
              3:145777302,
              4:181009364
             }

order_shop = shop_gb.get_group(test_cases[3])

# imagine order_shop df is passed in to this apply func
event_times = order_shop['event_time'].values
array_length = len(event_times)
counter_list = []


for start_idx, start_time in enumerate(event_times[:-2]):
    counts_for_start_time = {}
    
    max_end_time = start_time + np.timedelta64(1, 'h')   # prepare to find elements within this right bound of time
    max_end_idx = bisect_left(event_times, max_end_time) - 1    # find largest idx within time bound, this will be largest possible window for current start_time
    
    if max_end_idx < start_idx + 2:
        continue # no need to continue if cannot form at least 3 rows
    
    
    if start_idx:  
        left_timeshift_possible = start_time - event_times[start_idx-1]  # can be handled by df.diff outside apply
        min_end_time = max_end_time - left_timeshift_possible  
        
        # smallest window begins at 3 rows minimum, or idx of largest time less than min_end_time 
        min_end_idx = max(start_idx + 2, bisect_left(event_times, min_end_time)-1) 

        print('start: {}  min_end: {}  max_end: {}'.format(start_idx,min_end_idx,max_end_idx))
    else:
        min_end_idx = start_idx + 2
    
    for window_tail_idx in range(min_end_idx, max_end_idx + 1):
        current_window = order_shop.iloc[start_idx: window_tail_idx+1] #iloc excludes right edge
        concentration_ratio = len(current_window)/current_window['userid'].nunique()
        
        if concentration_ratio >= 3: 

            counts_for_start_time.update(dict(current_window['userid'].value_counts()))
            
            event_times[start_idx-1]
            event_times[window_tail_idx+1]
            current_window
            counts_for_start_time
        
    # prevent appending empty counts_for_start_time (for clean debugging prints, doesn't affect Counter summation later)
    if counts_for_start_time:
        counter_list.append(Counter(counts_for_start_time))  # prepare Counter type for accumulation later
        counter_list
            
                
if counter_list:  # if not empty [{}] for shops with no brushing:
    reduced_counter_list = reduce(operator.add,counter_list)
    reduced_counter_list
    max_value = max(reduced_counter_list.values())
    users = sorted(user for user,count in reduced_counter_list.items() if count == max_value)
    print('FINAL ANSWER BELOW')
    '&'.join(map(str,users))
else:
    '0'


# <a id='2'></a>
# ### Adding all windows in without updating 
# - Same start_idx will get all its user counts summed

# In[ ]:


test_cases = {1:8996761,
              2:27121667,
              3:145777302,
              4:181009364
             }

order_shop = shop_gb.get_group(test_cases[3])

# imagine order_shop df is passed in to this apply func
event_times = order_shop['event_time'].values
array_length = len(event_times)
counter_list = []


for start_idx, start_time in enumerate(event_times[:-2]):
    
    max_end_time = start_time + np.timedelta64(1, 'h')   # prepare to find elements within this right bound of time
    max_end_idx = bisect_left(event_times, max_end_time) - 1    # find largest idx within time bound, this will be largest possible window for current start_time
    
    if max_end_idx < start_idx + 2:
        continue # no need to continue if cannot form at least 3 rows
    
    
    if start_idx:  
        left_timeshift_possible = start_time - event_times[start_idx-1]  # can be handled by df.diff outside apply
        min_end_time = max_end_time - left_timeshift_possible  
        
        # smallest window begins at 3 rows minimum, or idx of largest time less than min_end_time 
        min_end_idx = max(start_idx + 2, bisect_left(event_times, min_end_time)-1) 

        print('start: {}  min_end: {}  max_end: {}'.format(start_idx,min_end_idx,max_end_idx))
    else:
        min_end_idx = start_idx + 2
    
    for window_tail_idx in range(min_end_idx, max_end_idx + 1):
        current_window = order_shop.iloc[start_idx: window_tail_idx+1] #iloc excludes right edge
        concentration_ratio = len(current_window)/current_window['userid'].nunique()
        
        if concentration_ratio >= 3: 
            counter_list.append(Counter(current_window['userid']))
            
            event_times[start_idx-1]
            event_times[window_tail_idx+1]
            current_window
        
                
if counter_list:  # if not empty [{}] for shops with no brushing:
    reduced_counter_list = reduce(operator.add,counter_list)
    reduced_counter_list
    max_value = max(reduced_counter_list.values())
    users = sorted(user for user,count in reduced_counter_list.items() if count == max_value)
    print('FINAL ANSWER BELOW')
    '&'.join(map(str,users))
else:
    '0'


# <a id='3'></a>
# ### Windows with looking back (max user within each window)
# - prevent too few unique users in window

# In[ ]:


test_cases = {1:8996761,
              2:27121667,
              3:145777302,
              4:181009364
             }

order_shop = shop_gb.get_group(test_cases[1])

# imagine order_shop df is passed in to this apply func
event_times = order_shop['event_time'].values
array_length = len(event_times)
user_set = set()


for start_idx, start_time in enumerate(event_times[:-2]):
    
    max_end_time = start_time + np.timedelta64(1, 'h')   # prepare to find elements within this right bound of time
    max_end_idx = bisect_left(event_times, max_end_time) - 1    # find largest idx within time bound, this will be largest possible window for current start_time
    
    start_idx, max_end_idx
    
    if max_end_idx < start_idx + 2:
        print('skip')
        continue # no need to continue if cannot form at least 3 rows
    
    start_idx,max_end_idx
    
    if start_idx:  
        left_timeshift_possible = start_time - event_times[start_idx-1]  # can be handled by df.diff outside apply
        min_end_time = max_end_time - left_timeshift_possible  
        
        # smallest window begins at 3 rows minimum, or idx of largest time less than min_end_time 
        min_end_idx = max(start_idx + 2, bisect_left(event_times, min_end_time) - 1) 

        print('min_end: {} max_end: {}'.format(min_end_idx,max_end_idx))
    else:
        min_end_idx = start_idx + 2
    
    for window_tail_idx in range(min_end_idx, max_end_idx + 1):
        current_window = order_shop.iloc[start_idx: window_tail_idx+1] #iloc excludes right edge
        concentration_ratio = len(current_window)/current_window['userid'].nunique()
        
        if concentration_ratio >= 3: 
            current_window_counts = Counter(current_window['userid'])

            max_value = max(current_window_counts.values())
            user_set.update(user for user, count in current_window_counts.items() if count ==  max_value)
            
            current_window
            current_window_counts
                
if user_set:  # if not empty [{}] for shops with no brushing:
    users = sorted(user_set)
    print(users)
    print('FINAL ANSWER BELOW')
    '&'.join(map(str,users))
else:
    '0'
    
    # ADD RETURN STATEMENT WHEN PASTING INTO FUNCTION


# ### Refining bisect for min_end_idx

# In[ ]:


test_cases = {1:8996761,
              2:27121667,
              3:145777302,
              4:181009364,
              5:155143347,
              6:156883302
             }

order_shop = shop_gb.get_group(test_cases[6])

# imagine order_shop df is passed in to this apply func
event_times = order_shop['event_time'].values
array_length = len(event_times)
counter_list = []


for start_idx, start_time in enumerate(event_times[:-2]):
    counts_for_start_time = {}
    
    max_end_time = start_time + np.timedelta64(1, 'h')   # prepare to find elements within this right bound of time
    max_end_idx = bisect_left(event_times, max_end_time) - 1    # find largest idx within time bound, this will be largest possible window for current start_time
    
    if max_end_idx < start_idx + 2:
        continue # no need to continue if cannot form at least 3 rows
    
    
    if start_idx:  
        left_timeshift_possible = start_time - event_times[start_idx-1]  # can be handled by df.diff outside apply
        min_end_time = max_end_time - left_timeshift_possible  

        bisected_idx = bisect_left(event_times, min_end_time)
        # short-circuit prevents IndexError when event_times[bisected_idx] after or 
        if bisected_idx == array_length or event_times[bisected_idx] >  min_end_time:
            bisected_idx -= 1
            while bisected_idx > start_idx+2 and event_times[bisected_idx-1] == event_times[bisected_idx]: 
                bisected_idx -= 1
        # smallest window begins at 3 rows minimum, or idx of largest time less than min_end_time 
        min_end_idx = max(start_idx + 2,bisected_idx) 
    #   print('start: {}  min_end: {}  max_end: {}'.format(start_idx,min_end_idx,max_end_idx))
    else:
        min_end_idx = start_idx + 2
    
    for window_tail_idx in range(min_end_idx, max_end_idx + 1):
        current_window = order_shop.iloc[start_idx: window_tail_idx+1] #iloc excludes right edge
        concentration_ratio = len(current_window)/current_window['userid'].nunique()
        
        if concentration_ratio >= 3: 

            counts_for_start_time.update(dict(current_window['userid'].value_counts()))
            
            event_times[start_idx-1]
            event_times[window_tail_idx+1]
            current_window
            counts_for_start_time
        
    # prevent appending empty counts_for_start_time (for clean debugging prints, doesn't affect Counter summation later)
    if counts_for_start_time:
        counter_list.append(Counter(counts_for_start_time))  # prepare Counter type for accumulation later
        counter_list
            
                
if counter_list:  # if not empty [{}] for shops with no brushing:
    reduced_counter_list = reduce(operator.add,counter_list)
    reduced_counter_list
    max_value = max(reduced_counter_list.values())
    users = sorted(user for user,count in reduced_counter_list.items() if count == max_value)
    print('FINAL ANSWER BELOW')
    '&'.join(map(str,users))
else:
    '0'


# <a id='1'></a>
# ### Not recounting same orders in different windows 

# In[ ]:


test_cases = {1:8996761,
              2:27121667,
              3:145777302,
              4:181009364,
              5:155143347,
              6:156883302
             }

order_shop = shop_gb.get_group(test_cases[5])

# imagine order_shop df is passed in to this apply func
event_times = order_shop['event_time'].values
array_length = len(event_times)

order_user = {}

for start_idx, start_time in enumerate(event_times[:-2]):
    
    max_end_time = start_time + np.timedelta64(1, 'h')   # prepare to find elements within this right bound of time
    max_end_idx = bisect_left(event_times, max_end_time) - 1    # find largest idx within time bound, this will be largest possible window for current start_time
    
    if max_end_idx < start_idx + 2:
        continue # no need to continue if cannot form at least 3 rows
    
    
    if start_idx:  
        left_timeshift_possible = start_time - event_times[start_idx-1]  # can be handled by df.diff outside apply
        min_end_time = max_end_time - left_timeshift_possible  

        bisected_idx = bisect_left(event_times, min_end_time)
        # short-circuit prevents IndexError when event_times[bisected_idx] after or 
        if bisected_idx == array_length or event_times[bisected_idx] >  min_end_time:
            bisected_idx -= 1
            while bisected_idx > start_idx+2 and event_times[bisected_idx-1] == event_times[bisected_idx]: 
                bisected_idx -= 1
        # smallest window begins at 3 rows minimum, or idx of largest time less than min_end_time 
        min_end_idx = max(start_idx + 2,bisected_idx) 
    #   print('start: {}  min_end: {}  max_end: {}'.format(start_idx,min_end_idx,max_end_idx))
    else:
        min_end_idx = start_idx + 2
    
    for window_tail_idx in range(min_end_idx, max_end_idx + 1):
        current_window = order_shop.iloc[start_idx: window_tail_idx+1] #iloc excludes right edge
        concentration_ratio = len(current_window)/current_window['userid'].nunique()
        
        if concentration_ratio >= 3: 
            
            order_user.update(dict(zip(current_window['orderid'],current_window['userid'])))            
            
            event_times[start_idx-1]
            event_times[window_tail_idx+1]
            current_window
        

if order_user:
    user_counts = {userid:list(order_user.values()).count(userid) for userid in set(order_user.values())}
    max_value = max(user_counts.values())
    users = sorted(user for user,count in user_counts.items() if count == max_value)
    print('FINAL ANSWER BELOW')
    '&'.join(map(str,users))
else:
    '0'


# # Running on full data

# ### all windows aggregated 

# In[ ]:


def find_brush_enum_window_aggregate(order_shop):
    
    event_times = order_shop['event_time'].values
    array_length = len(event_times)
    counter_list = []


    for start_idx, start_time in enumerate(event_times[:-2]):
        counts_for_start_time = {}

        max_end_time = start_time + np.timedelta64(1, 'h')   # prepare to find elements within this right bound of time
        max_end_idx = bisect_left(event_times, max_end_time) - 1    # find largest idx within time bound, this will be largest possible window for current start_time

        if max_end_idx < start_idx + 2:
            continue # no need to continue if cannot form at least 3 rows

        if start_idx:  
            left_timeshift_possible = start_time - event_times[start_idx-1]  # can be handled by df.diff outside apply
            min_end_time = max_end_time - left_timeshift_possible  

            # smallest window begins at 3 rows minimum, or idx of largest time less than min_end_time 
            min_end_idx = max(start_idx + 2, bisect_left(event_times, min_end_time)-1) 
        else:
            # no row before start_idx == 0, so no restriction from bisected_idx
            min_end_idx = start_idx + 2

        for window_tail_idx in range(min_end_idx, max_end_idx + 1):
            current_window = order_shop.iloc[start_idx: window_tail_idx+1] #iloc excludes right edge
            concentration_ratio = len(current_window)/current_window['userid'].nunique()

            if concentration_ratio >= 3: 
                counts_for_start_time.update(dict(current_window['userid'].value_counts()))
        

        # prevent appending empty counts_for_start_time (for clean debugging prints, doesn't affect Counter summation later)
        if counts_for_start_time:
            counter_list.append(Counter(counts_for_start_time))  # prepare Counter type for accumulation later

    if counter_list:  # if not empty [{}] for shops with no brushing:
        reduced_counter_list = reduce(operator.add,counter_list)
        max_value = max(reduced_counter_list.values())
        users = sorted(user for user,count in reduced_counter_list.items() if count == max_value)
        return '&'.join(map(str,users))
    else:
        return '0'


# In[ ]:


#result_enum_window_aggregate = orders_sorted.groupby(['shopid']).apply(find_brush_enum_window_aggregate)


# In[ ]:


#result_enum_window_aggregate = result_enum_window_aggregate.reset_index(name='userid')
#result_enum_window_aggregate.to_csv('enum_window_aggregate.csv',index=False)


# ### all windows aggregated - no update 

# In[ ]:


def find_brush_enum_window_no_update(order_shop):
    event_times = order_shop['event_time'].values
    array_length = len(event_times)
    counter_list = []


    for start_idx, start_time in enumerate(event_times[:-2]):

        max_end_time = start_time + np.timedelta64(1, 'h')   # prepare to find elements within this right bound of time
        max_end_idx = bisect_left(event_times, max_end_time) - 1    # find largest idx within time bound, this will be largest possible window for current start_time

        if max_end_idx < start_idx + 2:
            continue # no need to continue if cannot form at least 3 rows


        if start_idx:  
            left_timeshift_possible = start_time - event_times[start_idx-1]  # can be handled by df.diff outside apply
            min_end_time = max_end_time - left_timeshift_possible  

            # smallest window begins at 3 rows minimum, or idx of largest time less than min_end_time 
            min_end_idx = max(start_idx + 2, bisect_left(event_times, min_end_time)-1)
            
        else:
            # no row before start_idx == 0, so no restriction from bisected_idx
            min_end_idx = start_idx + 2

        for window_tail_idx in range(min_end_idx, max_end_idx + 1):
            current_window = order_shop.iloc[start_idx: window_tail_idx+1] #iloc excludes right edge
            concentration_ratio = len(current_window)/current_window['userid'].nunique()

            if concentration_ratio >= 3: 
                counter_list.append(Counter(current_window['userid']))


    if counter_list:  # if not empty [{}] for shops with no brushing:
        reduced_counter_list = reduce(operator.add,counter_list)
        max_value = max(reduced_counter_list.values())
        users = sorted(user for user,count in reduced_counter_list.items() if count == max_value)

        return '&'.join(map(str,users))
    else:
        return '0'


# In[ ]:


#result_enum_window_no_update = orders_sorted.groupby(['shopid']).apply(find_brush_enum_window_no_update)


# In[ ]:


#result_enum_window_no_update = result_enum_window_no_update.reset_index(name='userid')
#result_enum_window_no_update.to_csv('enum_window_no_update.csv',index=False)


# ### Window look back (max user in each window)  

# In[ ]:


def find_brush_enum_window(order_shop):


    event_times = order_shop['event_time'].values
    array_length = len(event_times)
    user_set = set()


    for start_idx, start_time in enumerate(event_times[:-2]):

        max_end_time = start_time + np.timedelta64(1, 'h')   # prepare to find elements within this right bound of time
        max_end_idx = bisect_left(event_times, max_end_time) - 1    # find largest idx within time bound, this will be largest possible window for current start_time

        if max_end_idx < start_idx + 2:
            continue # no need to continue if cannot form at least 3 rows

        if start_idx:  
            left_timeshift_possible = start_time - event_times[start_idx-1]  # can be handled by df.diff outside apply
            min_end_time = max_end_time - left_timeshift_possible  

            # smallest window begins at 3 rows minimum, or idx of largest time less than min_end_time 
            min_end_idx = max(start_idx + 2, bisect_left(event_times, min_end_time)-1)

        else:
            # no row before start_idx == 0, so no restriction from bisected_idx
            min_end_idx = start_idx + 2

        for window_tail_idx in range(min_end_idx, max_end_idx + 1):
            current_window = order_shop.iloc[start_idx: window_tail_idx+1] #iloc excludes right edge
            concentration_ratio = len(current_window)/current_window['userid'].nunique()

            if concentration_ratio >= 3: 
                current_window_counts = Counter(current_window['userid'])
                current_window_counts
                max_value = max(current_window_counts.values())
                user_set.update(user for user, count in current_window_counts.items() if count ==  max_value)


    if user_set:  # if not empty [{}] for shops with no brushing:
        users = sorted(user_set)
        return '&'.join(map(str,users))
    else:
        return '0'


# In[ ]:


#result_enum_window = orders_sorted.groupby(['shopid']).apply(find_brush_enum_window)


# In[ ]:


#result_enum_window = result_enum_window.reset_index(name='userid')
#result_enum_window.to_csv('enum_window.csv',index=False)


# ### Refine bisect if there's case of  event_times[bisected_idx] ==  min_end_time (do not -1)

# In[ ]:


def find_brush_enum_window_bisect(order_shop):
    event_times = order_shop['event_time'].values
    array_length = len(event_times)
    counter_list = []


    for start_idx, start_time in enumerate(event_times[:-2]):
        counts_for_start_time = {}

        max_end_time = start_time + np.timedelta64(1, 'h')   # prepare to find elements within this right bound of time
        max_end_idx = bisect_left(event_times, max_end_time) - 1    # find largest idx within time bound, this will be largest possible window for current start_time

        if max_end_idx < start_idx + 2:
            continue # no need to continue if cannot form at least 3 rows


        if start_idx:  
            left_timeshift_possible = start_time - event_times[start_idx-1]  # can be handled by df.diff outside apply
            min_end_time = max_end_time - left_timeshift_possible  

            bisected_idx = bisect_left(event_times, min_end_time)
            
            # short-circuit prevents IndexError when event_times[bisected_idx] after or 
            if bisected_idx == array_length or event_times[bisected_idx] >  min_end_time:
                bisected_idx -= 1
                while bisected_idx > start_idx+2 and event_times[bisected_idx-1] == event_times[bisected_idx]: 
                    bisected_idx -= 1
            # smallest window begins at 3 rows minimum, or idx of largest time less than min_end_time 
            min_end_idx = max(start_idx + 2,bisected_idx) 

        else:
            # no row before start_idx == 0, so no restriction from bisected_idx 
            min_end_idx = start_idx + 2

        for window_tail_idx in range(min_end_idx, max_end_idx + 1):
            current_window = order_shop.iloc[start_idx: window_tail_idx+1] #iloc excludes right edge
            concentration_ratio = len(current_window)/current_window['userid'].nunique()

            if concentration_ratio >= 3: 
                counts_for_start_time.update(dict(current_window['userid'].value_counts()))
                
        # prevent appending empty counts_for_start_time (for clean debugging prints, doesn't affect Counter summation later)
        if counts_for_start_time:
            counter_list.append(Counter(counts_for_start_time))  # prepare Counter type for accumulation later

    if counter_list:  # if not empty [{}] for shops with no brushing:
        reduced_counter_list = reduce(operator.add,counter_list)
        max_value = max(reduced_counter_list.values())
        users = sorted(user for user,count in reduced_counter_list.items() if count == max_value)
        return '&'.join(map(str,users))
    else:
        return '0'


# In[ ]:


#result_enum_window_bisect = orders_sorted.groupby(['shopid']).apply(find_brush_enum_window_bisect)


# In[ ]:


#result_enum_window_bisect = result_enum_window_bisect.reset_index(name='userid')
#result_enum_window_bisect.to_csv('enum_window_bisect.csv',index=False)


# ### Not recounting same order in different window 

# In[ ]:


def find_brush_enum_window_dedup(order_shop):
    
    event_times = order_shop['event_time'].values
    array_length = len(event_times)

    order_user = {}

    for start_idx, start_time in enumerate(event_times[:-2]):

        max_end_time = start_time + np.timedelta64(1, 'h')   # prepare to find elements within this right bound of time
        max_end_idx = bisect_left(event_times, max_end_time) - 1    # find largest idx within time bound, this will be largest possible window for current start_time

        if max_end_idx < start_idx + 2:
            continue # no need to continue if cannot form at least 3 rows


        if start_idx:  
            left_timeshift_possible = start_time - event_times[start_idx-1]  # can be handled by df.diff outside apply
            min_end_time = max_end_time - left_timeshift_possible  

            bisected_idx = bisect_left(event_times, min_end_time)
            # short-circuit prevents IndexError when event_times[bisected_idx] after or 
            if bisected_idx == array_length or event_times[bisected_idx] >  min_end_time:
                bisected_idx -= 1
                while bisected_idx > start_idx+2 and event_times[bisected_idx-1] == event_times[bisected_idx]: 
                    bisected_idx -= 1
            # smallest window begins at 3 rows minimum, or idx of largest time less than min_end_time 
            min_end_idx = max(start_idx + 2,bisected_idx) 

        else:
            # no row before start_idx == 0, so no restriction from bisected_idx
            min_end_idx = start_idx + 2

        for window_tail_idx in range(min_end_idx, max_end_idx + 1):
            current_window = order_shop.iloc[start_idx: window_tail_idx+1] #iloc excludes right edge
            concentration_ratio = len(current_window)/current_window['userid'].nunique()

            if concentration_ratio >= 3: 
                order_user.update(dict(zip(current_window['orderid'],current_window['userid'])))            


    if order_user:
        user_counts = {userid:list(order_user.values()).count(userid) for userid in set(order_user.values())}
        max_value = max(user_counts.values())
        users = sorted(user for user,count in user_counts.items() if count == max_value)
        
        return '&'.join(map(str,users))
    else:
        return '0'


# In[ ]:


result_enum_window_dedup = orders_sorted.groupby(['shopid']).apply(find_brush_enum_window_dedup)


# In[ ]:


result_enum_window_dedup = result_enum_window_dedup.reset_index(name='userid')
result_enum_window_dedup.to_csv('enum_window_dedup.csv',index=False)


# ### Using np.diff to test speed up (not much effect) 

# In[ ]:


def find_brush_enum_window_dedup_diff(order_shop):
    
    event_times = order_shop['event_time'].values
    array_length = len(event_times)
    
    order_user = {}
    # insert to shift right 1 position for natural indexing using start_idx
    event_times_diff = np.insert(np.diff(event_times),values=0,obj=0)
    
    for start_idx, start_time in enumerate(event_times[:-2]):

        max_end_time = start_time + np.timedelta64(1, 'h')   # prepare to find elements within this right bound of time
        max_end_idx = bisect_left(event_times, max_end_time) - 1    # find largest idx within time bound, this will be largest possible window for current start_time

        if max_end_idx < start_idx + 2:
            continue # no need to continue if cannot form at least 3 rows


        if start_idx:
            min_end_time = max_end_time - event_times_diff[start_idx]  

            bisected_idx = bisect_left(event_times, min_end_time)
            # short-circuit prevents IndexError when event_times[bisected_idx] after or 
            if bisected_idx == array_length or event_times[bisected_idx] >  min_end_time:
                bisected_idx -= 1
                while bisected_idx > start_idx+2 and event_times[bisected_idx-1] == event_times[bisected_idx]: 
                    bisected_idx -= 1
            # smallest window begins at 3 rows minimum, or idx of largest time less than min_end_time 
            min_end_idx = max(start_idx + 2,bisected_idx)

        else:
            # no row before start_idx == 0, so no restriction from bisected_idx
            min_end_idx = start_idx + 2

        for window_tail_idx in range(min_end_idx, max_end_idx + 1):
            current_window = order_shop.iloc[start_idx: window_tail_idx+1] #iloc excludes right edge
            concentration_ratio = len(current_window)/current_window['userid'].nunique()

            if concentration_ratio >= 3: 
                order_user.update(dict(zip(current_window['orderid'],current_window['userid'])))            


    if order_user:
        user_counts = {userid:list(order_user.values()).count(userid) for userid in set(order_user.values())}
        max_value = max(user_counts.values())
        users = sorted(user for user,count in user_counts.items() if count == max_value)
        
        return '&'.join(map(str,users))
    else:
        return '0'


# In[ ]:


#result_enum_window_dedup_diff = orders_sorted.groupby(['shopid']).apply(find_brush_enum_window_dedup_diff)


# # Clarifying bisect
# **Why is `max_end_time` using bisect_left - 1**
# - Goal of max_end_time is to find the index of largest event_time < max_end_time.
# - When value to search does not match values in array, both bisect_left/bisect return one idx higher than the above requirement, so -1 for this case
# - When value to search match one or more(duplicated) values in array, bisect_left returns first idx of matching value, but we want a strictly smaller than max_end_time, so -1 for this case too  
# 
# **Why is `min_end_time` using bisect_left without -1 if min_end_time == event_times[bisected_idx] and with -1 when `bisected_idx == array_length` or `min_end_time < event_times[bisected_idx]`** 
# - Goal of min_end_time is to find the index of largest event_time <= min_end_time
#     - Note the difference from `max_end_time` is the <= here 
#     - Complication here is if such largest event_time is duplicated, we want the first idx among duplications to avoid missing out rows in the window with the least orders for the current start_time
# - When value to search does not match values in array, same explanation as above for `max_end_time`
#     - Additionally, because we want the smallest possible window for min_end_time, if there are duplicate times for min_end_time, keep moving to left until 1st value (implemented as while-loop)
# - When value to search match one or more(duplicated) values in array, bisect_left returns first idx of matching value, which is exactly what's needed --> Do not -1 

# In[ ]:


from bisect import bisect, bisect_left

time = [10,10,20,20,20,30,30]
# bisect finds index of array to insert new value to keep array sorted.
# bisect and bisect_left differences show up when the value to be inserted matches exactly one of the values in the array
# such a difference is magnified if that matched value is duplicated in the array

bisect(time,20)
bisect_left(time,20)

# No difference between bisect and bisect_left when value inserted does not clash
bisect(time,21)
bisect_left(time,21) 


# In[ ]:


bisect_left([1,1,2,2,3,3,3],3.1) 


# In[ ]:




