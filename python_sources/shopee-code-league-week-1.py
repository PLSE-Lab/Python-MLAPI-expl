#!/usr/bin/env python
# coding: utf-8

# ## Shopee Code League Week 1 
# By: Gabriel Lewis and Jackie Chu

# <h3>Task Summary:</h3> <br>
# We first identified and dropped unhelpful columns like orderid. Next, we grouped the dataframe by shops, since the first part of the task was to detect shop that have conducted order brushing. For each shop, we iterated through each timestamp of each order, and created 'sub-frames' of orders that fit within an hour of the order in question. We subjected that window to the concentration rate test, and if threshold is met, we record the userid whose presence had the highest proportion in that window. We repeat for all timestamps in the shop, recording all highly represented userids of each window. After some formatting we end up with lists of userids that are suspected brushers for shops with suspicious 1h windows.

# In[ ]:


#packages
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
pd.set_option('display.max_rows', None)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


shopdf = pd.read_csv("/kaggle/input/orderbrushing/order_brush_order.csv")
shopdf.head()


# In[ ]:


#dropping order id because all of them are unique, making it unusable as a predictor
print(shopdf['orderid'].nunique() == len(shopdf))
shopdf = shopdf.drop('orderid', axis=1)


# In[ ]:


#converting time string to Timestamp object 
shopdf['event_time'] = pd.to_datetime(shopdf['event_time'])


# In[ ]:


def sus_checkv3(x):
    #per shop
    shop_user = []
    abs_count = []
    sus_id = 0
    
    #sort time
    x = x.sort_values('event_time')
    
    #for each possible brush order hour
    for timestamp in x['event_time']:
        
        #per timeframe
        result_user = []
        
        #create a dataframe of orders occuring within 1 hour from the timestamp
        subframe = x[(x['event_time'] >= timestamp) & (x['event_time'] <= (timestamp + timedelta(hours=1)))]
        
        #find number of unique users within that timeframe
        unique_user = subframe['userid'].nunique()

        if unique_user > 0:
            conc_rate = len(subframe)/unique_user
        else:
            conc_rate = 0
        
        #if that window exceeds concentration threshold
        if conc_rate >= 3:
            #ranking the highest proportion users
            user_prop = subframe['userid'].value_counts()
            #putting user_prop into a dataframe
            user_frame = pd.DataFrame({'id': user_prop.index, 'count': user_prop.values})
            nummax = user_prop.max()
            
            #selecting the userid(s) that have highest proportion
            result_user = list(user_frame[user_frame['count'] == nummax]['id'])
            shop_user.append(result_user)
            abs_count.append(nummax)
        
    if len(abs_count) > 0:
        
        total_max = max(abs_count)

        count_frame = pd.DataFrame({'id': shop_user, 'count':abs_count})
        
        #this series will contain both lists containing userids that are tied for highest proportion 
        #as well as single values of the same
        sus_id = count_frame[count_frame['count'] == total_max]['id']
        
        sus_id_result = []
        
        #getting the list values out
        for i in sus_id:
            if type(i) is list:
                for j in i:
                    sus_id_result.append(j)
            else:
                sus_id_result.append(i)

        return sus_id_result
    
    #handling for non-brusher shops
    else:
        return 0


# In[ ]:


#application of above function
susdf = shopdf.groupby('shopid').apply(sus_checkv3)


# In[ ]:


#formatting to follow submission rules
userid_col = []
for i in susdf.values:
    if i != 0:
        intermed_list = []
        #sort list, convert all values into string and append unique values only
        for i in [str(j) for j in sorted(i)]:
            if i not in intermed_list:
                intermed_list.append(i)
        
        #join with &
        userid_col.append("&".join(intermed_list))
    else:
        userid_col.append('0')


# In[ ]:


final_df = pd.DataFrame({'shopid': susdf.index, 'userid': userid_col})
final_df.to_csv('submission_df.csv', index=False)


# We obtained a final score of <strong>0.89933</strong>. <br>
# 
# <h3>Possible improvements</h3> <br>
# From the kaggle evaluations formula for this challenge, it can be calculated that there are 315 suspicious shopids. <br>

# In[ ]:


from IPython.display import Image
Image("/kaggle/input/eval-formulapng/eval_formula.png")


# In[ ]:


#number of unique shops
len_counter = []
for i in userid_col:
    if i != '0':
        len_counter.append(i)
print("Total number of unique suspicious shops found: " + str(len(len_counter)))


# Our first improvement task is to discover the 39 suspicious shopids that we missed in our analysis
