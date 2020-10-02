#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Import libraries.
import pandas as pd
import numpy as np
from datetime import datetime


# In[ ]:


# Import input CSV file and store it as a pandas dataframe in df.
df = pd.read_csv('/kaggle/input/students-order-brushing-1/order_brush_order.csv')


# In[ ]:


# Create a new column `unix_time` by converting `event_time` to the corresponding Unix time.
time_zero = pd.to_datetime('1970-01-01')
df["unix_time"] = pd.to_datetime(df["event_time"]).apply(lambda x: int((x - time_zero).total_seconds()))

# From here onwards, we actually only need `unit_time`, `shopid`, `userid` columns. But I'm too lazy to remove the other columns.

# Sort the rows of `df` ascendingly by `unit_time`. 
df = df.sort_values('unix_time')


# In[ ]:


# get_suspicious_buyers will first get the suspicious buyers for each shopid.
# Then, it formats the buyerid(s) and store this string as a value in shopid_to_userids dictionary, with the shopid as the key.

def get_suspicious_buyers(df):
    # df is a dataframe that contains all the orders for a unique shopid, sorted ascendingly by `unix_time`.
    assert df['shopid'].nunique() == 1
    
    # Reset the index of df so that we can reference each order by the dataframe index later.
    df = df.reset_index(drop=True)
    # Stores the no. of orders of df.
    N = len(df)
    # Create new column `suspicious_buyer` to record the orders that occurred in a brushing period.
    df['suspicious_buyer'] = False
    # Stores the current shopid.
    shopid = df['shopid'].iloc[0]
    
    # st_idx denotes the index of the first order in the 1hr time interval.
    # en_idx denotes the index of the last order in the 1hr time interval.
    for st_idx in range(N):
        for en_idx in range(st_idx, N):
            
            # If time difference between the current last order and the first order is more than 3600s (1hr),
            # we do not need to consider the next order as a possible last order because the orders are sorted
            # ascendingly by `unix_time`.
            if df.loc[en_idx, 'unix_time'] - df.loc[st_idx, 'unix_time'] > 3600:
                break
                
            # The first condition checks for all the possible cases whereby the 1hr time interval could occur at. 
            # For st_idx == 0, we are considering the cases whereby the start of the 1hr time interval occurs at or before the first order in the dataset. 
            # For en_idx == 0, we are considering the cases whereby the end of the 1hr time interval occurs at or after the last order in the dataset.
            # For df.loc[en_idx+1, 'unix_time'] - df.loc[st_idx-1, 'unix_time'] > 3600,
            # we are considering the cases whereby the start and end of the 1hr time interval occurs between two orders in the dataset, 
            # more specifically, the start of the 1hr interval occurs within interval (df.loc[st_idx-1, 'unix_time'], df.loc[st_idx, 'unix_time']]
            # while the end of the 1hr interval occurs within interval [df.loc[en_idx, 'unix_time'], df.loc[en_idx+1, 'unix_time']).
    
            # The second condition checks that the 1hr time interval has concentration >= 3.
            if (st_idx == 0 or en_idx == N - 1 or df.loc[en_idx+1, 'unix_time'] - df.loc[st_idx-1, 'unix_time'] > 3600) and                 (en_idx-st_idx+1 >= 3 * df.loc[st_idx:en_idx, 'userid'].nunique()):
                df.loc[st_idx:en_idx, 'suspicious_buyer'] = True
    
    # Get all orders from suspicious buyers.
    df = df[df['suspicious_buyer']]                               
    
    # Dictionary that maps userid to the total no. of orders for that userid during brushing periods.
    # We do not need to calculate the order proportion for each userid since that is proportionate to the total no. of orders.
    occur_dict = df['userid'].value_counts().to_dict()
    
    # Stores formatted output.
    output = ''
    # Stores maximum no. of orders amoung all userids.
    max_count = 0
    # Stores the userids which have the maximum no. of orders.
    userid_list = []
    
    # Creates the formatted output and stores it in shopid_to_userids dictionary.
    for k, v in sorted(occur_dict.items()):
        if v > max_count:
            max_count = v
            userid_list = [k]
        elif v == max_count:
            userid_list.append(k) 
    for userid in userid_list:
        if output == '':
            output = str(userid)
        else:
            output += ('&' + str(userid))
            
    shopid_to_userids[shopid] = output


# In[ ]:


# Use groupby on `shopid` and call get_suspicious_buyers for each shopid.

# Running this cell takes about 1hr to complete.
shopid_to_userids = {}
df.groupby('shopid', as_index=False).apply(get_suspicious_buyers)


# In[ ]:


# Create df containing output.
output_df = pd.DataFrame({'shopid': df['shopid'].unique()})
output_df['userid'] = '0'

# Transfer values from shopid_to_userids to output dataframe.
for i, row in output_df.iterrows():
    shopid = row['shopid']
    if shopid_to_userids[shopid] != '':
        
        output_df.loc[i, 'userid'] = shopid_to_userids[shopid]
        
# Store output in CSV file.    
output_df.to_csv('submission.csv', index=False)


# ## My personal learning point from this Order Brushing competition is that neither the start nor end of the 1hr time intervals need to occur exactly at the time when the orders occured, which were suggested by the explanations of the examples given. Perhaps it might be best to focus on the Description rather than the Examples next time.
# 
# ## I hope that the code + comments were helpful. Leave a comment if you need more clarification(s) and we can try to help each other out.
# 
# ## Feel free to create your own public notebooks if you think that my method or explanation is not good enough :P
# 
# ## Even though Shopee has decided that they *will not* provide any solutions or pseudocodes. I believe that, as partipants, we can take the initiative to help each other to learn from these competitions. 
# 
# ## Nonetheless, kudos to Shopee for having the audacity to hold such an immense regional competition. This is probably one of the largest data science/ coding competition ever held, if not the largest, in terms of the number of participants (I think).
# 
# 
# ## This notebook would not be possible without the help from some fellow participants. I will give a shout-out to each of them in the shout-outs section below.
# 
# 
# ### Big big shout-outs to:
# 
# - [Wen Yuen](https://www.kaggle.com/pwypeanut) for generously sharing and discussing his team's [solution](https://ideone.com/RfoSod) which managed to get the perfect score during the competition. Congratz to his team!
# 
# - [mtherfuk](https://www.kaggle.com/phuccoi96) for taking the initiative to create a [Telegram group](https://www.kaggle.com/c/students-order-brushing-1/discussion/158093#884125) for us to discuss about the solutions.
# 
# - [Nguyen Diep Xuan Quang](https://www.kaggle.com/xuanquang1999) for generously sharing his team's [solution](https://www.kaggle.com/xuanquang1999/3-hooman-1-cat-order-brushing).
# 
# - [Tong Hui Kang](https://www.kaggle.com/huikang), one of my my teammates, for [discovering the total no. of suspicious shops during the competition](https://www.kaggle.com/c/students-order-brushing-1/discussion/158120). Do check out [his notebook](https://www.kaggle.com/huikang/week-1-baseline-methods-and-visualisations).
# 
# 
