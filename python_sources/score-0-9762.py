#!/usr/bin/env python
# coding: utf-8

# <h1> Order Brushing Detection. Score 0.9762 </h1>

# In[ ]:


#Import libraries
import pandas as pd
import numpy as np
import datetime

#And data
data = pd.read_csv('/kaggle/input/order-brushing-shopee-code-league/order_brush_order.csv', parse_dates=[3])

#Sort the data by 'event_time', so the orders are in sequence
data.sort_values('event_time', ascending = True, inplace=True)
data.reset_index(drop=True, inplace=True)


# The result evaluation is by Concentrate rate = Number of Orders within 1 hour / Number of Unique Buyers within 1 hour, therefore create a string composed of 'shopid' and 'userid'  to identify multiple instances of such combination within time period (in this case 1 hour)

# In[ ]:


data['shopid'] = data['shopid'].astype(str)
data['userid'] = data['userid'].astype(str)
data['string_shop_user'] = data[['shopid', 'userid']].apply(lambda x: ''.join(x), axis=1)


# To reduce calculation leadtime - delete the customers, who order from a certain shop less than 3 times overall.
# This operation reduces total number of orders from 222750 to 135977

# In[ ]:


customers_more_two_orders = pd.DataFrame(data.groupby('string_shop_user').orderid.count() > 2)
customers_more_two_orders = list(customers_more_two_orders[customers_more_two_orders.orderid == True].index)
mask = data['string_shop_user'].isin(customers_more_two_orders)
data_filtered = data[mask]
data_filtered.reset_index(drop=True, inplace=True)


# In order to identify shop / user combination with >=3 instances within an hour, iterate through orders (data_filtered) in 1 hour time deltas. Save all the identifies 'shopid' / 'userid' in dataframe (summary)

# In[ ]:


#Create blank dataframe for final results
columns = ['orderid', 'shopid', 'userid', 'event_time', 'string_shop_user']
summary = pd.DataFrame(columns = columns)

#Iterate through orders by 1 hour time frame
for i in range(data_filtered.shape[0]):
    
    #set 1 hour timeframe
    limit = datetime.timedelta(0,3600) #1 hour time frame
    start = data_filtered.loc[i, 'event_time'] #start time of time frame
    end = data_filtered.loc[i, 'event_time'] + limit #end time of time frame
    
    #filter data of one hour
    data_hour = data_filtered[(data_filtered['event_time'] >= start) & (data_filtered['event_time'] <= end)]
    
    #select users who ordered more than twice
    data_hour_users = pd.DataFrame(data_hour.groupby('string_shop_user').orderid.count() > 2)
    result_data_hour = data_hour['string_shop_user'].isin(list(data_hour_users[data_hour_users.orderid == True].index))
    result_hour = data_hour[result_data_hour]
    
    #Store results in summary dataframe
    summary = summary.append(result_hour)


# The timeframes in iteration overlapped, therefore there are duplicates in the summary dataframe. Remove the duplicates and unnecessary columns.

# In[ ]:


#remove duplicates
summary.drop('orderid', axis=1, inplace=True)
summary.drop('event_time', axis=1, inplace=True)
summary.drop('string_shop_user', axis=1, inplace=True)
summary = summary.drop_duplicates(keep='first')


# Summary dataframe contains the result: combinations of 'shopid' and 'userid' but not in the submission format. Therefore it is necessary: <br>
#     - to sort 'userid' in ascending format
#     - combine multiple 'userid' for same 'shopid' in 111&222&333 format

# In[ ]:


#Clean and put the data in submission format
summary['shopid'] = summary['shopid'].astype(int)
summary['userid'] = summary['userid'].astype(int)
summary_sorted = summary.sort_values(by=['shopid', 'userid'], ascending=True)
summary_sorted.reset_index(drop=True, inplace=True)

#Consolidate userid
for i in range(summary_sorted.shape[0]):
    if summary_sorted.loc[i, 'shopid'] == summary_sorted.loc[i+1, 'shopid']:
        summary_sorted.loc[i, 'userid'] = str(summary_sorted.loc[i, 'userid']) + '&' + str(summary_sorted.loc[i+1, 'userid'])
        summary_sorted.drop(i+1, inplace=True)
        summary_sorted.reset_index(drop=True, inplace=True)        


# The summary_sorted contains only the 'shopid' who employed order brushing. Add remaining shops with the 'userid' = 0

# In[ ]:


#Identify missing shops
all_shops = list(set(data['shopid'])) #all shops in source data
identified_shops = list(summary_sorted['shopid']) #shops identified as order brushing

#Iterate through all shops to remove identified shops
for shop in all_shops:
    if shop in identified_shops:
        all_shops.remove(shop)
        
#Add the missing shops in the dataframe
missing_shops = pd.DataFrame(total_shops) #Create dataframe with missing shops and 'userid' = 0
missing_shops['userid'] = np.zeros(len(total_shops))
missing_shops.columns = ['shopid','userid']
missing_shops['userid'] = missing_shops['userid'].astype(int)

#Append to the summary_sorted fianl result dataframe
final = summary_sorted.append(missing_shops)


# This is final result of the analysis
