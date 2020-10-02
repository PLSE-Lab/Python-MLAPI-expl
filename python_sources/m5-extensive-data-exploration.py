#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#data manipulation packages
import pandas as pd
import numpy as np
from scipy import stats
import re

#visualization tools
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import cycle

import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 400)


# **calendar.csv** - Contains information about the dates on which the products are sold.
#  
# **sales_train_validation.csv** - Contains the historical daily unit sales data per product and store [d_1 - d_1913]
# 
# **sample_submission.csv** - The correct format for submissions. Reference the Evaluation tab for more info.
# 
# **sell_prices.csv** - Contains information about the price of the products sold per store and date.

# In[ ]:


path = '/kaggle/input/m5-forecasting-accuracy'
cal = pd.read_csv(f'{path}/calendar.csv', parse_dates = ['date'])
sales = pd.read_csv(f'{path}/sales_train_validation.csv')
prices = pd.read_csv(f'{path}/sell_prices.csv')

print(cal.shape)
print(sales.shape)
print(prices.shape)


# In[ ]:


cal.head()


# In[ ]:


sales.head()


# In[ ]:


prices.head()


# Since both Sales and Cal dataframes both share 'd_' values, let's merge them together so we can create an informed time series plot with actual date values.

# In[ ]:


colors = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
d_cols = [col for col in sales.columns if 'd_' in col]

def plot_item(item_id):
    '''Plot the selling history of "item" by day, month, and year.
    
    Args:
        item(str): id of the item we are wanting to plot.
    
    Returns:
        matplotlib plot object of item sale history
    '''
    global item
    item = 'FOODS_3_090_CA_3'
    
    item_df = sales.loc[sales['id'] == item_id][d_cols].T
    item_df = item_df.rename(columns={sales.index[sales['id']==item_id].to_list()[0]:item_id}) # Name it correctly
    item_df = item_df.reset_index().rename(columns={'index': 'd'}) # make the index "d"
    item_df = item_df.merge(cal, how='left', validate='1:1')

    fig, axes = plt.subplots(1, 1, figsize=(15, 6), dpi=100)
    item_df[['date', item_id]].set_index('date').resample('D').mean()[item_id].plot(ax=axes, label='By day', alpha=0.8).set_ylabel('Amount of Item Sold', fontsize=14);
    item_df[['date', item_id]].set_index('date').resample('M').mean()[item_id].plot(ax=axes, label='By month', alpha=1).set_ylabel('Amount of Item Sold', fontsize=14);
    item_df[['date', item_id]].set_index('date').resample('Y').mean()[item_id].plot(ax=axes, label='By year', alpha=1).set_ylabel('Amount of Item Sold', fontsize=14);
    axes.set_title('Mean '+str(item_id)+ ' sold by hour, day and month', fontsize=16);
    axes.legend()


# In[ ]:


plot_item('HOBBIES_1_004_CA_1_validation')


# In[ ]:


plot_item('FOODS_1_068_TX_1_validation')


# We can see from some of our plots that some of the items weren't sold throughout the entire date range. We will need to fix this so our data isn't skewed.

# In[ ]:


'''As observed above, some of the items, for one reason or another, have limited history throughout the time series.
The following script will find the items that have limited history, the amount of consecutive zeros within the time
series, and where exactly they occur in the series (i.e. 'beginning', 'middle', 'end').
'''

import itertools

limited_items = {}

for item_id in sales['id']:
    df = sales.loc[sales['id'] == item_id][d_cols].T
    df = df.rename(columns={sales.index[sales['id']==item_id].to_list()[0]:item_id}) # Name it correctly
    df = df.reset_index().rename(columns={'index': 'd'}) # make the index "d"
    df = df.merge(cal, how='left', validate='1:1')
    
    rolled = np.asarray(df.iloc[:,1].astype(int))
    
    zero_consec, zero_count = [0], 0
    
    condition = np.where(rolled==0,'true','false')
    zero_groups = [ sum( 1 for _ in group ) for key, group in itertools.groupby( condition ) if key ]
    zero_gap = (zero_groups.index(max(zero_groups))/len(zero_groups))*100

    if zero_gap<35:
        zero_location = 'beginning'
    elif zero_gap <= 35 or zero_gap<=75:
        zero_location = 'middle'
    else:
        zero_location = 'end'

    for val in rolled:
        condition = val == 0
        if val==0:
            zero_count+=1
        else:
            if zero_count>zero_consec[0]:
                zero_consec[0] = zero_count
            else:
                zero_count = 0
    if zero_consec[0]>150:
        limited_items[item_id] = zero_consec[0], zero_location
    else:
        continue


# In[ ]:


print(dict(itertools.islice(limited_items.items(), 3)))


# In[ ]:


print('The percentage of items that have more than 150 consecutive days without the item being sold is: {}'.format((len(limited_items)/sales.shape[0])*100))


# Almost 70% of all of the items have more than 150 days without being sold -- given the date range. From the graphs we've obeserved above, at least part of this could be explained by the fact that the item wan't being sold throughout the entire date range. Perhaps there are other reasons as well. Let's pump out a few more graphs so that we can better understand these gaps.

# In[ ]:


'''Plotting how much of the items with limited history occur in the beginning, middle, and end'''

from collections import Counter 

location = []
for key, value in limited_items.items():
    location.append(value[1])
    
totals = Counter(location)

plt.bar(totals.keys(), totals.values())


# Based on the graph, an overwhelming amount of items having missing values in the beginning of the time series, though some zero gaps occur in the middle in the end.

# In[ ]:


selling_history = sales[d_cols]

plt.figure(figsize = (14,6))
plt.plot((np.count_nonzero(selling_history.values, axis=0)/selling_history.shape[0]), color = 'darkcyan')
plt.title('Proportion of Items Sold vs. No Items Sold')
plt.xlabel('Day')
plt.ylabel('Percentage of Nonzero')


# We can see that, as the time series progress, the percentage of items with non-zero days rises. This shows that there are a lot of items that aren't introduced from the beginning of the time series.

# In[ ]:


plot_item('HOBBIES_1_288_CA_1_validation')


# Well, in this case, this particular item has been around from beginning till end, but did observe a complete hault in the middle. These unexpected fluctuations will surely skew our models. We will have to fix these later.

# Now that we have a function that plots the selling history of an item, let's look at them from a categorical perspective; Let's sum these items by category to better understand which category of items have high volume of sales.

# In[ ]:


def agg_plot(*args):
    '''Plot the selling history of items aggregated by categories.
    
    Args:
        category(str): item(s) type as described by the id column (i.e. HOBBIES, HOUSEHOLD, FOODS).
    
    Returns:
        matplotlib plot object of category sale history
    '''
    temp = sales.copy()
    temp['item_cat'] = temp['id'].str.split('_',1).str.get(0).str.lower()
    
    if len(args)==1:
        cat_df = temp[temp['item_cat']==args[0]][d_cols]
        cat_totals = cat_df.sum(axis=0).T

        cat_totals.plot(figsize=(14,10), title = str(args[0])+'_aggregated', color = next(colors))
        plt.legend('')
        plt.show()

    else:
        aggs = []
        for i in args:
            cat_df = temp[temp['item_cat']==i][d_cols]
            aggs.append(np.asarray(cat_df.sum(axis=0).T))
        plt.figure(figsize=(14,10))
        for x in aggs:
            vis = plt.plot(x)
            vis = plt.xlabel('Day Sold')
            vis = plt.ylabel('Amount of Items Sold')
        plt.legend([i for i in args])
        plt.show()


# In[ ]:


agg_plot('foods','hobbies','household')


# We can see that Food items are sold consistantly more often than the other categories. On the other hand, even though items in hobbies are sold less frequently, the series is much less noisy than the other categories. The categories seem to have very similar seasonal/cyclical fluctuations; no lead/lag relationship.
# 
# Let's reduce the number of categories to see if we can index on any trends. Reducing the categories will reduce the y axis range, so trends should be more pronounced.

# In[ ]:


agg_plot('household', 'hobbies')


# We can see that 'household' has a moderate upwards trend, whereas 'hobbies' consolidates in the 3,000/low 4,000 items per day range.

# Given the hierarchical nature of the data, lets graph the selling history of these items by store id.

# In[ ]:


##Thanks to https://www.kaggle.com/robikscube/m5-forecasting-starter-data-exploration for providing this.

past_sales = sales.set_index('id')[d_cols]     .T     .merge(cal.set_index('d')['date'],
           left_index=True,
           right_index=True,
            validate='1:1') \
    .set_index('date')

store_list = prices['store_id'].unique()
for s in store_list:
    store_items = [c for c in past_sales.columns if s in c]
    past_sales[store_items]         .sum(axis=1)         .rolling(50).mean()         .plot(figsize=(15, 5),
              alpha=0.8,
              title='90 Day Moving Average Total Sales (10 stores)')
plt.legend(store_list)
plt.show()


# On average, the stores located in CA seem to sell more than the stores in the other states. Let's create a similar graph that index the selling history by state, not just by store.

# In[ ]:


'''Plotting the 150/50 Day Moving Average of Items Sold by State'''

state_list = sales['state_id'].unique()

fig, (ax1,ax2,ax3) = plt.subplots(3, 1, figsize=(14, 9))
axes = (ax1,ax2,ax3)
fig.tight_layout(pad=3.0)

for s in range(len(state_list)):
    store_items = [c for c in past_sales.columns if state_list[s] in c]
    past_sales[store_items]         .sum(axis=1)         .rolling(150).mean()         .plot(kind='line',
          title='150/50 Day Moving Average of Items Sold by {} Stores'.format(state_list[s]),
          lw=2,
          color='firebrick',
          ax= axes[s])
    past_sales[store_items]         .sum(axis=1)         .rolling(50).mean()         .plot(kind='line',
          title='150/50 Day Moving Average of Items Sold by {} Stores'.format(state_list[s]),
          lw=2,
          color= 'darksalmon',
          ax=axes[s]
        )
    axes[s].legend(['150 Day', '50 Day'], loc='upper left')
 


# Though all states are slightly trending upwards, it is clear that CA maintains a higher selling range than TX and WI.

# Now that we've observed the selling history by state, Lets observe the prices of the items - grouped by category - in a violin plot.

# In[ ]:


temp = prices.copy()
temp['item_cat'] = temp['item_id'].str.split('_',1).str.get(0).str.lower()
temp['state'] = temp['store_id'].str.split('_').str.get(0)

fig, ax = plt.subplots(figsize=(12,8))
sns.set(style ='darkgrid')
sns.violinplot(x=temp['sell_price'],y=temp['item_cat'], hue=temp['state'], ax = ax, scale='width', cut=0, palette='muted')
ax.set_xlabel('Sale Price')
ax.set_ylabel('Item Category')
ax.set_title('Item Sale Price by Item Category for Each State')


# What we can gather from this is that food items cost the least and, in general, household items and hobby items cost the same on average -- withstanding a few household items that blow up the price range. Additionally, based on the graph, the stores sell the same items regardless of state; the only exception to this is that it seems that WI and TX sell a few more expensive items that CA doesn't sell.

# It's important to take into account the sales during holidays. Lets gather this information.

# In[ ]:


#credit: https://www.kaggle.com/williamhuybui/holiday-s-visualization

#List of all events
event_list=[i for i in cal.event_name_1.fillna(0).unique() if i != 0] 

#Extract all the days an event has in the span of 1916 days
day_event_list=[cal[cal.event_name_1==i].d.tolist() for i in event_list]

#Create the Event_df dataframe which we will use throughout the notebook
event_df=pd.DataFrame({'Event Name' : event_list, 'Event day':day_event_list})
restricted_day= set(['d_'+ str(i) for i in np.arange(1916,1970)])
quantity=[]

for i in day_event_list:
    # Making sure that we exclude all the days thats are not in the training set
    clean_i=list(set(i)-restricted_day)
    temp=sales[clean_i].sum().sum() #Adding columns and then rows
    quantity.append(temp)

event_df['Quantity']=quantity
event_df


# In[ ]:


#Top 2 and bottom 2 in terms of total sales
a=event_df.sort_values('Quantity',ascending=False)[['Event Name','Quantity']].head(2)
b=event_df.sort_values('Quantity',ascending=False)[['Event Name','Quantity']].tail(3)
a.append(b)


# We've just grabbed the top and bottom values in terms of items sold. It looks like, wth exception to Christmas, even the worst performing holidays still do a good job in Sales. Christmas is the only day where we see a huge deviation from holiday sales.

# In[ ]:


average_quantity=sales.iloc[:,6:].sum().sum()/1913
name=['SuperBowl','Purim End','NewYear','Thanksgiving', 'All days average']
values=event_df[event_df['Event Name'].isin(name)].Quantity.tolist()
values.append(average_quantity)


plt.figure(figsize=(8,4))
plt.xlabel("Event")
plt.ylabel("Quantity")
plt.bar(name,values)
plt.title("Sale quantity in some holidays compare to the average sale")
plt.show()


# As shown in the graph, even though New Year and Thanksgiving rank among the worst selling holidays, they still perform better than the average day. Once again, the only exception we observe is Christmas.

# In[ ]:




