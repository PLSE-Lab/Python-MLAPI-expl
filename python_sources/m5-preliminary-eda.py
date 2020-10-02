#!/usr/bin/env python
# coding: utf-8

# This notebook will grow over time as I continue with my analysis. So far, I've only been looking at total sales between states and stores.

# In[ ]:


import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt


# ## Load data and get shapes

# In[ ]:


# get pandas dataframes
calendar = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/calendar.csv')
sell_prices = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sell_prices.csv')
sales_train = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sales_train_validation.csv')

# check shapes
print("calendar: {0}\nsell_prices: {1}\nsales_train: {2}".format(calendar.shape, sell_prices.shape, sales_train.shape))


# ## Reduce memory usage

# In[ ]:


def reduce_mem_usage(df, var):
    
    # get memory usage before conversion (in Mb)
    mem_before = df.memory_usage().sum() / 1024 ** 2
    
    # get integer and float columns
    int_columns = df.select_dtypes(include=["int"]).columns
    float_columns = df.select_dtypes(include=["float"]).columns

    # reduce integer columns
    for col in int_columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")

    # reduce float columns
    for col in float_columns:
        df[col] = pd.to_numeric(df[col], downcast="float")

    # get memory usage after conversion (in Mb)
    mem_after = df.memory_usage().sum() / 1024 ** 2
    # difference before --> after
    mem_diff = 100 * (mem_before - mem_after) / mem_before
    
    print("{} decreased from {:.2f} Mb to {:.2f} Mb --> ({:.1f}% reduction)".format(var, mem_before, mem_after, mem_diff))
    
    return df

calendar = reduce_mem_usage(calendar, 'Calendar')
sell_prices = reduce_mem_usage(sell_prices, 'Sell prices')
sales_train = reduce_mem_usage(sales_train, 'Sales train')


# ## Take a peek

# #### Calendar

# In[ ]:


calendar.head(5)


# #### Sell prices

# In[ ]:


sell_prices.head(5)


# #### Sales train

# In[ ]:


sales_train.head(5)


# ## Least sold and most sold items

# In[ ]:


# columns of sales data
d_cols = [c for c in sales_train.columns if 'd_' in c]
    
# item-wise sum of all sales
sum_sales = sales_train[d_cols].sum(axis=1)
min_sales_ID = sum_sales.idxmin()
max_sales_ID = sum_sales.idxmax()

# plotting function
def plot_item_sales(ID):
    item = sales_train.loc[sales_train['id']==sales_train.iloc[ID]['id']] # select item (row)
    item = item[d_cols] # get only the columns representing the sales time series
    item = pd.DataFrame([calendar['date'][:1913], item.iloc[0,:].reset_index(drop=True)], index=['date', 'sales']).T
    item = item.set_index('date')
    item.plot(figsize=(14,4), title=sales_train.iloc[ID]['id'], legend=None) # plot time series

# plotting
plot_item_sales(min_sales_ID)
plot_item_sales(max_sales_ID)
plt.show()


# **Takeaways:**
# 
# - Spikes in sales on certain dates
# - Periods of zero in between --> possibly items out of stock

# ## Total sales per state

# In[ ]:


# get indices for items sold in each state
CA = sales_train[sales_train['state_id']=='CA'].index
TX = sales_train[sales_train['state_id']=='TX'].index
WI = sales_train[sales_train['state_id']=='WI'].index

# get corresponding sales data
sales_CA = sales_train.iloc[CA][d_cols]
sales_TX = sales_train.iloc[TX][d_cols]
sales_WI = sales_train.iloc[WI][d_cols]

# compare total sales between states
total_CA = sales_CA.sum().sum()
total_TX = sales_TX.sum().sum()
total_WI = sales_WI.sum().sum()

plt.figure(figsize=(8,4))
sns.set_style('white', {'axes.spines.top': False, 'axes.spines.right': False})
sns.barplot([total_CA, total_TX, total_WI], ['CA', 'TX', 'WI'], palette=['gold', 'red', 'green']).set_title('Total sales')
plt.show()


# **Takeaways:**
# 
# - California has the most total sales, followed by Texas and Wisconsin
# - Texas and Wisconsin have roughly 2/3 of California's total sales

# In[ ]:


sales_CA = pd.DataFrame([calendar['date'][:1913], sales_CA.sum().reset_index(drop=True)], index=['date', 'sales']).T
sales_CA_date = sales_CA.set_index('date')
sales_CA_date.plot(figsize=(15,3), title='California', color='gold', legend=None)

sales_TX = pd.DataFrame([calendar['date'][:1913], sales_TX.sum().reset_index(drop=True)], index=['date', 'sales']).T
sales_TX_date = sales_TX.set_index('date')
sales_TX_date.plot(figsize=(15,3), title='Texas', color='red', legend=None)

sales_WI = pd.DataFrame([calendar['date'][:1913], sales_WI.sum().reset_index(drop=True)], index=['date', 'sales']).T
sales_WI_date = sales_WI.set_index('date')
sales_WI_date.plot(figsize=(15,3), title='Wisconsin', color='green', legend=None)
plt.show()


# Interestingly, there are these periodic dips in sales that occur at the same time across all three states. 
# 
# Let's take a closer look:

# In[ ]:


# select a dip
sales_CA_frac = sales_CA[300:400]

# get row representing the minimum (dip)
sales_CA_frac[sales_CA_frac['sales']==sales_CA_frac['sales'].min()]

# check in plot
sales_CA_frac.plot(figsize=(15,3), title='California', color='gold', legend=None)
plt.plot([330, 330], [0, 20000], color='k', alpha=0.5, label='2011-12-25')
plt.legend()
plt.show()


# This dip occurs on Christmas day 2011. Let's double check if this is true for the other years as well.

# In[ ]:


# get all Christmas dates
christmas_dates = [i for i in sales_CA['date'] if '12-25' in i]
# get their indices
inds = [sales_CA[sales_CA['date']==i].index[0] for i in christmas_dates]

# check dips over entire time series of sales
sales_CA_date.plot(figsize=(15,3), title='California', color='gold', label='sales')
for i, ind in enumerate(inds):
    plt.plot([ind, ind], [0, 25000], color='k', alpha=0.3, label = christmas_dates[i])
plt.legend(loc=3)
plt.show()


# **Takeaways:**
# 
# - It appears as though these dips are caused by the closure of stores every year on Christmas day
# - There are also dips a few weeks before Christmas day
# - Sales seem to be highest in the summer, then go down over Christmas, and then start going up again in the new year

# ## Total sales per category

# In[ ]:


# looking at categories
print('Unique categories: {}'.format(set(sales_train['cat_id'])))


# In[ ]:


# get HOBBIES items per state
hobbies_CA = sales_train.iloc[CA][sales_train.iloc[CA]['cat_id']=='HOBBIES'][d_cols].sum().sum()
hobbies_TX = sales_train.iloc[TX][sales_train.iloc[TX]['cat_id']=='HOBBIES'][d_cols].sum().sum()
hobbies_WI = sales_train.iloc[WI][sales_train.iloc[WI]['cat_id']=='HOBBIES'][d_cols].sum().sum()

# get FOODS items per state
foods_CA = sales_train.iloc[CA][sales_train.iloc[CA]['cat_id']=='FOODS'][d_cols].sum().sum()
foods_TX = sales_train.iloc[TX][sales_train.iloc[TX]['cat_id']=='FOODS'][d_cols].sum().sum()
foods_WI = sales_train.iloc[WI][sales_train.iloc[WI]['cat_id']=='FOODS'][d_cols].sum().sum()

# get HOUSEHOLD items per state
household_CA = sales_train.iloc[CA][sales_train.iloc[CA]['cat_id']=='HOUSEHOLD'][d_cols].sum().sum()
household_TX = sales_train.iloc[TX][sales_train.iloc[TX]['cat_id']=='HOUSEHOLD'][d_cols].sum().sum()
household_WI = sales_train.iloc[WI][sales_train.iloc[WI]['cat_id']=='HOUSEHOLD'][d_cols].sum().sum()

fig, axs = plt.subplots(1, 3, sharey=True, figsize=(20,5))
axs[0] = sns.barplot(['CA', 'TX', 'WI'], [hobbies_CA, hobbies_TX, hobbies_WI], palette=['gold', 'red', 'green'], orient='v', ax=axs[0]).set_title('HOBBIES sales')
axs[1] = sns.barplot(['CA', 'TX', 'WI'], [foods_CA, foods_TX, foods_WI], palette=['gold', 'red', 'green'], orient='v', ax=axs[1]).set_title('FOODS sales')
axs[2] = sns.barplot(['CA', 'TX', 'WI'], [household_CA, household_TX, household_WI], palette=['gold', 'red', 'green'], orient='v', ax=axs[2]).set_title('HOUSEHOLD sales')
plt.show()


# **Takeaways:**
# 
# - FOODS clearly sells the most items, followed by HOUSEHOLD and lastly HOBBIES

# ## Total sales per store

# In[ ]:


# get unique store IDs
stores = sorted(list(set(sales_train['store_id'])))
stores


# In[ ]:


# plot time series of total sales of each store
def sales_per_store():
    
    fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(16,20), sharey=True)
    
    for i, store in enumerate(stores):
        
        # compute total sales per store
        indices = sales_train[sales_train['store_id']==store].index
        sales_of_store = sales_train.iloc[indices][d_cols].sum()
        sales_of_store = pd.DataFrame([calendar['date'][:1913], sales_of_store.reset_index(drop=True)], index=['date', 'sales']).T
        sales_of_store = sales_of_store.set_index('date')
        
        # for subplots
        r = int(i/2) if i%2 == 0 else int(i/2-0.5)
        c = 0 if i%2 == 0 else 1
        
        # set color
        if 'CA' in store:
            color = 'gold'
        elif 'TX' in store:
            color = 'red'
        elif 'WI' in store:
            color = 'green'
    
        # plot
        sales_of_store.plot(ax=axes[r,c], color=color, title=store, legend=None, rot=20)
        
    plt.subplots_adjust(hspace=0.7)

sales_per_store()


# **Takeaways:**
# 
# - Apart from the dips at Christmas day, stores show additional dips that seem to be non-period and not across all stores
# - This could perhaps be state holidays
# - CA_3 shows a big increase around April 2015
# - Texas store show a spike up around March/April 2015
# - WI_1 and WI_2 show sudden increases in sales around May/June 2012

# In[ ]:


# bar plot of total sales per store
all_sales, color = [], []

for i, store in enumerate(stores):
    # compute total sales per store
    indices = sales_train[sales_train['store_id']==store].index
    sales_of_store = sales_train.iloc[indices][d_cols].sum().sum()
    all_sales.append(sales_of_store)
    
    # set color
    if 'CA' in store:
        color.append('gold')
    elif 'TX' in store:
        color.append('red')
    elif 'WI' in store:
        color.append('green')
    
fig, axs = plt.subplots(1, 2, sharey=True, figsize=(18,6))
axs[0].bar(stores, all_sales, color=color)
axs[0].set_title('Individual store sales')
axs[1].bar(['CA', 'TX', 'WI'], [np.mean(all_sales[:4]), np.mean(all_sales[4:7]), np.mean(all_sales[7:])], color=['gold', 'red', 'green'])
axs[1].set_title('Average store sales per state')
plt.show()

