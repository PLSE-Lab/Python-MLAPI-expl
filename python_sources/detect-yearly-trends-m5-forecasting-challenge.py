#!/usr/bin/env python
# coding: utf-8

# ### Upvote if you like my notebook, Your support and encouragement are greatly appreciated!!!
# 
# ### Suggestions and Criticisms are welcomed !!
# 
# ### Thank you!

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# ### Importing necessary modules and Reading the data

# In[ ]:


import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
sns.set(rc={'figure.figsize':(16,6)})
sns.set(style='whitegrid')
from itertools import cycle
color_cycle = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])


# In[ ]:


calender = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/calendar.csv')
sales_train_validation = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sales_train_validation.csv')
sell_prices = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sell_prices.csv')
sample_submission = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sample_submission.csv')


# In[ ]:


sales_train_validation.sample(5)


# In[ ]:


calender.sample(3)


# In[ ]:


print(sales_train_validation.shape)
print(calender.shape)


# ### Visualizing department level sales

# In[ ]:


dept_df = sales_train_validation.dept_id.value_counts().rename_axis('dept').reset_index(name='count')
sns.barplot(x='dept', y='count', data=dept_df, palette='gist_gray')
plt.title('Department level Sales')
plt.show()


# ### Visualizing state level Sales

# In[ ]:


state_df = sales_train_validation.state_id.value_counts().rename_axis('state').reset_index(name='count')
sns.barplot(x='state', y='count', data=state_df, palette='gist_gray')
plt.title('State level Sales')
plt.show()


# ### Visualizing Sales by Category

# In[ ]:


cat_df = sales_train_validation.cat_id.value_counts().rename_axis('cat').reset_index(name='count')
sns.barplot(x='cat', y='count', data=cat_df, palette='gist_gray')
plt.title('Sales by Category')
plt.show()


# ### Visualizing Sales by Category and State

# In[ ]:


sales_train_validation['sales'] = sales_train_validation.sum(axis=1)
sns.barplot(x='cat_id', y='sales', data=sales_train_validation, hue='state_id', color=next(color_cycle))
plt.title('Sales by Category and State')
plt.show()


# ### Visualizing Sales by State and Category

# In[ ]:


sns.barplot(x='state_id', y='sales', data=sales_train_validation, hue='cat_id', color=next(color_cycle))
plt.title('Sales by State and Category')
plt.show()


# ### Categorywise sales by different stores

# In[ ]:


sns.barplot(x='store_id', y='sales', data=sales_train_validation, hue='cat_id', color=next(color_cycle))
plt.title('Categorywise sales by different stores')
plt.show()


# ### Looking at Sales of Random Items

# In[ ]:


day_cols = [col for col in sales_train_validation.columns if 'd_' in col]
sales_train_validation.sort_values('sales', ascending=False)[day_cols].sample(5)


# In[ ]:


random_items = ['FOODS_3_090_CA_3_validation','FOODS_3_661_CA_1_validation','FOODS_3_377_TX_3_validation']
fig, axes = plt.subplots(3,1,figsize=(16,12), sharex=True)
axes = axes.flatten()
axx=0
for item in random_items:
    sales_train_validation.loc[sales_train_validation.id==item][day_cols].T.plot(color=next(color_cycle), ax=axes[axx], label='sales')
    axes[axx].set_title(item)
    axx+=1
plt.suptitle('Plotting random item sales')
plt.show()


# ### Time series of Sales across the States

# In[ ]:


# Yearly sales in all the stores
mean_sales_CA = sales_train_validation[sales_train_validation.state_id=='CA'][day_cols].mean(axis=0).reset_index().set_index(calender[0:1913]['date'])
mean_sales_CA = mean_sales_CA.drop('index', axis=1)
mean_sales_CA.index = pd.to_datetime(mean_sales_CA.index)
mean_sales_CA.columns = ['mean_sale_items']

mean_sales_TX = sales_train_validation[sales_train_validation.state_id=='TX'][day_cols].mean(axis=0).reset_index().set_index(calender[0:1913]['date'])
mean_sales_TX = mean_sales_TX.drop('index', axis=1)
mean_sales_TX.index = pd.to_datetime(mean_sales_TX.index)
mean_sales_TX.columns = ['mean_sale_items']

mean_sales_WI = sales_train_validation[sales_train_validation.state_id=='WI'][day_cols].mean(axis=0).reset_index().set_index(calender[0:1913]['date'])
mean_sales_WI = mean_sales_WI.drop('index', axis=1)
mean_sales_WI.index = pd.to_datetime(mean_sales_WI.index)
mean_sales_WI.columns = ['mean_sale_items']

# Plotting sale of items in all three cities
fig, (ax1,ax2,ax3) = plt.subplots(3,1, figsize=(18,16), sharex=True)

ax1.plot(mean_sales_CA, label='sales - CA', color=next(color_cycle))
ax1.legend(loc='upper left')

ax2.plot(mean_sales_TX, label='sales - TX', color=next(color_cycle))
ax2.legend(loc='upper left')

ax3.plot(mean_sales_WI, label='sales - WI', color=next(color_cycle))
ax3.legend(loc='upper left')
plt.title('Yearly sales of all goods across states')
plt.tight_layout()
plt.show() 


# ### Plotting the moving avrage sales across the states
# 

# In[ ]:


# Plotting the moving avrage
fig, ax = plt.subplots(figsize=(18,6))
ax.plot(mean_sales_CA.rolling(window=80).mean(), label='moving average sales CA', color=next(color_cycle))
ax.plot(mean_sales_TX.rolling(window=80).mean(), label='moving average sales TX', color=next(color_cycle))
ax.plot(mean_sales_WI.rolling(window=80).mean(), label='moving average sales WI', color=next(color_cycle))
plt.legend()
plt.title('Moving average sales of the ctates')
plt.show()


# ### Consolidated monthly mean of sales across States

# In[ ]:


# Monthly sales across the states
mean_sales_CA['month'] = pd.DatetimeIndex(mean_sales_CA.index).month_name()
mean_sales_CA['weekday_name'] = pd.DatetimeIndex(mean_sales_CA.index).weekday_name

mean_sales_TX['month'] = pd.DatetimeIndex(mean_sales_TX.index).month_name()
mean_sales_TX['weekday_name'] = pd.DatetimeIndex(mean_sales_TX.index).weekday_name

mean_sales_WI['month'] = pd.DatetimeIndex(mean_sales_WI.index).month_name()
mean_sales_WI['weekday_name'] = pd.DatetimeIndex(mean_sales_WI.index).weekday_name

new_order = ['January','February','March','April','May','June','July','August','September','October','November','December']

mean_sales_CA_grouped = mean_sales_CA.groupby(['month']).mean().reindex(new_order, axis=0)
mean_sales_TX_grouped = mean_sales_TX.groupby(['month']).mean().reindex(new_order, axis=0)
mean_sales_WI_grouped = mean_sales_WI.groupby(['month']).mean().reindex(new_order, axis=0)

fig, ax = plt.subplots(figsize=(18,6))

ax.plot(mean_sales_CA_grouped, label='mothly sales in CA', c='red', linewidth=4)
ax.plot(mean_sales_TX_grouped, label='mothly sales in TX', c='green', linewidth=4)
ax.plot(mean_sales_WI_grouped, label='mothly sales in WI', c='blue', linewidth=4)

ax.legend(loc='upper right')
plt.title('Consolidated monthly sales across States')
plt.tight_layout()
plt.show()


# ### Consolidated weekly sales acrosss states

# In[ ]:


new_order = ['Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday']
mean_sales_CA_grouped = mean_sales_CA.groupby(['weekday_name']).sum().reindex(new_order, axis=0)
mean_sales_TX_grouped = mean_sales_TX.groupby(['weekday_name']).sum().reindex(new_order, axis=0)
mean_sales_WI_grouped = mean_sales_WI.groupby(['weekday_name']).sum().reindex(new_order, axis=0)
fig, ax = plt.subplots(figsize=(18,6))
ax.plot(mean_sales_CA_grouped, label='weekly sales in CA')
ax.plot(mean_sales_TX_grouped, label='weekly sales in TX')
ax.plot(mean_sales_WI_grouped, label='weekly sales in WI')
plt.title('Consolidated weekly sales acrosss states')
plt.legend()
plt.show()


# ### Specific on Hobbey item sales in California

# In[ ]:


# Hobbey item sales in California
fig, axes = plt.subplots(3, 1, figsize=(16,12), sharex=True)
axes = axes.flatten()
axx = 0
for cat in sales_train_validation.cat_id.unique():
    sales = sales_train_validation[(sales_train_validation.cat_id==cat) & (sales_train_validation.state_id=='CA')][day_cols].T.    mean(axis=1).reset_index().set_index(calender[0:1913]['date']).drop('index', 1)
    sales.columns = ['sales_CA_'+str(cat)]
    sales.index = pd.to_datetime(sales.index)
    sales.plot(color=next(color_cycle), ax=axes[axx])
    axx += 1
plt.suptitle('Hobbey item sales in California')
plt.tight_layout()
plt.show()


# ### Looking at mean sales of all the stores

# In[ ]:


# Sales by store
sales_train_validation = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sales_train_validation.csv')
sales_by_store = sales_train_validation.groupby(['store_id']).mean().T
sales_by_store = sales_by_store.set_index(calender[0:1913]['date'])
sales_by_store.index = pd.to_datetime(sales_by_store.index)

fig,ax = plt.subplots()
for col in sales_by_store.columns:
    sales_by_store[col].plot(color=next(color_cycle), label='avg sales '+str(col), ax=ax)
plt.title('Mean sales across the shops')
plt.show()


# ### Moving average sales of all stores: better visualization, window=100

# In[ ]:


# moving average sales in all the 10 shops
import matplotlib.dates as mdates

sales_rolling = sales_by_store.rolling(100).mean()
fig,ax = plt.subplots(figsize=(18,7))
for col in sales_by_store.columns:
    sales_rolling[col].plot( label='avg sales '+str(col), ax=ax, color=next(color_cycle), linewidth=1)
#ax.xaxis.set_major_locator(mdates.YearLocator())
#ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.title('moving average sales across the shops')
plt.legend(fontsize=12)
plt.show()


# ### Visualizing monthly trends across the stores
# 

# In[ ]:


sales_by_store['month'] = pd.DatetimeIndex(sales_by_store.index).month_name()
sales_by_store['weekdays'] = pd.DatetimeIndex(sales_by_store.index).weekday_name

# Monthly sales across the stores
new_order = ['January','February','March','April','May','June','July','August','September','October','November','December']
monthly_store_sales = sales_by_store.groupby(['month']).mean().reindex(new_order, axis=0)
fig, axes = plt.subplots(5,2,figsize=(18,12), sharex=True)
axes = axes.flatten()
axx = 0
for col in monthly_store_sales.columns[0:11]:
    monthly_store_sales[col].plot(color=next(color_cycle), ax=axes[axx])
    axes[axx].set_title(str(col))
    axx+=1
plt.suptitle('Monthly trend in different stores')
plt.tight_layout()
plt.show()


# ### Weekly trends across the stores

# In[ ]:


# Weekly sales across the stores
new_order = ['Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday']
weekly_store_sales = sales_by_store.groupby(['weekdays']).mean().reindex(new_order, axis=0)
fig, axes = plt.subplots(5,2,figsize=(18,12), sharex=True)
axes = axes.flatten()
axx = 0
for col in weekly_store_sales.columns[0:11]:
    weekly_store_sales[col].plot(color=next(color_cycle), ax=axes[axx])
    axes[axx].set_title(str(col))
    axx+=1
plt.suptitle('Weekly trend in different stores')
plt.tight_layout()
plt.show()


# ### Yearly trends across the stores

# In[ ]:


#new_order = ['January','February','March','April','May','June','July','August','September','October','November','December']
#monthly_store_sales = sales_by_store.groupby(['month']).mean().reindex(new_order, axis=0)
fig, axes = plt.subplots(5,2,figsize=(18,12), sharex=True)
axes = axes.flatten()
axx = 0
for col in monthly_store_sales.columns[0:11]:
    sales_by_store[col].plot(color=next(color_cycle), ax=axes[axx])
    axes[axx].set_title(str(col))
    axx+=1
plt.suptitle('Yearly trend in different stores')
plt.tight_layout()
plt.show()


# ### Monthly and weekly trends by Categories

# In[ ]:


# Trend in sales of items
for cat in sales_train_validation.cat_id.unique():
    df = sales_train_validation[sales_train_validation.cat_id==cat][day_cols].    T.mean(axis=1).reset_index().set_index(calender[0:1913]['date']).drop('index', 1)
    df.columns = ['mean_sales']
    df.index = pd.to_datetime(df.index)
    df['month'] = pd.DatetimeIndex(df.index).month_name()
    df['weekday_name'] = pd.DatetimeIndex(df.index).weekday_name


    fig, (ax1,ax2) = plt.subplots(2,1,figsize=(16,8))
    sns.boxplot(x='month', y='mean_sales', data=df, ax=ax1)
    ax1.set_title('monthly trend')
    sns.boxplot(x='weekday_name', y='mean_sales', data=df, ax=ax2)
    ax2.set_title('weekly trend')
    plt.suptitle('Trend across ' + str(cat))
    # plt.tight_layout()
    plt.show()


# ### Detecting trends for all the stores

# In[ ]:


# Detect trend
import matplotlib.dates as mdates
def detect_trend(X_df):
    coefficients, residuals, _, _, _ = np.polyfit(range(len(X_df)), X_df, 1, full=True)
    mse = residuals[0]/len(X_df)
    nrmse = np.sqrt(mse)/(X_df.max()-X_df.min())
    
    print('slope = ', str(float(coefficients[0])))
    print('nrmse = ', str(float(nrmse)))
    
    fig, ax = plt.subplots(figsize=(9,5))
    new_df = pd.DataFrame([coefficients[0]*x+coefficients[1] for x in range(len(X_df))], columns=['trend'])    
    new_df = new_df.reset_index().set_index(calender[0:1913]['date']).drop('index', 1)
    X_df.plot(color=next(color_cycle), ax=ax, label='Original')
    new_df.plot(ax=ax, color='red', linewidth=4)
    ax.legend()
    plt.show()
    
for col in sales_by_store.columns:
    X_df = sales_by_store[col].reset_index().set_index(calender[0:1913]['date']).drop('date', 1)
    detect_trend(X_df)


# ### Efforts from a humble beginner!
# 
# ### To be continued!
# 
# ### Cast an upvote if it was useful!

# In[ ]:




