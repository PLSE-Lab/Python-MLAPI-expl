#!/usr/bin/env python
# coding: utf-8

# Hi, 
# I am Ajay !
# In this kernel, we will perform EDA on Store Sales.

# In[ ]:


########### Library Imports #######
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import seaborn as sns ## Data plotting
print(os.listdir("../input"))
train_df = train_df = pd.read_csv("../input/train.csv")
train_df.head()

# Any results you write to the current directory are saved as output.


# Converting date column to datetime object and looking at:
# * Start and end date
# * total number of stores
# * total number of items

# In[ ]:


train_df['date'] = pd.to_datetime(train_df['date'],format='%Y-%m-%d')
start_date = train_df['date'].min()
end_date = train_df['date'].max()
number_of_stores = len(train_df['store'].unique().tolist())
number_of_items = len(train_df['item'].unique().tolist())
print("start_date",start_date,"\n end_date",end_date,"\n number_of_stores",number_of_stores,"\n number_of_items",number_of_items)


# Training data is available from Jan 2013 to December 2017
# 
# Let's look at distribution total sales per year per store: 

# In[ ]:


train_df['year'] = train_df['date'].dt.year
total_sales_per_store_per_year = train_df.groupby(['store','year'])['sales'].sum().reset_index()

sns.catplot(x= 'store',y='sales',data=total_sales_per_store_per_year,hue='year',aspect= 1.5,kind='bar')


# From Plot we can observe following things:
# *  Store 2 & Store 8 perform best while Store 7 and Store 6 have low sales
# *  Total Sales increase over the years and rate of increase in sales is diminishing over the years
# 
# Now, we will take highest selling store and lowest selling store see if there is any seasonal variation over the year
# 

# In[ ]:


mask = (((train_df['store'] == 1) | (train_df['store'] == 7)) & (train_df['year'] == 2017))
train_df  = train_df[mask]
train_df['date1'] = train_df['date'].dt.strftime("%b")
train_df_weekly = train_df.groupby(['store','date1'],as_index=False)['sales'].sum()
sort_order = train_df['date1'].unique().tolist()
g = sns.catplot(x= 'date1',y='sales',data=train_df_weekly,hue='store',aspect= 3,kind='bar',height = 5,order=sort_order)
g.set_xticklabels(rotation=70)


# We can observe following things regarding seasonal variation:
# * Both highest and lowest selling store have similar sales trend across the months.
# * Sales are least during middle of the year and peaks around July then starts reducing
# 
