#!/usr/bin/env python
# coding: utf-8

# # Introduction

# Hello Viewers, 
# 
# Thanks for viewing my notebook. We have a dataset on Avocado prices and volume sold by region and date. I thought this would be a great dataset for any beginner to start playing around with visualizations. I'm sharing few basic concepts of visualizations using this data here. 
# 
# I've mainly focused on creating simple graphs and customizing them to pose a great look. I've used Seaborn predominantly throughout this kernel. However Seaborn has many sophisticated visuals, I have not used everything here as it is not necessary. Let's have a look at what we have for today.

# # Importing Libraries

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import scipy as sp
import warnings
warnings.filterwarnings('ignore')
plt.style.use('ggplot')

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


df = pd.read_csv('../input/avocado-prices/avocado.csv')
original = df.copy()


# In[ ]:


df.shape


# In[ ]:


df.info()


# In[ ]:


df.head()


# I do not need all the columns, hence I'm creating a subset of the dataset. You can also drop the unnecessary columns otherwise.

# In[ ]:


df = df[['Date', 'AveragePrice', 'Total Volume', '4046', '4225', '4770', 'Total Bags', 'type', 'region']]


# In[ ]:


df.head()


# In[ ]:


df.describe()


# We've a date feature which is stored as an object. Let's convert it to a datetime object as this would help us manipulating the dataset better.

# In[ ]:


df['Date'] = pd.to_datetime(df['Date'])


# In[ ]:


df.columns


# As mentioned in the description of the dataset, the codes correspond to various types of Avocados. Let's rename the column names as this looks confusing.

# In[ ]:


new_cols = {'4046' : 'Small Haas', '4225' : 'Large Haas', '4770' : 'XLarge Haas'}
df.rename(columns = new_cols, inplace = True)


# As these features represent the total number of units sold, let's convert it to integer type to get rid of the decimal places.

# In[ ]:


for i in ['Total Volume', 'Small Haas', 'Large Haas','XLarge Haas', 'Total Bags']:
    df[i] = df[i].astype('int64')


# In[ ]:


df.info()


# Okay, I'm done with the type conversions. Let's also create two more features - month and year which can be obtained from the date feature. You need not try splicing this feature, but the datetime type allows us to create date related features by inbuilt functions.

# In[ ]:


df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month


# In[ ]:


df.head()


# As we are done with manipulating the data, let's create a summary table which shows us the following,
# * Total Volume
# * Average Price
# * Total Bags

# Pivot tables are the greatest means of aggregating the dataset, it also allows us to customize it. If you notice, different aggregation types can be applied to different features. You can also try a color mapping to get a better view of it! Sounds cool!

# In[ ]:


summary = pd.pivot_table(index = 'Year', values = ['Total Volume', 'AveragePrice', 'Total Bags'], data = df, 
               aggfunc = {'Total Volume' : sum, 'AveragePrice' : np.mean, 'Total Bags' : sum}).style.background_gradient(cmap = 'Set2')
summary


# ## Total Avocados Sold by Year

# Throughout this kernel, I'll be grouping the data then visualize it. Even though there are sophisticated visuals which could do it in a click, let's use this opportunity to get us equipped with knowledge on grouping data.
# 
# * Group the data based on a feature of interest (Year)
# * Apply the desired aggregation type (Sum of Total Volume)
# * Visualize
# 
# Here, we are trying to visualize the sum of avocados sold by year. So let's group the data by year, then sum up the total volume. We'll use the barplot to visualize then.
# 
# Got it, Right?

# In[ ]:


plt.rcParams['figure.figsize'] = (10, 7)

grouped = df.groupby('Year')['Total Volume'].sum().reset_index()
ax = sns.barplot(x = 'Year', y = 'Total Volume', linewidth = 1, edgecolor = 'k', data = grouped, palette = 'plasma')
for index, row in grouped.iterrows():
    ax.text(row.name, row['Total Volume'], str(round(row['Total Volume'] / 10000000, 2)) + 'Cr.', color = 'k', ha = 'center', va = 'bottom')
plt.title('Total Volume of Avocados Sold by Year', fontsize = 16)
plt.show()


# As seen in the graph, the sales of avocados increase with time except 2018. This sounds to be a sharp decline in sales, but it is possibly due to the fact that we do not have the data for the entire year of 2018.

# ## Total Avocados Sold by Month

# In[ ]:


grouped_month = df.groupby('Month')['Total Volume'].sum().reset_index()

ax = sns.barplot(y = 'Month', x = 'Total Volume', data = grouped_month, palette = 'plasma', linewidth = 1, edgecolor = 'k', orient = 'h')
for index, row in grouped_month.iterrows():
    ax.text(row['Total Volume'], row.name, str(round(row['Total Volume'] / 10000000, 2)) + 'Cr.', color = 'k', va = 'bottom')
plt.title('Total Volume of Avocados Sold by Month', fontsize = 16)
plt.show()


# The aggregation of total sales by months clearly shows that the start of the year have seen higher sales compared to the rest of the year.

# ## Average Price of Avocados by Month

# To visualize the change in price over time, lineplot is the best option. We can group the data as we did before and then use the lineplot functionality to generate a lineplot for the same.
# 
# We'll also use the axhline to plot a horizontal line of minimum and maximum prices as reference lines.

# In[ ]:


grouped_price = df.groupby('Month')['AveragePrice'].mean().reset_index()

ax = sns.lineplot(x = 'Month', y = 'AveragePrice', data = grouped_price, palette = 'plasma', marker = 'v')
plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
for index, row in grouped_price.iterrows():
    ax.text(row.name, row['AveragePrice'], str(round(row['AveragePrice'], 2)) + '$')
plt.axhline((grouped_price['AveragePrice']).min(), color = 'k', label = 'Min. Avg. Price')
plt.axhline((grouped_price['AveragePrice']).max(), color = 'g', label = 'Max. Avg. Price')
plt.title('Avg Price of Avocados by Month', fontsize = 16)
plt.legend(loc = 'best')
plt.show()


# The price of avocados seem to come down by the end of year.

# ## Avocado Sales by Region

# I'm interested in four insights when it comes to region. We'll see the trends on sales made in different regions based on the avocado type.
# * Total Sales by Region
# * Sales of Small Haas by Region
# * Sales of Large Haas by Region
# * Sales of Extra Large Haas by Region

# In[ ]:


grouped_shaas = df.groupby('region')['Small Haas'].sum().sort_values().reset_index()
grouped_lhaas = df.groupby('region')['Large Haas'].sum().sort_values().reset_index()
grouped_xlhaas = df.groupby('region')['XLarge Haas'].sum().sort_values().reset_index()
grouped_reg_total = df.groupby('region')['Total Volume'].sum().sort_values().reset_index()


# In[ ]:


plt.rcParams['figure.figsize'] = (19, 6)

sns.barplot(x = 'region', y = 'Total Volume', data = grouped_reg_total, palette = 'plasma', linewidth = 1, edgecolor = 'k')
plt.xticks(rotation = 90)
plt.title('Total Avocados Sold by Region', fontsize = 16)
plt.show()


# In[ ]:


sns.barplot(x = 'region', y = 'Small Haas', data = grouped_shaas, palette = 'plasma', linewidth = 1, edgecolor = 'k')
plt.xticks(rotation = 90)
plt.title('Small Haas Avocados Sold by Region', fontsize = 16)
plt.show()


# In[ ]:


sns.barplot(x = 'region', y = 'Large Haas', data = grouped_lhaas, palette = 'plasma', linewidth = 1, edgecolor = 'k')
plt.xticks(rotation = 90)
plt.title('Large Haas Avocados Sold by Region', fontsize = 16)
plt.show()


# In[ ]:


sns.barplot(x = 'region', y = 'XLarge Haas', data = grouped_xlhaas, palette = 'plasma', linewidth = 1, edgecolor = 'k')
plt.xticks(rotation = 90)
plt.title('Extra Large Haas Avocados Sold by Region', fontsize = 16)
plt.show()


# ## Avocado Sales by Type

# In[ ]:


plt.rcParams['figure.figsize'] = (10, 7)

grouped_type_total = df.groupby('type')['Total Volume'].sum().reset_index()
ax = sns.barplot(x = 'type', y = 'Total Volume', data = grouped_type_total, palette = 'plasma', linewidth = 1, edgecolor = 'k')
for index, row in grouped_type_total.iterrows():
    ax.text(row.name, row['Total Volume'], str(round(row['Total Volume'] / 10000000, 2)) + 'Cr.', color = 'k', va = 'bottom')
plt.title('Total Volume of Avocados Sold by Type', fontsize = 16)
plt.show()


# Thanks for viewing! Please leave an upvote and comment, if you like this kernel. 
