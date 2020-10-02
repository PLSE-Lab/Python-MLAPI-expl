#!/usr/bin/env python
# coding: utf-8

# # E-commerce data
#  Analysing dummy data for an ecommerce company, I will use Python/Pandas to analyse the data with the goal of optimising sales revenue.
# The aim of this analysis
# is to find the optimal conditions for an online ad campaign that will maximise sales revenue. The variables to optimise are:
#     - Month of advertisement
#     - Time of day of advertisement
#     - Area to target with the Ads
#     - Which products to advertise individually
#     - Which products to advertise as a bundle

# #### Import necessary libraries

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# #### Merge the 12 months of sales data into a single CSV file

# In[ ]:


import os


# In[ ]:


files = [file for file in os.listdir('../input/monthly-sales-2019')]

for file in files:
    print(file)


# In[ ]:


df = pd.DataFrame()

for file in files:
    filedf =pd.read_csv('../input/monthly-sales-2019/'+file)
    df = pd.concat([df,filedf])
    
df.head()


# In[ ]:


df.to_csv("all_months_data.csv",index=False)


# In[ ]:


all_data = pd.read_csv('all_months_data.csv')


# In[ ]:


all_data.head()


# # Data Cleaning

# ### Drop NaN rows

# In[ ]:


len(all_data)


# In[ ]:


all_data.isnull().sum()


# Since the number of rows containing empty values accumulate less than 1% of the total rows, we can just remove them without affecting the overall analysis.

# In[ ]:


all_data = all_data.dropna(how='any')
all_data.head()


# ### Altering the columns

# ####  Remove the strings in the Order Date column

# In[ ]:


all_data = all_data[all_data['Order Date'].str[0:2] != 'Or']
all_data.head()


# ####  Add a month column

# In[ ]:


all_data['Month'] = all_data['Order Date'].str[0:2]
all_data['Month'] = all_data['Month'].astype('int32')


# #### Convert columns to the correct type

# In[ ]:


all_data['Quantity Ordered'] = pd.to_numeric(all_data['Quantity Ordered'])
all_data['Price Each'] = pd.to_numeric(all_data['Price Each'])
all_data['Order Date'] = pd.to_datetime(all_data['Order Date'])


# In[ ]:


all_data.head()


# #### Add an hour column

# In[ ]:


all_data['Hour'] = all_data['Order Date'].dt.hour


# #### Add a city column

# In[ ]:


def get_city(address):
    return address.split(',')[1]
def get_state(address):
    return address.split(',')[2].split(' ')[1]

all_data['City'] = all_data['Purchase Address'].apply(lambda x:get_city(x)+' ('+get_state(x)+')')
all_data.head()


# #### Add a sales column

# In[ ]:


all_data['Sales'] = all_data['Quantity Ordered']*all_data['Price Each']
all_data.head()


# Now that the data is cleaned to a suitable format we can begin to perform exploratory data analysis. The aim of this analysis
# is to find the optimal conditions for an online ad campaign that will maximise sales revenue. The variables to optimise are:
#     - Month of advertisement
#     - Time of day of advertisement
#     - Area to target with the Ads
#     - Which products to advertise individually
#     - Which products to advertise as a bundle

# # Data Exploration and visualization

# ### When to advertise to maximise sales

# #### Most popular month for ordering.

# Below we can see that the best month for sales was December. This is an intuitive result because we would expect a larger sales volume in December due to the christmas holidays. 

# In[ ]:


months = range(1,13)

bymonth = all_data.groupby('Month').sum()

plt.bar(months,bymonth['Sales'])
plt.xticks(months)
plt.ylabel('Sales in $')
plt.xlabel('Month number')
plt.show()


# #### Most popular time of the day for ordering.

# In order to maximise sales, we can see that the most orders are placed at 12pm and 19pm. This is intuitive since most people will be on their lunch breaks at 12pm and also they will be home from daily tasks at 19pm. In order to maximise the effectiveness of ads, we should aim to display the ads between the hours 11am-1pm and 6pm to 8pm.

# In[ ]:


all_data['Hour'] = all_data['Order Date'].dt.hour
all_data['Minute'] = all_data['Order Date'].dt.hour


# In[ ]:


all_data.head()


# In[ ]:


hours = [hour for hour, df in all_data.groupby('Hour')]
plt.plot(hours, all_data.groupby(['Hour']).count())
plt.xticks(hours)
plt.xlabel('time (hour)')
plt.ylabel('Number of orders')
plt.grid()
plt.show()


# ### Where to advertise to maximise sales

# #### Most popular city for ordering.

# We can see from the data below that Austin (TX) had the highest amount of sales during the year.

# In[ ]:


citysales =all_data[['City','Sales']]


# In[ ]:


results = all_data.groupby('City').sum()


# In[ ]:


cities = [city for city, df in all_data.groupby('City')]

plt.bar(cities, results['Sales'],width =0.7)
plt.xticks(cities,rotation='vertical')
plt.xlabel('Cities')
plt.ylabel('Total Sales $')
plt.show()


# ### Which products to advertise to maximise sales.

# #### Most popular individual product.

# Looking at the most popular product on an individual basis, we can see that the AAA batteries sold in the highest quantities. This makes sense because they are one of the cheapest items.

# In[ ]:


product_group = all_data.groupby('Product')

quantity_ordered = product_group.sum()['Quantity Ordered']

products = [p for p, df in product_group]

prices = all_data.groupby('Product').mean()['Price Each']


# In[ ]:


fig, ax1 = plt.subplots()

ax2 = ax1.twinx()
ax1.bar(products, quantity_ordered)
ax2.plot(products, prices, 'b-',color='red',markersize=10, linewidth=4, linestyle='dashed')

ax1.set_xlabel('Product name')
ax1.set_ylabel('Quantity ordered',color='blue')
ax2.set_ylabel('Price ($)',color='red')
ax1.set_xticklabels(products, rotation = 'vertical',size=8)

plt.show()


# #### Most popular bundle of products.

# When considering the bundling of products ordered, we can see below that the most popular bundles are the 'iPhone,Lightning Charging Cable' and the 'Google Phone,USB-C Charging Cable' bundles. 

# In[ ]:


df = all_data[all_data['Order ID'].duplicated(keep=False)]
df.head(5)


# In[ ]:


df['Grouped'] = df.groupby('Order ID')['Product'].transform(lambda x:','.join(x))


# In[ ]:


df['Grouped'].value_counts()


# # Conclusions:
#  - If the company is looking to optimise its sales by focusing on the best performing factors:
#     - The company should aim to target the majority of its ads to the consumer during peak hours (11am-1pm & 6pm-8pm).
#     - The company should ensure that a higher volume of ads are targeted at customers between peak hours in the day and certain months of the year.
#     - The company should ensure that they are weighting their ads according to the locations (cities) that produce the most orders.
#     - The company should focus the majority of the content of their Ads to the best selling items and bundles, namely phones and chargers.
# 
#  - If the company is looking to optimise its sales by improving the performance of other factors:
#     - The time of day that an ad should be aimed at the consumer remains unchanged, because the time at which a consumer decides to shop is out of the companies control. Nevertheless, the company could attempt to flatten the peak of the graph by targeting ads uniformly from 11am-7pm.
#     - The company should try to test different ad campaigns at points throughout the year that did not perform as well as christmas and April, in order to increase sales consistently throughout the year.
#     - The company should try to increase Ad volume in cities that are not performing as well as Austin, in order to diversify their sales revenue geographically and decrease the risk related to their income.

# In[ ]:




