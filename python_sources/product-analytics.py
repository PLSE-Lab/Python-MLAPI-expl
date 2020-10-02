#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


import matplotlib.pyplot as plt
import pandas as pd


# In[ ]:


import os

print(os.listdir('../input'))


# # 1. Load Data
# 
# The data we are going to load is a transnational data set which contains all the transactions occurring between 01/12/2010 and 09/12/2011 for a UK-based and registered non-store online retail.
# <br>The company mainly sells unique all-occasion gifts. 
# <br>Many customers of the company are wholesalers.

# In[ ]:


df = pd.read_csv("../input/Online Retail.csv", sep = ',',encoding = "ISO-8859-1", header= 0)
df.head()


# In[ ]:


df.shape


# In[ ]:


df.dtypes


# In[ ]:


# parse date
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], format = "%d-%m-%Y %H:%M")
df.head()


# # 2. Product Analytics

# ### - Quantity Distribution

# In[ ]:


# Visualize the distributuion of Quantity using a box plot

ax = df['Quantity'].plot.box(
    showfliers=False, # without outliers
    grid=True,
    figsize=(10, 7)
)

ax.set_ylabel('Order Quantity')
ax.set_title('Quantity Distribution')

plt.suptitle("")
plt.show()


# As we can see, some orders have negative quantities because the cancelled or refunded orders are recorded with negative values in the Qunatity column of our dataset.

# In[ ]:


pd.DataFrame(df['Quantity'].describe())


# In[ ]:


# Filter out all the cancelled orders

df.loc[df['Quantity'] > 0].shape


# In[ ]:


df = df.loc[df['Quantity'] > 0]


# In[ ]:


df.shape


# ## Time Series Analyses
# 
# Understanding the overall time series trends in the revenue and the numbers of orders or purchases, this helps us realize whether the business is growing or shrinking in terms of both the overall revenue and the numbers of orders we receive over time.

# In[ ]:


# Look into the numbers of orders received over time

monthly_orders_df = df.set_index('InvoiceDate')['InvoiceNo'].resample('M').nunique()


# Resample function resamples and converts time series data into the monthly time series data, by using 'M'.

# In[ ]:


monthly_orders_df


# In[ ]:


# Visualize this monthly data with a line chart

ax = pd.DataFrame(monthly_orders_df.values).plot(
    grid=True,
    figsize=(10, 7),
    legend=False
)

ax.set_xlabel('date')
ax.set_ylabel('number of orders/invoices')
ax.set_title('Total Number of Orders Over Time')

plt.xticks(
    range(len(monthly_orders_df.index)), 
    [x.strftime('%m.%Y') for x in monthly_orders_df.index], 
    rotation=45
)

plt.show()


# We can notice that there is a sudden radical drop in the number of orders in December 2011.

# In[ ]:


# Look at the data in December 2011

invoice_dates = df.loc[
    df['InvoiceDate'] >= '2011-12-01',
    'InvoiceDate'
]


# In[ ]:


print('Min date: %s\nMax date: %s' % (invoice_dates.min(), invoice_dates.max()))


# The drop we noticed before is due to the fact that we only have the data from December 1, to December 9, 2011. 
# So, it would be a misrepresentation to consider the month of December, 2011, for doing our time series analysis.

# In[ ]:


# Remove data for December, 2011

df = df.loc[df['InvoiceDate'] < '2011-12-01']


# In[ ]:


df.shape


# In[ ]:


monthly_orders_df = df.set_index('InvoiceDate')['InvoiceNo'].resample('M').nunique()


# In[ ]:


monthly_orders_df


# In[ ]:


ax = pd.DataFrame(monthly_orders_df.values).plot(
    grid=True,
    figsize=(10,7),
    legend=False
)

ax.set_xlabel('date')
ax.set_ylabel('number of orders')
ax.set_title('Total Number of Orders Over Time')

ax.set_ylim([0, max(monthly_orders_df.values)+500])

plt.xticks(
    range(len(monthly_orders_df.index)), 
    [x.strftime('%m.%Y') for x in monthly_orders_df.index], 
    rotation=45
)

plt.show()


# __The monthly number of orders seems to float around 1,500 from December 2010 to August 2011.
# <br>It increases significantly from September 2011, and almost doubles by November 2011.__
# <br>This spike in sales could be provoked by a growth in business or by seasonal effects.
# <br>To know this, we should compare the current year's data against the previuos year's data.

# ### - Time-series Revenue

# In[ ]:


# Built the monthly revenue data column

df['Sales'] = df['Quantity'] * df['UnitPrice']


# In[ ]:


# Get the monthly revenue data

monthly_revenue_df = df.set_index('InvoiceDate')['Sales'].resample('M').sum()


# In[ ]:


monthly_revenue_df


# In[ ]:


# Visualize this data with a line plot

ax = pd.DataFrame(monthly_revenue_df.values).plot(
    grid=True,
    figsize=(10,7),
    legend=False
)

ax.set_xlabel('date')
ax.set_ylabel('sales')
ax.set_title('Total Revenue Over Time')

ax.set_ylim([0, max(monthly_revenue_df.values)+100000])

plt.xticks(
    range(len(monthly_revenue_df.index)), 
    [x.strftime('%m.%Y') for x in monthly_revenue_df.index], 
    rotation=45
)

plt.show()


# We see a similar pattern to the previous Total Number of Orders Over Time chart.
# <br>Here, __the monthly revenue floats around 700,000 from December 2010 to August 2011.
# <br> It increases significantly from September 2011.__
# <br>To know the reason of this spike, we need to look further back in the sales history.

# ### - Time-series Repeat Customers
# 
# A typical strong and stable business has a steady stream of sales from existing customers.
# <br>We are going to see how much of the sales are from repeat and existing customers of the current online retail business.

# In[ ]:


df.head()


# In[ ]:


# Aggregate data for each order using InvoiceNo and InvoiceDate

invoice_customer_df = df.groupby(
    by=['InvoiceNo', 'InvoiceDate'],
).agg({
    'Sales': sum,
    'CustomerID': max,
    'Country': max,
}).reset_index()


# In[ ]:


invoice_customer_df.head()


# In[ ]:


# Aggregate this data per month and 
# compute the number of customers who made more than one purchase in a given month

monthly_repeat_customers_df = invoice_customer_df.set_index('InvoiceDate').groupby([
    pd.Grouper(freq='M'), 'CustomerID'# group the index InvoiceDate by each month and by CustomerID
]).filter(lambda x: len(x) > 1). resample('M').nunique()['CustomerID']# only those customers with more than one order


# In[ ]:


monthly_repeat_customers_df


# In[ ]:


monthly_unique_customers_df = df.set_index('InvoiceDate')['CustomerID'].resample('M').nunique()


# In[ ]:


monthly_unique_customers_df


# In[ ]:


# Calculate the percentages of repeat customers for each month

monthly_repeat_percentage = monthly_repeat_customers_df / monthly_unique_customers_df * 100.0
monthly_repeat_percentage


# Roughly about 20 to 30% of the customers are repeat customers.

# In[ ]:


# Visualize all this data in a chart

ax = pd.DataFrame(monthly_repeat_customers_df.values).plot(
    figsize=(10,7)
)

pd.DataFrame(monthly_unique_customers_df.values).plot(
    ax=ax,
    grid=True
)


ax2 = pd.DataFrame(monthly_repeat_percentage.values).plot.bar(
    ax=ax,
    grid=True,
    secondary_y=True, # add another y-axis on the rightside of the chart
    color='green',
    alpha=0.2
)

ax.set_xlabel('date')
ax.set_ylabel('number of customers')
ax.set_title('Number of All vs. Repeat Customers Over Time')

ax2.set_ylabel('percentage (%)')

ax.legend(['Repeat Customers', 'All Customers'])
ax2.legend(['Percentage of Repeat'], loc='upper right')

ax.set_ylim([0, monthly_unique_customers_df.values.max()+100])
ax2.set_ylim([0, 100])

plt.xticks(
    range(len(monthly_repeat_customers_df.index)), 
    [x.strftime('%m.%Y') for x in monthly_repeat_customers_df.index], 
    rotation=45
)

plt.show()


# __The numbers of both reapeat and all customers start to rise significantly from September 2011.
# <br>The percentage of <u>repeat  customers</u> seems to stay pretty consistent at about 20% to 30%.__
# <br>This online retail business will benefit from this steady stream of repeat customers, as they will help the business to generate a stable stream of sales.

# ### - Revenue from Repeat Customers
# 
# Now we analyze how much of the monthly revenue comes from these __<u>repeat customers</u>__.

# In[ ]:


monthly_rev_repeat_customers_df = invoice_customer_df.set_index('InvoiceDate').groupby([
    pd.Grouper(freq='M'), 'CustomerID'# group the index InvoiceDate by each month and by CustomerID
]).filter(lambda x: len(x) > 1). resample('M').sum()['Sales']# sum to add all the sales from repeat customers for a given month


# In[ ]:


monthly_rev_repeat_customers_df


# In[ ]:


# Calculate the percentages of the monthly revenue generated by the repeat customers

monthly_rev_perc_repeat_customers_df = monthly_rev_repeat_customers_df/monthly_revenue_df * 100.0


# In[ ]:


monthly_rev_perc_repeat_customers_df


# In[ ]:


# Visualize this monthly revenue

ax = pd.DataFrame(monthly_revenue_df.values).plot(figsize=(12,9))

pd.DataFrame(monthly_rev_repeat_customers_df.values).plot(
    ax=ax,
    grid=True,
)

ax.set_xlabel('date')
ax.set_ylabel('sales')
ax.set_title('Total Revenue vs. Revenue from Repeat Customers')

ax.legend(['Total Revenue', 'Repeat Customer Revenue'])

ax.set_ylim([0, max(monthly_revenue_df.values)+100000])

ax2 = ax.twinx()

pd.DataFrame(monthly_rev_perc_repeat_customers_df.values).plot(
    ax=ax2,
    kind='bar',
    color='g',
    alpha=0.2
)

ax2.set_ylim([0, max(monthly_rev_perc_repeat_customers_df.values)+30])
ax2.set_ylabel('percentage (%)')
ax2.legend(['Repeat Revenue Percentage'])

ax2.set_xticklabels([
    x.strftime('%m.%Y') for x in monthly_rev_perc_repeat_customers_df.index
])

plt.show()


# We see a similar pattern as before, where there is a significant increase in the revenue from September 2011.
# <br>Whereas we have learnt that roughly 20-30% of the customers who made purchases are <u>repeat customer</u>, 40-50% of the total revenue is from <u>repeat customers</u>:
# <br>__this means that roughly half of the revenue was driven by the 20-30% of the customer base who repeats purchases.__
# <br>__<u>This demonstrates how important it is to retain customers!</u>__

# ### - Popular Items Over Time
# 
# We are going to analyze how the customers interact with products that are sold, specifically with the top five best-sellers.

# In[ ]:


# Calculate the number  of items sold for each product for each period

date_item_df = pd.DataFrame(
    df.set_index('InvoiceDate').groupby([
        pd.Grouper(freq='M'), 'StockCode'
    ])['Quantity'].sum()
)

date_item_df


# In[ ]:


# Rank items by the last month sales
# Specifically, see what items were sold the most on November 30, 2011

last_month_sorted_df = date_item_df.loc['2011-11-30'].sort_values(
    by = 'Quantity', ascending=False
).reset_index()

last_month_sorted_df.head(5)


# In the list above, we got the top five best-sellers of November 2011 with the codes: 23084, 84826, 22197, 22086, and 85099B.

# In[ ]:


# Aggregate the monthly sales data for these 5 products

date_item_df = pd.DataFrame(
    df.loc[
        df['StockCode'].isin(['23084', '84826', '22197', '22086', '85099B'])
    ].set_index('InvoiceDate').groupby([
        pd.Grouper(freq='M'), 'StockCode'
    ])['Quantity'].sum()
)

date_item_df


# In[ ]:


# Transform this data into a tabular format

trending_items_df = date_item_df.reset_index().pivot('InvoiceDate','StockCode').fillna(0)
trending_items_df.head(2)


# In[ ]:


trending_items_df = trending_items_df.reset_index()
trending_items_df.head(2)


# In[ ]:


trending_items_df = trending_items_df.set_index('InvoiceDate')
trending_items_df.head(2)


# In[ ]:


trending_items_df.columns = trending_items_df.columns.droplevel(0)
trending_items_df


# In[ ]:


# Visualize this time series for the top five best-sellers

ax = pd.DataFrame(trending_items_df.values).plot(
    figsize=(10,7),
    grid=True,
)

ax.set_ylabel('number of purchases')
ax.set_xlabel('date')
ax.set_title('Item Trends over Time')

ax.legend(trending_items_df.columns, loc='upper left')

plt.xticks(
    range(len(trending_items_df.index)), 
    [x.strftime('%m.%Y') for x in trending_items_df.index], 
    rotation=45
)

plt.show()


# The sales of the five products spiked in November 2011, especially with the product with the stock code, __84826, which were close to $0$ from February 2010 to October 2011.Then, it suddenly spiked in November 2011.
# <br>The popularity of the other top five products seems to have built up in the few months priors to November 2011.__
# <br> As a marketer, we should verify the reasons or the drivers behind this buildup of rising popularity for all the five products.

# __<u>Analyzing trends and changes in the popularity of products</u> helps us:__
# <br>
# - __understand what customers like and purchase the most__
# <br>
# - __tailor the marketing messages__
# <br>
# - __improve customer engagement__
# <br>
# - __get higher customer conversion rate__
# <br>
# - __build a product recommendation engine.__
