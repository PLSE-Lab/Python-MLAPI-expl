#!/usr/bin/env python
# coding: utf-8

# Discount Metric computation
# ----------

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
# print(os.listdir("../input"))

files_data = []
for file in os.listdir("../input"):
    file_data = pd.read_csv('../input/' + file)
    files_data.append(file_data)

stores_data = pd.concat(files_data)

stores_data['Discount'] = stores_data['MRP'] - stores_data['Sales Price']
stores_data['Discount Rate'] = np.where( stores_data['MRP'] != 0, 
                    (stores_data['Discount'] * 100) / stores_data['MRP'], 0)
    
stores_data.loc[(stores_data['MRP'] == 0) & (stores_data['Sales Price'] != 0), 'MRP Anamoly'] = True
stores_data.loc[~((stores_data['MRP'] == 0) & (stores_data['Sales Price'] != 0)), 'MRP Anamoly'] = False

stores_data['Sales Anamoly'] =  stores_data['Sales Price'] > stores_data['MRP']

stores_data['Sale Date'] = stores_data['Sale Date'].astype('datetime64[ns]')
stores_data['Month'] = pd.DatetimeIndex(stores_data['Sale Date']).month
stores_data['Year'] = pd.DatetimeIndex(stores_data['Sale Date']).year


def wavg(group, avg_name, weight_name):
    d = group[avg_name]
    w = group[weight_name]
    try:
        return (d * w).sum() / w.sum()
    except ZeroDivisionError:
        return 0


# It is to be noted that there are three anamolies found:
#     1. MRP Anamoly - MRP is zero (Free item) but Sales Price is positive (Sold for money). Since the MRP is zero, the Discount Rate for these data is set as 0.
#     2. Zero Data - Data with both MRP and Sales Price as zero. Since MRP is zero, the Discount Rate for these data is also set as 0.
#     3. Negative Discount - Sales Price is higher than the MRP. This Anamoly needs to be handled when computing the Discount percent by store and month. These data can either be removed in the computation or separately shown as anamolous sales along with the MRP Anamoly data using the '*Sales Anamoly*' boolean.
#     
# Currently all the three anamolies are pruned and the remaining data is used for aggregating various interesting results. Moving forward, all the condition are set as *Discount Rate > 0* to prune the MRP and Zero Data Anamonlies along with Negative Discount Anamoly.

# In[ ]:


aggregator = {'Discount': ['count', 'sum', 'min', 'max'], 'Discount Rate': ['min', 'max', 'mean'], 'Sales Qty': ['sum', 'min', 'max'], 'Sales Price': ['sum', 'min', 'max']}
sort_by = ('Sales Qty', 'sum')
discounts_count_store_wise = stores_data[stores_data['Discount Rate'] > 0].groupby(
                'Store Code').agg(aggregator)
discounts_count_store_wise['Weighted Average Discount'] = stores_data[
    stores_data['Discount Rate'] > 0].groupby('Store Code').apply(wavg, 'Discount', 'Sales Qty')
discounts_count_store_wise['Weighted Average Sales Price'] = stores_data[
    stores_data['Discount Rate'] > 0].groupby('Store Code').apply(wavg, 'Sales Price', 'Sales Qty')
discounts_count_store_wise.sort_values(sort_by, ascending=False).to_csv('Discounts - Store Wise.csv')


# ***Discount Rate***
# A direct conclusion is that Store 5 has the highest number of discount and the higher sum of discount rates over the period. 
# 
# Store 6 gives the highest average discount on items but has the lowest number of discounts among all the stores. Hence, it is not considered among the higher discount stores.
# 
# Although, Store 2 has the highest average discount rate, that is very close to 100%, it's weighted average discount is still lesser than Store 1 and 3. 
# 
# Store 3 surprisingly achives the highest valid weighted average discount (omitting Store 6).
# 
# Solely based on average *Discount Rate*, Store 2 easily win. But by the sheer number of Discounts and the amount of Discounts, Store 5 wins over every other store. If we look at average discount per item and the quantity sold, Store 3 wins omitting Store 6 for practical reasons.
# 
# Further inspection is necessary over the month-wise discounts given and brands or categories that each store provides discounts on.

# In[ ]:


grouped_discount_month_wise = stores_data[stores_data['Discount Rate'] > 0].groupby(['Year', 'Month', 'Store Code'])
grouped_discount_store_wise = stores_data[stores_data['Discount Rate'] > 0].groupby(['Store Code', 'Year', 'Month'])

discounts_month_wise = grouped_discount_month_wise.agg(aggregator)
discounts_month_wise['Weighted Average Discount'] = grouped_discount_month_wise.apply(wavg, 'Discount', 'Sales Qty')
discounts_month_wise.to_csv('Discounts - Month Store Wise.csv')

discounts_store_wise = grouped_discount_store_wise.agg(aggregator)
discounts_store_wise['Weighted Average Discount'] = grouped_discount_store_wise.apply(wavg, 'Discount', 'Sales Qty')
discounts_store_wise.to_csv('Discounts - Store Month Wise.csv')


# **Discount Rate**
# 
# From the two sheets, it is evident that Store 5 has been giving consistent discounts over the year and has given higher discounts in year ending and new year (Nov 2017, Dec 2017, Jan 2018) than any other months.
# 
# Based on the Sales Qty with Discount, the end of year and new year (Nov 2017, Dec 2017, Jan 2018, Feb 2018) has higher number of sales with discount than any other months.
# 
# Store 2 has shown very high Discount Rate during the year 2017 (Aug - Dec) but its number of discounts and weighted average discount per item is relatively much lesser than few other stores. This is seen to change since the new year i.e., the Discount Rate has fallen to 70+% on an average while the weighted average discount has rosen to very high values. This could correspond to lesser item sales with discount.
# 
# The trend in the month-wise report clearly shows that discounts from all the store increase to a peak during the year and (Dec 2017) and New Year (Jan 2018) in comparison to any other months.
# 
# These few months, Store 5 has given more and higher discounts than any other store (Oct, Dec,2017, Jan 2018)
# 
# Next we check the trend with brand and category. First, we need to compare and check the Brand Code and Category values. There is a possibility that a brand code can have items in multiple categories.

# In[ ]:


sort_by_discount_rate = [('Discount Rate', 'mean')]
category_grouped_data = stores_data[stores_data['Discount Rate'] > 0].groupby(['Store Code', 'Category'])
discounts_category_wise = category_grouped_data.agg(aggregator)
discounts_category_wise['Weighted Average Discount'] = category_grouped_data.apply(wavg, 'Discount', 'Sales Qty')
discounts_category_wise.to_csv('Discounts Category Wise.csv')


# Category 4 has the highest Discount and Discount Rate in all the stores with at least 90%. Let us now check which brands of Category 4 mark higher Discount and Discount Rate.

# In[ ]:


brand_grouped = stores_data[(stores_data['Discount Rate'] > 0)].groupby(['Brand Code'])
discounts_brand_cat4_overall = brand_grouped.agg(aggregator)
discounts_brand_cat4_overall['Weighted Average Discount'] = brand_grouped.apply(wavg, 'Discount', 'Sales Qty')
discounts_brand_cat4_overall.sort_values('Weighted Average Discount', ascending=False).to_csv('Discounts Brand Wise.csv')


# Clearly, Brand 50 has the highest discount rate (100%) closely followed by Brand 27 (91%). Brand 41 can be eliminated since it is a returned product.
# 
# Based on the previous result of store wise data, Brand 27 has higher number and amount of discounts more than any other brand in the Category 4. But this can be attributed to higher sales of the brand.
# 
# When weighted average discount is taken into consideration, Brand 49 has the highest discount per item than any other brand and Brand 42 the highest in Category 4, as this category has higher Discount Rate than any other.
# 
# This result can be generalized to any Category and generally said that Brand 50 and Brand 27 have higher Discount Rates than any other brand with Brand 27 having higher number and amount of discounts.

# Comparing both Sales Quantity and Average Selling Price together and concluding monthly and store wise observations
# -------

# In[ ]:


grouped_sales_month_wise = stores_data.groupby(['Year', 'Month', 'Store Code'])
grouped_sales_store_wise = stores_data.groupby(['Store Code', 'Year', 'Month'])
grouped_sales_store_cateogry_wise = stores_data.groupby(['Store Code', 'Category'])
grouped_sales_store_brand_wise = stores_data.groupby(['Store Code', 'Brand Code'])
aggregator2 = {'Sales Qty': ['sum', 'min', 'max'], 'Sales Price': ['sum', 'min', 'max'], 'Discount Rate': ['min', 'max', 'mean']}
sort_by = 'Weighted Sales Price Average'
sort_by_second = ('Sales Qty', 'sum')

sales_store_wise = stores_data.groupby('Store Code').agg(aggregator2)
sales_store_wise['Weighted Sales Price Average'] = stores_data.groupby('Store Code').apply(wavg, 'Sales Price', 'Sales Qty')
sales_store_wise.sort_values(sort_by, ascending=False).to_csv('Sales - Store Wise.csv')


# **Sales Quantity**
# 
# Store 5 clearly has the highest number of sales in the given data. Further inspection is necessary on monthly sales to inspect if this is due to consistency or some other parameter.
# 
# **Weighted Average Sales Price**
# 
# In terms of average sales price, Store 6 has the highest sales price average among all the stores and hence notably has the lowest sales quantity. 
# 
# Other than Store 6, Store 2 has higher sales price average with decent sales quantity.

# In[ ]:


sales_month_wise = grouped_sales_month_wise.agg(aggregator2)
sales_month_wise['Weighted Sales Price Average'] = grouped_sales_month_wise.apply(wavg, 'Sales Price', 'Sales Qty')
sales_month_wise.to_csv('Sales Month Store Wise.csv')

sales_store_month_wise = grouped_sales_store_wise.agg(aggregator2)
sales_store_month_wise['Weighted Sales Price Average'] = grouped_sales_store_wise.apply(wavg, 'Sales Price', 'Sales Qty')
sales_store_month_wise.to_csv('Sales Store Month Wise.csv')


# **Sales Quantity**
# 
# Store 5 and Store 1 has been head on head for sales during the year and end with record highest total sales (around 2000+). But Store 5 has consistent sales throughtout the year securing the highest spot in terms of sales quantity.
# 
# The trend clearly suggests that there is a higher sale during the end of the year 2017 (Nov, Dec) than any other months.
# 
# **Weighted Average Sales Price**
# 
# Store 6 can easily be generalized as the store with highest sales price average overall. It can also be noted that Store 2 has been comparitively showing higher sales price average over months other than year ends and new year (Nov 2017 - Feb 2018).
# 
# It is also worth to be noted that the sales price average drops during the year ending and new year (Nov 2017 - Feb 2018).

# In[ ]:


sales_store_category_wise = grouped_sales_store_cateogry_wise.agg(aggregator2)
sales_store_category_wise['Weighted Sales Price Average'] = grouped_sales_store_cateogry_wise.apply(wavg, 'Sales Price', 'Sales Qty')
sales_store_category_wise.sort_values(sort_by, ascending=False).to_csv('Sales Category Wise.csv')


# **Sales Quantity**
# 
# Among the Categories, Cat04 has shown the highest sales in several stores (Store 1, 5, 2, 4, 3 - in the descending order).
# 
# Store 5 has higher number of categories with high sales (more than 2000) which are Cat04, 07 & 01 and so stand victorious in sales.
# 
# **Weighted Average Sales Price**
# 
# Category 8 in Store 1 has one of the highest average selling price and subsequently very low sales quantity.
# 
# But, Category 1 has consistently shown higher average selling price in several stores (Store 1, 5, 6, 2, 3).

# In[ ]:


sales_store_brand_wise = grouped_sales_store_brand_wise.agg(aggregator2)
sales_store_brand_wise['Weighted Sales Price Average'] = grouped_sales_store_brand_wise.apply(wavg, 'Sales Price', 'Sales Qty')
sales_store_brand_wise.sort_values(sort_by_second, ascending=False).to_csv('Sales Store Brand Wise.csv')


# **Sales Quantity**
# 
# In terms of sales quantity, Brand 27 definitely dominates the charts with very low average sales price. It is followed by Brand 50 and 33 with a sufficiently high difference.
# 
# **Weighted Average Sales Price**
# 
# Among the Brands, Brand49 definitely has the highest average selling price and can been seen to have among the lowest total sales quantity. Following closely is Brand 47 and 2.

# Profit Calculation
# -----------

# In[ ]:


margin = 0.5
stores_data['Margin'] = stores_data['Sales Price'] * margin
stores_data['Margin Quantity'] = stores_data['Margin'] * stores_data['Sales Qty']

brand_grouped = stores_data.groupby('Brand Code')
aggregator3 = {'Sales Price': ['min', 'max', 'sum', 'mean'], 'Discount Rate':['min', 'max', 'mean'], 'Sales Qty': ['min', 'max', 'sum'], 'Margin Quantity': ['min', 'max', 'sum']}
sort_by2 = ('Margin Quantity', 'sum')
sort_by3 = 'Weighted Margin Average'
brand_margin = brand_grouped.agg(aggregator3)
brand_margin['Weighted Margin Average'] = brand_grouped.apply(wavg, 'Margin', 'Sales Qty')
brand_margin.sort_values(sort_by2, ascending=False).to_csv('Brand Margin.csv')


# By weighted average margin, Brand 49 definitely wins with highest weighted average followed by Brand 47 but has relatively very low total margin (Sum of Margin Quantity) due to very low sales quantity. This could be due to low demand and very high average sales price.
# 
# When comparing the brands with total margin (Sum of Margin Quantity - Margin x Quantity sold or returned), Brand17 clearly wins over the rest of the brand while Brand 18 trails with around 30,000 difference in total margin quantity. Even though both these brands are only third and fourth in terms of overall sales quantity, these brands have dominated total margin.
# 
# Brand 27 and 50 are found very low in the margin chart which can be attributed to higher discount rates on these brands. Most of the top brands on this chart have a negative mean discount rate, meaning these brands are sold at prices that are higher than its MRP.
# 
# If the sales of Brand 49 and Brand 47 can be improved, these brands would definitely give higher margins to the store. But to improve the sales, if the sales price is reduced, a direct consequence can be reduced average margin. Hence, these products cannot be firmly stated as high margin products.
# 
# Brand 17 can definitely be said as the best brand for higher margin for its relatively lower sales price and high sales quantity probably due to higher demand. Brand 18 can also be trailed but has relatively lower margin with similar sales quantity as Brand 17.
