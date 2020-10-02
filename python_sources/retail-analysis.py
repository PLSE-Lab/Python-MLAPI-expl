#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # Data visualization
import seaborn as sns # another data visualization

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


annual_sales_data = pd.read_csv("/kaggle/input/retail-business-sales-20172019/business.retailsales2.csv")
order_sales_data = pd.read_csv("/kaggle/input/retail-business-sales-20172019/business.retailsales.csv")


# In[ ]:


annual_sales_data.head()


# In[ ]:


order_sales_data.head()


# Based on the two datasets, I could answer a few questions:
# 
# 1. What is the average value per order in terms of sales, disounts and returns for each month
# 1. What product types contribute 80% of the sales
# 1. What is the typical volume per order by product type
# 1. What is the range of discounts and returns for each product type
# 1. Is there a relationship between discounts rate and returns rate?
# 
# However, there isn't a way to link both datasets together. Without that connection, I would not be able to answer a few questions:
# 1. Without the order ID, I would not be able to identify what product type is typically ordered with another product type. This could help the seller to identify what product bundle could be created for sales.
# 1. Without a timestamp, I would not be able to identify whether is there a trend to the monthly product profile.
# 1. Without a timestamp, it is not possible to identify the expected sales for the year 2020 (if coronovirus is not around) and what level of inventory should be kept for each month.

# # 1. What is the average value per order in terms of sales, disounts and returns for each month

# In[ ]:


annual_sales_data["Avg order gross sales"] = annual_sales_data["Gross Sales"]/annual_sales_data["Total Orders"]
annual_sales_data["Avg order total sales"] = annual_sales_data["Total Sales"]/annual_sales_data["Total Orders"]
annual_sales_data["Avg order discounts"] = -annual_sales_data["Discounts"]/annual_sales_data["Total Orders"]
annual_sales_data["Avg order returns"] = -annual_sales_data["Returns"]/annual_sales_data["Total Orders"]
annual_sales_data["Avg order shipping"] = annual_sales_data["Shipping"]/annual_sales_data["Total Orders"]
annual_sales_data["Avg order returns and discounts"] = -(annual_sales_data["Returns"] + annual_sales_data["Discounts"])/annual_sales_data["Total Orders"]


# In[ ]:


annual_sales_data.head()
# Total orders = monthly total orders
# Gross sales = monthly total sales from the total orders, excluding discounts and returns 
# Net sales = gross sales - (discounts + returns)
# Total sales = net sales + shipping (I assume that shipping cost is paid by the customers)


# In[ ]:


fig, ax = plt.subplots(3,1, figsize = (16,10), sharex = "all")
sns.lineplot(x = "Month", y = "Total Orders", data = annual_sales_data, hue = "Year", ax = ax[0], palette = "bright", sort = False)
ax[0].set_title("Monthly total orders")

sns.barplot(x = "Month", y = "Gross Sales", data = annual_sales_data, hue = "Year", ax = ax[1])
ax[1].set_title("Monthly gross sales")

sns.barplot(x = "Month", y = "Total Sales", data = annual_sales_data, hue = "Year", ax = ax[2])
ax[2].set_title("Monthly total sales")
plt.xticks(rotation=45)


# In[ ]:


fig, ax = plt.subplots(5,1, figsize = (10,15), sharex = "all")
sns.lineplot(x = "Month", y = "Avg order gross sales", data = annual_sales_data, hue = "Year", ax = ax[0], palette = "bright", sort = False)
ax[0].set_title("Monthly average gross sales per order")
ax[0].set_ylim(0)

sns.lineplot(x = "Month", y = "Avg order returns", data = annual_sales_data, hue = "Year", ax = ax[1], palette = "bright", sort = False)
ax[1].set_title("Monthly average returns per order")

sns.lineplot(x = "Month", y = "Avg order discounts", data = annual_sales_data, hue = "Year", ax = ax[2], palette = "bright", sort = False)
ax[2].set_title("Monthly average discounts per order")

sns.lineplot(x = "Month", y = "Avg order shipping", data = annual_sales_data, hue = "Year", ax = ax[3], palette = "bright", sort = False)
ax[3].set_title("Monthly average shipping per order")
ax[3].set_ylim(0)

#sns.lineplot(x = "Month", y = "Avg order returns and discounts", data = annual_sales_data, hue = "Year", ax = ax[2], palette = "bright", sort = False)
#ax[2].set_title("Monthly average returns and discounts per order")
#ax[2].set_ylim(0)

sns.lineplot(x = "Month", y = "Avg order total sales", data = annual_sales_data, hue = "Year", ax = ax[4], palette = "bright", sort = False)
ax[4].set_title("Monthly average total sales per order")
ax[4].set_ylim(0)

plt.xticks(rotation=45)


# As you can see, the monthly total orders have been quite consistent in the first 10 months it averages about 100 orders per month. The total orders start to pick up October onwards and peak in December before declining signficantly back to the average level in January. Likewise, the gross and total sales averages about 8k-10k per month before picking up October onwards and peak in Nov/Dec with an average of 15k. 
# 
# However, in 2019, the trend has changed a little. There are two peaks, one in March and another in December. In March, the gross and total sales is about 30% higher than the average sales of 10k and in November and December, the sales are about 100% and 130% higher than the average sales of 15k.
# 
# The average shipping cost per order has been steadily increasing. In 2019, the average shipping cost is between 15 to 20 as compared to the average ahipping cost of 10 to 15 in 2017. Despite that, the average returns and discount per order has been increasing and outweights the increase in shipping cost. That would explain why average sales per order has been at the all time low in 2019 especially in the second half of 2019. The The average gross and total sales per order have typically hover in the range 75-125 and 100-150 respectively.
# 

# # 2. What product types contribute 80% of the sales

# In[ ]:


pivot_1 = order_sales_data.groupby(["Product Type"])[["Total Net Sales"]].agg("sum")
pivot_1["Sales proportions"] = (pivot_1["Total Net Sales"]/pivot_1["Total Net Sales"].sum())
pivot_1.sort_values("Sales proportions", ascending = False, inplace = True)
pivot_1["Cumulative sales proportions"] = pivot_1["Sales proportions"].cumsum().apply(lambda x: "%.2f" % round(100*x,2))

pivot_2 = order_sales_data.groupby(["Product Type"])[["Total Net Sales"]].agg("mean")
pivot_2["Total Net Sales"] = pivot_2["Total Net Sales"].apply(lambda x: "%.2f" % round(x,2))
pivot_2.rename(columns = {"Total Net Sales":"Average sales per order"}, inplace = True)

pivot_3 = order_sales_data.groupby(["Product Type"])[["Total Net Sales"]].agg("count")
pivot_3.rename(columns = {"Total Net Sales":"No of orders"}, inplace = True)

pivot_1 = pivot_1.join(pivot_2)
pivot_1 = pivot_1.join(pivot_3)
pivot_1


# Basket, Art & Sculpture, Jewelry and Home Decor are the top 4 product type (out of 24) that contribute over 80% of your total net sales. Also, we can see that these 4 product types are some of the product types that generate the most sales per order as well.
# 
# However, it is unknown whether the these 4 product types are the same products for each of the 3 years. Moreover, with the lack of data, I cannot determine whether these top 4 product types are due to the surge in sales in November and December. To do so, the timestamp of the orders are needed to enhance the analysis.
# 
# If these 4 product types are the same throughout the entire year, the seller should focus more effort on generating more orders and sales for these products.

# # 3. What is the typical volume per order by product type

# In[ ]:


plt.figure(figsize=(20, 6))
sns.boxplot(y = "Net Quantity", data = order_sales_data, x = "Product Type")
plt.xticks(rotation = 45)


# A boxplot is a ay to visualize the distribution of the data. Here we can observe that most of the order quantity are rather skewed. Calculating the average quantity per order for each product type could be misleading and may lead the seller to overestimate the inventory needed for each order. Hence, the use of median would be more appropriate.

# In[ ]:


pivot_4 = order_sales_data.groupby(["Product Type"])[["Net Quantity"]].agg(["count","sum", "median", "mean"])
pivot_4[("Net Quantity", "mean")] = pivot_4[("Net Quantity", "mean")].apply(lambda x: int(x))

pivot_4


# Based on the above table, we can observe that the quantity for each product type are rather small with typical quantity between 1 to 2 unit. Here, we can tell that the seller can expect most of the orders to be small quantity with the rare occasion that a bulk order is made. 

# # 4. What is the range of discounts and returns for each product type

# In[ ]:


order_sales_data["Discounts proportions"] = (order_sales_data["Discounts"].apply(lambda x: abs(x))/order_sales_data["Gross Sales"]).apply(lambda x: round(100*x, 2))
order_sales_data["Returns proportions"] = (order_sales_data["Returns"].apply(lambda x: abs(x))/order_sales_data["Gross Sales"]).apply(lambda x: round(100*x, 2))


# In[ ]:


plt.figure(figsize=(20, 6))
sns.stripplot(x='Product Type', y='Discounts proportions', data = order_sales_data, jitter=True, split=True)
plt.xticks(rotation = 45)
plt.title("Distribution of discounts proportions for each product type")


# Here we can see that discounts can go up to 25% and are usually between 0% to 10%. The discount rate is determined by the seller, so there is not much to study here. The more important thing to study is the returns rate.

# In[ ]:


plt.figure(figsize=(20, 6))
sns.stripplot(x='Product Type', y='Returns proportions', data = order_sales_data, jitter=True, split=True)
plt.xticks(rotation = 45)
plt.title("Distribution of returns proportions for each product type")


# Here, we can observe that most orders for most product types are typically 0%. However, products that are often returned are the Arts & Sculpture, Basket, Home Decor, Jewelry and Kitchen. The rest of the product types are not typically returned. The seller will need to take a deeper look into these few product types to identify the reason these products are returned, especially the top 4 products that generate 80% of the total net sales.

# # 5. Is there a relationship between discounts rate and returns rate?
# 
# Such relationships are investigated for Arts & Sculpture, Basket, Home Decor, Jewelry and Kitchen as these products have high returns rate

# In[ ]:


order_sales_data


# In[ ]:





# In[ ]:


product_type = ["Art & Sculpture", "Basket", "Home Decor", "Jewelry","Kitchen"]
for product in product_type:
    temp = order_sales_data[order_sales_data["Product Type"] == product]
    plt.figure(figsize = (5,5))
    sns.scatterplot(x = "Discounts proportions", y = "Returns proportions", data = temp)
    plt.title("Returns vs discounts rate for %s" %product)


# There isn't an obvious relationship between returns and discounts rate for these product types. It seems to be totally random and the returns is not dependent on the discounts rate. The seller will need to deep dive and have a better look at the returned items to identify the reason. The sales could be much higher if the goods are not returned.
