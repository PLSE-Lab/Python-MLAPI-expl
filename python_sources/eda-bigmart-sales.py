#!/usr/bin/env python
# coding: utf-8

# # BIG MART SALES

# ## Problem Statement 

# The data scientists at BigMart have collected 2013 sales data for 1559 products across 10 stores in different cities. Also, certain attributes of each product and store have been defined. The aim is to build a predictive model and find out the sales of each product at a particular store.
# Using this model, BigMart will try to understand the properties of products and stores which play a key role in increasing sales.
#  
# Please note that the data may have missing values as some stores might not report all the data due to technical glitches. Hence, it will be required to treat them accordingly.

# In[ ]:


import numpy as np
import pandas as pd
from pandas import Series,DataFrame
import matplotlib.pyplot as plt
import seaborn as sns


# ### Reading the train_dataset

# In[ ]:


data=pd.read_csv('/kaggle/input/bigmart-sales-dataset/EDA_BIGMART_SALES.csv',index_col=0)
data.head()


# In[ ]:


data.columns


# ### Identifying the numerical & categorical variables in the dataset

# In[ ]:


num_var = [key for key in dict(data.dtypes) if dict(data.dtypes)[key] in ['int64','float64']]
cat_var = [key for key in dict(data.dtypes) if dict(data.dtypes)[key] in ['object']]


# In[ ]:


print(num_var)
print(cat_var)


# ### Data Visualization

# In[ ]:


sns.distplot(data.Item_Outlet_Sales)


# Inference:
#  - 'Item_Outlet_Sales' is the target variable
#  - Majority of the sales(amount) have been made within a range of 0 and 2000

# In[ ]:


sns.scatterplot(x=data.Item_MRP,y=data.Item_Outlet_Sales,hue=data.Item_Type_Category)


# Inference :
#   - Items with price >200 are generating more Sales
#   - Majorly Food items are being purchased by the customers 

# In[ ]:


sns.scatterplot(x=data.Item_Visibility,y=data.Item_Outlet_Sales,hue=data.Item_Type_Category)


# Inference :
#  - Food & Non-consumable products are made easily available to the customers compared to Drinks

# In[ ]:


plt.bar(x=data.Outlet_Establishment_Year,height=data.Item_Outlet_Sales)


# Inference:
#  - Customers tend to purchase more items in the store which has been in the market for longer duration

# In[ ]:


sns.boxplot(x=data.Item_Type_Category,y=data.Item_Outlet_Sales)


# Inference:
#  - Average sales(value) for different category of items made available are similar

# In[ ]:


plt.figure(figsize=(8,5))
sns.barplot(x="Outlet_Type",data=data,hue='Item_Type_Category',y="Item_Outlet_Sales")
plt.show()


# Inference :
#  - Supermarket Type3 is generating more revenue in the market.
#  - Revenue generated in each store under each category is similar. 

# In[ ]:


plt.figure(figsize=(8,5))
sns.barplot(x="Item_Fat_Content",data=data,hue="Item_Type_Category",y="Item_Outlet_Sales")
plt.show()


# In[ ]:


plt.figure(figsize=(8,5))
sns.countplot(x="Item_Fat_Content",data=data,hue="Item_Type_Category")
plt.show()


# In[ ]:


plt.pie(data.Item_Fat_Content.value_counts(),labels=data.Item_Fat_Content.unique(),autopct='%1.1f%%')


# In[ ]:


plt.pie(data.Item_Type_Category.value_counts(),labels=data.Item_Type_Category.unique(),autopct='%1.1f%%')


# Inference:
#  - Majorly food items are being purchased by the customers

# In[ ]:


plt.pie(data.Outlet_Location_Type.value_counts(),labels=data.Outlet_Location_Type.unique(),autopct='%1.1f%%')


# Inference:
#  - Customers from Tier1 locations tend to make frequent purchases compared to the other cities

# In[ ]:




