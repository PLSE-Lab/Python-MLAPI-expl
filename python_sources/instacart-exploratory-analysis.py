#!/usr/bin/env python
# coding: utf-8

# # Instacart Exploratory Analysis
# 
# ## Introduction
# 
# Instacart is a same-day delivery and pick up service that allows consumers to shop through the company's mobile app or website from selected grocery stores such as Trader Joe's, Costco and Fairway. The company has dedicated employees who shop groceries from the selected stores and deliver to consumers' doorsteps. The way the company generates revenue is by adding markup on prices for specific stores, delivery and membership fee. Hence, the main business objective for the organization is to not only increase the amount of customer memberships but also improve the repeat visit and orders. Predicting consumer engagement and behavior with the products and grocery stores have a huge impact on instacart's success.
# 
# The main purpose of this analysis is to perform an explotary analysis and formulate incremental business problems to increase revenue. In order to do that, we will rely on historical data provided by Instacart. The further information about this publicly released dataset can be found here: https://tech.instacart.com/3-million-instacart-orders-open-sourced-d40d29ead6f2 

# ## About the Data
# 
# The dataset is provided by instacart for a Kaggle Competition which are relational set of files describing costumers' orders over time. The dataset is anaymized and contains a sample of over 3 million grocery orders from more than 200,000 instacart users. For each user, the dataset provides between 4 and 100 of their orders, with the sequence of products purcahsed in each order. The dataset also provides the week and hour of day the order was placed, and relative measure of time between orders. 
# 
# The description of each variable is outlined as below;
# 
# **orders dataset**
# 
# *order_id: order identifier*
# 
# *user_id: customer identifier*
# 
# *eval_set: which evaluation set this order belongs in (see train and prior dataset described below)*
# 
# *order_number: the order sequence number for this user (1 = first, n = nth)*
# 
# *order_dow: the day of the week the order was placed on*
# 
# *order_hour_of_day: the hour of the day the order was placed on*
# 
# *days_since_prior: days since the last order, capped at 30 (with NAs for order_number = 1)*
# 
# **products dataset**
# 
# *product_id: product identifier*
# 
# *product_name: name of the product*
# 
# *aisle_id: foreign key*
# 
# *department_id: foreign key*
# 
# **aisles dataset**
# 
# *aisle_id: aisle identifier*
# 
# *aisle: the name of the aisle*
# 
# **deptartments dataset**
# 
# *department_id: department identifier*
# 
# *department: the name of the department*
# 
# **order_products_prior and train datasets**
# 
# *order_id: foreign key*
# 
# *product_id: foreign key*
# 
# *add_to_cart_order: order in which each product was added to cart*
# 
# *reordered: 1 if this product has been ordered by this user in the past, 0 otherwise*
# 
# 
# *"prior": orders prior to that users most recent order*
# 
# *"train": training data supplied to participants of the competion*
# 
# *"test": test data reserved for machine learning competition*

# ## Data Collection and Cleaning
# 
# Ideally, prior to the data collection phase, we would have liked to define the data requirements based on problem statement in order to collect the needed data. However, we are executing a different approach in this case, where we really dont know what problems instacart business have and we want to formulate problems with the available data. This is completely different from a data science project process perscpective, where we usually want to outline business understanding with the domain experts and stake holders as a first step and furhter define collection methods, data collecting, cleaning and explatory analysis. 

# In[ ]:


# import required packages
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


# load the data to dataframes
orders=pd.read_csv("../input/instacart/orders.csv")
products=pd.read_csv("../input/instacart/products.csv")
aisles=pd.read_csv("../input/instacart/aisles.csv")
departments=pd.read_csv("../input/instacart/departments.csv")
order_products_prior=pd.read_csv("../input/instacart/order_products__prior.csv")


# In[ ]:


# quick overview of all the orders
orders.head()


# In[ ]:


orders.info()


# In[ ]:


orders.isnull().sum()


# Looking at all the orders from the customers, we have total of over 3 million orders, numerical and string data types are as expected and does not require us to update. We have over 200,000 missing values on the day since the last order. However, as it is explained in the description of the variables, NA represents the order_number 1 of that particular customer. 

# In[ ]:


# quick overview of the products
products.head()


# In[ ]:


products.info()


# In[ ]:


products.isnull().sum()


# We have around 50,000 products with product id, name, aisle id and department id. All data types are as expected and we dont have any missing values.

# In[ ]:


aisles.head()


# In[ ]:


aisles.info()


# In[ ]:


aisles.isnull().sum()


# There are 134 aisles with aisle id and aisle name. The data types are as expected and we dont have any missing values. As you may notice, aisle_id variable is a common variable with products dataframe. We can merge these two dataframes to see what product belong to what aisle.

# In[ ]:


# merge aisles and products
products=pd.merge(aisles, products, on="aisle_id")
products.head()


# In[ ]:


# quick overview of departments 
departments.head()


# In[ ]:


departments.info()


# In[ ]:


departments.isnull().sum()


# There are total of 21 departments with department names and ids. The data types are as expected and we dont have any missing values. We have another common variable "department_id" with products dataframe. We can merge the departments dataframe to see what product belongs to which department.

# In[ ]:


# merge departments and products
products=pd.merge(departments, products, on="department_id")
products.head()


# In[ ]:


# quick overview of orders prior to the most recent user order
order_products_prior.head()


# In[ ]:


order_products_prior.info()


# In[ ]:


order_products_prior.isnull().sum()


# As explained in the variable descriptions, order products prior data set gives us the insights on the orders that are ordered prior to users most recent order. We have over 3 million of these orders. The data types are correct and there are no missing values. We also have product_id common variable where we can merge this dataset with products to see what product maps to what order.

# In[ ]:


# merge order_products_prior with products dataframe
products=pd.merge(order_products_prior, products, on="product_id")
products.head()


# We also have another common variable order_id between products and orders dataset. We can merge these two dataframes to help gather insight between products and orders further.

# In[ ]:


# merge orders and products with order_id common
products_and_orders=pd.merge(products, orders, on="order_id")
products_and_orders.head()


# We have been provided over 1.3 million training data, with correct data types and no missing values. At this point, the data sets provided by instacart seems to be pretty clean and besides merging aisles, departments, products and orders data sets, we didnt need to do much of data wrangling. 

# ## Data Exploration
# 
# At this stage, we will look at the updated version of the data set individually, and gather insights to start creating possible business problems. 

# In[ ]:


orders.head()


# In[ ]:


# look at orders per customer
orders_per_customer=orders.groupby("user_id")["order_number"].max().reset_index()
orders_per_customer.head()


# In[ ]:


orders_per_customer["order_number"].value_counts()


# In[ ]:


plt.figure(figsize=(25,15))
sns.countplot(orders_per_customer["order_number"])
plt.title("Number of Orders Vs Number of Customer Makes These Orders")
plt.xlabel("Number of Orders By Customers")
plt.ylabel("Number of Customers")


# What we are seeing here is that 23986 customers made only 4  orders, 19590 customers only made 5 orders and so on... As the number of orders made by customers goes up, the number of customers that makes those orders goes down. Majority of the customers make 4 to 12 orders. If business finds way to increase the amount of orders from repeat customers, it can increase the revenue. 

# In[ ]:


# Look at which day of the week customers order
orders_dow = orders["order_dow"].value_counts()
orders_dow


# In[ ]:


plt.figure(figsize=(20,15))
sns.countplot(orders["order_dow"])
plt.title("Purchase Day of the Week Distribution")
plt.xlabel("Day of the week")
plt.ylabel("Count")


# Majority of the purchases are being made on Monday and Tuesday followed by Sunday. Consumers might be making their weekly grocessory shopping at the first and second day of the week. There isnt a huge gap between the other days of the week either. For example the difference between Wednesday and Tuesday is not that significant. 

# In[ ]:


# Purchase hour of the day
orders_how=orders["order_hour_of_day"].value_counts()
orders_how


# In[ ]:


plt.figure(figsize=(15,10))
sns.countplot(orders["order_hour_of_day"])
plt.title("Purchase Hour Distribution")
plt.xlabel("Hour of the Day")
plt.ylabel("Count")


# Majority of the purchases are made between 10am to 4pm. 

# In[ ]:


# frequency of orders
order_frequency=orders.groupby("order_id")["days_since_prior_order"].max().reset_index()
order_frequency.head()


# In[ ]:


plt.figure(figsize=(20,15))
sns.countplot(order_frequency["days_since_prior_order"])
plt.title("How often Customers Purchase")
plt.xlabel("Day Since Prior Order")
plt.ylabel("Count")


# Majority of the customers do their purchases weekly and monthly. Considering the fact that majority of the customers also place their orders Monday or Tuesday of the week, placing an order might require planning effort. There is a slight chance this might be due to a non seamless ordering process. 

# In[ ]:


#looking at the products
products.head()


# In[ ]:


# products per order
product_amount_per_order=products.groupby("order_id")["add_to_cart_order"].max().reset_index()
product_amount_per_order.head()


# In[ ]:


product_amount_per_order["add_to_cart_order"].value_counts()


# In[ ]:


plt.figure(figsize=(20,15))
sns.countplot(product_amount_per_order["add_to_cart_order"])
plt.title("Number of Products per Order")
plt.xlabel("Number of Products")
plt.ylabel("Frequency")


# We can see that, based on the basket size distribution, majority of the customers purchased 5 to 6 products per order. Considering the low amount of orders from repeat customers, low product amount per customer can be a huge problem for the business in terms of revenue. 

# In[ ]:


# most ordered products
top_ten_products=products["product_name"].value_counts().head(10)
top_ten_products


# In[ ]:


plt.figure(figsize=(15,10))
sns.countplot(x="product_name", hue="department", data=products, order=products.product_name.value_counts().iloc[:10].index)
plt.title("Top Ten Purchased Products")
plt.xlabel("Product Name")
plt.ylabel("Count")


# In[ ]:


# look at the reorders
product_reorders=products.groupby(['product_id', 'product_name'])['reordered'].count().reset_index()
product_reorders_top_ten=product_reorders.nlargest(10, "reordered")
product_reorders_top_ten


# In[ ]:


plt.figure(figsize=(15,10))
sns.barplot(x="product_name", y="reordered", data=product_reorders_top_ten)


# In[ ]:


# top ten reorder products and aisles
product_orders=products.groupby(['product_id', 'product_name', "aisle"])[['order_id']].count().reset_index()
product_orders.columns=["product_id", "product_name", "aisle", "order_amount"]
product_orders.head()


# In[ ]:


product_orders_top_ten=product_orders.nlargest(10, 'order_amount')
product_orders_top_ten


# Top ten reordered items match exactly to the top ten items ordered in general. Additionaly most of the items on top ten purchases are in the fresh fruits aisle. 

# In[ ]:


# look at the top aisles
top_aisle_in_one_order=products.groupby("aisle")["order_id"].count().reset_index()
top_aisle_in_one_order=top_aisle_in_one_order.nlargest(10, "order_id")
top_aisle_in_one_order


# In[ ]:


plt.figure(figsize=(20,10))
sns.barplot(x="aisle", y="order_id", data=top_aisle_in_one_order)


# Top aisles are fresh fruits, fresh vegetables, packaged vegatables and fruit, followed by yogurt and packaged cheese. 

# In[ ]:


#look at the products, customer and order mapping
products_and_orders.head()


# In[ ]:


#look at the day of order and the basket size
percentage_of_orders=products_and_orders.groupby("order_dow")["order_id"].count().reset_index()
percentage_of_orders["percentage"]=percentage_of_orders["order_id"]/percentage_of_orders["order_id"].sum()
percentage_of_orders.head()


# In[ ]:


plt.figure(figsize=(20,10))
sns.barplot(x="order_dow", y="percentage", data=percentage_of_orders)


# As we figured out earlier, the highest percentage of orders and the highest basket size is Monday and Tuesday.

# In[ ]:


# look at orders and products
orders_in_a_day=products_and_orders.groupby(["order_dow", "product_name"])["order_id"].count().reset_index()
orders_in_a_day["percentage"]=orders_in_a_day["order_id"]/orders_in_a_day["order_id"].sum()
orders_in_a_day=orders_in_a_day.nlargest(10, "percentage")
orders_in_a_day


# In[ ]:


plt.figure(figsize=(15,10))
sns.barplot(x="order_dow", y="percentage", data=orders_in_a_day, hue="product_name")


# As we found out earlier, the highest ordered products are Banana, Bag of Organic Bananas, Organic Baby Spinach and Organic Strawberries. Interesting insight here is that, out of the top 4 products, Banana has the most proportion of orders in a day for every but the other four top products do not have any contribution to the proportion of orders on Wednesday, Thursday, Friday, Saturday and Sunday. Can this be related to an item inventory issue?

# In[ ]:


# Looking at departments
top_departments=products_and_orders.groupby(["department"])["order_id"].count().reset_index()
top_departments=top_departments.nlargest(10, "order_id")
top_departments


# In[ ]:


plt.figure(figsize=(15,10))
sns.barplot(x="department", y="order_id", data=top_departments)


# Top department is produce, followed by dairy eggs and snacks. Addition to the general individual preference and needs, there might be other reasons why consumers are not purchasing from other departments. For example, if instacart has the biggest mark up for canned goods, but the user experience within the instacart website or app does not entice the consumer for canned goods, regardless of the consumer needs the order amount on canned goods will be low. 
# 
# The datasets are quite big and have many variables that we can analyze. We can further look at learning associations in detail between products (even though unstacking and converting the existing large dataset to a transactional data set would be challenging), investigate consumer reordering patterns in terms of days and hours, reorder ratio vs the hour of the order and so on...

# ## Conclusion
# 
# Looking at variety of data science processes and methodologies, you will realize almost all of them have the similar approach which start with business understanding outlining a business problem statement to validate and solve. However, we might run into situations where the domain expert, brand owner, business unit or stakeholder may not neccessarily know or does not have the time to figure out what the business problem is. In most of these cases, they are looking at the product manager to identify, formulate the business problems and provide tangible recommendations to solve them. 
# 
# In our instacart analysis, we can summarize possible business problems that might have impact on revenue are;
# 
# - Frequency of customer orders are low ranging from 4-12. What actions can business take in order the improve the frequncy of customer orders? 
# 
# - Majority of the customers planning their purchases as weekly and monthly cycles. What actions can business take to improve the purchases for any day of the week and month? As an example; Amazon made their entire ordering product and process very seamlessly that I personaly do not do extensive planning on my grocessory purchases. If I am in need of a product from amazon, I order that day. 
# 
# - Besides banana, top items are sold mostly on Monday and Tuesday. Can this be related to an item inventory issue from the retailers? For example; would it be possible Trader Joe's runs out of organic strawberries by Wednesday and instacart employee substitutes that order with a regular strawbery based on customer request?
# 
# - Besides the general consumer and market needs, are there any reaons why certain department sales are lower than the others? (such as instacart website or mobileapp usability issues) 
# 
# Formulating and defining business problem statements is extremly important to find the right solution for the business goals. In my earlier article on linear regression, I mentioned how problematic it is for some statisticians to start with data analysis without creating a problem statement, however there are many cases where stakeholders expect data to drive and formulate a problem. 
