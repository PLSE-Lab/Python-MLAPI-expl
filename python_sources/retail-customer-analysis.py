#!/usr/bin/env python
# coding: utf-8

# # Retail Customer Analysis

# ## Introduction
# 
# In this article we are going to perform an exploratary analysis for an online retail store data set, in order to understand its customers. Let's assume, we own a retail store that has been doing incredibly well and we want to find a way to scale our business efficiencly and effectively. In order to do this, we need to make sure we understand our customers and customize our marketing or expansion efforts based on specific subset of our customers. The main business problem in question in this case is "How can I scale my current business that is doing really well, in the most effective way?". The sub question that might follow to support the main business objectives can be "What type of marketing initiatives can we perform for each customer in order to get the best ROI?".

# ## About the Data
# 
# The dataset is a very common one that can be found in many publicly available data sources and intended to be used as an example online retail store data. The descriptions of each variable is straightforward and self explanatory. 

# ## Data Collection and Cleaning

# In[ ]:


# Import Standard packages
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


# Load the available data and overview
df=pd.read_csv("https://raw.githubusercontent.com/anilak1978/ecommerce/master/OnlineRetail.csv", encoding="ISO-8859-1")
df.head()


# In[ ]:


# Look for missing values
df.isnull().sum()


# In[ ]:


# Look for datatypes
df.dtypes


# There are 1454 missing "Description" and 135,080 missing "CustomerID" values. Keeping the missing "Description" values won't have any impact on the analysis as we will be focusing on understanding our customers. Based on the assumption that each missing "CustomerID" represents a new customer (meaning the missing "CustomerID" values are unique to the dataset), we can also assume that the "CustomerID" missing values wont have any impact to our analysis as well.
# 
# We notice that the "InvoiceDate" data type is object and requires to be updated to date data type. We can also take the month and the year from the "InvoiceDate" variable and create new columns with "InvoiceMonth" and "InvoiceYear". 

# In[ ]:


# Add Revenue variable and update InvoiceDate 
df["Revenue"]=df["UnitPrice"]*df["Quantity"]
df["InvoiceDate"]=pd.to_datetime(df["InvoiceDate"]).dt.date
df["InvoiceMonth"]=pd.DatetimeIndex(df["InvoiceDate"]).month
df["InvoiceYear"]=pd.DatetimeIndex(df["InvoiceDate"]).year
df.head()


# We loaded and performed minor cleaning on our online store data set. We can further analyze the customers.

# ## Data Exploration
# 
# Initially we are going to perform basic exploratory statistical analysis in order to understand the variable distribution. Further we will look at how we are doing in terms of revenue within the customers. Revenue is one of the most common and important metrics when it comes to online retail store business models. We would want to make sure our revenue is growing as we are defining scalibility within our business. 

# In[ ]:


# basic statistical analysis
df.info()


# In[ ]:


df.describe()


# In[ ]:


#Monthly Revenue Overview
df_revenue=df.groupby(["InvoiceMonth", "InvoiceYear"])["Revenue"].sum().reset_index()
plt.figure(figsize=(15,10))
sns.barplot(x="InvoiceMonth", y="Revenue", hue="InvoiceYear", data=df_revenue)
plt.title("Monthly Revenue")
plt.xlabel("Month")
plt.ylabel("Revenue")


# In[ ]:


# Monthly Revenue Overview
plt.figure()
sns.relplot(x="InvoiceMonth", y="Revenue", hue="InvoiceYear", kind="line", data=df_revenue, height=10, aspect=15/10)
plt.title("Monthly Revenue")
plt.xlabel("Month")
plt.ylabel("Revenue")


# When we look at the mothly revenue growth, we realize that our data set starts in December 2010 and goes all the way to December 2011. We notice that the revenue slumps around December 2011. We need to see if this is due to customers not purchasing our merchandise or if it is related to an issue within the datset.

# In[ ]:


# Look at the December 2011 data
df_december_2011=df.query("InvoiceMonth==12 and InvoiceYear==2011")
df_december_2011


# Based on the december 2011 data, we understand that the data set does not include any purchases after December 9th 2011. We need to make sure, we consider this within our analysis and conclusion.
# 
# On a separate note; we can see that the revenue grows steadily starting from September 2011 all the way up to December 2011, November being the best month in terms of revenue.

# In[ ]:


# Monthly Items Sold Overview
df_quantity=df.groupby(["InvoiceMonth", "InvoiceYear"])["Quantity"].sum().reset_index()
plt.figure(figsize=(15,10))
sns.barplot(x="InvoiceMonth", y="Quantity", data=df_quantity)
plt.title("Monthly Items Sold")
plt.xlabel("Month")
plt.ylabel("Items Sold")


# In[ ]:


# Monthly Active Customers
df_active=df.groupby(["InvoiceMonth", "InvoiceYear"])["CustomerID"].nunique().reset_index()
plt.figure(figsize=(15,10))
sns.barplot(x="InvoiceMonth", y="CustomerID", hue="InvoiceYear", data=df_active)
plt.title("Monthly Active Users")
plt.xlabel("Month")
plt.ylabel("Active Users")


# In[ ]:


# Average Revenue per Month
df_revenue_avg=df.groupby(["InvoiceMonth", "InvoiceYear"])["Revenue"].mean().reset_index()
plt.figure(figsize=(15,10))
sns.barplot(x="InvoiceMonth", y="Revenue", data=df_revenue)
plt.title("Monthly Average Revenue ")
plt.xlabel("Month")
plt.ylabel("Revenue")


# As expected, the monthly items sold, monthly active users and average revenue per month shows possitive correlation with the monthly revenue growth.

# In[ ]:


# New vs Existing Users
df_first_purchase=df.groupby(["CustomerID"])["InvoiceDate"].min().reset_index()
df_first_purchase.columns=["CustomerID", "FirstPurchaseDate"]
df=pd.merge(df, df_first_purchase, on="CustomerID")
df["UserType"]="New"
df.loc[df["InvoiceDate"]>df["FirstPurchaseDate"], "UserType"]="Existing"
df.head()


# In[ ]:


# New vs Existing User Revenue Analysis
df_new_revenue=df.groupby(["InvoiceMonth", "InvoiceYear", "UserType"])["Revenue"].sum().reset_index()
plt.figure()
sns.relplot(x="InvoiceMonth", y="Revenue", hue="UserType", data=df_new_revenue, kind="line", height=12, aspect=18/10)
plt.title("New vs Existing Customer Revenue Overview")
plt.xlabel("Month")
plt.ylabel("Revenue")


# In most cases, with the exception of the level, we see alignment on positive and negative revenue growth for new and existing customers. However, when we look at the revenue from January to Febuary and October to November of 2011, we see that even though existing customer revenue grows, the new customer revenue declines. 

# ## Customer Segementation
# 
# We analyzed our customers based on revenue, activity, new and existing customer monthly revenue and we definately have some insights that we can take action from. We can also segment our customers in order to target our actions based on the main business problem we are working to solve. 
# 
# We are going to use the RFM (Recency, Frequency and Monetary Value) strategy to analyze and estimate value of each customer and further segment them accordingly. Looking at Recency gives us how recently customers make a purhcase, Freqeuncy, how often they make a purchase, and Monetary Value shows us how often do they spend.

# In[ ]:


# Recency Calculation
df_user=pd.DataFrame(df["CustomerID"].unique())
df_user.columns=["CustomerID"]
df_last_purchase=df.groupby(["CustomerID"])["InvoiceDate"].max().reset_index()
df_last_purchase.columns=["CustomerID", "LastPurchaseDate"]
df_last_purchase["Recency"]=(df_last_purchase["LastPurchaseDate"].max()-df_last_purchase["LastPurchaseDate"]).dt.days
df_recency=pd.merge(df_user, df_last_purchase[["CustomerID", "Recency"]])
df_recency.head()


# In[ ]:


# Look at the distribution of Recency
plt.figure(figsize=(15,10))
sns.distplot(df_recency["Recency"])
plt.title("Recency Distribution Within the Customers")
plt.xlabel("Recency")
plt.ylabel("Customer Count")


# In[ ]:


# use KMeans Clustering for Recency Clustering
from sklearn.cluster import KMeans
# find out how many clusters are optimal
y=df_recency[["Recency"]] # label what we are clustering
dic={} # store the clustering values in a dictionary
for k in range(1,10):
    kmeans=KMeans(n_clusters=k, max_iter=1000).fit(y)
    y["clusters"]=kmeans.labels_
    dic[k]=kmeans.inertia_
plt.figure(figsize=(15,10))
plt.plot(list(dic.keys()), list(dic.values()))
plt.show()


# In[ ]:


# Cluster Customer based on Recency
kmodel_recency=KMeans(n_clusters=4)
kmodel_recency.fit(y)
kpredict_recency=kmodel_recency.predict(y)
kpredict_recency[0:5]
df_recency["RecencyCluster"]=kpredict_recency
df_recency.head()


# In[ ]:


# get statistical analysis for each cluster
df_recency.groupby(["RecencyCluster"])["Recency"].describe()


# We segmented our customers into 4 different clusters which are from 0 to 4. We have 524 customers in Cluster 0, 2157 customers in Cluster 1, 632 customers in Cluster 2 and 1059 Customers in Cluster 3. When we compare the recency, the cluster 0 is the best performing customer set and cluster 1 is the worst performaning cluster set.
# 
# We can further look at frequency and segment customers based on how ofthen do they purchase.

# In[ ]:


# frequency of orders
df_frequency=df.groupby(["CustomerID"])["InvoiceDate"].count().reset_index()
df_frequency.columns=["CustomerID", "Frequency"]
df_frequency=pd.merge(df_user, df_frequency, on="CustomerID")
df_frequency.head()


# In[ ]:


# Review of Frequency Distribution
plt.figure(figsize=(15,10))
sns.distplot(df_frequency.query("Frequency<1000")["Frequency"])
plt.title("Frequency Distribution")
plt.xlabel("Frequency")
plt.ylabel("Count")


# In[ ]:


# Customer Segmentation based on Frequency
x=df_frequency[["Frequency"]]
k_model_frequency=KMeans(n_clusters=4)
k_model_frequency.fit(x)
k_model_frequency_predict=k_model_frequency.predict(x)
df_frequency["FrequencyCluster"]=k_model_frequency_predict
df_frequency.head()


# In[ ]:


# Statistical Analysis of clusters based on frequency
df_frequency.groupby(["FrequencyCluster"])["Frequency"].describe()


# Cluster 0 has the most customers and Cluster 2 has the least based on Frequency. Based on this segmentation we see that cluster 0 has the least frequency customers however has the most amount of customers. 
# 
# Finally, we can segment our customers based on their monetary value.

# In[ ]:


df_customer_revenue=df.groupby(["CustomerID"])["Revenue"].sum().reset_index()
df_customer_revenue=pd.merge(df_user, df_customer_revenue, on="CustomerID")
df_customer_revenue.head()


# In[ ]:


# Revenue Distribution
plt.figure(figsize=(15,10))
sns.distplot(df_customer_revenue.query("Revenue < 10000")["Revenue"])


# In[ ]:


# Segmenting Customers Based on their Monetary Value
a=df_customer_revenue[["Revenue"]]
k_model_revenue=KMeans(n_clusters=4)
k_model_revenue.fit(a)
k_model_revenue_pred=k_model_revenue.predict(a)
df_customer_revenue["RevenueCluster"]=k_model_revenue_pred
df_customer_revenue.groupby(["RevenueCluster"])["Revenue"].describe()


# Based on this analysis, our main cluster based on the customers monetary value is Cluster 0. 

# ## Conclusion
# 
# With the recency, frequency, monetary value segmentation and our exploratory analysis of our customers, we can further decide our purchase cycle of our products, prioritize and define our marketing campaigns. For example, we can look at our current marketing campaigns, inventory purchase strategies and operations for the months of January and October to see if we can entice more new users in order to turn the negative revenue growth of new users to positive. We can investigate the histroical marketing campaigns in order to see what impact they provide for each recency and frequency clusters. We can further apply this to our main business goal to scale the business. We can also assign a 1-10 score for each category of RFM. For example customer will get a recency score of 1 if they havent made a purchase for a year and gets a recency score of 10 if they made a purchase wihtin the last month. We can further add all these scores for each category and create a customer value for each cluster. These actions will contribute drastically to our success for scalibility of our business.

# In[ ]:




