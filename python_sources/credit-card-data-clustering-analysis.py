#!/usr/bin/env python
# coding: utf-8

# **A simple clustering example**
# The data provider @arjunbhasin2013 says: 
# > This case requires to develop a customer segmentation to define marketing strategy. The sample Dataset summarizes the usage behavior of about 9000 active credit card holders during the last 6 months. The file is at a customer level with 18 behavioral variables.
# 
# As usual we begin by importing libraries and the data. We'll check for missing values.

# In[10]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data =pd.read_csv("../input/CC GENERAL.csv")
missing = data.isna().sum()
print(missing)


# Since the number of missing values is low (the total number of samples is 8950), we'll impute with median of the columns. 
# 
# We'll then use the elbow method to find a good number of clusters with the KMeans++ algorithm

# In[11]:


data = data.fillna( data.median() )

# Let's assume we use all cols except CustomerID
vals = data.iloc[ :, 1:].values

from sklearn.cluster import KMeans
# Use the Elbow method to find a good number of clusters using WCSS
wcss = []
for ii in range( 1, 30 ):
    kmeans = KMeans(n_clusters=ii, init="k-means++", n_init=10, max_iter=300) 
    kmeans.fit_predict( vals )
    wcss.append( kmeans.inertia_ )
    
plt.plot( wcss, 'ro-', label="WCSS")
plt.title("Computing WCSS for KMeans++")
plt.xlabel("Number of clusters")
plt.ylabel("WCSS")
plt.show()


# Let's choose n=8 clusters. As it's difficult to visualize clusters when we have more than 2-dimensions, we'll see if Seaborn's pairplot can show how the clusters are segmenting the samples.

# In[16]:


kmeans = KMeans(n_clusters=8, init="k-means++", n_init=10, max_iter=300) 
y_pred = kmeans.fit_predict( vals )

# As it's difficult to visualise clusters when the data is high-dimensional - we'll see
# if Seaborn's pairplot can help us see how the clusters are separating out the samples.   
import seaborn as sns
data["cluster"] = y_pred
cols = list(data.columns)
cols.remove("CUST_ID")

sns.pairplot( data[ cols ], hue="cluster")


# Repeat but with only those columns that clustering seems to have separated the samples more clearly on - this will make each subplot larger.

# In[15]:


best_cols = ["BALANCE", "PURCHASES", "CASH_ADVANCE","CREDIT_LIMIT", "PAYMENTS", "MINIMUM_PAYMENTS"]
kmeans = KMeans(n_clusters=8, init="k-means++", n_init=10, max_iter=300) 
best_vals = data[best_cols].iloc[ :, 1:].values
y_pred = kmeans.fit_predict( best_vals )

data["cluster"] = y_pred
best_cols.append("cluster")
sns.pairplot( data[ best_cols ], hue="cluster")


# The goal was to segment the customers in order to define a marketing strategy. Unfortunately the colors of the plots change when this kernel is rerun - but here are some thoughts:
# 
# * **Big Spenders with large Payments** - they make expensive purchases and have a credit limit that is between average and high.  This is only a small group of customers.
# * **Cash Advances with large Payments** - this group takes the most cash advances. They make large payments, but this appears to be a small group of customers.
# * **Medium Spenders with third highest Payments **- the second highest Purchases group (after the Big Spenders).
# * **Highest Credit Limit but Frugal** - this group doesn't make a lot of purchases. It looks like the 3rd largest group of customers.
# * **Cash Advances with Small Payments **- this group likes taking cash advances, but make only small payments. 
# * **Small Spenders and Low Credit Limit** - they have the smallest Balances after the Smallest Spenders, their Credit Limit is in the bottom 3 groups, the second largest group of customers.
# * **Smallest Spenders and Lowest Credit Limit** - this is the group with the lowest credit limit but they don't appear to buy much. Unfortunately this appears to be the largest group of customers.
# * **Highest Min Payments** - this group has the highest minimum payments (which presumably refers to "Min Payment Due" on the monthly statement. This might be a reflection of the fact that they have the second lowest Credit Limit of the groups, so it looks like the bank has identified them as higher risk.)
# 
# So a marketing strategy that targeted the first five groups might be effective. 
# 
