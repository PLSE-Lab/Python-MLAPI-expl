#!/usr/bin/env python
# coding: utf-8

# # About The Project

# In this project, you have been hired as a data scientist at a bank and you have been provided with extensive data on the bank's customers for the past 6 monthhs.
# 
# Data include transactions frequency , amount , tenure etc,
# 
# The bank marketing team would like to leverage AI/ML to launch a targeted marketing ad campaign that is tailored to specific group of customers.
# 
# In order for this campaign to be successful , the bank has divided its customers into at least 3 distinctive groups.
# 
# This porcess is known as "maketing segmentation" and it crucial for maximizing marketing campaign conversion rate.

# ## IMPORT LIBRARIES AND DATASETS

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


# You have to include the full link to the csv file containing your dataset
creditcard_df = pd.read_csv('/kaggle/input/marketing_data.csv')

# CUSTID: Identification of Credit Card holder 
# BALANCE: Balance amount left in customer's account to make purchases
# BALANCE_FREQUENCY: How frequently the Balance is updated, score between 0 and 1 (1 = frequently updated, 0 = not frequently updated)
# PURCHASES: Amount of purchases made from account
# ONEOFFPURCHASES: Maximum purchase amount done in one-go
# INSTALLMENTS_PURCHASES: Amount of purchase done in installment
# CASH_ADVANCE: Cash in advance given by the user
# PURCHASES_FREQUENCY: How frequently the Purchases are being made, score between 0 and 1 (1 = frequently purchased, 0 = not frequently purchased)
# ONEOFF_PURCHASES_FREQUENCY: How frequently Purchases are happening in one-go (1 = frequently purchased, 0 = not frequently purchased)
# PURCHASES_INSTALLMENTS_FREQUENCY: How frequently purchases in installments are being done (1 = frequently done, 0 = not frequently done)
# CASH_ADVANCE_FREQUENCY: How frequently the cash in advance being paid
# CASH_ADVANCE_TRX: Number of Transactions made with "Cash in Advance"
# PURCHASES_TRX: Number of purchase transactions made
# CREDIT_LIMIT: Limit of Credit Card for user
# PAYMENTS: Amount of Payment done by user
# MINIMUM_PAYMENTS: Minimum amount of payments made by user  
# PRC_FULL_PAYMENT: Percent of full payment paid by user
# TENURE: Tenure of credit card service for user


# In[ ]:


creditcard_df


# In[ ]:


creditcard_df.info()
# Let's apply info and get additional insights on our dataframe
# 18 features with 8950 points  


# ##### Let's apply describe() and get more statistical insights on our dataframe
# Mean balance is $1564           
# 
# Balance frequency is frequently updated on average ~0.9
# 
# Purchases average is $1000
# 
# One off purchase average is ~$600
# 
# Average purchases frequency is around 0.5
# 
# average ONEOFF_PURCHASES_FREQUENCY, PURCHASES_INSTALLMENTS_FREQUENCY, and CASH_ADVANCE_FREQUENCY are generally low
# 
# Average credit limit ~ 4500
# 
# Percent of full payment is 15%
# 
# Average tenure is 11 years

# In[ ]:


creditcard_df.describe()


# Obtain the features (row) of the customer who made the maximim "ONEOFF_PURCHASES"
# 
# Obtain the features of the customer who made the maximum cash advance transaction? how many cash advance transactions did that customer make? how often did he/she pay their bill?

# In[ ]:


creditcard_df[creditcard_df['ONEOFF_PURCHASES'] == 40761.25]


# In[ ]:


creditcard_df['CASH_ADVANCE'].max()


# ## VISUALIZE AND EXPLORE DATASET

# In[ ]:


creditcard_df[creditcard_df['CASH_ADVANCE'] == 47137.211760000006]


# In[ ]:


# Let's see if we have any missing data, luckily we don't have many!
sns.heatmap(creditcard_df.isnull(), yticklabels = False, cbar = False, cmap="Blues")


# In[ ]:


creditcard_df.isnull().sum()


# In[ ]:


# Fill up the missing elements with mean of the 'MINIMUM_PAYMENT' 
creditcard_df.loc[(creditcard_df['MINIMUM_PAYMENTS'].isnull() == True), 'MINIMUM_PAYMENTS'] = creditcard_df['MINIMUM_PAYMENTS'].mean()


# 
# Fill out missing elements in the "CREDIT_LIMIT" column
# Double check and make sure that no missing elements are present

# In[ ]:


creditcard_df.fillna({'CREDIT_LIMIT': creditcard_df.CREDIT_LIMIT.mean()} , inplace= True)


# In[ ]:


creditcard_df.isnull().sum()


# In[ ]:


# Let's see if we have duplicated entries in the data
creditcard_df.duplicated().sum()


# 
# Drop Customer ID column 'CUST_ID' and make sure that the column has been removed from the dataframe

# In[ ]:


creditcard_df.drop('CUST_ID' , axis = 1 , inplace = True)


# In[ ]:


creditcard_df


# In[ ]:


n = len(creditcard_df.columns)
n


# In[ ]:


creditcard_df.columns


# In[ ]:


# distplot combines the matplotlib.hist function with seaborn kdeplot()
# KDE Plot represents the Kernel Density Estimate
# KDE is used for visualizing the Probability Density of a continuous variable. 
# KDE demonstrates the probability density at different values in a continuous variable. 

# Mean of balance is $1500
# 'Balance_Frequency' for most customers is updated frequently ~1
# For 'PURCHASES_FREQUENCY', there are two distinct group of customers
# For 'ONEOFF_PURCHASES_FREQUENCY' and 'PURCHASES_INSTALLMENT_FREQUENCY' most users don't do one off puchases or installment purchases frequently 
# Very small number of customers pay their balance in full 'PRC_FULL_PAYMENT'~0
# Credit limit average is around $4500
# Most customers are ~11 years tenure

plt.figure(figsize=(10,150))
for i in range(len(creditcard_df.columns)):
  plt.subplot(17, 1, i+1)
  sns.distplot(creditcard_df[creditcard_df.columns[i]], kde_kws={'color': 'b', 'lw': 3, 'label': 'KDE' , 'bw' : 1.5}, hist_kws={'color': 'g'})
  plt.title(creditcard_df.columns[i])

plt.tight_layout()


# In[ ]:


correlations = creditcard_df.corr()
f , ax = plt.subplots(figsize = (20,10))
sns.heatmap(correlations , annot = True)


# ## FIND THE OPTIMAL NUMBER OF CLUSTERS USING ELBOW METHOD

# - The elbow method is a heuristic method of interpretation and validation of consistency within cluster analysis designed to help find the appropriate number of clusters in a dataset. 
# - If the line chart looks like an arm, then the "elbow" on the arm is the value of k that is the best.

# In[ ]:


# Let's scale the data first
scaler = StandardScaler()
creditcard_df_scaled = scaler.fit_transform(creditcard_df)


# In[ ]:


creditcard_df_scaled.shape


# In[ ]:


creditcard_df_scaled


# In[ ]:


# Index(['BALANCE', 'BALANCE_FREQUENCY', 'PURCHASES', 'ONEOFF_PURCHASES',
#       'INSTALLMENTS_PURCHASES', 'CASH_ADVANCE', 'PURCHASES_FREQUENCY',
#       'ONEOFF_PURCHASES_FREQUENCY', 'PURCHASES_INSTALLMENTS_FREQUENCY',
#       'CASH_ADVANCE_FREQUENCY', 'CASH_ADVANCE_TRX', 'PURCHASES_TRX',
#       'CREDIT_LIMIT', 'PAYMENTS', 'MINIMUM_PAYMENTS', 'PRC_FULL_PAYMENT',
#       'TENURE'], dtype='object')

scores_1 = []
range_values = range(1,20)
for i in range_values:
    kmeans = KMeans(n_clusters =i)
    kmeans.fit(creditcard_df_scaled)
    
    scores_1.append(kmeans.inertia_)
plt.plot(scores_1 , 'bx-')

# From this we can observe that, 4th cluster seems to be forming the elbow of the curve. 
# However, the values does not reduce linearly until 8th cluster. 
# Let's choose the number of clusters to be 7 or 8.


# In[ ]:


# scores_1 = []
# range_values = range(1,20)
# for i in range_values:
#     kmeans = KMeans(n_clusters =i)
#     kmeans.fit(creditcard_df_scaled[: , :7])
    
#     scores_1.append(kmeans.inertia_)
# plt.plot(scores_1 , 'bx-')


# ## APPLY K-MEANS METHOD

# In[ ]:


kmeans = KMeans(7)
kmeans.fit(creditcard_df_scaled)
labels = kmeans.labels_


# In[ ]:


kmeans.cluster_centers_.shape


# In[ ]:


cluster_centers = pd.DataFrame(data = kmeans.cluster_centers_, columns = [creditcard_df.columns])
cluster_centers


# In[ ]:


# In order to understand what these numbers mean, let's perform inverse transformation
cluster_centers = scaler.inverse_transform(cluster_centers)
cluster_centers = pd.DataFrame(data = cluster_centers, columns = [creditcard_df.columns])
cluster_centers

# First Customers cluster (Transactors): Those are customers who pay least amount of intrerest charges and careful with their money, Cluster with lowest balance ($104) and cash advance ($303), Percentage of full payment = 23%
# Second customers cluster (revolvers) who use credit card as a loan (most lucrative sector): highest balance ($5000) and cash advance (~$5000), low purchase frequency, high cash advance frequency (0.5), high cash advance transactions (16) and low percentage of full payment (3%)
# Third customer cluster (VIP/Prime): high credit limit $16K and highest percentage of full payment, target for increase credit limit and increase spending habits
# Fourth customer cluster (low tenure): these are customers with low tenure (7 years), low balance 


# In[ ]:


labels.shape # Labels associated to each data point


# In[ ]:


labels.max()


# In[ ]:


labels.min()


# In[ ]:


y_kmeans = kmeans.fit_predict(creditcard_df_scaled)
y_kmeans


# In[ ]:


# concatenate the clusters labels to our original dataframe
creditcard_df_cluster = pd.concat([creditcard_df, pd.DataFrame({'cluster':labels})], axis = 1)
creditcard_df_cluster.head()


# In[ ]:


# Plot the histogram of various clusters
for i in creditcard_df.columns:
  plt.figure(figsize = (35, 5))
  for j in range(7):
    plt.subplot(1,7,j+1)
    cluster = creditcard_df_cluster[creditcard_df_cluster['cluster'] == j]
    cluster[i].hist(bins = 20)
    plt.title('{}    \nCluster {} '.format(i,j))
  
  plt.show()


# ## APPLY PRINCIPAL COMPONENT ANALYSIS AND VISUALIZE THE RESULTS

# In[ ]:


# Obtain the principal components 
pca = PCA(n_components=2)
principal_comp = pca.fit_transform(creditcard_df_scaled)
principal_comp


# In[ ]:


# Create a dataframe with the two components
pca_df = pd.DataFrame(data = principal_comp, columns =['pca1','pca2'])
pca_df.head()


# In[ ]:


# Concatenate the clusters labels to the dataframe
pca_df = pd.concat([pca_df,pd.DataFrame({'cluster':labels})], axis = 1)
pca_df.head()


# In[ ]:


plt.figure(figsize=(10,10))
ax = sns.scatterplot(x="pca1", y="pca2", hue = "cluster", data = pca_df, palette =['red','green','blue','pink','yellow','gray','purple'])
plt.show()

