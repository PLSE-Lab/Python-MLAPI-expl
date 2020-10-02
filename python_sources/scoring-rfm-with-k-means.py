#!/usr/bin/env python
# coding: utf-8

# # Segmenting customers by RFM Score
# 
# The purpose of this notebook is to use K-Means Clustering to segment a customer base in 3 groups based on their RFM Score.
# 
# 
# ## RFM
# 
# The RFM Score (Recency, Frequency, Monetary Value) is a metric that analyzes the customer based on three data points:
# 
# - **Recency**: How recently the customer made a purchase
# - **Frequency**: How often do they purchase
# - **Monetary Value**: How much they spent
# 
# RFM analysis classifies customers with a numerical ranking of the three categories, with the ideal customer earning the highest on each category.
# 
# 
# ## The Data
# 
# The dataset contains about a year worth of transactions (dec-2010 to dec-2011) from an **online retail company** based on in the UK. More information can be found on [Kaggle](https://www.kaggle.com/vijayuv/onlineretail). 
# 
# 
# 

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# K-Means to cluster the users
from sklearn.cluster import KMeans

# Yellowbrick for Model visualization
from yellowbrick.cluster import KElbowVisualizer


# In[ ]:


# Define the functions
def order_clusters(cluster, target, df, ascending):
    new_cluster = 'new' + cluster
    
    temp = df.groupby(cluster)[target].mean().reset_index()
    temp = temp.sort_values(by=target, ascending=ascending).reset_index(drop=True)
    temp['index'] = temp.index
    
    cluster_df = pd.merge(df, temp[[cluster, 'index']], on=cluster)
    cluster_df = cluster_df.drop([cluster], axis=1)
    cluster_df = cluster_df.rename(columns={'index':cluster})
    
    return cluster_df

def rfm_cluster(df, cluster_variable, n_clusters, ascending):
    
    # Create and fit the k-means 
    model = KMeans(n_clusters=n_clusters)
    model.fit(df[[cluster_variable]])
    
    # predict the cluster and pass it to the dataframe
    df[cluster_variable + 'Cluster'] = model.predict(df[[cluster_variable]])
    
    # order the cluster numbers
    df = order_clusters(cluster_variable + 'Cluster', cluster_variable, df, ascending)
    
    return df


# In[ ]:


# Import the data
df = pd.read_csv(r"../input/onlineretail/OnlineRetail.csv", encoding='cp1252', parse_dates=['InvoiceDate'])

# I'll only keep UK sales
df = df[df.Country == 'United Kingdom']


# In[ ]:


df.head()


# In[ ]:


df.info()


# In[ ]:


maxdate = df['InvoiceDate'].dt.date.max()
mindate = df['InvoiceDate'].dt.date.min()
customers = df['CustomerID'].nunique()
stock = df['StockCode'].nunique()
quantity = df['Quantity'].sum()

print(f'Transactions timeframe: {mindate} to {maxdate}.')
print(f'Unique customers: {customers}.')
print(f'Unique items sold: {stock}.')
print(f'Quantity sold in period {quantity}')


# In[ ]:


# Create a users dataframe to segment
users = pd.DataFrame(df['CustomerID'].unique())
users.columns = ['CustomerID']


# ### Recency
# For the purpose of this project, I'll consider the last day in the dataset as if it were the present day to calculate Recency

# In[ ]:


# Get the latest purchase date for each customer and pass it to a df
max_purchase = df.groupby('CustomerID').InvoiceDate.max().reset_index()
max_purchase.columns = ['CustomerID', 'MaxPurchaseDate']
max_purchase['MaxPurchaseDate'] = max_purchase['MaxPurchaseDate'].dt.date

# Calculate Recency
max_purchase['Recency'] = (
    max_purchase['MaxPurchaseDate'].max() - max_purchase['MaxPurchaseDate']).dt.days 

# Merge the dataframe with the users to get the Recency value for each customer
users = pd.merge(users, max_purchase[['CustomerID', 'Recency']], on='CustomerID')
users.head()


# In[ ]:


# plot a histogram of Recency
fig = plt.figure(figsize=(10, 6))
plt.hist(users['Recency']);


# The distribution of Recency is very skewed which, in this case, is actually good. We see that the majority of clients have bought something between 0 and 50 days ago.

# In[ ]:


# Describe Recency
users['Recency'].describe()


# The number of optimal clusters seems to be 2, but I'll use 3 just for the sake of demonstration

# ### Frequency

# In[ ]:


# calculate the frequency score, that is how 
# frequently the customer buy from the store
frequency_score = df.groupby('CustomerID')['InvoiceDate'].count().reset_index()
frequency_score.columns = ['CustomerID', 'Frequency']


# In[ ]:


users = pd.merge(users, frequency_score, on='CustomerID')
users.head()


# In[ ]:


# Plot the distribution 
plt.hist(users['Frequency']);


# Unlike the recency cluster, a higher median (50%) frequency indicates a better customer.

# ### Monetary Value (Revenue)

# In[ ]:


plt.hist(df['Quantity']);


# There are negative values for `Quantity` these could be returned goods, but the Kaggle page doesn't say anything about it, so I'll exclude those from the analysis just for demonstration.

# In[ ]:


df.drop(df[df['Quantity']<0].index, axis=0, inplace=True)


# In[ ]:


# Calculate revenue for each individual customer
df['Revenue'] = df['UnitPrice']*df['Quantity']


# In[ ]:


# Calculate revenue for each individual customer
df['Revenue'] = df['UnitPrice']*df['Quantity']
revenue = df.groupby('CustomerID')['Revenue'].mean().reset_index()

# Merge the revenue with users dataframe
users = pd.merge(users, revenue, on='CustomerID')


# In[ ]:


# Plot the data
plt.hist(users['Revenue']);


# ## Creating the clusters

# Now, we'll apply K-means clustering to assign the scores. 
# 
# To find out how many clusters we need, we'll apply the Elbow Method on 'Recency' to work as a standard for the entire RFM analysis

# In[ ]:


model = KMeans()

recency_score = users[['Recency']]

visualizer = KElbowVisualizer(model, k=(1, 11))

visualizer.fit(recency_score)
visualizer.show();


# The optimum amount of clusters is 2, but I'll use 3 for this project

# ### Recency Cluster

# In[ ]:


# Create the Recency cluster, smaller recency is 
# better, so we set ascending to False
users = rfm_cluster(users, 'Recency', 3, False)

# Check the df with the clusters
users.head()


# In[ ]:


# Now every customer has been assigned to a cluster based on their Recency
# and the clusters are ordered from lowest to highest
users.groupby('RecencyCluster')['Recency'].describe()


# ### Monetary Value (Revenue) Cluster

# In[ ]:


rfm_cluster(users, 'Revenue', 4, True)


# In[ ]:


users.groupby('RevenueCluster')['Revenue'].describe()


# ### Frequency cluster

# In[ ]:


# Create the Frequency Clusters
users = rfm_cluster(users,'Frequency', 3, True)

# describe the clusters
users.groupby('FrequencyCluster')['Frequency'].describe()


# ## Overall RFM Score

# For this RFM model, all the features will have the same weight, but this can easily be adjusted in the formula below:

# In[ ]:


# Calculate OverallScore
users['OverallScore'] = users['FrequencyCluster'] + users['RevenueCluster'] - users['RecencyCluster'] 

# Show the mean of the features for each OverallScore value
users.groupby('OverallScore')[['Recency', 'Frequency', 'Revenue']].mean()


# We can arbitrarily assign labels for the different Overall Score clusters for the sake of simplicity
# 
# | Value   | Label  |
# |---------|--------|
# | (-2,-1) | Low    |
# | (0,1)   | Medium |
# | (2,4)   | High   |

# In[ ]:


# Create the Segment variable based on the OverallScore
x = users['OverallScore']
conditions = [x<0, x<2]
choices = ['Low', 'Medium']

users['Segment'] = np.select(conditions, choices, default='High')


# In[ ]:


fig = plt.figure(figsize=(10, 6))
plt.title('Segments')
plt.xlabel('Revenue')
plt.ylabel('Frequency')
    
plt.scatter(x=users.query("Segment == 'Low'")['Revenue'],
            y= users.query("Segment == 'Low'")['Frequency'],
            c='green', alpha=0.7, label='Low')

plt.scatter(x=users.query("Segment == 'Medium'")['Revenue'],
            y=users.query("Segment == 'Medium'")['Frequency'],
            c='red', alpha=0.6, label='Medium')

plt.scatter(x=users.query("Segment == 'High'")['Revenue'],
            y=users.query("Segment == 'High'")['Frequency'],
            c='blue', alpha=0.5, label='High')

plt.legend();


# In[ ]:


fig = plt.figure(figsize=(10, 6))
plt.title('Segments')
plt.xlabel('Frequency')
plt.ylabel('Revenue')
    
plt.scatter(x=users.query("Segment == 'Low'")['Frequency'],
            y= users.query("Segment == 'Low'")['Revenue'],
            c='green', alpha=0.7, label='Low')

plt.scatter(x=users.query("Segment == 'Medium'")['Frequency'],
            y=users.query("Segment == 'Medium'")['Revenue'],
            c='red', alpha=0.6, label='Medium')

plt.scatter(x=users.query("Segment == 'High'")['Frequency'],
            y=users.query("Segment == 'High'")['Revenue'],
            c='blue', alpha=0.5, label='High')

plt.legend();


# In[ ]:


fig = plt.figure(figsize=(10, 6))
plt.title('Segments')
plt.xlabel('Recency')
plt.ylabel('Revenue')

plt.scatter(x=users.query("Segment == 'Low'")['Recency'],
            y= users.query("Segment == 'Low'")['Revenue'],
            c='green', alpha=0.7, label='Low')

plt.scatter(x=users.query("Segment == 'Medium'")['Recency'],
            y=users.query("Segment == 'Medium'")['Revenue'],
            c='red', alpha=0.6, label='Medium')

plt.scatter(x=users.query("Segment == 'High'")['Recency'],
            y=users.query("Segment == 'High'")['Revenue'],
            c='blue', alpha=0.5, label='High')

plt.legend();


# In[ ]:


fig = plt.figure(figsize=(10, 6))
plt.title('Segments')
plt.xlabel('Recency')
plt.ylabel('Frequency')
    
plt.scatter(x=users.query("Segment == 'Low'")['Recency'],
            y= users.query("Segment == 'Low'")['Frequency'],
            c='green', alpha=0.7, label='Low')

plt.scatter(x=users.query("Segment == 'Medium'")['Recency'],
            y=users.query("Segment == 'Medium'")['Frequency'],
            c='red', alpha=0.6, label='Medium')

plt.scatter(x=users.query("Segment == 'High'")['Recency'],
            y=users.query("Segment == 'High'")['Frequency'],
            c='blue', alpha=0.5, label='High')

plt.legend();


# We can see the segments behave differently so we can apply adequate strategies to the each group.

# In[ ]:


users.head()

