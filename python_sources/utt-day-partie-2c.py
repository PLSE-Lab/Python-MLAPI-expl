#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# In[ ]:


dateparse = lambda x: pd.datetime.strptime(x, '%m/%d/%Y %H:%M')


# In[ ]:


tx_data = pd.read_csv('/kaggle/input/onlineretail/OnlineRetail.csv', parse_dates=['InvoiceDate'], date_parser=dateparse, encoding = 'unicode_escape')
#If you specify the date_parser, you increase the speed of loading


# In[ ]:


print("Number of lines:" + str(len(tx_data)))
tx_data.head(10)


# In[ ]:


tx_data['InvoiceDate'] = pd.to_datetime(tx_data['InvoiceDate'])


# In[ ]:


tx_data['InvoiceDate'].describe()


# In[ ]:


tx_uk = tx_data[tx_data.Country == 'United Kingdom'].reset_index(drop=True)


# In[ ]:


tx_user = pd.DataFrame(tx_data['CustomerID'].unique())
tx_user.columns = ['CustomerID']


# # Data viz

# In[ ]:


tx_data.plot.scatter(x="Quantity", y="UnitPrice")


# In[ ]:


tx_data.Quantity[tx_data.Quantity < 0].count()


# In[ ]:


tx_data.UnitPrice[tx_data.UnitPrice < 0]


# # Null analysis

# In[ ]:


tx_data.isnull().sum()


# Think carefully about each column.
# 
# When `CustomerID` is null, well we can't use it for the customer segmentation. We can drop it / ignore it.
# 
# If `Description` is null, it can still be useful for our segmentation.

# # Recency

# In[ ]:


tx_max_purchase = tx_uk.groupby('CustomerID').InvoiceDate.max().reset_index()


# In[ ]:


tx_max_purchase.columns = ['CustomerID','MaxPurchaseDate']


# In[ ]:


tx_max_purchase['Recency'] = (tx_max_purchase['MaxPurchaseDate'].max() - tx_max_purchase['MaxPurchaseDate']).dt.days
tx_max_purchase.head()


# In[ ]:


tx_user = pd.merge(tx_user, tx_max_purchase[['CustomerID','Recency']], on='CustomerID')


# In[ ]:


tx_user.head()


# In[ ]:


tx_user.Recency.describe()


# In[ ]:


tx_user.Recency.hist(bins=40)


# In[ ]:


#Using elbow method
from sklearn.cluster import KMeans

sse={}
tx_recency = tx_user[['Recency']]
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(tx_recency)
    tx_recency["clusters"] = kmeans.labels_
    sse[k] = kmeans.inertia_ 
plt.figure()
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of cluster")
plt.show()


# In[ ]:


from sklearn.metrics import silhouette_score
from yellowbrick.cluster import SilhouetteVisualizer
import matplotlib.cm as cm

#Using silhouettes
def plot_silhouettes(X, range_n_clusters):
  for n_clusters in range_n_clusters:
      clusterer = KMeans(n_clusters=n_clusters, random_state=10)
      cluster_labels = clusterer.fit_predict(X)

      # The silhouette_score gives the average value for all the samples.
      # This gives a perspective into the density and separation of the formed
      # clusters
      silhouette_avg = silhouette_score(X, cluster_labels)
      print("For n_clusters =", n_clusters,
            "The average silhouette_score is :", silhouette_avg)

      visualizer = SilhouetteVisualizer(clusterer)
      visualizer.fit(X)        # Fit the data to the visualizer
      plt.show()

tx_recency = tx_user[['Recency']]
range_n_clusters = [2, 3, 4, 5, 6]
plot_silhouettes(tx_recency, range_n_clusters)


# In[ ]:


kmeans = KMeans(n_clusters=4)
kmeans.fit(tx_user[['Recency']])
tx_user['RecencyCluster'] = kmeans.predict(tx_user[['Recency']])


# In[ ]:


tx_user.groupby('RecencyCluster')['Recency'].describe()


# In[ ]:


def order_cluster(cluster_field_name, target_field_name, df, ascending):
    new_cluster_field_name = 'new_' + cluster_field_name
    df_new = df.groupby(cluster_field_name)[target_field_name].mean().reset_index()
    df_new = df_new.sort_values(by=target_field_name, ascending=ascending).reset_index(drop=True)
    df_new['index'] = df_new.index
    df_final = pd.merge(df, df_new[[cluster_field_name,'index']], on=cluster_field_name)
    df_final = df_final.drop([cluster_field_name], axis=1)
    df_final = df_final.rename(columns={"index":cluster_field_name})
    return df_final


# In[ ]:


tx_user = order_cluster('RecencyCluster', 'Recency', tx_user, False)
print(tx_user.head())
print(tx_user.tail())


# In[ ]:


tx_user.groupby('RecencyCluster')['Recency'].describe()


# # Frequency

# In[ ]:


tx_frequency = tx_uk.groupby('CustomerID').InvoiceDate.count().reset_index()


# In[ ]:


tx_frequency.columns = ['CustomerID', 'Frequency']


# In[ ]:


tx_frequency.head()


# In[ ]:


tx_user = pd.merge(tx_user, tx_frequency, on='CustomerID')


# In[ ]:


tx_user.head()


# In[ ]:


tx_user.Frequency.describe()


# In[ ]:


tx_user.Frequency.hist()
#We can't see anything here


# In[ ]:


tx_user.Frequency[tx_user.Frequency < 1000].hist(bins=100)


# In[ ]:


sse={}
tx_frequency = tx_user[['Frequency']].copy()
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(tx_frequency)
    tx_frequency["clusters"] = kmeans.labels_
    sse[k] = kmeans.inertia_ 
plt.figure()
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of cluster")
plt.show()


# In[ ]:


kmeans = KMeans(n_clusters=4)
kmeans.fit(tx_user[['Frequency']])
tx_user['FrequencyCluster'] = kmeans.predict(tx_user[['Frequency']])


# In[ ]:


tx_user.groupby('FrequencyCluster')['Frequency'].describe()


# In[ ]:


tx_user = order_cluster('FrequencyCluster', 'Frequency', tx_user, True)
tx_user.head()


# # Monetary Value

# In[ ]:


tx_uk['Revenue'] = tx_uk['UnitPrice'] * tx_uk['Quantity']


# In[ ]:


tx_revenue = tx_uk.groupby('CustomerID').Revenue.sum().reset_index()


# In[ ]:


tx_revenue.head()


# In[ ]:


tx_user = pd.merge(tx_user, tx_revenue, on='CustomerID')


# In[ ]:


tx_user.Revenue.describe()


# In[ ]:


tx_user.Revenue.hist(bins=50)


# In[ ]:


tx_user.Revenue[tx_user.Revenue < 10000].hist(bins=50)


# In[ ]:


sse={}
tx_revenue = tx_user[['Revenue']].copy()
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(tx_revenue)
    tx_revenue["clusters"] = kmeans.labels_
    sse[k] = kmeans.inertia_ 
plt.figure()
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of cluster")
plt.show()


# In[ ]:


kmeans = KMeans(n_clusters=4)
kmeans.fit(tx_user[['Revenue']])
tx_user['RevenueCluster'] = kmeans.predict(tx_user[['Revenue']])


# In[ ]:


tx_user = order_cluster('RevenueCluster', 'Revenue',tx_user,True)


# In[ ]:


tx_user.groupby('RevenueCluster')['Revenue'].describe()

