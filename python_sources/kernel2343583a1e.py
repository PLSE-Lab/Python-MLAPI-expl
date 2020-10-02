#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
credit_df=pd.read_csv("../input/ccdata/CC GENERAL.csv")
credit_df.columns


# before clustering the segments, the features need to be normalized as these are on different scales.

# In[ ]:


from sklearn.preprocessing import StandardScaler

credit_df=credit_df.drop("CUST_ID",axis=1)
credit_df.fillna(method="ffill",inplace=True)

scaler= StandardScaler()
scaled_beer_df=scaler.fit_transform(credit_df[[ 'BALANCE', 'BALANCE_FREQUENCY', 'PURCHASES',
       'ONEOFF_PURCHASES', 'INSTALLMENTS_PURCHASES', 'CASH_ADVANCE',
       'PURCHASES_FREQUENCY', 'ONEOFF_PURCHASES_FREQUENCY',
       'PURCHASES_INSTALLMENTS_FREQUENCY', 'CASH_ADVANCE_FREQUENCY',
       'CASH_ADVANCE_TRX', 'PURCHASES_TRX', 'CREDIT_LIMIT', 'PAYMENTS',
       'MINIMUM_PAYMENTS', 'PRC_FULL_PAYMENT', 'TENURE']])
credit_df.head(6)


# In[ ]:


#we'll use elbow curve method to find the optimal no. of clusters
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
get_ipython().run_line_magic('matplotlib', 'inline')

cluster_range=range(1,10)
cluster_errors=[]

for num_clusters in cluster_range:
    clusters=KMeans(num_clusters)
    clusters.fit(scaled_beer_df)
    cluster_errors.append(clusters.inertia_)
    
plt.figure(figsize=(6,4))
plt.plot(cluster_range,cluster_errors,marker="o");


# In[ ]:


k=4

clusters=KMeans(k,random_state=42)
clusters.fit(scaled_beer_df)
credit_df["clusterid"]=clusters.labels_


# In[ ]:


for c in credit_df:
    grid= sn.FacetGrid(credit_df, col='clusterid')
    grid.map(plt.hist, c)


# In[ ]:


credit_df.groupby('clusterid').mean()


# Cluster 0 :
# 
# Balance is very high and it gets updated very frequently as well. no. of purchases are extremely high and majority of their purchases are done either in one-go or in installments. Purchase frequency also very high indicating purchasing happening at high frequency. Also, these have the highest credit limit.

# Cluster 1 :
# 
# Comparitively high balance but the balance does not get updated frequently ie. less no. of transactions. No. of purchases from account are quite low and very low purchases in one go or in installments. Majority of purchases being done by paying cash in advance. Purchase frequency is also quite low.

# Cluster 2:
# 
# Low balance but the balance gets updated frequently ie. more no. of transactions. No of purchases from the account are also quite large and majority of the purchases are done either in one go or in installments but not by paying cash in advance.

# Cluster 3 : 
# 
# Balance is very high and it gets updated very frequently as well. No. of purchases are comparitively less and almost all the purchases are done with cash in advance. Purchase frequency is also quite low.
