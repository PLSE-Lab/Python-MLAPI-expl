#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


credit_df=pd.read_csv("../input/CreditCardUsage.csv")
credit_df.shape


# In[ ]:


credit_df.describe().transpose()


# In[ ]:


credit_df.info()


# In[ ]:


credit_df.head(10)


# In[ ]:


credit_df.isna().sum()


# In[ ]:


credit_df["MINIMUM_PAYMENTS"]=credit_df.MINIMUM_PAYMENTS.fillna(np.mean(credit_df.MINIMUM_PAYMENTS))
credit_df["CREDIT_LIMIT"]=credit_df.CREDIT_LIMIT.fillna(np.mean(credit_df.CREDIT_LIMIT))
credit_df.isna().sum()


# In[ ]:


credit_df.cov()


# In[ ]:


credit_corr=credit_df.corr()


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


plt.figure(figsize=(20,10))
sns.heatmap(credit_corr,vmin=-1,vmax=1,center=0,annot=True)


# **BALANCE HAVE CORRELATION WITH CASH_ADVANCE(0.5),CASH_ADVANCE_FREQUENCE(0.45),CASH_ADVANCE_TXN(0.39),CREDIT_LIMIT(0.53) **
# 
# **PURCHASES HAVE CORRELATION WITH ONE-OFF-PURCHASE(0.92),INSTALLEMENT_PURCHASE(0.68),PURCHASE_TXN(0.69),PAYMENT(0.6) **
# 
# **ONE-OFF PURCHASE HAVE CORRELATION WITH ONE-OFF-PURCHASE-FREQUENCE(0.52),PURCHASE_TXN(0.55),PAYMENTS(0.57)**
# 
# **INSTALLEMENT_PURCHASE HAVE CORRELATION WITH INSTALLEMENT_PURCHASE-FREQUENCE(0.51),& PURCHASE_TXN(0.63)**
# 
# **CASH-ADVANCE-FREQUENCY HAVE CORRELATION WITH CASH_ADVANCE_FREQUENCE (0.63) & CASH_ADVANCE_TXN(0.66)**
# 
# **PURCHASE-FREQUENCE HAVE CORRELATION WITH INSTALLEMENT_PURCHASE-FREQUENCE(.86) & PURCHASE_TXN(.57)**
# 
# **ONE-OFF-PURCHASE-FREQUENCE HAVE CORRELATION WITH PURCHASE_TXN(0.54)**
# 
# **INSTALLEMENT_PURCHASE-FREQUENCE HAVE CORRELATION WITH PURCHASE_TXN(0.53)**

# In[ ]:


df = credit_df.drop('CUST_ID', axis=1)
df.head()


# In[ ]:


from sklearn.preprocessing import StandardScaler
X = df.values[:]
X = np.nan_to_num(X)
Clus_dataSet = StandardScaler().fit_transform(X)
Clus_dataSet


# In[ ]:


from sklearn.cluster import KMeans 
from sklearn.datasets.samples_generator import make_blobs 


# In[ ]:


clusterNum = 7
k_means = KMeans(init = "k-means++", n_clusters = clusterNum, n_init = 12)
k_means.fit(X)
labels = k_means.labels_
centroids = k_means.cluster_centers_
print(labels)
print(centroids)


# In[ ]:


Nc = range(1, 20)
kmeans = [KMeans(n_clusters=i) for i in Nc]
kmeans
score = [kmeans[i].fit(X).score(X) for i in range(len(kmeans))]
score
plt.plot(Nc,score)
plt.xlabel('Number of Clusters')
plt.ylabel('Score')
plt.title('Elbow Curve')
plt.show()


# In[ ]:


Nc = range(1, 20)
kmeans = [KMeans(n_clusters=i) for i in Nc]
kmeans
score = [kmeans[i].fit(X).inertia_ for i in range(len(kmeans))]
score
plt.plot(Nc,score)
plt.xlabel('Number of Clusters')
plt.ylabel('Sum of within sum square')
plt.title('Elbow Curve')
plt.show()


# In[ ]:


df["Clus_km"] = labels
df.head(5)


# In[ ]:


df.columns
#df.iloc[:,:-1]
#df.iloc[:,-1]


# In[ ]:


plt.figure(figsize=(10,50))
key_col=['BALANCE',  'PURCHASES', 'ONEOFF_PURCHASES',
       'INSTALLMENTS_PURCHASES', 'CASH_ADVANCE',  'CASH_ADVANCE_TRX', 'PURCHASES_TRX',
       'CREDIT_LIMIT', 'PAYMENTS', 'MINIMUM_PAYMENTS']
sns.pairplot(data=df,vars=key_col,hue='Clus_km')


# In[ ]:


#fig, ax = plt.subplots(15, 15,figsize=(30, 40))
#for subplot,tuple_col in zip(ax,col_combination):
    #print(tuple_col[0],tuple_col[1])
    #area = np.pi * ( X[:, 1])**2 
 #   plt.scatter(x=tuple_col[0],y=tuple_col[1], label='Clus_km', alpha=0.5,ax=subplot)
  #  plt.show()
sns.lmplot(data=df,x='BALANCE',y='CASH_ADVANCE',hue='Clus_km')
sns.lmplot(data=df,x='BALANCE',y='CREDIT_LIMIT',hue='Clus_km')
sns.lmplot(data=df,x='BALANCE',y='PAYMENTS',hue='Clus_km')

sns.lmplot(data=df,x='PURCHASES',y='ONEOFF_PURCHASES',hue='Clus_km')
sns.lmplot(data=df,x='PURCHASES',y='INSTALLMENTS_PURCHASES',hue='Clus_km')
sns.lmplot(data=df,x='PURCHASES',y='PAYMENTS',hue='Clus_km')

sns.lmplot(data=df,x='ONEOFF_PURCHASES',y='ONEOFF_PURCHASES_FREQUENCY',hue='Clus_km')
sns.lmplot(data=df,x='ONEOFF_PURCHASES',y='PURCHASES_TRX',hue='Clus_km')
sns.lmplot(data=df,x='ONEOFF_PURCHASES',y='PAYMENTS',hue='Clus_km')

sns.lmplot(data=df,x='INSTALLMENTS_PURCHASES',y='PURCHASES_INSTALLMENTS_FREQUENCY',hue='Clus_km')
sns.lmplot(data=df,x='INSTALLMENTS_PURCHASES',y='PURCHASES_TRX',hue='Clus_km')

sns.lmplot(data=df,x='CASH_ADVANCE',y='CASH_ADVANCE_FREQUENCY',hue='Clus_km')
sns.lmplot(data=df,x='CASH_ADVANCE',y='CASH_ADVANCE_TRX',hue='Clus_km')

sns.lmplot(data=df,x='PURCHASES_FREQUENCY',y='PURCHASES_INSTALLMENTS_FREQUENCY',hue='Clus_km')
sns.lmplot(data=df,x='PURCHASES_FREQUENCY',y='PURCHASES_TRX',hue='Clus_km')


sns.lmplot(data=df,x='ONEOFF_PURCHASES_FREQUENCY',y='PURCHASES_TRX',hue='Clus_km')
sns.lmplot(data=df,x='PURCHASES_INSTALLMENTS_FREQUENCY',y='PURCHASES_TRX',hue='Clus_km')


# **This visualization give little bit inference on columns considered for naming column. lets try scatter plot considering key attributes**

# In[ ]:


#area = np.pi * ( X[:, 13])**2  
#colors = ['b', 'c', 'y', 'm', 'r']

#lo = plt.scatter(X[:,0], X[:,2], marker='x', c=labels.astype(np.float))
#ll = plt.scatter(X[:,0], X[:,2], marker='o', c=labels.astype(np.float))
#l  = plt.scatter(X[:,0], X[:,2], marker='o', c=labels.astype(np.float))
#a  = plt.scatter(X[:,0], X[:,2], marker='o', c=labels.astype(np.float))
#h  = plt.scatter(X[:,0], X[:,2], marker='o', c=labels.astype(np.float))
#hh = plt.scatter(X[:,0], X[:,2], marker='o', c=labels.astype(np.float))
#ho = plt.scatter(X[:,0], X[:,2], marker='x', c=labels.astype(np.float))

#plt.legend((lo, ll, l, a, h, hh, ho),
#           ('Low Outlier', 'LoLo', 'Lo', 'Average', 'Hi', 'HiHi', 'High Outlier'),
#           scatterpoints=1,
#           loc='upper right',
#           ncol=3,
#           fontsize=8)

plt.scatter(X[:,0], X[:,2], c=labels.astype(np.float), alpha=0.5)
plt.xlabel('BALANCE', fontsize=18)
plt.ylabel('PURCHASES', fontsize=16)



plt.show()


# In[ ]:


#area = np.pi * ( X[:, 13])**2  
plt.scatter(X[:,0], X[:,4], c=labels.astype(np.float), alpha=0.5)
plt.xlabel('BALANCE', fontsize=18)
plt.ylabel('INSTALLEMENT_PURCHASES', fontsize=16)

plt.show()


# In[ ]:


#area = np.pi * ( X[:, 13])**2  
plt.scatter(X[:,0], X[:,12], c=labels.astype(np.float), alpha=0.5)
plt.xlabel('BALANCE', fontsize=18)
plt.ylabel('CREDIT_LIMIT', fontsize=16)

plt.show()


# In[ ]:


#area = np.pi * ( X[:, 13])**2  
plt.scatter(X[:,0], X[:,13], c=labels.astype(np.float), alpha=0.5)
plt.xlabel('BALANCE', fontsize=18)
plt.ylabel('PAYMENT', fontsize=16)

plt.show()


# In[ ]:


plt.scatter(X[:,2], X[:,12], c=labels.astype(np.float), alpha=0.5)
plt.xlabel('PURCHASES', fontsize=18)
plt.ylabel('CREDIT_LIMIT', fontsize=16)

plt.show()


# In[ ]:


#area = np.pi * ( X[:, 13])**2  
plt.scatter(X[:,2], X[:,13], c=labels.astype(np.float), alpha=0.5)
plt.xlabel('PURCHASES', fontsize=18)
plt.ylabel('PAYMENT', fontsize=16)

plt.show()


# In[ ]:


plt.scatter(X[:,3], X[:,12], c=labels.astype(np.float), alpha=0.5)
plt.xlabel('ONEOFF_PURCHASES', fontsize=18)
plt.ylabel('CREDIT_LIMIT', fontsize=16)

plt.show()


# In[ ]:


#area = np.pi * ( X[:, 13])**2  
plt.scatter(X[:,3], X[:,13], c=labels.astype(np.float), alpha=0.5)
plt.xlabel('ONEOFF_PURCHASES', fontsize=18)
plt.ylabel('PAYMENT', fontsize=16)

plt.show()


# In[ ]:


#area = np.pi * ( X[:, 13])**2  
plt.scatter(X[:,4], X[:,12], c=labels.astype(np.float), alpha=0.5)
plt.xlabel('INSTALLEMENT_PURCHASES', fontsize=18)
plt.ylabel('CREDIT_LIMIT', fontsize=16)

plt.show()


# In[ ]:


#area = np.pi * ( X[:, 13])**2  
plt.scatter(X[:,4], X[:,13], c=labels.astype(np.float), alpha=0.5)
plt.xlabel('INSTALLEMENT_PURCHASES', fontsize=18)
plt.ylabel('PAYMENT', fontsize=16)

plt.show()


# In[ ]:


plt.scatter(X[:,5], X[:,12], c=labels.astype(np.float), alpha=0.5)
plt.xlabel('CASH_ADVANCE', fontsize=18)
plt.ylabel('CREDIT_LIMIT', fontsize=16)

plt.show()


# In[ ]:


plt.scatter(X[:,12], X[:,13], c=labels.astype(np.float), alpha=0.5)
plt.xlabel('CREDIT_LIMIT', fontsize=18)
plt.ylabel('PAYMENT', fontsize=16)

plt.show()


# In[ ]:


df.groupby('Clus_km').mean().sort_values(by='BALANCE')


# **Cluster-5 : BALANCE -M  ,PURCHASES -L  ,ONE-OFF PURCHASES -VVL,INSTALLMENT PURCHASE -H  ,CASH ADVANCE -M  ,CREDIT_LIMIT -VL ,PAYMENT -VL ,MINIIMUM PAYMENT -VVH**
# 
# **Cluster-1 : BALANCE -VVL,PURCHASES -VVL,ONE-OFF PURCHASES -VL ,INSTALLMENT PURCHASE -VVL,CASH ADVANCE -VVL,CREDIT_LIMIT -VVL,PAYMENT -VVL,MINIIMUM PAYMENT -VVL**
# 
# **Cluster-6 : BALANCE -VL ,PURCHASES -VL ,ONE-OFF PURCHASES -L  ,INSTALLMENT PURCHASE -VL ,CASH ADVANCE -L  ,CREDIT_LIMIT -L  ,PAYMENT -L  ,MINIIMUM PAYMENT -VL **
# 
# **Cluster-3 : BALANCE -VH ,PURCHASES -VVH,ONE-OFF PURCHASES -VVH,INSTALLMENT PURCHASE -VVH,CASH ADVANCE -H  ,CREDIT_LIMIT -VVH,PAYMENT -VVH,MINIIMUM PAYMENT -VH **
# 
# **Cluster-2 : BALANCE -VVH,PURCHASES -M  ,ONE-OFF PURCHASES -M  ,INSTALLMENT PURCHASE -L  ,CASH ADVANCE -VH ,CREDIT_LIMIT -VH ,PAYMENT -M  ,MINIIMUM PAYMENT -M  **
# 
# **Cluster-4 : BALANCE -L  ,PURCHASES -VH ,ONE-OFF PURCHASES -VH ,INSTALLMENT PURCHASE -VH ,CASH ADVANCE -VL ,CREDIT_LIMIT -M  ,PAYMENT -H  ,MINIIMUM PAYMENT -L **
# 
# **Cluster-0 : BALANCE -H  ,PURCHASES -H  ,ONE-OFF PURCHASES -H ,INSTALLMENT PURCHASE -M  ,CASH ADVANCE -VVH,CREDIT_LIMIT -H  ,PAYMENT -VH ,MINIIMUM PAYMENT -H  **
# 
# **INFERENCE**
# 
# With above groups we see below inferences
# 
# Cluster-3 group of people with their high balance they tends to spend more and they able to pay very high amount(very good credit history)
# Cluster-0 group of people with high balance they tends to spend high but medium installement purchase  and they maintain good credit history with both payment and minimum payment (Very Good credit history)
# 
# Cluster-4 group of people inspite of low balance they tends to purchase more and they pay high but with low minimum payment(good credit history).
# Cluster-2 group of people with very very high balance but they tends spend moderate and they pay moderate payment only (Moderate credit history)
# 
# Cluster-5 group of people with moderate balance they do installment purchase high with high pay in minimum payment(Can be candiate of installement )
# Cluster-6 group of people with very low balance they tends to purchase low and pay low(Not good Credit history)
# Cluster-1 group of people with very very low balance they also purchase and pay low (Bad credit history)
# 
# > CONCLUSION
# > With above inference we see good credit history for 3,6,5 and for cluster 5 we can increase credit limit.
# For cluster-0 they more interested with installement purchase so we can avail more option of EMI for him

# In[ ]:





# In[ ]:




