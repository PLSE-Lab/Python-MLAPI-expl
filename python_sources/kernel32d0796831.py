#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('reset', '-f')
import pandas as pd
import numpy as np
import os


# In[ ]:


customer=pd.read_csv("../input/customer-segmentation-tutorial-in-python/Mall_Customers.csv")


# In[ ]:


customer.head()


# In[ ]:


customer.shape


# In[ ]:


customer.dtypes


# In[ ]:


NewColumns= {
              'CustomerID'         : 'customerid',
              'Gender'             : 'n_gender',
              'Age'                : 'age',
              'Annual Income (k$)' : 'annual_income',
              'Spending Score (1-100)' : 'spending_score'
            }
customer.rename(NewColumns,inplace='True',axis=1)


# In[ ]:


customer['age'].max()


# In[ ]:


customer['age'].min()


# In[ ]:


customer = customer.astype({'age': 'int32', 'annual_income': 'int32'})

#customer['age']=customer['age'].astype('int32')
#customer['annual_income']=customer['annual_income'].astype('int32')   


# In[ ]:


customer.dtypes


# In[ ]:


customer.drop(columns=['customerid'],inplace=True)


# In[ ]:



customer.head()


# In[ ]:


cust_group=customer.groupby(['n_gender','age']).sum().reset_index()


# In[ ]:


import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


# In[ ]:


plt.figure(figsize=(15,3))
sns.barplot(data=cust_group,x='age',y='spending_score',hue='n_gender')


# In[ ]:


g=sns.FacetGrid(cust_group,row='n_gender',palette='set3',size=6,aspect=2)
g.map(sns.barplot,'age','spending_score',color='#CC0066' )
# check all colors in https://www.rapidtables.com/web/color/RGB_Color.html


# In[ ]:


g=sns.FacetGrid(cust_group,row='n_gender',palette='set3',size=6,aspect=2)
g.map(sns.barplot,'annual_income','spending_score')


# In[ ]:


sns.pairplot(data=cust_group, hue='n_gender')


# In[ ]:


sns.jointplot(x='annual_income',y='spending_score',data=customer,kind='reg')


# In[ ]:


customer


# In[ ]:


from sklearn.preprocessing import LabelEncoder as le
enc=le()


# In[ ]:


customer['n_gender']=enc.fit_transform(customer['n_gender'])


# In[ ]:


customer.head()


# In[ ]:


from sklearn.cluster import KMeans
find_cls=[]
for i in range(1,15):
    kmean = KMeans(n_clusters=i)
    kmean.fit(customer)
    find_cls.append(kmean.inertia_)


# In[ ]:


find_cls


# In[ ]:


fig, axs = plt.subplots(figsize=(12,5))
sns.lineplot(range(1,15),find_cls, ax=axs,marker='X')
axs.axvline(5, ls="--", c="crimson") # CRIMSON is color, ls - line style
axs.axvline(6, ls="--", c="crimson")
plt.grid() #  square lines in back ground
plt.show


# In[ ]:


from sklearn.cluster import KMeans
#kmean_clust_data=[]
kmean=KMeans(n_clusters=1)
kmean.fit(customer)


# In[ ]:


kmean.inertia_


# In[ ]:


kmean=KMeans(n_clusters=2)
kmean.fit(customer)


# In[ ]:


kmean.inertia_


# In[ ]:


from sklearn.cluster import KMeans
find_cls=[]
for i in range(1,15):
    kmean = KMeans(n_clusters=i)
    kmean.fit(customer)
    find_cls.append(kmean.inertia_)


# In[ ]:


find_cls


# In[ ]:


kmean=KMeans(n_clusters=5)   # we found that best clusters are 5
kmean.fit(customer)


# In[ ]:


kmean.cluster_centers_


# In[ ]:


clust_centers=kmean.cluster_centers_


# In[ ]:


customer.head()


# In[ ]:


kmean.labels_


# In[ ]:


customer['center_cluster']=kmean.labels_


# In[ ]:


customer.head()


# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


fig,(ax1,ax2)=plt.subplots(nrows=1, ncols=2,figsize=(15,5))

sns.scatterplot(data=customer, x='age',y='spending_score', ax=ax1, hue='center_cluster',palette='Set1')  # good color use palette =1
sns.scatterplot(data=customer, x='annual_income',y='spending_score', ax=ax2, hue='center_cluster',palette='Set1')


# In[ ]:


fig,(ax1,ax2)=plt.subplots(nrows=1, ncols=2,figsize=(15,5))

ax1.scatter(kmean.cluster_centers_[:,1], kmean.cluster_centers_[:,3],marker='X',color='red')      
ax2.scatter(kmean.cluster_centers_[:,2], kmean.cluster_centers_[:,3],marker='X',color='red')         

#(data=customer, x='age',y='spending_score', ax=ax1, hue='center_cluster',palette='Set1')  # good color use palette =1
#sns.scatterplot(data=customer, x='annual_income',y='spending_score', ax=ax2, hue='center_cluster',palette='Set1')


# In[ ]:


fig,(ax1,ax2)=plt.subplots(nrows=1, ncols=2,figsize=(15,5))

ax1.scatter(kmean.cluster_centers_[:,1], kmean.cluster_centers_[:,3],marker='X',color='red')      
ax2.scatter(kmean.cluster_centers_[:,2], kmean.cluster_centers_[:,3],marker='X',color='red')         

plt.show() # this will remove <matplotlib.collections.PathCollection at 0x2793b435f08>


# In[ ]:


fig,(ax1,ax2)=plt.subplots(nrows=1, ncols=2,figsize=(15,5))

sns.scatterplot(data=customer, x='age',y='spending_score', ax=ax1, hue='center_cluster',palette='Set1')  # good color use palette =1
ax1.scatter(kmean.cluster_centers_[:,1], kmean.cluster_centers_[:,3],marker='X',color='black') # to mege in above plot
sns.scatterplot(data=customer, x='annual_income',y='spending_score', ax=ax2, hue='center_cluster',palette='Set1')
ax2.scatter(kmean.cluster_centers_[:,2], kmean.cluster_centers_[:,3],marker='X',color='black') 
plt.show()


# In[ ]:




