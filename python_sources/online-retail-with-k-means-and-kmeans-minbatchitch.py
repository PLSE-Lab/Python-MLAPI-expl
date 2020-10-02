#!/usr/bin/env python
# coding: utf-8

# 

# # Online Retail Dataset

# ### 1. Business Understanding

# This dataset is about Products in Stock Market. Each Product has a Code, Each Customer will buy a product with a provided code 

# ### Features
# 
# InvoiceNo (numeric)
# 
# StockCode (numeric)
# 
# Gender (text: male, female)
# 
# Job (numeric: 0 - unskilled and non-resident, 1 - unskilled and resident, 2 - skilled, 3 - highly skilled)
# 
# Housing (text: own, rent, or free)
# 
# Saving accounts (text - little, moderate, quite rich, rich)
# 
# Checking account (numeric, in DM - Deutsch Mark)
# 
# Credit amount (numeric, in DM)
# 
# Duration (numeric, in month)
# 
# Purpose (text: car, furniture/equipment, radio/TV, domestic appliances, repairs, education, business, vacation/others)

# In[ ]:


#importing modules
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Objective
# We are trying to find the creditworthiness of the customer on the German Credit DataSet.

# #### 1. Loading the Dataset

# In[ ]:


#Load Dataset
TextFileReader = pd.read_csv("/kaggle/input/onlineretail/OnlineRetail.csv",engine='python', chunksize=100000)  # the number of rows per chunk

DS = []
for df in TextFileReader:
    DS.append(df)

df = pd.concat(DS,sort=False)


# In[ ]:


DS_2 = DS[0]
print (DS_2.columns)
DS_2.head(10)


# #### 2.Descriptive Statistics

# In[ ]:


#Some data Stats
DS_2.shape # Shape
DS_2.info() # information
DS_2.describe() #Summary Stastics


# #### 3. Missing Values identification and handling

# In[ ]:


#Looking out for missing values and handling them
DS_2.isnull().sum()


# In[ ]:


#finding out unique variables
print("Description : ",DS_2.Description.unique())
print("Country  : ",DS_2.Country.unique())


# In[ ]:


# using Pandas function to_numeric to check if there is other value than number it will convert into Nan
DS_2['InvoiceNo'] = pd.to_numeric(DS_2['InvoiceNo'], errors='coerce')

DS_2.isnull().sum()


# In[ ]:


# using Pandas function to_numeric to check if there is other value than number it will convert into Nan

DS_2['StockCode'] = pd.to_numeric(DS_2['StockCode'], errors='coerce')
DS_2.isnull().sum()


# In[ ]:


# replace the matching strings 
DS_2['Description'] = DS_2['Description'].replace('lost', np.NaN)
DS_2.isnull().sum()


# In[ ]:


# Applying abs() to Quantity column which will change negative values to positive 
DS_2['Quantity'] = DS_2['Quantity'].abs() 


# In[ ]:


# Applying abs() to UnitPrice column which will change negative values to positive 
DS_2['UnitPrice'] = DS_2['UnitPrice'].abs() 


# In[ ]:


#Filling Values with mode in all Categorical columns which have NaN values in it
stringcols = DS_2.select_dtypes(exclude=np.number)
for cat in stringcols:
    DS_2[cat] = DS_2[cat].fillna(DS_2[cat].mode().values[0])
DS_2.isnull().sum()


# In[ ]:


#Changing Datatype of InvoiceDate to_datetime 
DS_2['InvoiceDate'] = pd.to_datetime(DS_2['InvoiceDate'])


# In[ ]:


#Filling Values with mean in all number columns which have NaN values in it
numcols = DS_2.select_dtypes(include=np.number)
for cat in numcols:
    DS_2[cat] = DS_2[cat].fillna(DS_2[cat].median())
DS_2.isnull().sum()


# In[ ]:


#label encoder
from numpy import array
from sklearn.preprocessing import LabelEncoder

label_encoder=LabelEncoder()
DS_2['DescriptionCode']=label_encoder.fit_transform(DS_2.Description)
DS_2.head()


# #### 4. Visualize

# In[ ]:


sns.pairplot(DS_2)


# The above diagram shows pairplot of all the numerical features.

# In[ ]:


#create correlation
corr = DS_2.corr(method = 'pearson')

#convert correlation to numpy array
mask = np.array(corr)

#to mask the repetitive value for each pair
mask[np.tril_indices_from(mask)] = False
fig, ax = plt.subplots(figsize = (15,12))
fig.set_size_inches(15,15)
sns.heatmap(corr, mask = mask, vmax = 0.9, square = True, annot = True)


# The heatmap shows best correlation between Stock Code and Customer ID.

# In[ ]:


DS_2_cluster = pd.DataFrame()
DS_2_cluster['Quantity'] = DS_2['Quantity']
DS_2_cluster['UnitPrice'] = DS_2['UnitPrice']
DS_2_cluster['DescriptionCode'] = DS_2['DescriptionCode']
DS_2_cluster.head()


# Plotting Box plots to find outliers - The box plot shows outliers in the numerical features

# In[ ]:


BOLD = '\033[1m'
END = '\033[0m'
for col in numcols:

    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(12,3))
    sns.boxplot(DS_2[col], linewidth=1, ax = ax1)
    DS_2[col].hist(ax = ax2)

    plt.tight_layout()
    plt.show()
    print(BOLD+col.center(115)+END)


# In[ ]:


fig, (ax1, ax2, ax3) = plt.subplots(3,1, figsize=(8,8))
sns.distplot(DS_2["Quantity"], ax=ax1)
sns.distplot(DS_2["UnitPrice"], ax=ax2)
sns.distplot(DS_2["DescriptionCode"], ax=ax3)
plt.tight_layout()
plt.legend()


# ### Positive Skewness
# means when the tail on the right side of the distribution is longer or fatter. The mean and median will be greater than the mode.

# # Feature Engineering
# ## IQR
# We can use IQR to reduce the outliers and distribution skewness

# In[ ]:


Q1 = DS_2_cluster.quantile(0.25)
Q3 = DS_2_cluster.quantile(0.75)
IQR = Q3 - Q1
print(IQR)


# In[ ]:


DS_2_clusters = DS_2_cluster[~((DS_2_cluster < (Q1 - 1.5 * IQR)) |(DS_2_cluster > (Q3 + 1.5 * IQR))).any(axis=1)]
DS_2_clusters


# In[ ]:


fig, (ax1, ax2, ax3) = plt.subplots(3,1, figsize=(8,8))
sns.distplot(DS_2_clusters["Quantity"], ax=ax1)
sns.distplot(DS_2_clusters["UnitPrice"], ax=ax2)
sns.distplot(DS_2_clusters["DescriptionCode"], ax=ax3)
plt.tight_layout()


# This has caused the skewness to be removed.

# In[ ]:


#Fit and transform
DS_2_clusters.head()


# In[ ]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
cluster_scaled = scaler.fit_transform(DS_2_clusters)


# # K-means

# K-means
# First we use the Elbow Method to determine the optimal k value for the k-means

# In[ ]:


from sklearn.cluster import KMeans

Sum_of_squared_distances = []
K = range(1,15)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(cluster_scaled)
    Sum_of_squared_distances.append(km.inertia_)
plt.figure(figsize=(20,5))
plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()


# From the figure above we can see that the most optimal values are 4. So we choose 4 as the k values of the k-means model.

# In[ ]:


from mpl_toolkits.mplot3d import Axes3D

model = KMeans(n_clusters=4)
model.fit(cluster_scaled)
kmeans_labels = model.labels_

fig = plt.figure(num=None, figsize=(15, 10), dpi=80, facecolor='w', edgecolor='k')
ax = plt.axes(projection="3d")

ax.scatter3D(DS_2_clusters['DescriptionCode'],DS_2_clusters['Quantity'],DS_2_clusters['UnitPrice'],c=kmeans_labels, cmap='rainbow')

xLabel = ax.set_xlabel('DescriptionCode', linespacing=3.2)
yLabel = ax.set_ylabel('Quantity', linespacing=3.1)
zLabel = ax.set_zlabel('UnitPrice', linespacing=3.4)
print("K-Means")


# In[ ]:


DS2_clustered_kmeans = DS_2_clusters.assign(Cluster=kmeans_labels)
grouped_kmeans = DS2_clustered_kmeans.groupby(['Cluster']).mean().round(1)
grouped_kmeans


# The table shows centroids of each clusters that could determine the clusters rule.
# These are:
# 
# Cluster 0: Less Quantity, Mid Unit Price, Description Code High
# 
# Cluster 1: Less Quantity, Mid Unit Price, Description Code low
# 
# Cluster 2: Less Quantity, High Unit Price, Description Code High
#     
# Cluster 3: Less Quantity, low Unit Price, Description Code Mid
#     
# Description Code low: It means that codes which label encoder makes by its own first 1000 is describe as Low
# 
# Description Code Mid: It means that codes which label encoder makes by its own from 1000-2000 is describe as Mid
# 
# Description Code High: It means that codes which label encoder makes by its own first 2000-3000 is describe as High

# In[ ]:


from sklearn.cluster import MiniBatchKMeans
kmeans = MiniBatchKMeans(n_clusters=4,random_state=0,batch_size=6,max_iter=10).fit(cluster_scaled)
kmeans_labels = kmeans.labels_

fig = plt.figure(num=None, figsize=(15, 10), dpi=80, facecolor='w', edgecolor='k')
ax = plt.axes(projection="3d")

ax.scatter3D(DS_2_clusters['DescriptionCode'],DS_2_clusters['Quantity'],DS_2_clusters['UnitPrice'],c=kmeans_labels, cmap='rainbow')

xLabel = ax.set_xlabel('DescriptionCode', linespacing=3.2)
yLabel = ax.set_ylabel('Quantity', linespacing=3.1)
zLabel = ax.set_zlabel('UnitPrice', linespacing=3.4)
print("K-Means")


# In[ ]:


DS2_clustered_kmeans = DS_2_clusters.assign(Cluster=kmeans_labels)
grouped_kmeans = DS2_clustered_kmeans.groupby(['Cluster']).mean().round(1)
grouped_kmeans


# The table shows centroids of each clusters that could determine the clusters rule.
# These are:
# 
# Cluster 0: Less Quantity, Mid Unit Price, Description Code High
# 
# Cluster 1: More Quantity, Low Unit Price, Description Code Mid
# 
# Cluster 2: Less Quantity, High Unit Price, Description Code Mid
#     
# Cluster 3: Less Quantity, Mid Unit Price, Description Code low
#     
# Description Code low: It means that codes which label encoder makes by its own first 1000 is describe as Low
# 
# Description Code Mid: It means that codes which label encoder makes by its own from 1000-2000 is describe as Mid
# 
# Description Code High: It means that codes which label encoder makes by its own first 2000-3000 is describe as High

# In[ ]:





# In[ ]:




