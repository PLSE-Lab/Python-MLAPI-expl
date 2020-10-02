#!/usr/bin/env python
# coding: utf-8

# # INTRODUCTION

# The data set refers to clients of a wholesale distributor. It includes the annual spending in monetary units on diverse product categories.
# 
# ### Attribute Information:
# - 1)	FRESH: annual spending (m.u.) on fresh products (Continuous); 
# - 2)	MILK: annual spending (m.u.) on milk products (Continuous); 
# - 3)	GROCERY: annual spending (m.u.)on grocery products (Continuous); 
# - 4)	FROZEN: annual spending (m.u.)on frozen products (Continuous) 
# - 5)	DETERGENTS_PAPER: annual spending (m.u.) on detergents and paper products (Continuous) 
# - 6)	DELICATESSEN: annual spending (m.u.)on and delicatessen products (Continuous); 
# - 7)	CHANNEL: customers Channel - Horeca (Hotel/Restaurant/Cafe) or Retail channel (Nominal) 
# - 8)	REGION: customers Region - Lisbon, Oporto or Other (Nominal) 
# 
# ### Region	Frequency 
# Lisbon	77 
# Oporto	47 
# Other Region	316 
# Total	440 
# 
# ### Channel	Frequency 
# Horeca	298 
# Retail	142 
# Total	440 
# 
# Our project goal is to use various clustering techniques to segment customers. Clustering is an unsupervised learning algorithm that tries to cluster data based on their similarity. Thus, there is no outcome to be predicted, and the algorithm just tries to find patterns in the data.

# In[ ]:


#Importing required numerical and data manipulation libraries
import numpy as np 
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

#plotting libraries
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

import os
print(os.listdir("../input"))


# In[ ]:


data = pd.read_csv("../input/Wholesale customers data.csv")
data.shape


# In[ ]:


data.head()


# In[ ]:


type(data)


# ## Exploratory Data Analysis(EDA)

# In[ ]:


print('Descriptive Statastics of our Data:')
data.describe().T


# In[ ]:


print('Showing Meta Data :')
data.info()


# In[ ]:


#Checking for missing values
pd.isnull(data).sum()


# In[ ]:


data.Region.value_counts()


# In[ ]:


data.Channel.value_counts()

Now,lets convert the channel and Region to categorical variable for only EDA purpose 
mapping channels as 1:Horeca 2:Retail and 
also regions 1:lisbon 2:oporto 3:other
# ### Univarient Analysis

# In[ ]:


dataset = data.copy()


# In[ ]:


dataset['Channel'] = dataset['Channel'].map({1:'Horeca', 2:'Retail'})


# In[ ]:


dataset['Region'].replace([1,2,3],['Lisbon','Oporto','other'],inplace=True)


# In[ ]:


dataset.head()


# In[ ]:


def continous_data(i):
    if dataset[i].dtype!='object':
        print('--'*60)
        sns.boxplot(dataset[i])
        plt.title("Boxplot of "+str(i))
        plt.show()
        plt.title("histogram of "+str(i))        
        dataset[i].plot.hist(bins = 20)
        plt.show()
        plt.clf()


# In[ ]:


sns.set() #Sets the default seaborn style
j=['Fresh','Milk','Grocery','Frozen','Detergents_Paper','Delicassen']
for k in j:
    continous_data(i=k)


# ## Data Transformation

# In[ ]:


dataset.head()


# In[ ]:


# Scale the data using the natural logarithm
log_data = np.log(dataset[['Fresh','Milk','Grocery','Frozen','Detergents_Paper','Delicassen']].copy())


# In[ ]:


log_data.head()


# In[ ]:


def categorical_data(i):
    dataset[i].value_counts().plot(kind='bar')

j_1 = ['Channel','Region']

for k in j_1:
    categorical_data(i=k)
    plt.show()    


# In[ ]:


dataset.corr()


# In[ ]:


print('Correlation Heat map of the data')
plt.figure(figsize=(10,6))
sns.heatmap(dataset.corr(),annot=True,fmt='.2f',vmin=-1,vmax=1,cmap='Spectral')
plt.show()


# In[ ]:


def scatterplot(i,j):
    sns.regplot(data=log_data,x=i,y=j)
    plt.show()


# In[ ]:


scatterplot(i='Milk',j='Grocery')


# In[ ]:


scatterplot(i='Milk',j='Detergents_Paper')


# In[ ]:


scatterplot(i='Detergents_Paper',j='Grocery')


# In[ ]:


def categorical_multi(i,j):
    pd.crosstab(dataset[i],dataset[j]).plot(kind='bar')
    plt.show()
    print(pd.crosstab(dataset[i],dataset[j]))

categorical_multi(i='Channel',j='Region')    


# ## Treatment of Outliers

# In[ ]:


list(log_data.columns)


# In[ ]:


# replacing the outliers with their Inner fences
for k in list(log_data.columns):
    IQR = np.percentile(log_data[k],75) - np.percentile(log_data[k],25)
    
    Outlier_top = np.percentile(log_data[k],75) + 1.5*IQR
    Outlier_bottom = np.percentile(log_data[k],25) - 1.5*IQR
    
    log_data[k] = np.where(log_data[k] > Outlier_top,Outlier_top,log_data[k])
    log_data[k] = np.where(log_data[k] < Outlier_bottom,Outlier_bottom,log_data[k])


# In[ ]:


def continous_data(i):
    if log_data[i].dtype!='object':
        print('--'*60)
        sns.boxplot(log_data[i])
        plt.title("Boxplot of "+str(i))
        plt.show()
        plt.title("histogram of "+str(i))
        log_data[i].plot.kde()
        plt.show()
        plt.clf()

for k in j:
    continous_data(i=k)        


# In[ ]:


sns.pairplot(log_data,diag_kind = 'kde')


# In[ ]:


dataset1 = log_data.copy()
list(dataset1.columns)


# In[ ]:


## replacing with median to treat the outliers
for k in list(dataset1.columns):
    IQR=np.percentile(dataset1[k],75) - np.percentile(dataset1[k],25)
    
    Outlier_top=np.percentile(dataset1[k],75)+1.5*IQR
    Outlier_bottom=np.percentile(dataset1[k],25)-1.5*IQR
    
    dataset1[k]=np.where(dataset1[k] > Outlier_top,np.percentile(dataset1[k],50),dataset1[k])
    dataset1[k]=np.where(dataset1[k] < Outlier_bottom,np.percentile(dataset1[k],50),dataset1[k])


# In[ ]:


sns.pairplot(dataset1,diag_kind = 'kde')


# ## Creating dummies for categorical varibles

# In[ ]:


df  =  pd.concat([dataset[['Channel','Region']],log_data],axis=1)
df.head()


# In[ ]:


df = pd.get_dummies(df,columns=['Channel','Region'],drop_first=True)
df.head()


# # Feature Scaling

# In[ ]:


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df_std = scaler.fit_transform(df)
df_std = pd.DataFrame(df_std,columns=df.columns)
df_std.head()


# In[ ]:


from scipy.spatial.distance import pdist,squareform
from scipy.cluster.hierarchy import linkage,dendrogram,cut_tree


# In[ ]:


eu_d = pdist(df_std,metric='euclidean')
clus = linkage(eu_d,method='average')
names = np.arange(0,df_std.shape[0]).tolist()


# In[ ]:


plt.figure(figsize=[14,8])
dendrogram(clus,labels=names)
plt.xlabel('hclust')
plt.ylabel('distance')
plt.title('cluster dendogram')


# In[ ]:


data_hier = data.copy()
data_hier.head()


# In[ ]:


data_hier['clusters'] = cut_tree(clus,6)


# In[ ]:


clust_profile = data_hier.groupby(['clusters'],as_index=False).mean()
clust_profile


# ## Kmeans Clustering

# In[ ]:


X = df_std.copy()

from sklearn.cluster import KMeans
cluster_range = range(1,20)
cluster_wss=[] 
for cluster in cluster_range:
    model = KMeans(cluster)
    model.fit(X)
    cluster_wss.append(model.inertia_)


# In[ ]:


#PLotting Elbow curve for finding Optimal K value
plt.figure(figsize=[10,6])
plt.title('WSS curve for finding Optimul K value')
plt.xlabel('No. of clusters')
plt.ylabel('Inertia or WSS')
plt.plot(list(cluster_range),cluster_wss,marker='o')
plt.show()


# ### Clustering Using K-means with K=6

# In[ ]:


from sklearn.cluster import KMeans
model = KMeans(n_clusters=6,random_state=0)
model.fit(X)


# In[ ]:


dataset_final = data.copy()
dataset_final.head()


# In[ ]:


dataset_final['clusters']=model.predict(X)
dataset_final.head()


# In[ ]:


#cluster profiles
clust_prof = dataset_final.groupby(['clusters'],as_index=False).mean()
clust_prof


# ### PCA for getting the first 2 Principle components

# In[ ]:


from sklearn.decomposition import PCA
pca2 = PCA(n_components=2)
pc = pca2.fit_transform(df_std)
pc_df = pd.DataFrame(pc)
pc_df.head()


# In[ ]:


pca = pd.concat([pc_df,dataset_final['clusters']],axis=1)
pca.columns = ['pc1','pc2','clusters']
print(pca.shape)
pca.head()


# In[ ]:


pca.clusters.value_counts()


# In[ ]:


plt.figure(figsize=[16,8])
sns.scatterplot(x='pc1', y='pc2', hue= 'clusters', data=pca,palette='Set1')
plt.show()


# ## Explain what you have done and how this segments customers

# In[ ]:


dataset_final.groupby('clusters').Fresh.mean().plot(kind='bar')


# In[ ]:


dataset_final.groupby('clusters').Milk.mean().plot(kind='bar')


# In[ ]:


dataset_final.groupby('clusters').Grocery.mean().plot(kind='bar')


# In[ ]:


dataset_final.groupby('clusters').Frozen.mean().plot(kind='bar')


# In[ ]:


dataset_final.groupby('clusters').Detergents_Paper.mean().plot(kind='bar')

