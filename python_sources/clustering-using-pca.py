#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# 
# ###### Reading the dataset 

# In[ ]:


data=pd.read_csv('../input/country-socioeconomic-data/Country-data.csv')
data.head()


# **Aim is to find out the countries which are in dire need of funding .**

# In[ ]:


data['country'].value_counts()


# In[ ]:


print('Categorical columns : ',list(data.select_dtypes(include='object').columns))
print('Numeric columns : ',list(data.select_dtypes(exclude='object').columns))


# #####  five-point summary for numerical variables 

# In[ ]:


num=data.select_dtypes(exclude='object')
num.head()


# In[ ]:


num.describe().T


# ###### Summarizing observations for categorical variables 

# In[ ]:


print('No of categories in the county column are :',data['country'].nunique())


# In[ ]:


print('% observations in each category :\n',data['country'].value_counts(normalize=True)*100)


# ######  covariance and correlation tables for the data 

# In[ ]:


corr=num.corr() #### correlation table
corr


# In[ ]:


cov=num.cov() #### covariance table
cov


# In[ ]:


plt.figure(figsize=(15,8))
sns.heatmap(corr,annot=True)
plt.show()


# ######  Data Preparation 
# 

# In[ ]:


num.isnull().sum() ### no null values in the data


# In[ ]:


cols=list(num.columns)
for a in cols:
    sns.distplot(num[a])
    plt.show()


# In[ ]:


### Lowest 10 countries based on child_mortality
df=data[['country','child_mort']].sort_values('child_mort', ascending = False).head(10)
sns.barplot(x='country',y='child_mort',data=df)
plt.xticks(rotation=90)
plt.show()


# In[ ]:


### Lowest 10 countries based on 'total_fer'
df=data[['country','total_fer']].sort_values('total_fer', ascending = False).head(10)
sns.barplot(x='country',y='total_fer',data=df)
plt.xticks(rotation=90)
plt.show()


# In[ ]:


### Lowest 10 countries based on 'life_expec'
df=data[['country','life_expec']].sort_values('life_expec', ascending = False).head(10)
sns.barplot(x='country',y='life_expec',data=df)
plt.xticks(rotation=90)
plt.show()


# In[ ]:


### Lowest 10 countries based on 'health'
df=data[['country','health']].sort_values('health', ascending = False).head(10)
sns.barplot(x='country',y='health',data=df)
plt.xticks(rotation=90)
plt.show()


# In[ ]:


### Lowest 10 countries based on 'gdpp'
df=data[['country','gdpp']].sort_values('gdpp', ascending = False).head(10)
sns.barplot(x='country',y='gdpp',data=df)
plt.xticks(rotation=90)
plt.show()


# In[ ]:


### Lowest 10 countries based on 'income'
df=data[['country','income']].sort_values('income', ascending = False).head(10)
sns.barplot(x='country',y='income',data=df)
plt.xticks(rotation=90)
plt.show()


# In[ ]:


### Lowest 10 countries based on 'inflation'
df=data[['country','inflation']].sort_values('inflation', ascending = False).head(10)
sns.barplot(x='country',y='inflation',data=df)
plt.xticks(rotation=90)
plt.show()


# In[ ]:


### Lowest 10 countries based on 'exports'
df=data[['country','exports']].sort_values('exports', ascending = False).head(10)
sns.barplot(x='country',y='exports',data=df)
plt.xticks(rotation=90)
plt.show()


# In[ ]:


### Lowest 10 countries based on 'imports'
df=data[['country','imports']].sort_values('imports', ascending = False).head(10)
sns.barplot(x='country',y='imports',data=df)
plt.xticks(rotation=90)
plt.show()


# In[ ]:


##### Scaling the data:
from scipy.stats import zscore
scale=num.apply(zscore)
scale.head()


# In[ ]:


cols=list(scale.columns)
for a in cols:
    sns.boxplot(scale[a])
    plt.show()


# #### Their are few ouliers in each feature and gdpp has a lot of outliers but our dataset has very less information  and every row has data about a single county, hence removing outliers would lead to loss of information and we might loose the rows whcih will help determine the lowest countries in each category.
# #### Henec we will proceed without removal of outliers

# #####  Dimensionality Reduction Using PCA
# 

# ###### PCA is a dimentianlity reduction technique where we can reduce the number of features on the basis of correlation.What it does is create components which will capture majority of the variance in the data .All the components will be orthogonal to each other and The first component will capture the most amount of information.By seeing the correlation among features , PCA will determine which features to combine so as to capture majority of the information.

# ######  Applying PCA on the above dataset and determining the number of PCA components to be used 

# In[ ]:


cov_matrics=np.cov(scale.T)
cov_matrics


# In[ ]:



eign_values , eign_vect = np.linalg.eig(cov_matrics)
print ( "Eigen Values:\n" , eign_values)
print('\n Eigen vectors : \n',eign_vect)


# In[ ]:


eig_pairs = [(eign_values[index], eign_vect[:,index]) for index in range(len(eign_values))]
eig_pairs


# In[ ]:


total = sum( eign_values )
var_exp = [ ( i / total ) * 100 for i in sorted ( eign_values , reverse = True ) ]
cum_var_exp = np.cumsum ( var_exp )
print("Cumulative Variance Explained", cum_var_exp)


# ##### 5 principal components explain 94% of the variance in the data , hence instead of using  9 features we can just use 5 principal components

# In[ ]:


plt.bar(range(1,eign_values.size + 1), var_exp, alpha=0.5, align='center', label='individual explained variance')
plt.step(range(1,eign_values.size + 1),cum_var_exp, where= 'mid', label='cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.legend(loc = 'best')
plt.show()


# In[ ]:


#### KMeans without PCA:
from sklearn.cluster import KMeans

wcss = []

for k in range(1,10):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(scale)
    wcss.append(kmeans.inertia_)


# In[ ]:


plt.figure(figsize=(12,6))
plt.plot( range(1,10), wcss, marker = "o" )


# ###### without pca  their are 3 clusters as per the elbow curve

# In[ ]:


kmeans = KMeans(n_clusters=3)
kmeans.fit(scale)


# In[ ]:


kmeans.labels_


# In[ ]:


scale['Labels']=kmeans.labels_


# In[ ]:


#####  Agglomerative clustering without PCA

### MAKING OF DENDOGRAM:
from scipy.cluster.hierarchy import linkage, dendrogram,cophenet
from scipy.spatial.distance import pdist
plt.figure(figsize=[10,10])
merg = linkage(scale, method='ward')
dendrogram(merg, leaf_rotation=90)
plt.title('Dendrogram')
plt.xlabel('Data Points')
plt.ylabel('Euclidean Distances')
plt.show()


# The dendogram shows that the optimal number of clusters are 3 

# In[ ]:


from sklearn.cluster import AgglomerativeClustering

hie_clus = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
df_AC = scale.drop('Labels',1).copy(deep=True)
cluster2 = hie_clus.fit_predict(df_AC)


df_AC['label'] = cluster2


# In[ ]:


df_AC


# In[ ]:


scale


# In[ ]:





# ###### Plot to see the distribution of clusters using two features

# In[ ]:


### PLOTTING: WITHOUT PCA

plt.title('K-Means Classes')
sns.scatterplot(x='child_mort', y='life_expec', hue='Labels', data=scale)
plt.show()
plt.title('Hierarchical Classes')
sns.scatterplot(x='child_mort', y='life_expec', hue='label', data=df_AC)
plt.show()


# #####  Evaluating the clusters formed using appropriate  silhouette score
# 

# In[ ]:


###### silhouette_score without pca for AC
from sklearn.metrics import silhouette_score , cohen_kappa_score
x_pca_AC=df_AC.drop('label',1)
print('silhouette_score for AC with pca :',silhouette_score (x_pca_AC , df_AC['label'] ))


# In[ ]:


###### silhouette_score without pca for Kmeans
from sklearn.metrics import silhouette_score , cohen_kappa_score
x_pca_km=scale.drop('Labels',1)
print('silhouette_score for Kmeans without pca :',silhouette_score (x_pca_km , scale['Labels'] ))


# ###### USING PCA DETERMINING THE OPTIMAL CLUSTERS

# In[ ]:


from sklearn.decomposition import PCA


# In[ ]:


###### WITH PCA
s=scale.drop('Labels',1)
p=PCA(n_components=5)
d=p.fit_transform(s)
d=pd.DataFrame(d,columns=['PC1','PC2','PC3','PC4','PC5'])
d.shape


# In[ ]:


from sklearn.cluster import KMeans

wcss = []

for k in range(1,10):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(d)
    wcss.append(kmeans.inertia_)
    


# In[ ]:


# Visualization of k values:

plt.plot(range(1,10), wcss, color='red',marker='*')
plt.title('Graph of k values and WCSS')
plt.xlabel('k values')
plt.ylabel('wcss values')
plt.show()


# In[ ]:


#### Optimal clusters are 3
km= KMeans(n_clusters=3)
km.fit(d)
d['Labels']=km.labels_


# In[ ]:


ac=d.drop('Labels',1)


# In[ ]:


###### Agglomerative clustering with PCA

from scipy.cluster.hierarchy import linkage, dendrogram,cophenet
from scipy.spatial.distance import pdist
plt.figure(figsize=[10,10])
merg = linkage(ac, method='ward')
dendrogram(merg, leaf_rotation=90)
plt.title('Dendrogram')
plt.xlabel('Data Points')
plt.ylabel('Euclidean Distances')
plt.show()


# In[ ]:


##### Optimal clusters are 3

#### HIERARCHICAL CLUSTERING:
from sklearn.cluster import AgglomerativeClustering

hie_clus = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
cluster2 = hie_clus.fit_predict(ac)


ac['label'] = cluster2


# In[ ]:


plt.title('K-Means Classes')
sns.scatterplot(x='PC1', y='PC2', hue='Labels', style='Labels', data=d)
plt.show()
plt.title('Hierarchical Classes')
sns.scatterplot(x='PC1', y='PC2', hue='label', style='label', data=ac)
plt.show()


# #####  Evaluateing the clusters formed using  silhouette score
# 

# In[ ]:


###### silhouette_score with pca AC 
from sklearn.metrics import silhouette_score , cohen_kappa_score
x_pca_AC=ac.drop('label',1)
print('silhouette_score for AC with pca :',silhouette_score (x_pca_AC , ac['label'] ))


# In[ ]:


###### silhouette_score with PCA Kmeans
from sklearn.metrics import silhouette_score , cohen_kappa_score
x_pca_km=d.drop('Labels',1)
print('silhouette_score for AC with pca :',silhouette_score (x_pca_km , d['Labels'] ))


# ##### After PCA the silhouette score increases 
# #### From silhouette score the KMeans clustering is doing good 
# #### A high value indicates that the object is well matched to its own cluster and poorly matched to neighboring clusters. 

#  ##### Using Kmeans lables with PCA for further Analysis

# In[ ]:


#### WIth pca kmean
d.groupby('Labels').agg({'PC1':'mean','PC2':'mean','PC3':'mean','PC4':'mean','PC5':'mean'}).T


# ###### Plotting a box plot to visualize the cluster means across different attributes 

# In[ ]:


#### Kmeans boxplot
d.groupby('Labels').agg({'PC1':'mean','PC2':'mean','PC3':'mean','PC4':'mean','PC5':'mean'}).T.plot(kind='box')


# In[ ]:


#### With PCA
d1=pd.concat([d,data['country']],axis=1)


# In[ ]:


d1.groupby('Labels').agg({'country':'count'})


# In[ ]:


##### Without PCA
scale1=pd.concat([scale,data['country']],axis=1)


# In[ ]:


scale1.groupby('Labels').agg({'country':'count'})


# In[ ]:


###### USing Kmeans clustering Labels
clust_df = d1[['country','Labels']].merge(data, on = 'country')
clust_df.head()


# In[ ]:


clust_exports = pd.DataFrame(clust_df.groupby(['Labels']).exports.mean())
clust_health = pd.DataFrame(clust_df.groupby(['Labels']).health.mean())
clust_imports = pd.DataFrame(clust_df.groupby(['Labels']).imports.mean())
clust_income = pd.DataFrame(clust_df.groupby(['Labels']).income.mean())
clust_inflation = pd.DataFrame(clust_df.groupby(['Labels']).inflation.mean())
clust_life_expec = pd.DataFrame(clust_df.groupby(['Labels']).life_expec.mean())
clust_total_fer = pd.DataFrame(clust_df.groupby(['Labels']).total_fer.mean())
clust_gdpp = pd.DataFrame(clust_df.groupby(['Labels']).gdpp.mean())
clust_child_mort=pd.DataFrame(clust_df.groupby(['Labels']).child_mort.mean())


# In[ ]:


df2 = pd.concat([pd.Series(list(range(0,5))), clust_child_mort,clust_exports, clust_health, clust_imports,
               clust_income, clust_inflation, clust_life_expec,clust_total_fer,clust_gdpp], axis=1)
df2.columns = ["Labels", "child_mort_mean", "exports_mean", "health_mean", "imports_mean", "income_mean", "inflation_mean",
               "life_expec_mean", "total_fer_mean", "gdpp_mean"]
df2


# In[ ]:



fig, axs = plt.subplots(3,3,figsize = (15,15))
sns.barplot(x=df2.Labels, y=df2.child_mort_mean, ax = axs[0,0])
sns.barplot(x=df2.Labels, y=df2.exports_mean, ax = axs[0,1])
sns.barplot(x=df2.Labels, y=df2.health_mean, ax = axs[0,2])
sns.barplot(x=df2.Labels, y=df2.imports_mean, ax = axs[1,0])
sns.barplot(x=df2.Labels, y=df2.income_mean, ax = axs[1,1])
sns.barplot(x=df2.Labels, y=df2.life_expec_mean, ax = axs[1,2])
sns.barplot(x=df2.Labels, y=df2.inflation_mean, ax = axs[2,0])
sns.barplot(x=df2.Labels, y=df2.total_fer_mean, ax = axs[2,1])
sns.barplot(x=df2.Labels, y=df2.gdpp_mean, ax = axs[2,2])
plt.tight_layout()


# In[ ]:


clust_df[clust_df.Labels == 1].country.values


# The above are the countries that need highest attention and aid as they have low income means, low gdpp and have high child mortality rate along with high import along with health issues.

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


import pandas as pd
Country_data = pd.read_csv("../input/country-socioeconomic-data/Country-data.csv")
data_dictionary = pd.read_csv("../input/country-socioeconomic-data/data-dictionary.csv")

