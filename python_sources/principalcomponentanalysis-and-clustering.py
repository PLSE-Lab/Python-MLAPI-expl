#!/usr/bin/env python
# coding: utf-8

# # HELP International NGO (Clustering and Principal Component Analysis)
# - HELP International is an international humanitarian NGO that is committed to fighting poverty and providing the people of backward countries with basic amenities and relief during the time of disasters and natural calamities. It runs a lot of operational projects from time to time along with advocacy drives to raise awareness as well as for funding purposes.
# - After the recent funding programmes, they have been able to raise around $ 10 million. Now the CEO of the NGO needs to decide how to use this money strategically and effectively. The significant issues that come while making this decision are mostly related to choosing the countries that are in the direst need of aid.
# The job is to categorise the countries using some socio-economic and health factors that determine the overall development of the country and then suggest the countries which the CEO needs to focus on the most.The solution below will help to derive suggestions to be given the CEO

# #### Import all the important libraries

# In[ ]:


# Supress Warnings
import warnings
warnings.filterwarnings('ignore')

# import all important libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import cut_tree


# In[ ]:


countryData = pd.read_csv("../input/Country-data.csv")


# ### Data Analysis

# In[ ]:


countryData.shape


# In[ ]:


countryData.head()


# In[ ]:


countryData.describe()


# In[ ]:


countryData.info()


# ##### Checking for null values

# In[ ]:


countryData.isnull().sum()


# From above, you can see that there are no nulls in the data. All columns have data.

# ### Data Visualisation

# Lets look at the population factors in the countries and see what inferences we get

# In[ ]:


plt.figure(figsize=(35, 20))

plt.subplot(3,1,1)
plt.xticks(rotation=90,fontsize=12)
sorted_child_mort = countryData[['country','child_mort']].sort_values('child_mort', ascending = False)
ax1=sns.barplot(x='country', y='child_mort', data=sorted_child_mort, estimator=np.mean)
ax1.set(xlabel = 'Countries', ylabel= 'Child Mortality Rate')

plt.subplot(3,1,2)
plt.xticks(rotation=90,fontsize=12)
sorted_total_fer = countryData[['country','total_fer']].sort_values('total_fer', ascending = False)
ax2=sns.barplot(x='country', y='total_fer', data=sorted_total_fer, estimator=np.mean)
ax2.set(xlabel = 'Countries', ylabel= 'Fertility Rate')

plt.subplot(3,1,3)
plt.xticks(rotation=90,fontsize=12)
sorted_life_expec = countryData[['country','life_expec']].sort_values('life_expec', ascending = True)
ax2=sns.barplot(x='country', y='life_expec', data=sorted_life_expec, estimator=np.mean)
ax2.set(xlabel = 'Countries', ylabel= 'Life Expectancy')

plt.subplots_adjust(hspace=1)
plt.show()
#plt.tight_layout()


# Inference :
# - Haiti has a highest child mortality rate.The next 10 countries where the child mortality rate is high are mostly all the <b> African countries</b>
# - Similarly the fertility rate is more in the African and middle east countries but lowest in most European / Asia Pacific nations.
# - The life expectancy is highest in Japan (approx 85 years) and lowest in Haiti (approx 30 years).  In general the life expectancy is high in the European and Asia Pacific nations.

# Now lets look at the socio economic factors in the countries and see what inference we get

# <b>Perform data conversion from percentage to actual values for fields like imports,exports, health spend</b>

# In[ ]:


countryData['exports'] = (countryData['exports'] * countryData['gdpp'])/100
countryData['imports'] = (countryData['imports'] * countryData['gdpp'])/100
countryData['health'] = (countryData['health'] * countryData['gdpp'])/100


# In[ ]:


countryData.head(10)


# In[ ]:


plt.figure(figsize=(16, 16))

plt.subplot(3,2,1)
plt.xticks(rotation=45,fontsize=12,horizontalalignment='right')
avg15_country_exports = countryData[['country','exports']].sort_values('exports', ascending = False).head(15)
ax1=sns.barplot(x='country', y='exports', data=avg15_country_exports, estimator=np.mean)

plt.subplot(3,2,2)
plt.xticks(rotation=45,fontsize=12,horizontalalignment='right')
avg15_country_imports = countryData[['country','imports']].sort_values('imports', ascending = False).head(10)
ax2=sns.barplot(x='country', y='imports', data=avg15_country_imports, estimator=np.mean)

plt.subplot(3,2,3)
plt.xticks(rotation=45,fontsize=12)
plt.xticks(rotation=45,fontsize=12,horizontalalignment='right')
avg15_country_healthspend = countryData[['country','health']].sort_values('health', ascending = False).head(10)
ax3=sns.barplot(x='country', y='health', data=avg15_country_healthspend, estimator=np.mean)

plt.subplots_adjust(hspace=1)
plt.show()


# Inference:
# - Luxembourg has good exports and imports
# - United States has good amount of health spend followed by most of the European nations and Canada who equally have good health spend !

# ##### Outlier Analysis (followed by removal)

# In[ ]:


# Checking for outliers in the continuous variables
factors = countryData[['child_mort', 'exports', 'health', 'imports', 'income','inflation', 'life_expec','total_fer', 'gdpp']]


# In[ ]:


plt.figure(figsize=(20, 10))
plt.subplot(3,3,1)
sns.boxplot(factors['child_mort'],orient="v")
plt.subplot(3,3,2)
sns.boxplot(factors['life_expec'],orient="v")
plt.subplot(3,3,3)
sns.boxplot(factors['health'],orient="v")
plt.subplot(3,3,4)
sns.boxplot(factors['income'],orient="v")
plt.subplot(3,3,5)
sns.boxplot(factors['inflation'],orient="v")
plt.subplot(3,3,6)
sns.boxplot(factors['imports'],orient="v")
plt.subplot(3,3,7)
sns.boxplot(factors['exports'],orient="v")
plt.subplot(3,3,8)
sns.boxplot(factors['gdpp'],orient="v")
plt.subplot(3,3,9)
sns.boxplot(factors['total_fer'],orient="v")
plt.show()


# In[ ]:


# Checking outliers at 25%, 50%, 75%, 90%, 95% and 99%
factors.describe(percentiles=[.05,.25, .5, .75, .90, .95, .99])


# As we can see from above table and the box plots there are a number of outliers in the data.We will treat the outliers to values accordingly for analysis. 

# ##### Outlier Removal

# In[ ]:


# # removing (statistical) outliers
Q1 = countryData.child_mort.quantile(0.05)
Q3 = countryData.child_mort.quantile(0.95)
IQR = Q3 - Q1
countryData = countryData[(countryData.child_mort >= Q1 - 1.5*IQR) & (countryData.child_mort <= Q3 + 1.5*IQR)]

Q1 = countryData.health.quantile(0.05)
Q3 = countryData.health.quantile(0.95)
IQR = Q3 - Q1
countryData = countryData[(countryData.health >= Q1 - 1.5*IQR) & (countryData.health <= Q3 + 1.5*IQR)]

Q1 = countryData.income.quantile(0.05)
Q3 = countryData.income.quantile(0.95)
IQR = Q3 - Q1
countryData = countryData[(countryData.income >= Q1 - 1.5*IQR) & (countryData.income <= Q3 + 1.5*IQR)]

Q1 = countryData.inflation.quantile(0.05)
Q3 = countryData.inflation.quantile(0.95)
IQR = Q3 - Q1
countryData = countryData[(countryData.inflation >= Q1 - 1.5*IQR) & (countryData.inflation <= Q3 + 1.5*IQR)]

Q1 = countryData.imports.quantile(0.05)
Q3 = countryData.imports.quantile(0.95)
IQR = Q3 - Q1
countryData = countryData[(countryData.imports >= Q1 - 1.5*IQR) & (countryData.imports <= Q3 + 1.5*IQR)]

Q1 = countryData.exports.quantile(0.05)
Q3 = countryData.exports.quantile(0.95)
IQR = Q3 - Q1
countryData = countryData[(countryData.exports >= Q1 - 1.5*IQR) & (countryData.exports <= Q3 + 1.5*IQR)]

Q1 = countryData.gdpp.quantile(0.05)
Q3 = countryData.gdpp.quantile(0.95)
IQR = Q3 - Q1
countryData = countryData[(countryData.gdpp >= Q1 - 1.5*IQR) & (countryData.gdpp <= Q3 + 1.5*IQR)]

Q1 = countryData.life_expec.quantile(0.05)
Q3 = countryData.life_expec.quantile(0.95)
IQR = Q3 - Q1
countryData = countryData[(countryData.life_expec >= Q1 - 1.5*IQR) & (countryData.life_expec <= Q3 + 1.5*IQR)]

Q1 = countryData.total_fer.quantile(0.05)
Q3 = countryData.total_fer.quantile(0.95)
IQR = Q3 - Q1
countryData = countryData[(countryData.total_fer >= Q1 - 1.5*IQR) & (countryData.total_fer <= Q3 + 1.5*IQR)]


# In[ ]:


countryData.head()


# In[ ]:


countryData.shape


# Not much data is lost after removing outliers as compared to original data

# ### Check Correlations between the variables in the dataset

# In[ ]:


plt.figure(figsize = (12, 10))
sns.heatmap(countryData.corr(), annot = True, cmap="bwr" , linewidths=.5)
plt.show()


# Inference:
#  - From above it is seen that there are highly correlated factors. We will need to do PCA to solve this.

# ### Scaling the dataset

# In[ ]:


df_countryData=countryData.copy()
df_countryData.drop(['country'],axis=1,inplace=True)


# In[ ]:


# Scaling
# instantiate
scaler = StandardScaler()

# fit_transform
df_scaled_countryData = scaler.fit_transform(df_countryData)


# In[ ]:


df_scaled_countryData = pd.DataFrame(df_scaled_countryData)
df_scaled_countryData.head()


# ### PCA on the data

# In[ ]:


pca = PCA(svd_solver='randomized',random_state=42)


# In[ ]:


pca.fit(df_scaled_countryData)


# #### Components from the PCA

# In[ ]:


pca.components_


# #### Variance ration between components

# In[ ]:


pca.explained_variance_ratio_


# ##### Making a scree plot for the explained variance

# In[ ]:


fig = plt.figure(figsize=[10,8])
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel("Number of components")
plt.ylabel("Cumulative variance explained")
plt.show()


# Inference :
# - <b>Looks like 4 components are enough to explain 95% of variance in the dataset.</b>
# - Perform PCA with 4 components. 

# In[ ]:


cols = list(df_countryData.columns)
df_principal_comp = pd.DataFrame({'PC1':pca.components_[0] , 'PC2':pca.components_[1],'PC3':pca.components_[2],'PC4':pca.components_[3],'Features':cols})
df_principal_comp.head(10)


# In[ ]:


fig = plt.figure(figsize = (8,8))
plt.scatter(df_principal_comp.PC1, df_principal_comp.PC2)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
for i, txt in enumerate(df_principal_comp.Features):
    plt.annotate(txt, (df_principal_comp.PC1[i],df_principal_comp.PC2[i]))


# Inference:
# - From the above plot, we can see that the 
#     - First component(PC1) is in the direction where the imports, exports, gdpp,income, health,life_expec are more. 
#     - Second component(PC2) is in the direction where child_mort , total_fer is more

# ### Incremental PCA

# In[ ]:


# let's go ahead and do dimenstionality reduction using the four Principal Components
pca_final = IncrementalPCA(n_components=4)


# In[ ]:


df_pca_final = pca_final.fit_transform(df_scaled_countryData)


# In[ ]:


df_pca_final.shape


# In[ ]:


df_pca_final = pd.DataFrame(df_pca_final)
df_pca_final.head()


# Lets check the correlation between the principal components by plotting the correlation matrix.

# In[ ]:


corrmat = np.corrcoef(df_pca_final.transpose())


# In[ ]:


corrmat.shape


# In[ ]:


plt.figure(figsize=[10,5])
sns.heatmap(corrmat, annot=True)


# In[ ]:


df_pca_final.corr()


# Inference:
# - From above it is seen that <b>after performing PCA there are no correlations in the data.</b>

# In[ ]:


df_pca = df_pca_final.copy()


# In[ ]:


df_pca.columns


# In[ ]:


df_pca_final.columns


# In[ ]:


#Adding country column to PCA dataframe
df_pca_final['country'] = countryData['country']
df_pca_final.columns = ['PC1','PC2','PC3','PC4','country']
df_pca_final.head()


# In[ ]:


sns.scatterplot(x='PC1',y='PC2',data=df_pca_final)


# In[ ]:


plt.figure(figsize=(20,10))
sns.regplot(x='PC1', y='PC2', data=df_pca_final, fit_reg=False)

def label_point(x, y, val, ax):
    a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
    for i, point in a.iterrows():
        ax.text(point['x']+.02, point['y'], str(point['val']))

label_point(df_pca_final.PC1, df_pca_final.PC2, df_pca_final.country, plt.gca())


# ## Hopkins Statistics:
# The Hopkins statistic, is a statistic which gives a value which indicates the cluster tendency, in other words: how well the data can be clustered.It is used to assess the clustering tendency of a data set by measuring the probability that a given data set is generated by a uniform data distribution.
# 
# - If the value is between {0.01, ...,0.3}, the data is regularly spaced.
# 
# - If the value is around 0.5, it is random.
# 
# - If the value is between {0.7, ..., 0.99}, it has a high tendency to cluster.
# 
# The null and the alternative hypotheses are defined as follow:
# - Null hypothesis: the data set is uniformly distributed (i.e. no meaningful clusters)
# - Alternative hypothesis: the data set  is not uniformly distributed (i.e. contains meaningful clusters)
# 

# Checking Hopkins Test score and performing Hypothesis test

# In[ ]:


from sklearn.neighbors import NearestNeighbors
from random import sample
from numpy.random import uniform
import numpy as np
from math import isnan
 
def hopkins(X):
    d = X.shape[1]
    #d = len(vars) # columns
    n = len(X) # rows
    m = int(0.1 * n) 
    nbrs = NearestNeighbors(n_neighbors=1).fit(X.values)
 
    rand_X = sample(range(0, n, 1), m)
 
    ujd = []
    wjd = []
    for j in range(0, m):
        u_dist, _ = nbrs.kneighbors(uniform(np.amin(X,axis=0),np.amax(X,axis=0),d).reshape(1, -1), 2, return_distance=True)
        ujd.append(u_dist[0][1])
        w_dist, _ = nbrs.kneighbors(X.iloc[rand_X[j]].values.reshape(1, -1), 2, return_distance=True)
        wjd.append(w_dist[0][1])
 
    H = sum(ujd) / (sum(ujd) + sum(wjd))
    if isnan(H):
        print(ujd, wjd)
        H = 0
 
    return H


# In[ ]:


#Use the Hopkins Statistic function by passing the above dataframe as a paramter
hopkins(df_pca)


# As per the above Hopkins score, we reject the null hypothesis. This means that Alternative Hypothesis is true and the given dataset contains meaningful clusters(It has significantly clusterable data) 

# ## K Means Clustering

# In[ ]:


#First we'll do the silhouette score analysis
from sklearn.metrics import silhouette_score
sse_ = []
for k in range(2, 10):
    kmeans = KMeans(n_clusters=k).fit(df_pca)
    sse_.append([k, silhouette_score(df_pca, kmeans.labels_)])


# In[ ]:


plt.plot(pd.DataFrame(sse_)[0], pd.DataFrame(sse_)[1]);


# From the above analysis we see  that 3 seems to be a good number of clusters for K means

# In[ ]:


# elbow-curve/SSD
ssd = []
for num_clusters in list(range(1,10)):
    kmeans = KMeans(n_clusters=num_clusters, max_iter=50)
    kmeans.fit(df_pca)
    
    ssd.append(kmeans.inertia_)
    
# plot the SSDs for each n_clusters
# ssd
plt.plot(ssd)


# Here also we see that 3 seems to be a good number of clusters for K means clustering.

# ### Silhouette Analysis
# 
# $$\text{silhouette score}=\frac{p-q}{max(p,q)}$$
# 
# $p$ is the mean distance to the points in the nearest cluster that the data point is not a part of
# 
# $q$ is the mean intra-cluster distance to all the points in its own cluster.
# 
# * The value of the silhouette score range lies between -1 to 1. 
# 
# * A score closer to 1 indicates that the data point is very similar to other data points in the cluster, 
# 
# * A score closer to -1 indicates that the data point is not similar to the data points in its cluster.

# In[ ]:


# silhouette analysis
range_n_clusters = [2,3,4,5,6]

for num_clusters in range_n_clusters:
    
    # intialise kmeans
    kmeans = KMeans(n_clusters=num_clusters, max_iter=50)
    kmeans.fit(df_pca)
    
    cluster_labels = kmeans.labels_
    
    # silhouette score
    silhouette_avg = silhouette_score(df_pca, cluster_labels)
    print("For n_clusters={0}, the silhouette score is {1}".format(num_clusters, silhouette_avg))


# In[ ]:


kmeans_cluster = KMeans(n_clusters=3, max_iter=50,random_state=50)
kmeans_cluster.fit(df_pca)


# In[ ]:


kmeans_cluster.labels_


# In[ ]:


df_kmeans=df_pca_final
df_kmeans.index = pd.RangeIndex(len(df_kmeans.index))
df_kmeans = pd.concat([df_kmeans, pd.Series(kmeans_cluster.labels_)], axis=1)
df_kmeans.columns = ['PC1', 'PC2','PC3', 'PC4','country','ClusterID']
df_kmeans


# In[ ]:


df_kmeans['ClusterID'].value_counts()


# In[ ]:


#Now lets visualizise the same data as shown above on a scatter plot
sns.scatterplot(x='PC1',y='PC2',hue='ClusterID',palette="Paired",legend='full',data=df_kmeans)


# In[ ]:


merge_df=pd.merge(countryData,df_kmeans, how='inner',on='country')
merge_df.head()


# In[ ]:


#Removing the principal components to keep the original features as it is 
df_final=merge_df.drop(['PC1','PC2','PC3','PC4'],axis=1)
df_final.shape


# Let us now analyse the clusters by comparing how the three variables [gdpp, child_mort and income] vary for each cluster of countries to recognise and differentiate the clusters of developed countries from the clusters of under-developed countries.

# In[ ]:


avg_child_mort =  pd.DataFrame(df_final.groupby(["ClusterID"]).child_mort.mean())
avg_gdpp = pd.DataFrame(df_final.groupby(["ClusterID"]).gdpp.mean())
avg_income = pd.DataFrame(df_final.groupby(["ClusterID"]).income.mean())


# In[ ]:


analyse_df = pd.concat([avg_child_mort,avg_gdpp,avg_income],axis=1)


# In[ ]:


analyse_df


# Inference
# - Average child mortality rate for cluster 2 seems to the highest and lowest for cluster 1
# - Average gdpp for is the lowest for cluster 2 and highest for cluster 1
# - Average income for is the lowest for cluster 2 and highest for cluster 1

# In[ ]:


plt.figure(figsize=(16, 8))

plt.subplot(1,3,1)
s=sns.barplot(x=analyse_df.index,y='child_mort',data=analyse_df)
plt.xlabel('Country Clusters on basis of Child Mortality', fontsize=8)
plt.ylabel('Child Mortality', fontsize=8)


plt.subplot(1,3,2)
s=sns.barplot(x=analyse_df.index,y='gdpp',data=analyse_df)
plt.xlabel('Country Clusters on basis of GDP', fontsize=8)
plt.ylabel('GDP per capita', fontsize=8)


plt.subplot(1,3,3)
s=sns.barplot(x=analyse_df.index,y='income',data=analyse_df)
plt.xlabel('Country Clusters on basis of Income', fontsize=8)
plt.ylabel('Net Income per person', fontsize=8)


plt.subplots_adjust(wspace=2)

plt.show()


# Inference:
# - Child mortality rate is significantly high for countries in cluster 2 as compared to other clusters
# - GDP of countries in cluster 2 is  very less as compared to other clusters
# - Net Income Per Person of countries in cluster 2 is lowest as compared to other clusters
# - <b>Cluster 2</b> seems to be the list of <b>Under Developed countries</b>
# - <b>Cluster 1</b> seems to be the list of <b>Developed countries</b>
# - <b>Cluster 0</b> seems to be list of <b>Developing countries</b>

# Let's find out the names of under developed countries now which from  are in direst need of aid  

# In[ ]:


analyse_df.rename(index={0: 'Developing Countries'},inplace=True)
analyse_df.rename(index={1: 'Developed Countries'},inplace=True)
analyse_df.rename(index={2: 'Under-Developed countries'},inplace=True)
analyse_df


# ### Analysis of the clusters by comparing how these three variables - [gdpp, child_mort and income] vary for each cluster of countries to recognise and differentiate the clusters of developed countries from the clusters of under-developed countries/developing countries

# Filtering out the 5 countries(Hierarchical) from clusters formed by KMeans

# In[ ]:


country_child_mortality=df_final.loc[df_final['ClusterID'] == 2].sort_values(by='child_mort',ascending=False).head(5)
country_child_mortality


# In[ ]:


country_gdpp=df_final.loc[df_final['ClusterID'] == 2].sort_values(by='gdpp',ascending=True).head(5)
country_gdpp


# In[ ]:


country_income=df_final.loc[df_final['ClusterID'] == 2].sort_values(by='income',ascending=True).head(5)
country_income


# ### Combining the 3 dataframes to list out countries which are in direst need of aid based on analysis of 3 variables - child mortality,gdpp and income

# In[ ]:


pd.concat([country_child_mortality,country_gdpp,country_income],ignore_index=True).drop_duplicates().reset_index(drop=True).head(9)


# ## Hiearchical Clustering

# In[ ]:


df_pca2= df_pca_final.copy()


# In[ ]:


df_hc=df_pca2.drop('country',axis=1)


# In[ ]:


df_hc.head()


# In[ ]:


# single linkage
mergings = linkage(df_hc, method="single", metric='euclidean')
dendrogram(mergings)
plt.show()


# Inference :
# - Single linkage type as seen from above has produced a dendrogram which is not structured properly

# In[ ]:


# complete linkage
mergings = linkage(df_hc, method="complete", metric='euclidean')
dendrogram(mergings)
plt.show()


# Inference:
# - Complete Linkage has given a proper tree liked structure(dendrogram) and making it easier to determine clusters
# - <b> Cutting the above dendrogram at 3 clusters</b>

# In[ ]:


# 3 clusters
cluster_labels = cut_tree(mergings, n_clusters=3).reshape(-1, )
cluster_labels


# In[ ]:


# assign cluster labels
df_hc['cluster_labels'] = cluster_labels
df_hc.head()


# In[ ]:


df_hc['cluster_labels'].value_counts()


# In[ ]:


df_hc.columns


# In[ ]:


plt.figure(figsize= (16,6))
sns.scatterplot(x='PC1',y='PC2',hue='cluster_labels',legend='full', palette = 'Paired',data=df_hc)


# In[ ]:


# Merging the origianl final data frame with hierarchical cluster dataframe
df_hc2=df_pca2
df_hc2 = pd.concat([df_hc2, df_hc], axis=1)
df_hc2.head()


# In[ ]:


#Dropping the prinicpal components from dataframe
df_hc_merge=pd.merge(countryData,df_hc2, how='inner',on='country')
df_hc3=df_hc_merge.drop(['PC1','PC2','PC3','PC4'],axis=1)
df_hc3.head()


# In[ ]:


avg_child_mort_2= pd.DataFrame(df_hc3.groupby(['cluster_labels']).child_mort.mean())
avg_gdpp_2 = pd.DataFrame(df_hc3.groupby(['cluster_labels']).gdpp.mean())
avg_gdpp_income_2 = pd.DataFrame(df_hc3.groupby(['cluster_labels']).income.mean())


# In[ ]:


analyse_df_2 = pd.concat([avg_child_mort_2,avg_gdpp_2,avg_gdpp_income_2],axis=1)


# In[ ]:


analyse_df_2


# Inference
# - Average child mortality rate for cluster 0 seems to the highest and lowest for cluster 1
# - Average gdpp for is the lowest for cluster 0 and highest for cluster 2
# - Average income for is the lowest for cluster 0 and highest for cluster 1

# In[ ]:


plt.figure(figsize=(16, 6))

plt.subplot(1,3,1)
s=sns.barplot(x=analyse_df_2.index,y='child_mort',data=analyse_df_2)
plt.xlabel('Country Clusters on basis of Child Mortality', fontsize=8)
plt.ylabel('Child Mortality', fontsize=8)


plt.subplot(1,3,2)
s=sns.barplot(x=analyse_df_2.index,y='gdpp',data=analyse_df_2)
plt.xlabel('Country Clusters on basis of GDP', fontsize=8)
plt.ylabel('GDP per capita', fontsize=8)


plt.subplot(1,3,3)
s=sns.barplot(x=analyse_df_2.index,y='income',data=analyse_df_2)
plt.xlabel('Country Clusters on basis of Income', fontsize=8)
plt.ylabel('Net Income per person', fontsize=8)


plt.subplots_adjust(wspace=2)

plt.show()


# Inference:
# - Child mortality rate is significantly high for countries in cluster 0 as compared to other clusters
# - GDP of countries in cluster 0 is significantly lowest as compared to other clusters
# - Net Income Per Person of countries in cluster 0 is lowest as compared to other clusters
# - <b>Cluster 2</b> seems to be the list of <b>Developing countries</b>
# - <b>Cluster 1</b> seems to be the list of <b>Developed countries</b>
# - <b>Cluster 0</b> seems to be list of <b>Under Developed countries</b>

# In[ ]:


analyse_df_2.rename(index={0: 'Under Developed Countries'},inplace=True)
analyse_df_2.rename(index={1: 'Developed Countries'},inplace=True)
analyse_df_2.rename(index={2: 'Developing countries'},inplace=True)
analyse_df_2


# ### Analysis of the clusters by comparing how these three variables - [gdpp, child_mort and income] vary for each cluster of countries to recognise and differentiate the clusters of developed countries from the clusters of under-developed countries/developing countries

# Filtering out the 5 countries(Hierarchical) from clusters formed by Hierarchical

# In[ ]:


country_child_mortality_2=df_hc3.loc[df_hc3['cluster_labels'] == 0].sort_values(by='child_mort',ascending=False).head(5)
country_child_mortality_2


# In[ ]:


country_gdpp_2=df_hc3.loc[df_hc3['cluster_labels'] == 0].sort_values(by='gdpp',ascending=True).head(5)
country_gdpp_2


# In[ ]:


country_income_2=df_hc3.loc[df_hc3['cluster_labels'] == 0].sort_values(by='income',ascending=True).head(5)
country_income_2


# ### Combining the 3 dataframes to list out countries which are in direst need of aid based on analysis of 3 variables - child mortality,gdpp and income

# In[ ]:


pd.concat([country_child_mortality_2,country_gdpp_2,country_income_2],ignore_index=True).drop_duplicates().reset_index(drop=True).head(9)


# ### From above we can observe that the almost same countries have been identified by both K-Means and Hierarchical clustering which are in direst need of aid
