#!/usr/bin/env python
# coding: utf-8

# ### Clustering the Countries by using K-means, Heirarchical and PCA for HELP International
# **Objective**: To categorise the countries using socio-economic and health factors that determine the overall development of the country.
# 
# **About organization**: HELP International is an international humanitarian NGO that is committed to fighting poverty and providing the people of backward countries with basic amenities and relief during the time of disasters and natural calamities.
# 
# **Problem Statement**: HELP International have been able to raise around $ 10 million. Now the CEO of the NGO needs to decide how to use this money strategically and effectively. So, CEO has to make decision to choose the countries that are in the direst need of aid.  Hence, My job as a Data analyst is to categorise the countries using some socio-economic and health factors that determine the overall development of the country. Then I need to suggest the countries which the CEO needs to focus on the most. 
# 
# 
# 
# ### Import all necessary libraries

# In[ ]:



import pandas as pd
import numpy as np

# For Visualisation
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

# For scaling the data
from sklearn.preprocessing import scale

# To perform K-means clustering
from sklearn.cluster import KMeans

# To perform PCA
from sklearn.decomposition import PCA

#To perform hierarchical clustering
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import cut_tree


# In[ ]:


#Reaidng the Dataset 

Country_data=pd.read_csv("../input/Country-data.csv")


# ### Let's understand the structure of our dataframe

# In[ ]:


#Reading the first 5 rows of the dataset
Country_data.head()


# - Here, exports, heath and imports variables are in percenatge of the total GDP format. So, It is converted into actual value on the basis of GDPP as below since, GDP data and population data is not present ihe data set.

# In[ ]:


Country_data['exports'] = Country_data['exports']*Country_data['gdpp']/100
Country_data['imports'] = Country_data['imports']*Country_data['gdpp']/100
Country_data['health'] = Country_data['health']*Country_data['gdpp']/100


# #### Let's read first 5 rows after converting the exports, heath and imports variables

# In[ ]:


Country_data.head()


# ### Checking of the outliers in each columns

# In[ ]:


# Checking outliers at 25%,50%,75%,90%,95% and 99%
Country_data.describe(percentiles=[.25,.5,.75,.90,.95,.99])


# - Let's plot the box plots to check outliers

# In[ ]:


fig = plt.figure(figsize = (12,8))
sns.boxplot(data=Country_data)
plt.show()


# - From the above table, we can see there are some outliers in case of exports,imports,income, gdpp, etc. We will handle this after performing PCA to derive principal components after considering the elemination of Country.

# In[ ]:


print("The number of countries are : ",Country_data.shape[0])


# #### Checking the datatypes of each variable

# In[ ]:


Country_data.info()


# - From above datatype information all the datattypes are in correct format.

# #### Checking of null or NaN values

# In[ ]:


Country_data.isnull().sum()


# In[ ]:


Country_data.isna().sum()


# #### As we can see there is no missing data(null,NaN values) in the list
# 
# #### Let's plot the heat map to check the multicollinearity of the variables.

# In[ ]:


#plotting the correlation matrix
get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure(figsize = (20,10))
sns.heatmap(Country_data.corr(),annot = True)
plt.show()


# - from above heatmap, we can see exports is highly correlated with import. health, exports, income,imports are highly correlated with gdpp. So, it will be treated after performing PCA.

# ### Data Preparation
# 
# #### Performing PCA on the data ( Principal Component Analysis)
# 
# - Applying scaling to the data

# In[ ]:


## First let us see if we can explain the dataset using fewer variables
from sklearn.preprocessing import StandardScaler
Country_data1=Country_data.drop('country',1) ## Droping string feature country name.
standard_scaler = StandardScaler()
Country_scaled = standard_scaler.fit_transform(Country_data1)


# #### Perfoming PCA

# In[ ]:


pca = PCA(svd_solver='randomized', random_state=42)


# fiting PCA on the dataset
pca.fit(Country_scaled)


# #### List of PCA components.

# In[ ]:


pca.components_


# #### Let's check the variance ratios of each features

# In[ ]:


pca.explained_variance_ratio_


# ### Plotting the scree plot

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
fig = plt.figure(figsize = (12,8))
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.show()


# #### From above scree plot, Around 98% of the information is being explained by 5 components.
# - Understanding how the original 9 variables are loaded on the principal components. It can be verified from above as well.

# In[ ]:


colnames = list(Country_data1.columns)
pcs_df = pd.DataFrame({ 'Feature':colnames,'PC1':pca.components_[0],'PC2':pca.components_[1],'PC3':pca.components_[2],
                      'PC4':pca.components_[3],'PC5':pca.components_[4]})
pcs_df


# ### Let's plot the principal components and try to make sense of them.
# 
# - We'll plot original features on the first 2 principal components as axes
# 
# - Let's plot them to visualise how these features are loaded

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
fig = plt.figure(figsize = (8,8))
plt.scatter(pcs_df.PC1, pcs_df.PC2)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
for i, txt in enumerate(pcs_df.Feature):
    plt.annotate(txt, (pcs_df.PC1[i],pcs_df.PC2[i]))
plt.tight_layout()
plt.show()


# - From the above plot, we can see the first component is in the dirction where the imports, exports, gdpp,income, health,life_expec are heavy and second component is in the direction where child_mort , total_fer is more.
# 
# - These variables also have the highest of the loadings
# 
# ### Performing Incremental PCA

# In[ ]:


#Finally let's go ahead and do dimenstionality reduction using the four Principal Components
from sklearn.decomposition import IncrementalPCA
pca_final = IncrementalPCA(n_components=5)


# In[ ]:


df_pca = pca_final.fit_transform(Country_scaled)
df_pca.shape


# ### Creating a transpose so that the each column is properly arranged

# In[ ]:


pc = np.transpose(df_pca)


#  #### Creating correlation matrix for the principal components

# In[ ]:


corrmat = np.corrcoef(pc)


# #### Plotting the correlation matrix of the principal components

# In[ ]:



get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure(figsize = (20,10))
sns.heatmap(corrmat,annot = True)
plt.show()


# - From above heat map, we can see all the compnents are not correlated to each other.

# ### Creating the dataframe of all 5 principal components

# In[ ]:


pcs_df2 = pd.DataFrame({'PC1':pc[0],'PC2':pc[1],'PC3':pc[2],'PC4':pc[3],'PC5':pc[4]})


# #### Checking outliers of all the principal complnents

# In[ ]:


fig = plt.figure(figsize = (12,8))
sns.boxplot(data=pcs_df2)
plt.show()


# - From above boxplots, we can see the Outliers in the data, So we will do the outlier treatment below

# In[ ]:


pcs_df2.shape


# In[ ]:


pcs_df2.head()


# In[ ]:


#Visualising the points on the PCs.
# one of the prime advatanges of PCA is that you can visualise high dimensional data
fig = plt.figure(figsize = (12,8))
sns.scatterplot(x='PC1',y='PC2',data=pcs_df2)
plt.show()


# ![](http://)- We can see some of the grouping as before and after 0 value of PC1

# ### Clustering Process
# 
# #### Let's go ahead and begin with the clustering process i.e first we are calculating the Hopkins statistic

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


pcs_df2.info()


# #### Let's check the Hopkins measure

# In[ ]:


hopkins(pcs_df2)


# - Since the value is > 0.5 the given dataset has a good tendency to form clusters.

# In[ ]:


pcs_df2.shape


# - Assigning pcs_df2 dataframe to a new variable

# In[ ]:


dat3_1 = pcs_df2


# ### Performing k-Means Clustering
# 
# #### First we'll do the silhouette score analysis

# In[ ]:


from sklearn.metrics import silhouette_score
sse_ = []
for k in range(2, 10):
    kmeans = KMeans(n_clusters=k).fit(dat3_1)
    sse_.append([k, silhouette_score(dat3_1, kmeans.labels_)])


# In[ ]:


plt.plot(pd.DataFrame(sse_)[0], pd.DataFrame(sse_)[1])
plt.show()


# - From the above analysis we find that 3 seems to be a good number of clusters for K means algorithm.

# ### Checking with elbow curve

# In[ ]:


ssd = []
for num_clusters in list(range(1,10)):
    model_clus = KMeans(n_clusters = num_clusters, max_iter=50)
    model_clus.fit(dat3_1)
    ssd.append(model_clus.inertia_)

plt.plot(ssd)
plt.show()


# - Here also we're seeing a distinct bend at around 3 clusters. Hence, checking with Silhouette Analysis also .
# 
# ### Again we are doing Silhouette Analysis with scores
# 

# In[ ]:


# silhouette analysis
range_n_clusters = [2, 3, 4, 5, 6, 7, 8]

for num_clusters in range_n_clusters:
    
    # intialise kmeans
    kmeans = KMeans(n_clusters=num_clusters, max_iter=50)
    kmeans.fit(dat3_1)
    
    cluster_labels = kmeans.labels_
    
    # silhouette score
    silhouette_avg = silhouette_score(dat3_1, cluster_labels)
    print("For n_clusters={0}, the silhouette score is {1}".format(num_clusters, silhouette_avg))


# - As per elbow curve, Let's perform K means using K=3

# In[ ]:


model_clus2 = KMeans(n_clusters = 3, max_iter=50,random_state = 50)
model_clus2.fit(dat3_1)


# In[ ]:


dat4=pcs_df2
dat4.index = pd.RangeIndex(len(dat4.index))
dat_km = pd.concat([dat4, pd.Series(model_clus2.labels_)], axis=1)
dat_km.columns = ['PC1', 'PC2','PC3','PC4','PC5','ClusterID']
dat_km


# In[ ]:


dat_km['ClusterID'].value_counts()


# In[ ]:


fig = plt.figure(figsize = (12,8))
sns.scatterplot(x='PC1',y='PC2',hue='ClusterID',legend='full',data=dat_km)

plt.title('Categories of countries on the basis of Components')
plt.show()


# - From above plots, we can clearly see the 3 clusters .
# 
# - Now we are merging clusters with the original dataset.

# In[ ]:


dat5=pd.merge(Country_data,dat_km, left_index=True,right_index=True)
dat5.head()


# #### Dropping all the principal components from merged dataframe.

# In[ ]:


dat6=dat5.drop(['PC1','PC2','PC3','PC4','PC5'],axis=1)
dat6.head()


# In[ ]:


dat6.shape


# - So we have performed the clustering using the PCs and have now allocated the clusterIDs back to GDPP, child_mort, income of the datapoints.
# 
# ### Analysis of the clusters
# 
# #### calculating the mean of all the variables of each clusters.

# In[ ]:


Cluster_GDPP=pd.DataFrame(dat6.groupby(["ClusterID"]).gdpp.mean())
Cluster_child_mort=pd.DataFrame(dat6.groupby(["ClusterID"]).child_mort.mean())
Cluster_exports=pd.DataFrame(dat6.groupby(["ClusterID"]).exports.mean())
Cluster_income=pd.DataFrame(dat6.groupby(["ClusterID"]).income.mean())
Cluster_health=pd.DataFrame(dat6.groupby(["ClusterID"]).health.mean())
Cluster_imports=pd.DataFrame(dat6.groupby(["ClusterID"]).imports.mean())
Cluster_inflation=pd.DataFrame(dat6.groupby(["ClusterID"]).inflation.mean())
Cluster_life_expec=pd.DataFrame(dat6.groupby(["ClusterID"]).life_expec.mean())
Cluster_total_fer=pd.DataFrame(dat6.groupby(["ClusterID"]).total_fer.mean())


# #### Concatenating all the grouped by data to create a new dataframe to find required mean.

# In[ ]:


df = pd.concat([Cluster_GDPP,Cluster_child_mort,Cluster_income,Cluster_exports,Cluster_health,
                Cluster_imports,Cluster_inflation,Cluster_life_expec,Cluster_total_fer], axis=1)


# - Creating a dataframe of mean of all the variables of all the clusters

# In[ ]:


df.columns = ["GDPP","child_mort","income","exports","health","imports","inflation","life_expec","total_fer"]
df


# - Analysing the clusters by comparing how the [gdpp, child_mort and income] vary for each cluster of countries to recognise and differentiate the clusters of developed countries from the clusters of under-developed countries.
# 
# - From above dataframe of means, we got the mean data of under-developed countries. So, creating a dataframe on the basis of same.

# In[ ]:


fig = plt.figure(figsize = (10,6))
df.rename(index={0: 'Developed Countries'},inplace=True)
df.rename(index={1: 'Developing Countries'},inplace=True)
df.rename(index={2: 'Under-developed Countries'},inplace=True)
s=sns.barplot(x=df.index,y='GDPP',data=df)
plt.xlabel('Country Groups', fontsize=10)
plt.ylabel('GDP per Capita', fontsize=10)
plt.title('Country Groups On the basis of GDPP')
plt.show()


# - Above bar chart shows that, all the developed countries are having high GDP per capita values, developing countries are having average GDP per capita values and poor countries are having the least GDPP values.

# In[ ]:


fig = plt.figure(figsize = (10,6))
sns.barplot(x=df.index,y='income',data=df)
plt.xlabel('Country Groups', fontsize=10)
plt.title('Country Groups On the basis of Income')
plt.show()


# - Similarly, Above bar chart shows that, all the developed countries are having high income per person, developing countries are having average income per person and poor countries are having the least income per person.

# In[ ]:


fig = plt.figure(figsize = (10,6))
sns.barplot(x=df.index,y='child_mort',data=df)
plt.xlabel('Country Groups', fontsize=10)
plt.title('Country Groups On the basis of Child_mort Rate')
plt.show()


# - So, Above bar chart shows that, all the developed countries are having low number of death of children under 5 years of age per 1000 live births, developing countries are having average death rate and poor countries are having the least daeth rate.

# In[ ]:


#Let's use the concept of binning
fin=Country_data[Country_data['gdpp']<=1909]
fin=fin[fin['child_mort']>= 92]
fin=fin[fin['income']<= 3897.35]


# - Merging to get the cluster ids

# In[ ]:


fin_k=pd.merge(fin,dat_km,left_index=True,right_index=True)


# In[ ]:


fin_k=fin_k.drop(['PC1','PC2','PC3','PC4','PC5'],axis=1)


# In[ ]:


fin_k.shape


# - After merging, we are getting 17 under-developed countries, where gdpp, income are less but child_mort is more.

# In[ ]:


fin_k_GDPP=fin_k.nsmallest(8,'gdpp')
fin_k_GDPP


# #### Above list shows all the top low GDPP countries.

# In[ ]:


fin_k_income=fin_k.nsmallest(8,'income')
fin_k_income


# #### Above list shows all the top low income countries.

# In[ ]:


fin_k_mort=fin_k.nlargest(8,'child_mort')
fin_k_mort


# #### Above list shows all the top high child mort countries.

# In[ ]:


fig = plt.figure(figsize = (12,8))
sns.scatterplot(x='gdpp',y='income',hue='ClusterID',legend='full',data=dat6)
plt.xlabel('GDP per Capita', fontsize=10)
plt.ylabel('Income per Person', fontsize=10)
plt.title('GDP per Capita vs Income per Person')
plt.show()


# - From above scatter plot of gdpp and income, we can see there is some clustering like where gdpp is more, then income is also more.

# In[ ]:


fig = plt.figure(figsize = (12,8))
sns.scatterplot(x='gdpp',y='child_mort',hue='ClusterID',legend='full',data=dat6)
plt.xlabel('GDP per Capita', fontsize=10)
plt.ylabel('Child_more rate', fontsize=10)
plt.title('GDP per Capita vs Child_more rate')
plt.show()


# - From above scatter plot of gdpp and child-mort, we can see there is some clustering where gdpp is more, there child-mort is low.

# In[ ]:


fig = plt.figure(figsize = (12,8))
sns.boxplot(x='ClusterID',y='gdpp',data=dat6)
plt.xlabel('Country Groups', fontsize=10)
plt.ylabel('GDP per Capita', fontsize=10)
plt.title('GDP per Capita of all the Country Groups')
plt.show()


# - Here, Developed countries are falling under 1st cluster because of high gdpp range. Poor countries are falling under cluster 2.

# In[ ]:


fig = plt.figure(figsize = (12,8))
sns.boxplot(x='ClusterID',y='income',data=dat6)
plt.xlabel('Country Groups', fontsize=10)
plt.ylabel('Income per person', fontsize=10)
plt.title('Income per person of all the Country Groups')
plt.show()


# - Here, As Developed countries are falling under 1st cluster that is 0, So the income is in high range i.e. under cluster 0. Poor contries are falling under cluster 2 as per the income also.

# In[ ]:


fig = plt.figure(figsize = (12,8))
sns.boxplot(x='ClusterID',y='child_mort',data=dat6)
plt.xlabel('Country Groups', fontsize=10)
plt.ylabel('Child_mort rate', fontsize=10)
plt.title('Child_mort rate of all the Country Groups')
plt.show()


# - From the above plots we can see poor countries are falling under cluster 2. So, the child_mort rate is more in these countries.

# In[ ]:


Developed_con_K=dat6[dat6['ClusterID']==0]
Avg_Developed_con_K=dat6[dat6['ClusterID']==1]
Poor_con_K=dat6[dat6['ClusterID']==2]


# In[ ]:


fig = plt.figure(figsize = (18,6))
s=sns.barplot(x='country',y='gdpp',data=Developed_con_K)
s.set_xticklabels(s.get_xticklabels(),rotation=90)
plt.xlabel('Country', fontsize=10)
plt.ylabel('GDP per Capita', fontsize=10)
plt.title('GDP per Capita of all the developed Countries ')
plt.show()


# - From the above barchart, we can see all the developed countries like Luxembourg, Australia, etc.

# In[ ]:


fig = plt.figure(figsize = (18,6))
s=sns.barplot(x='country',y='gdpp',data=Avg_Developed_con_K)
s.set_xticklabels(s.get_xticklabels(),rotation=90)
plt.xlabel('Country', fontsize=10)
plt.ylabel('GDP per Capita', fontsize=10)
plt.title('GDP per Capita of all the Developing Countries ')
plt.show()


# - From the above barchart, we can see all the Developing countries like Iran, Albania, etc.

# In[ ]:


fig = plt.figure(figsize = (18,6))
s=sns.barplot(x='country',y='gdpp',data=Poor_con_K)
s.set_xticklabels(s.get_xticklabels(),rotation=90)
plt.xlabel('Country', fontsize=10)
plt.ylabel('GDP per Capita', fontsize=10)
plt.title('GDP per Capita of all the Under-Developed Countries ')
plt.show()


# - From the above barchart, we can see all the Under-developed countries like Burundi, Afghanistan, etc.

# In[ ]:


fig = plt.figure(figsize = (18,6))
s=sns.barplot(x='country',y='child_mort',data=Poor_con_K)
s.set_xticklabels(s.get_xticklabels(),rotation=90)
plt.xlabel('Country', fontsize=10)
plt.ylabel('Child_mort Rate', fontsize=10)
plt.title('Child_mort Rate of all the Under-Developed Countries ')
plt.show()


# - From the above barchart, we can see all the Under-developed countries like Haiti, Sierra Leone, etc., where child_mort rate is more

# In[ ]:


fig = plt.figure(figsize = (18,6))
sns.barplot(x='country',y='gdpp',data=fin_k_GDPP)
plt.title('GDPP of Top 8 the Under-Developed Countries ')
plt.xlabel('Under-Developed Countries', fontsize=10)
plt.ylabel('GDPP', fontsize=10)
plt.show()


# - Top 8 under-developed countries in under-developement group, where GDPP is very low of countries like Burundi, Congo, etc.

# In[ ]:


fig = plt.figure(figsize = (18,6))
sns.barplot(x='country',y='child_mort',data=fin_k_mort)
plt.title('Child_mort rate of Top 8 the Under-Developed Countries ')
plt.xlabel('Under-Developed Countries', fontsize=10)
plt.ylabel('Child_mort rate', fontsize=10)
plt.show()


# - Top 8 under-developed countries in under-developement group, where child death is very high in countries like Haiti, Sierra, etc.

# In[ ]:


fig = plt.figure(figsize = (18,6))
sns.barplot(x='country',y='income',data=fin_k_income)
plt.title('Income of Top 8 the Under-Developed Countries ')
plt.xlabel('Under-Developed Countries', fontsize=10)
plt.ylabel('Income', fontsize=10)
plt.show()


# - Top 8 under-developed countries in under-developement group, where income is very low of countries like Congo, Burundi, etc.
# 
# 
# ### Performing Hirarchical Clustering
# 

# In[ ]:


pcs_df2.shape


# In[ ]:


pcs_df3 = pd.DataFrame({'PC1':pc[0],'PC2':pc[1],'PC3':pc[2],'PC4':pc[3],'PC5':pc[4]})


# In[ ]:


dat_km.head()


# #### Performing Single Linkage

# In[ ]:


mergings=linkage(pcs_df2,method='single',metric='euclidean')
dendrogram(mergings)
plt.show()


# #### Performing Complete Linkage

# In[ ]:


#mergings=linkage(fin,method='complete',metric='euclidean')
mergings=linkage(pcs_df2,method='complete',metric='euclidean')
dendrogram(mergings)
plt.show()


# - From above dendrograms, we can derive the 3 clusters.

# In[ ]:


cut_tree(mergings,n_clusters=3).shape


# - Let's reshape the cut_tree result array

# In[ ]:


cluser_labels=cut_tree(mergings,n_clusters=3).reshape(-1,)
cluser_labels


# In[ ]:


#assign cluster labels

dat_km['Cluster_lables']=cluser_labels
dat_km.head()


# In[ ]:


dat7=pd.merge(Country_data,dat_km, left_index=True,right_index=True)
dat7.head()


# In[ ]:


dat8=dat7.drop(['PC1','PC2','PC3','PC4','PC5'],axis=1)
dat8.shape


# In[ ]:


dat8.head()


# #### Analysis of the clusters

# In[ ]:


Cluster_GDPP_H=pd.DataFrame(dat8.groupby(["Cluster_lables"]).gdpp.mean())
Cluster_child_mort_H=pd.DataFrame(dat8.groupby(["Cluster_lables"]).child_mort.mean())
Cluster_income_H=pd.DataFrame(dat8.groupby(["Cluster_lables"]).income.mean())


# In[ ]:


df_H = pd.concat([Cluster_GDPP_H,Cluster_child_mort_H,Cluster_income_H], axis=1)


# In[ ]:


df_H.columns = ["GDPP","child_mort","income"]
df_H


# #### Filtering the final list of Under-Developed Countries where, more funding is required.

# In[ ]:


#Let's use the concept of binning
fin_H=Country_data[Country_data['gdpp']<=2330.000000]
fin_H=fin[fin['child_mort']>= 130.000000]
fin_H=fin[fin['income']<= 5150.000000]


# In[ ]:


fin_H=pd.merge(fin_H,dat_km,left_index=True,right_index=True)


# In[ ]:


fin_H=fin_H.drop(['PC1','PC2','PC3','PC4','PC5'],axis=1)
fin_H.shape


# In[ ]:


sns.boxplot(x='Cluster_lables',y='gdpp',data=dat8)
plt.show()


# - Here, Developed countries are falling under 2nd cluster because of high gdpp range. Poor countries are falling under cluster 0.

# In[ ]:


sns.boxplot(x='Cluster_lables',y='child_mort',data=dat8)
plt.show()


# - From the above plots we can see poor countries are falling under cluster 0. So, the child_mort rate is more in these countries.

# In[ ]:


sns.boxplot(x='Cluster_lables',y='income',data=dat8)
plt.show()


# - Here, As Developed countries are falling under cluster 1, So the income is in high range

# In[ ]:


fin_H.nsmallest(8,'gdpp')


# - After comparing both K-means and Heirarchical clustering method, I am going with the K-means outcomes as the plots are clearly visible. As in both the methods, the top 8 under-developed countries are similar. I am considering the result of k-means outcome.

# ## Conclusion
# 
# - After grouping all the countries into 3 groups by using some socio-economic and health factors, we can determine the overall development of the country.
# 
# - Here, the countries are categorised into list of developed countries, developeing countries and under-developed countries.
# 
# - In Developed countries, we can see the GDP per capita and income is high where as Death of children under 5 years of age per 1000 live births i.e. child-mort is very low, which is expected.
# 
# - In Developing countries and Under-developed countries, the GDP per capita and income are low and child-mort is high. Specifically, for under-developed countries, the death rate of children is very high.

# ## Recomendetions
# 
# - From bar chats, we can clearly see the socio-economic and heath situation of the under developed countries.
# - In countries like Haiti, Sierra Leone,Chad, etc., the death rate of children under 5 years of age per 1000 (child-mort) is high.
# - In countries like Burundi, Congo, Niger, etc., GDP per capita is very low. So, in those countries, the income per person is also low. So, these countries are considered as poor contries.
# - Finally, as per categories of the countries, top 8 under-developed countries which are in direst need of aid are as below:
# 
# 1. Burundi
# 2. Congo, Dem. Rep.
# 3. Niger
# 4. Sierra Leone
# 5. Haiti
# 6. Chad
# 7. Central African Republic
# 8. Mozambique

# In[ ]:




