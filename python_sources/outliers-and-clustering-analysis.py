#!/usr/bin/env python
# coding: utf-8

# Hello! I'm a begginer and I had the following ideia about this dataset:
# 
#     What if we were asked to answer the following question: 
#     Where should we place facilities in order to attend the emergency calls?
# 
# To do so, I intend to make a clustering analysis and show how outliers can be damaging in the results if not well treated. 
# 
# It's a simple analysis and the main goal here is to show the importance of treating outliers in a clusterization (mainly for begginers like me). So please, don't get alarmed if something doesn't make much sense for this particular dataset.
# 
# So, this notebook has the following objectives:
# 1. How to treat outliers?
# 1. Make a cluster analysis (KMeans) comparing outliers with treated data  
# 
# 
# Summary:
# 1. Loading and preparing data
#     * Quick overview of the data
# 2. Treating outliers
#     * Geographic distribution of the data (using Folium)
#     * Treating outliers with Boxplot
#     * Cleaning outliers
# 3. Clustering
#     * KMeans
#     * Elbow technique
#     * Treated data
#     * Non treated data
#     * Comparison
# 
# 
# Thanks in advance. I hope you enjoy it!
# 
# References: 
# 
# https://medium.com/analytics-vidhya/outlier-treatment-9bbe87384d02 
# 
# https://www.scikit-yb.org/en/latest/api/cluster/elbow.html
# 
# 

# In[ ]:


#Importing libraries
import pandas as pd
import seaborn as sns
import numpy as np
import folium as fo
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer


# # 1. Loading and preparing data

# In[ ]:


#reading data
df=pd.read_csv('/kaggle/input/montcoalert/911.csv')


# In[ ]:


# A quick loock in the data
df.head()


# In[ ]:


df.info()


# In[ ]:


#Creating columns that might be useful
df['timeStamp']=df['timeStamp'].apply(pd.Timestamp)
df['Year']=df.timeStamp.dt.year
df['Month']=df.timeStamp.dt.month


# In[ ]:


df['Year'].value_counts()


# In[ ]:


#since 2015 and 2020 are not complete, we're not using them
df=df[df['Year'].isin([2016,2017,2018,2019])]


# In[ ]:


df['title'].unique()


# In[ ]:


#There are 3 major categories: EMS (Emergency Medical Services), Traffic and Fire, 
#so let's use them to aggregate data
df['title']=df['title'].str.split(':').str.get(0)


# In[ ]:


#How many calls per year?
plt.figure(figsize=(10,6))
sns.set_palette('tab10')
sns.set_style('darkgrid')
sns.countplot(x='Year',data=df)
plt.title('Number of calls per year')


# The number doesn't change much from one to another year

# In[ ]:


#How many calls per month over the years?
plt.figure(figsize=(13,7))
sns.set_style('darkgrid')
sns.countplot(x='Month',hue='Year',data=df)
plt.title('Number of calls per month over the years')


# These 2 last plots show that the number of calls doesn't vary much over the months. Apparently there's not some kind of seasonality.

# In[ ]:


#How many calls per category (title)?
plt.figure(figsize=(10,6))
sns.set_style('darkgrid')
sns.countplot(x='Year',hue='title',data=df,hue_order=['EMS','Traffic','Fire'])
plt.title('Number of calls per per category over the years')


# In[ ]:


#How many calls per category (title) over the months?
plt.figure(figsize=(10,6))
sns.set_style('darkgrid')
sns.countplot(x='Month',hue='title',data=df,hue_order=['EMS','Traffic','Fire'])
plt.title('Number of calls per per category over the months')


# Apparently the number of calls doesn't change much per category. We could make a deeper analysis into each category, but that's not my goal here.

#  In order to answer the question presented in the introduction, we will consider only EMS (Emergency Medical Services) to make the clustering analysis, considering the hospitals are the ones responsible for attending this service category.

# In[ ]:


EMS=df[df['title']=='EMS']


# In[ ]:


#Top 25 towns with more calls
top_25=EMS['twp'].value_counts(ascending=False,normalize=True).head(25).index
plt.figure(figsize=(10,6))
sns.countplot(y='twp',data=EMS,order=top_25)
plt.title('Top 25 towns with more calls')


# # 2. Treating outliers

# In[ ]:


df.head(3)


# In[ ]:


#Let's plot a few points to take a look
Map=fo.Map([40.121354,-75.363829],zoom_start=7)
random_index=np.random.choice(df.index,1000) #getting some random points to plot
for ind in random_index:
    lat=df.loc[ind,'lat']
    long=df.loc[ind,'lng']
    fo.CircleMarker([lat,long],radius=2).add_to(Map)
Map


# Apparently the data is well concentrated in a particular region of PA. Let's make a scatterplot with the whole data

# In[ ]:


sns.scatterplot(x=EMS['lat'],y=EMS['lng'],data=df, alpha=0.3)


# Apparently there are some outliers. Let's use boxplot to do this analysis.

# In[ ]:


#Analysing Latitude
sns.boxplot(x=EMS['lat'])


# In[ ]:


#Getting outliers data (take a look at https://medium.com/analytics-vidhya/outlier-treatment-9bbe87384d02)
Q1=EMS['lat'].quantile(.25)
Q3=EMS['lat'].quantile(.75)
IQR=Q3-Q1
Lower_Whisker=Q1-1.5*IQR
Upper_Whisker = Q3+1.5*IQR


# In[ ]:


#How many outliers?
EMS[EMS['lat']>Upper_Whisker].shape[0]+EMS[EMS['lat']<Lower_Whisker].shape[0]


# In[ ]:


#Let's save the outliers for posteriori analysis
outliers=EMS[(EMS['lat']>Upper_Whisker)|(EMS['lat']<Lower_Whisker)]


# In[ ]:


#removing outliers
EMS_treated=EMS[(EMS['lat']<Upper_Whisker)&(EMS['lat']>Lower_Whisker)]


# In[ ]:


#Plotting the treated data 
sns.scatterplot(x=EMS_treated['lat'],y=EMS_treated['lng'],data=df)


# In[ ]:


#boxplot with treated data
sns.boxplot(x=EMS_treated['lat'])


# It's possible to see a big difference already. Those who knows a bit about latitude and longitude know that a small variance in any value (lat or long) can make a big difference. Let's repeat the process for longitude.

# In[ ]:


#the same process for long
sns.boxplot(x=EMS_treated['lng'])


# In[ ]:


Q1=EMS_treated['lng'].quantile(.25)
Q3=EMS_treated['lng'].quantile(.75)
IQR=Q3-Q1
Lower_Whisker=Q1-1.5*IQR
Upper_Whisker = Q3+1.5*IQR


# In[ ]:


#How many outliers?
EMS_treated[EMS_treated['lng']>Upper_Whisker].shape[0]+EMS_treated[EMS_treated['lng']<Lower_Whisker].shape[0]


# In[ ]:


#saving outliers
outliers=pd.concat([EMS_treated[(EMS_treated['lng']>Upper_Whisker)|(EMS_treated['lng']<Lower_Whisker)],outliers])


# In[ ]:


#cleaning longitude outliers
EMS_treated=EMS_treated[(EMS_treated['lng']<Upper_Whisker)&(EMS_treated['lng']>Lower_Whisker)]


# In[ ]:


#Total outliers
outliers.shape[0]


# In[ ]:


#Percentagem in relation to the whole data
(outliers.shape[0]/EMS.shape[0])*100


# We can just drop the outliers since they are not so representative. If this was the case, we could consider a way to treat these data in order to use them. 
# 

# In[ ]:


#Let's see again the treated data
#It's clear the improvement of this plot in relation to the first ones
sns.scatterplot(x=EMS_treated['lat'],y=EMS_treated['lng'],data=df)


# In[ ]:


sns.boxplot(x=EMS_treated['lng'])


# Now we can plot these data in a real map to see if there is really some difference

# In[ ]:


outliers.head(1)


# In[ ]:


#Outliers in red
Map_outliers=fo.Map([40.269061,-75.69959],zoom_start=6)
for index,row in outliers.iterrows():
    lat=row['lat']
    long=row['lng']
    fo.CircleMarker([lat,long],radius=2,color='red').add_to(Map_outliers)

# Treated data in blue
random_indexes=np.random.choice(EMS_treated.index,2000)
for rand_in in random_indexes:
    lat=EMS_treated.loc[rand_in,'lat']
    long=EMS_treated.loc[rand_in,'lng']
    fo.CircleMarker([lat,long],radius=2,color='blue').add_to(Map_outliers)
Map_outliers


# If we zoom out we can see points in other states of USA. Also there are points even in the ocean and in another continent.

# One may say that there are points next to the aggregated data (those near to West Grove and East Greenville, for instance)  and even so they were considered as outliers. Again, these data are not so representative in relation to the whole data and, in a practical way, these calls can easily be attended since they are not far from the massive data, where the facilities are to be located in order to attend the calls.

# Once we have treated the outliers, let's work on the clusterization.

# # 3. Clustering

# In[ ]:


#Getting lat long
X=np.array(EMS_treated[['lat','lng']])


# How many clusters should we use?
# I'm using the well known elbow technique to find out. (Take a look at https://www.scikit-yb.org/en/latest/api/cluster/elbow.html)
# 
# An alternative is to use an automatic clustering algorithm.

# In[ ]:


#creating instance of KMeans
kmeans=KMeans(init='k-means++')


# In[ ]:


#Creating instance of Elbow Visualizer to find how many clusters we use
visualizer = KElbowVisualizer(model=kmeans, k=(5,20))


# In[ ]:


#Fitting the model
visualizer.fit(X)      


# We will use k=10 according to the Visualizer.

# In[ ]:


clustering=KMeans(init='k-means++',n_clusters=10)
clustering.fit(X)    


# In[ ]:


clusters=clustering.cluster_centers_


# Let's see where the clusters were positioned with a few points

# In[ ]:


Map=fo.Map([ 40.13572425, -75.20909773],zoom_start=8)
for i in range(1000):
    random_index=np.random.choice(EMS_treated.index,1)
    lat=df.loc[random_index,'lat']
    long=df.loc[random_index,'lng']
    fo.Circle([lat,long],radius=2).add_to(Map)     
    
for c in clusters:
    lat=c[0]
    long=c[1]
    fo.RegularPolygonMarker([lat,long],radius=4,number_of_sides=3,color='black').add_to(Map) 
Map


# A clusterization can gives us an ideia of how to alocate our facilities in order to attend the anual emergency calls. Of course in a real case we cannot rely our answer only in a clustering result, but this is a good "first kick" to start our analysis, since KMeans clustering is based on data distribution.

# And what if we had used the whole data, that is, without treating the outliers?

# In[ ]:


#Clustering including outliers
X_outliers=np.array(EMS[['lat','lng']])
clustering_outliers=KMeans(init='k-means++',n_clusters=10)
clustering_outliers.fit(X_outliers)
clusters_outliers=clustering_outliers.cluster_centers_


# In[ ]:


#Adding the outlier clusters to the Map
for c in clusters_outliers:
    lat=c[0]
    long=c[1]
    fo.RegularPolygonMarker([lat,long],radius=4,number_of_sides=3,color='red').add_to(Map) 
Map


# If we zoom out, the algorithm has placed clusters far away from PA. That's because those outlier points we recognized as clusters. This shows us the importance of make a good data cleaning/preparing. That's crucial for any machine learning model.

# That's all folks. I'll be glad to hear your feedback and suggestions. Thank you so much!
