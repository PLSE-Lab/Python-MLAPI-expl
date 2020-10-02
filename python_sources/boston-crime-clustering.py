#!/usr/bin/env python
# coding: utf-8

# # Unsupervised Clustering
# 
# by : Hesham Asem
# ____
# 
# Here we'll take data about criminal accidents , which happend in Boston between 2015 & 2018 , and we need to classify it in unsupervised way , so we can take each segment to handle it later
# 
# 
# also you can find the data here : 
# 
# https://www.kaggle.com/AnalyzeBoston/crimes-in-boston/kernels
# 
# 
# ____
# 
# lets first import needed libraries
# 
# 

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch


# 
# # Loading Data
# 
# now lets read the data file , don't forget to specify the encoding to latin-1 to avoid any error

# In[ ]:


data = pd.read_csv( '/kaggle/input/crime.csv', encoding='latin-1')

data.head()


# 
# _____
# 
# 
# looks a mixed data which containsboth numerical & categorical values , lets check the shape
# 
# 

# In[ ]:


data.shape


# how about the range of numbers of numerical values . . 

# In[ ]:


data.describe()


# _____
# 
# when you look carfeully at the numbers , you can see the Offense Code is not a "numerical value" , but looks categorical 
# 
# also min year is 2015 & max is 2016
# 
# also for Lat & Long values , almost all of them around (42 , -71) , which refer to Boston city . . 
# 
# how about the Nulls ? 
# 

# In[ ]:


data.info()


# ____
# 
# # Data Processing
# 
# a huge amount of nulls at Shooting feature , also exists in ( District , UCR Part , Street , both Lat & Long )
# 
# before handling nulls or making categorical , we need to look at unique values , so we can drop features which have a very big amount of unique values , which will never help us in categorical features
# 
# 

# In[ ]:


for column in data.columns : 
    print('Length of unique data for {0} is {1} '.format(column , len(data[column].unique())))
    


# _____
# 
# Ok , Incident Number is anyway a reference number which will not help in training. . 
# 
# and the date will not help us so much since we'll keep the data of ( year , month & weekday ). 
# 
# also street feature have a huge amount of unique values so we can drop it since we'll keep long & lat , and we'll drop reporting area and ocation for the same reason . 
# 
# also we can drop offense code since we'llkeep offense code group to avoid make 222 values at get_dummies .
# 
# lets drop unhelpful features . . 
# 
# 
# 

# In[ ]:


data.drop(['INCIDENT_NUMBER' , 'OCCURRED_ON_DATE' ,'STREET' , 'OFFENSE_CODE' , 'REPORTING_AREA','Location'],axis=1, inplace=True)

data.head()


# lets check it now  

# In[ ]:


for column in data.columns : 
    print('Length of unique data for {0} is {1} '.format(column , len(data[column].unique())))


# ____
# 
# # Handling Location
# 
# now for Lat & Long data , it looks it got a huge amount of unique data , so we can never convert it to categorical data in this way , and also we cannot get rid of it , since it's very important imformation for clustering the data . . 
# 
# so we can round the Lat & Long data to two decimal numbers , so it will - geographically - classify the whole city into specific zones , & will reduce the amount of unique data for them
# 

# In[ ]:


data['Lat code'] = np.round(data['Lat'],2)
data['Long code'] = np.round(data['Long'],2)

data.drop(['Lat','Long'],axis=1, inplace=True)


# now lets check the data

# In[ ]:


data.head()


# now it looks more simple & smart , lets see the amount of unique data

# In[ ]:


for column in data.columns : 
    print('Length of unique data for {0} is {1} '.format(column , len(data[column].unique())))


# ____
# cool , 20 unique Lat & 22 unique Long , which will refer to 440 zones in Boston city , which is very helpful in making clusters 
# 
# 
# ___
# 
# # Handling Nulls
# 
# now we have to manage all missing data in other features , let's see it now 
# 

# In[ ]:


data.info()


# ____
# 
# shooting feature got the biggest amount of Nulls , lets see what it contains 

# In[ ]:


data['SHOOTING'].unique()


# 
# ok , it either Y for yes there was shooting in the crime , or nan for the crime contain not shooting , ok , let's make a new feature , which will got 1 for shooting & 0 for no shooting

# In[ ]:


data['shooting code'] = np.where(data['SHOOTING']=='Y' , 1 , 0)


# also we can know the percentage of shooting among all accidents happend here 

# In[ ]:


print( 'Shooting Percentage  is {} %'.format(round((data['shooting code'].sum() / data.shape[0]) * 100,2)))


# ok , a very tiny percentage  , so we'll keep the shooting code feature & drop the original shooting feature , also let's check other features nulls 

# In[ ]:


data.drop(['SHOOTING'],axis=1, inplace=True)


data.info()


# ___
# 
# more than 1700 nulls in the district feature , lets check the unique values

# In[ ]:


data['DISTRICT'].unique()


# since we cannot replace nulls with mean or median for categorical values , so we'll just put the word none instead of nulls , using fillna tool

# In[ ]:


data.DISTRICT.fillna('none', inplace=True)


# let's check it again 

# In[ ]:


data['DISTRICT'].unique()


# cool , no nulls now , lets move to UCR Part
# 

# In[ ]:


data['UCR_PART'].unique()


# again , we'll have to put any word since we cannot calculate mean , so lets use none

# In[ ]:


data.UCR_PART.fillna('none', inplace=True)


# check it now

# In[ ]:


data['UCR_PART'].unique()


# ____
# 
# now for Lat & Long , first lets get the mean values for them & round it 

# In[ ]:


lat_mean = data['Lat code'].sum() / 299074
long_mean =  data['Long code'].sum() / 299074
print(round(lat_mean,2))
print(round(long_mean,2))


# so we can fill null values in Lat & Long with that mean 

# In[ ]:


data['Lat code'].fillna(round(lat_mean,2), inplace=True)
data['Long code'].fillna(round(long_mean,2), inplace=True)


# now we filled all nulls & it's ready for categorical values , lets have last check it data 

# In[ ]:


data.info()


# cool , no nulls there , lets have a look to the features

# In[ ]:


data.head()


# ____
# 
# # Get Dummies
# 
# ok , before we run the clustering model , we've to convert all categorical values into numerical dummies . 
# 
# we have here five categorical features which are : 
# 
# Offense Groud
# 
# Offense Description
# 
# District
# 
# Day of week
# 
# UCR Part
# 
# 
# 
# so we'll use LabelEncoder model from Sklearn to do it quickly , then drop the original feature . 
# 
# 
# lets start with Offense Group . .
# 
# 
# 

# In[ ]:


enc  = LabelEncoder()
enc.fit(data['OFFENSE_CODE_GROUP'])
data['Offense Code'] = enc.transform(data['OFFENSE_CODE_GROUP'])
data.drop(['OFFENSE_CODE_GROUP'],axis=1, inplace=True)


# ok done , let's have a look

# In[ ]:


data.head()


# looks great , letl's repeat it to Offense Description

# In[ ]:


enc  = LabelEncoder()
enc.fit(data['OFFENSE_DESCRIPTION'])
data['Offense Desc Code'] = enc.transform(data['OFFENSE_DESCRIPTION'])
data.drop(['OFFENSE_DESCRIPTION'],axis=1, inplace=True)
data.head()


# then District

# In[ ]:


enc  = LabelEncoder()
enc.fit(data['DISTRICT'])
data['District Code'] = enc.transform(data['DISTRICT'])
data.drop(['DISTRICT'],axis=1, inplace=True)
data.head()


# & day of week

# In[ ]:


enc  = LabelEncoder()
enc.fit(data['DAY_OF_WEEK'])
data['Day Code'] = enc.transform(data['DAY_OF_WEEK'])
data.drop(['DAY_OF_WEEK'],axis=1, inplace=True)
data.head()


# then UCR part

# In[ ]:


enc  = LabelEncoder()
enc.fit(data['UCR_PART'])
data['UCR Code'] = enc.transform(data['UCR_PART'])
data.drop(['UCR_PART'],axis=1, inplace=True)
data.head()


# ok , lets have a look to the data 

# In[ ]:


data.describe()


# and make the last check or nulls

# In[ ]:


data.info()


# ___
# 
# # Run the Model
# 
# 
# so we are ready to run the unsupervised model for it now 
# 
# 
# first let's split the data to train & test

# In[ ]:


X_train = data[:250000]
X_test = data[250000:]


# how about the shape ? 

# In[ ]:


print('X Train Shape is {}'.format(X_train.shape))
print('X Test Shape is {}'.format(X_test.shape))


# ____
# 
# so we'll use 3 models to choose the best one
# 
# lets start with Kmeans , from Sklearn

# In[ ]:


KMeansModel = KMeans(n_clusters=5,init='k-means++', #also can be random
                     random_state=33,algorithm= 'auto') # also can be full or elkan
KMeansModel.fit(X_train)


# ____
# 
# now we need to have a look to its attributes

# In[ ]:


print('KMeansModel centers are : ' , KMeansModel.cluster_centers_)
print('---------------------------------------------------')
print('KMeansModel labels are : ' , KMeansModel.labels_[:20])
print('---------------------------------------------------')
print('KMeansModel intertia is : ' , KMeansModel.inertia_)
print('---------------------------------------------------')
print('KMeansModel No. of iteration is : ' , KMeansModel.n_iter_)
print('---------------------------------------------------')


# how about predicting from x_test

# In[ ]:


#Calculating Prediction
y_pred = KMeansModel.predict(X_test)
print('Predicted Value for KMeansModel is : ' , y_pred[:10])


# ____
# 
# then lets use KNN model for unsupervised training

# In[ ]:


NearestNeighborsModel = NearestNeighbors(n_neighbors=4,radius=1.0,algorithm='auto')#it can be:ball_tree,kd_tree,brute
NearestNeighborsModel.fit(X_train)


# then check the attributes

# In[ ]:


#Calculating Details
print('NearestNeighborsModel Train kneighbors are : ' , NearestNeighborsModel.kneighbors(X_train[: 5]))
print('----------------------------------------------------')
print('NearestNeighborsModel Train radius kneighbors are : ' , NearestNeighborsModel.radius_neighbors(X_train[:  1]))
print('----------------------------------------------------')
print('NearestNeighborsModel Test kneighbors are : ' , NearestNeighborsModel.kneighbors(X_test[: 5]))
print('----------------------------------------------------')
print('NearestNeighborsModel Test  radius kneighbors are : ' , NearestNeighborsModel.radius_neighbors(X_test[:  1]))
print('----------------------------------------------------')


# ____
# 
# also we can use Hierarchical clusering , it migh be useful , but we've to limit the using data to a tiny amount , lets say we'll check it in the first 1000 sample size

# In[ ]:


AggClusteringModel = AgglomerativeClustering(n_clusters=5,affinity='euclidean',# it can be l1,l2,manhattan,cosine,precomputed
                                             linkage='ward')# it can be complete,average,single

y_pred_train = AggClusteringModel.fit_predict(X_train[:1000])
y_pred_test = AggClusteringModel.fit_predict(X_test[:1000])


# now we can draw the dendogram using Scipy , for the first 30 record of training set

# In[ ]:


#draw the Hierarchical graph for Training set
dendrogram = sch.dendrogram(sch.linkage(X_train[:30], method = 'ward'))# it can be complete,average,single
plt.title('Training Set')
plt.xlabel('X Values')
plt.ylabel('Distances')
plt.show()


# and we can check it in the first 30 record in the test set

# In[ ]:


#draw the Hierarchical graph for Test set
dendrogram = sch.dendrogram(sch.linkage(X_test[:30], method = 'ward'))# it can be complete,average,single
plt.title('Test Set')
plt.xlabel('X Value')
plt.ylabel('Distances')
plt.show()


# ___
# 
# 
# # Finally
# 
# as we saw now , data processing is the most important step to manipulate data & make it ready for our model 
# 
# 
