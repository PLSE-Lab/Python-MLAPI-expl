#!/usr/bin/env python
# coding: utf-8

# # **Breast CANCER Prediction**

#  **diagnosis**:The diagnosis of breast tissues (M = malignant, B = benign)
# * **radius_mean**: of distances from center to points on the perimeter
# * **texture_mean**:standard deviation of gray-scale values
# * **perimeter_mean**: size of the core tumor
# * **area_mean**
# * **smoothness_meanmean** of local variation in radius lengths
# * compactness_meanmean of perimeter^2 / area - 1.0
# * concavity_meanmean of severity of concave portions of the contour
# * concave points_meanmean for number of concave portions of the contour
# * symmetry_mean
# * **fractal_dimension_mean**:mean for "coastline approximation" - 1
# * **radius_se**:standard error for the mean of distances from center to points on the perimeter
# * **texture_se**:standard error for standard deviation of gray-scale values
# * perimeter_se
# * area_se
# * **smoothness_se**:standard error for local variation in radius lengths
# * **compactness_se**:standard error for perimeter^2 / area - 1.0
# * **concavity_se**standard error for severity of concave portions of the contour
# * **concave points_se**standard error for number of concave portions of the contour
# * symmetry_se
# * fractal_dimension_sestandard error for "coastline approximation" - 1
# * radius_worst**"worst"** or largest mean value for mean of distances from center to points on the perimeter
# * texture_worst**"worst"** or largest mean value for standard deviation of gray-scale values
# * perimeter_worst
# * area_worst
# * smoothness_worst**"worst"** or largest mean value for local variation in radius lengths

# # BY USING BASICS LIBARIES AND ALOGRITIMS ANALYSISING AND PERFORMING  

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


df=pd.read_csv("../input/breast-cancer-wisconsin-data/data.csv")


# # Data Exploration

# In[ ]:


df


# In[ ]:


df.info()


# In[ ]:


df.columns


# In[ ]:


df.shape


# In[ ]:


df.drop(["id"],axis=1,inplace=True)


# In[ ]:


df.diagnosis.unique()


# # Data Cleaning

# In[ ]:


#check for missing values
df.isnull()


# In[ ]:


df.isnull().sum()


# In[ ]:


#check for duplicates
df.duplicated().sum()


# In[ ]:


df.head()


# # Feature Engineering

# In[ ]:


df.describe()


# In[ ]:


#removing outliers in mean_area
df.area_mean.quantile(0.999)


# In[ ]:


df.shape


# In[ ]:


df= df[df.area_mean<=df.area_mean.quantile(0.999999)]


# In[ ]:


df.shape


# In[ ]:


df.describe()


# # AGGREGATION

# In[ ]:


df.groupby(['diagnosis'])['radius_mean'].mean()


# In[ ]:


df.groupby(['diagnosis'])['texture_mean'].mean()


# In[ ]:


df.groupby(['diagnosis'])['area_mean'].mean()


# In[ ]:


df.groupby(['diagnosis'])['perimeter_mean'].mean()


# In[ ]:


df.groupby(['diagnosis'])['smoothness_mean'].mean()


# In[ ]:


df.


# # DATA VISUALIZATION

# In[ ]:


#analytics btw catergorical and numeric
plt.figure(figsize=(12,5))
sns.distplot(df.radius_mean[df.diagnosis=='M'])
sns.distplot(df.radius_mean[df.diagnosis=='B'])
plt.legend(['negative','positive'])
plt.show()


# At range radius 5 to 15 case is positive

# In[ ]:


plt.figure(figsize=(12,5))
sns.distplot(df.texture_mean[df.diagnosis=='M'])
sns.distplot(df.texture_mean[df.diagnosis=='B'])
plt.legend(['negative','positive'])
plt.show()


# In[ ]:


plt.figure(figsize=(12,5))
sns.distplot(df.perimeter_mean[df.diagnosis=='M'])
sns.distplot(df.perimeter_mean[df.diagnosis=='B'])
plt.legend(['negative','positive'])
plt.show()


# **at range 50 to 100  case is positive**

# In[ ]:


plt.figure(figsize=(12,5))
sns.distplot(df.area_mean[df.diagnosis=='M'])
sns.distplot(df.area_mean[df.diagnosis=='B'])
plt.legend(['negative','positive'])
plt.show()


# **At range 0 to 500 case is postive (mean_area might be a major factor)**

# In[ ]:


plt.figure(figsize=(12,5))
sns.distplot(df.smoothness_mean[df.diagnosis=='M'])
sns.distplot(df.smoothness_mean[df.diagnosis=='B'])
plt.legend(['negative','positive'])
plt.show()


# **At range of 0.06 to 0.08 is positive but not a effective factor**

# In[ ]:


#categorical v/s numerical v/s categorical 
#x ,y, hue>> x= cateogircal,y=numerical,hue= cateogrical
plt.figure(figsize=(12,5))
sns.scatterplot(x='radius_mean',y='area_mean',hue='diagnosis',data=df)
plt.show()


#  **radius and area are major impacting factor
#  as radius goes on  increasing and area goes on increasing ,case is becoming positive**

# In[ ]:


#categorical v/s numerical v/s categorical 
#x ,y, hue>> x= cateogircal,y=numerical,hue= cateogrical
plt.figure(figsize=(12,5))
sns.scatterplot(x='radius_mean',y='perimeter_mean',hue='diagnosis',data=df)
plt.show()


# In[ ]:


#categorical v/s numerical v/s categorical 
#x ,y, hue>> x= cateogircal,y=numerical,hue= cateogrical
plt.figure(figsize=(12,5))
sns.scatterplot(x='area_mean',y='perimeter_mean',hue='diagnosis',data=df)
plt.show()


# In[ ]:


plt.figure(figsize=(12,5))
sns.scatterplot(x='texture_mean',y='smoothness_mean',hue='diagnosis',data=df)
plt.show()


# In[ ]:


plt.figure(figsize=(12,5))
sns.scatterplot(x='radius_mean',y='smoothness_mean',hue='diagnosis',data=df)
plt.show()


# In[ ]:


plt.figure(figsize=(12,5))
sns.scatterplot(x='radius_mean',y='texture_mean',hue='diagnosis',data=df)
plt.show()


# # Correlation

# In[ ]:


#correlation analysis
cor = df.corr()
#heatmap for visualization correlation analysis
plt.figure(figsize=(10,8))
sns.heatmap(cor,annot=True,cmap='coolwarm')#if we will not write annot=True then the values will not show
plt.show()


# In[ ]:


x=df[['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean','smoothness_mean']]
y=df["diagnosis"]


# # Preprocessing

# In[ ]:


x.head()


# In[ ]:


y.head()


# In[ ]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x=sc.fit_transform(x)


# In[ ]:


#splitting data info train and test set
from sklearn.model_selection import train_test_split
xtr,xts,ytr,yts = train_test_split(x,y,test_size=0.2)
print(x.shape)
print(xtr.shape)
print(xts.shape)
print(y.shape)
print(ytr.shape)
print(yts.shape)


# # Apply ML algo

# # 1.Logistic Regression

# Logistic Regression is a classification algorithm that models the probability of the output class.
# 
# It estimates relationship between a dependent variable (target/label) and one or more independent variable (predictors) where dependent variable is categorical.
# *  Binary logistic regression requires the dependent variable to be binary.
# *  For a binary regression, the factor level 1 of the dependent variable should represent the desired outcome.
# *  Only the meaningful variables should be included.
# * The independent variables should be independent of each other. That is, the model should have little or no multicollinearity.
# *  The independent variables are linearly related to the log odds.
# *  Logistic regression requires quite large sample sizes.
#  

# This is a classification problem under supervised learning.This is a logistic problem because there are two classes and label is caterogrical

# In[ ]:


from sklearn.linear_model import LogisticRegression
model =LogisticRegression()


# In[ ]:


#train the model -using training data -xtr,ytr
model.fit(xtr,ytr)


# # Perfomance Analysis

# In[ ]:


# radius-16.65,texture-21.38,perimeter-110,area-904.6,smoothness-0.1121
new_case=[[16.65,21.38,110,904.6,0.1121]]
model.predict(new_case)


# In[ ]:


#here 0 is negative


# In[ ]:


#accuracy
#check perfomance of model on test data
# getting prediction for test data
ypred = model.predict(xts)
from sklearn import metrics
metrics.accuracy_score(yts,ypred)


# In[ ]:


yts.values.reshape(-1,1)


# In[ ]:


#calcuate recall
metrics.recall_score(yts,ypred,pos_label='M')


# # 2.KNN

# # Clustering 
# Clustering is the categorisation of objects into different groups, or more precisely, the partitioning of a data set into subsets (clusters), so that the data in each subset (ideally) share some common trait - often according to some defined distance measure.
# 

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
model2 = KNeighborsClassifier(n_neighbors = 3)#no of neighbors is hpyer parameter
model2.fit(xtr,ytr)


# In[ ]:


#accuracy
ypred2=model2.predict(xts)
metrics.accuracy_score(yts,ypred2)#here it checks for both class 1 and class 0( here 0 and 1  are told as class)


# In[ ]:


#recall
metrics.recall_score(yts,ypred2,pos_label='M')#here it ckecks only for class 1


# **AND,Here is end of my notebook**
#  if you find it usefull,then please upvote and share
# 

# In[ ]:




