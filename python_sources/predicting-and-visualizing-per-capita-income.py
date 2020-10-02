#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


dataset=pd.read_csv("/kaggle/input/farmers-markets-in-the-united-states/wiki_county_info.csv")
dataset.head()


# In[ ]:


dataset.isnull().sum()
dataset.dropna(inplace=True)
dataset.isnull().sum()
dataset.shape


# In[ ]:


dataset.reset_index()
dataset.shape


# In[ ]:


dataset['per capita income']


# In[ ]:


dataset['per capita income']=dataset['per capita income'].str.replace("$","")
dataset['per capita income']=dataset['per capita income'].str.replace(",","")
dataset['per capita income']=dataset['per capita income'].astype(int)


# In[ ]:


dataset['median family income']=dataset['median family income'].str.replace("$","")
dataset['median family income']=dataset['median family income'].str.replace(",","")
dataset['median family income']=dataset['median family income'].astype(int)


# In[ ]:


dataset['median household income']=dataset['median household income'].str.replace("$","")
dataset['median household income']=dataset['median household income'].str.replace(",","")
dataset['median household income']=dataset['median household income'].astype(int)


# In[ ]:


dataset['population']=dataset['population'].str.replace(",","")
dataset['population']=dataset['population'].astype(int)


# In[ ]:


dataset['number of households']=dataset['number of households'].str.replace(',',"")
dataset['number of households']=dataset['number of households'].astype(int)


# In[ ]:


dataset.head()


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sn
#Doing some visualizations on this data
plt.figure(figsize=(14,6))
dataset.groupby('State')['per capita income'].sum().sort_values(ascending=False).head(4).plot.bar()
#State with highest per capita income


# In[ ]:


plt.figure(figsize=(14,6))
dataset.groupby('county')['per capita income'].sum().sort_values(ascending=False).head(4).plot.bar()
#Highest county of per capita income 


# In[ ]:


plt.figure(figsize=(14,6))
sn.lineplot(data=dataset.iloc[:,[3,4,5,6,7]].head(20))


# In[ ]:


dataset.head()


# In[ ]:


plt.figure(figsize=(14,6))
sn.regplot(x=dataset['median household income'],y=dataset['per capita income'],color="orange")


# In[ ]:


plt.figure(figsize=(14,6))
sn.regplot(x=dataset['median family income'],y=dataset['per capita income'],color="red")


# In[ ]:


plt.figure(figsize=(14,6))
sn.regplot(x=dataset['population'],y=dataset['per capita income'],color="red")


# In[ ]:


plt.figure(figsize=(14,6))
sn.regplot(x=dataset['number of households'],y=dataset['per capita income'],color="red")


# In[ ]:


X=dataset.iloc[:,[3,4]].values
wcss=[]
from sklearn.cluster import KMeans
for i in range (1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++',n_init=10,max_iter=300,random_state=0)
    kmeans.fit_predict(X)
    wcss.append(kmeans.inertia_)
plt.figure(figsize=(14,6))
plt.plot(range(1,11),wcss)
#Within class sum of squares plot for X
#optimal number of clusters=4


# In[ ]:


kmeans=KMeans(n_clusters=4,init='k-means++',n_init=10,max_iter=300,random_state=0)
y_kmeans=kmeans.fit_predict(X)
plt.figure(figsize=(14,6))
plt.scatter(X[y_kmeans==0,0],X[y_kmeans==0,1],s=100,color="red")
plt.scatter(X[y_kmeans==1,0],X[y_kmeans==1,1],s=100,color="blue")
plt.scatter(X[y_kmeans==2,0],X[y_kmeans==2,1],s=100,color="green")
plt.scatter(X[y_kmeans==3,0],X[y_kmeans==3,1],s=100,color="orange")
plt.xlabel("median house hold income")
plt.ylabel("per capita income")
plt.show()


# In[ ]:


#Visualization of median family income with per capita income
X=dataset.iloc[:,[3,5]].values
wcss=[]
from sklearn.cluster import KMeans
for i in range (1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++',n_init=10,max_iter=300,random_state=0)
    kmeans.fit_predict(X)
    wcss.append(kmeans.inertia_)
plt.figure(figsize=(14,6))
plt.plot(range(1,11),wcss)
#Within class sum of squares plot for X
#optimal number of clusters=4


# In[ ]:


kmeans=KMeans(n_clusters=4,init='k-means++',n_init=10,max_iter=300,random_state=0)
y_kmeans=kmeans.fit_predict(X)
plt.figure(figsize=(14,6))
plt.scatter(X[y_kmeans==0,0],X[y_kmeans==0,1],s=100,color="red")
plt.scatter(X[y_kmeans==1,0],X[y_kmeans==1,1],s=100,color="blue")
plt.scatter(X[y_kmeans==2,0],X[y_kmeans==2,1],s=100,color="green")
plt.scatter(X[y_kmeans==3,0],X[y_kmeans==3,1],s=100,color="orange")
plt.xlabel("median house hold income")
plt.ylabel("per capita income")
plt.show()


# In[ ]:


#Applying multiple linear regression
dataset.head()


# In[ ]:


X=dataset.iloc[:,[4,5]]
Y=dataset.iloc[:,[3]]
Y.head()


# In[ ]:


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(X,Y,test_size=0.2,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(xtrain,ytrain)
ypred=regressor.predict(xtest)
from sklearn.model_selection import cross_val_score
accuracies=cross_val_score(estimator=regressor,X=xtrain,y=ytrain,cv=10)
print("accuracies:{:.2f}".format(accuracies.mean()*100))


# In[ ]:


#Accuracy testing using r2 score
from sklearn.metrics import r2_score
print(r2_score(ypred,ytest)*100)

