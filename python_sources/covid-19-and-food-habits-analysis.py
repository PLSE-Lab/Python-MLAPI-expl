#!/usr/bin/env python
# coding: utf-8

# # **Analysis based on Covid-19 Confirmed cases and Protein Intake **
# 
# To understand the relationship between Covid-19 and Source of Protein, I have applied Principal Component Analysis(PCA) to the data from Protein_Supply_Quantity_Data.csv.
# Dataset provides us the % of Protein Source (i.e. From Animal Products, Milk, Pulses etc.,) for different countries and confirmed COVID-19 cases. 
# 

# In[ ]:


# PCA
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
        
# Importing the dataset
dataset = pd.read_csv('/kaggle/input/covid19-healthy-diet-dataset/Protein_Supply_Quantity_Data.csv')
dataset=dataset.replace("<2.5", 0)

#dataset=dataset.dropna(inplace=True)
dataset = dataset.dropna()
dataset = dataset.reset_index(drop=True)

dataset.head()


# Using PCA Technique, I have tried to analyse and find out the most important factor of Protein Source. 

# In[ ]:


X_ori = dataset.iloc[:, 1:25].values
y_ori = dataset.iloc[:, 26].values

column_names=dataset.columns[1:25]

X=X_ori
y=y_ori

X = StandardScaler().fit_transform(X)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

X_train_ori = X_train
X_test_ori = X_test

# Applying PCA
from sklearn.decomposition import PCA
for i in range (5,0,-1):
    pca = PCA(n_components = i)
    X_train = pca.fit_transform(X_train_ori)
    X_test = pca.transform(X_test_ori)
    explained_variance = pca.explained_variance_ratio_
    print(explained_variance)
    n_pcs= pca.components_.shape[0]
    most_important = [np.abs(pca.components_[i]).argmax() for i in range(n_pcs)]
    most_important_names = [column_names[most_important[i]] for i in range(n_pcs)]
    #print(most_important)
    print(most_important_names)
    print("------")


# It is found out that Animal Products are the most important factor( Principal Component) with explained_variance_ratio_ of 0.25363972.
# 
# It is also found that "Vegetable Products" column is redundant.If we remove "Animal Products" feature from analysis then "Vegetable Products" becomes principle component.
# 

# I have plotted Principal Component Vs Confirmed Cases percentage to see relationship between them.

# In[ ]:



X_new = X_ori[:,most_important[0]]
y_new = y_ori

axes = plt.gca()
#axes.set_xlim([min(X_new[:])-.15,max(X_new[:])+.15])
#axes.set_ylim([min(y_new[:])-.09,max(y_new[:])+.001])
axes.spines["bottom"].set_color("purple")
axes.spines["left"].set_color("purple")
axes.tick_params(axis='x', colors='purple')
axes.tick_params(axis='y', colors='purple')

for i in range (0,X.shape[0]):
plt.scatter(X_new[i], y_new[i], s = 100, c = 'red')
if i%2 == 1:
    plt.annotate(dataset.iloc[i, 0], (X_new[i], y_new[i]), fontsize=16,rotation=-45,va='top')
else:
    plt.annotate(dataset.iloc[i, 0], (X_new[i], y_new[i]), fontsize=16,rotation=+45,va='bottom')

plt.title(column_names[most_important[0]] +' Consumed vs Confirmed cases',fontsize=20, fontweight='bold',c = 'purple')
plt.xlabel(column_names[most_important[0]] + ' Consumed Percentage',fontsize=16, fontweight='bold',c = 'purple')
plt.ylabel('Confirmed Cases Percentage',fontsize=16, fontweight='bold',c = 'purple')

figure = plt.gcf()  # get current figure
figure.set_size_inches(32, 18) # set figure's size manually to your full screen (32x18)
plt.show()


# From the above chart we can see that COVID-19 confirmed percentage is less in the contries where Animal Product Protein consumption is low like India.
# 
# Amongst the contries where Animal Product Protein consumption is high, we see confirmed percentage also high,like Spain, United States of America, Italy.
# 
# However, there are countries like Australia where confirmed percentage is low despite High % of Protein intake from Animal Product source, may be, perhaps, due to other fitness habits and wide spread population etc.
# 

# # **Analysis based on Confirmed Cases Percentage and Obesity Percentage**
# 
# To understand the relationship between Covid-19 and Obesity, I have applied K Means clustering to the data obtained from Fat_Supply_Quantity_Data.csv.
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

#import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
 
    #for filename in filenames:
     #   print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# -*- coding: utf-8 -*-
"""
Created on Mon May  4 16:48:47 2020

@author: rashmibh
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

from sklearn.preprocessing import StandardScaler

# Importing the INDIA dataset
dataset = pd.read_csv('/kaggle/input/covid19-healthy-diet-dataset/Fat_Supply_Quantity_Data.csv',usecols=[0,24,26])

dataset=dataset.replace("<2.5", 0)

dataset = dataset.dropna()
dataset = dataset.reset_index(drop=True)
dataset.head()


# Used Elbow method to find optimum number of clusters. In this case, it was found that 3 is the optimum number of clusters.

# In[ ]:


X = dataset.iloc[:, [1,2]].values

scalerX = StandardScaler().fit(X)
X_scaled = scalerX.transform(X)

# Using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
#plt.show()
figure = plt.gcf()  # get current figure
figure.set_size_inches(8, 4) # set figure's size manually to your full screen (32x18)
#plt.savefig("Elbow.png", bbox_inches='tight') # bbox_inches removes extra white spaces
plt.show()
#plt.clf()


# In[ ]:


num_opt_clusters=3

# Fitting K-Means to the dataset
kmeans = KMeans(n_clusters = num_opt_clusters, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X_scaled)

original_len=dataset.shape[0]
for i in range(0,original_len):
    dataset.loc[i,"Cluster"]=y_kmeans[i]
 
#dataset.to_csv('Cluster Assignment Result.csv') 
   
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.title('Cluster Analysis',fontsize=20, fontweight='bold')
plt.xlabel('Obesity',fontsize=16, fontweight='bold')
plt.ylabel('Confirmed',fontsize=16, fontweight='bold')

figure = plt.gcf()  # get current figure
figure.set_size_inches(32, 18) # set figure's size manually to your full screen (32x18)
plt.show()

Cluster chart with name of the countries.
# In[ ]:



axes = plt.gca()
axes.set_xlim([min(X[:,0])-.05,max(X[:,0])+.05])
axes.set_ylim([min(X[:,1])-.005,max(X[:,1])+.005])
axes.spines["bottom"].set_color("purple")
axes.spines["left"].set_color("purple")
axes.tick_params(axis='x', colors='purple')
axes.tick_params(axis='y', colors='purple')
    
for i in range (0,X.shape[0]):
    if y_kmeans[i] == 1 :
        plt.scatter(X[i,0], X[i,1], s = 100, c = 'green')
    elif y_kmeans[i] == 2 :
        plt.scatter(X[i,0], X[i,1], s = 100, c = 'red')
    elif y_kmeans[i] == 0 :
        plt.scatter(X[i,0], X[i,1], s = 100, c = 'blue')
   # plt.annotate(dataset.iloc[i, 0], (X[i,0], X[i,1]))
    if i%2 == 1:
        plt.annotate(dataset.iloc[i, 0], (X[i,0], X[i,1]), fontsize=14,rotation=35,va='bottom')
    else:
        plt.annotate(dataset.iloc[i, 0], (X[i,0], X[i,1]), fontsize=14,rotation=-35,va='top')

plt.title('Cluster Analysis',fontsize=20, fontweight='bold',c = 'purple')
plt.xlabel('Obesity Percentage',fontsize=16, fontweight='bold',c = 'purple')
plt.ylabel('Confirmed Cases Percentage',fontsize=16, fontweight='bold',c = 'purple')
#plt.show()

figure = plt.gcf()  # get current figure
figure.set_size_inches(32, 18) # set figure's size manually to your full screen (32x18)
#plt.savefig("Cluster.png", bbox_inches='tight') # bbox_inches removes extra white spaces
#plt.clf()
plt.show()


# Below charts provide closer look into 3 clusters.
# 

# In[ ]:



for j in range(0,num_opt_clusters):
    if j==1:
        colour = 'green'
    elif j==2:
        colour = 'red'
    elif j==0:
        colour = 'blue'
    else:
        print("Error:")
    
    axes = plt.gca()
    axes.set_xlim([min(X[y_kmeans==j,0])-.05,max(X[y_kmeans==j,0])+.05])
    axes.set_ylim([min(X[y_kmeans==j,1])-.005,max(X[y_kmeans==j,1])+.005])
    axes.spines["bottom"].set_color("purple")
    axes.spines["left"].set_color("purple")
    axes.tick_params(axis='x', colors='purple')
    axes.tick_params(axis='y', colors='purple')

    for i in range (0,X.shape[0]):
        if y_kmeans[i] == j :
            plt.scatter(X[i,0], X[i,1], s = 150, c = colour)
            if i%2 == 1:
                plt.annotate(dataset.iloc[i, 0], (X[i,0], X[i,1]), fontsize=14,rotation=-45,va='top')
            else:
                plt.annotate(dataset.iloc[i, 0], (X[i,0], X[i,1]), fontsize=14,rotation=+45,va='bottom')
        
    plt.title('Cluster_' + str(j) + ' Detail',fontsize=22, fontweight='bold',c = 'purple')
    plt.xlabel('Obesity Percentage',fontsize=20, fontweight='bold',c = 'purple')
    plt.ylabel('Confirmed Cases percentage',fontsize=20, fontweight='bold',c = 'purple')
    
    
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(32, 18) # set figure's size manually to your full screen (32x18)
    plt.show()
    #plt.savefig("Cluster"+str(j)+".png", bbox_inches='tight') # bbox_inches removes extra white spaces
    #plt.clf()


# 1. Cluster 0 -- Low COVID-19 confirmed percentage and Low Obesity percentage.
#              Ex. India,Kenya, Thailand
# 2. Cluster 1 -- Low COVID-19 confirmed percentage and High Obesity percentage.
#              Ex. Australia, Jordan, Mexico
# 3. Cluster 2 -- High COVID-19 confirmed percentage and High Obesity percentage.
#              Ex. United States of America, Spain, Iceland             
# 
# So we can see that contries where obesity is low COVID-19 cases are low. 

# Dataset:[](//https://www.kaggle.com/mariaren/covid19-healthy-diet-dataset)
# 
