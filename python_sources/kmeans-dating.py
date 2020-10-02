#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')  # To ignore python warnings

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


# Import the data set
dataset = pd.read_csv('../input/speed-dating-experiment/Speed Dating Data.csv', encoding="ISO-8859-1")


# In[ ]:


# Age distribution of the participants
plt.figure(figsize = (15,10))
sns.countplot(x = 'age', data= dataset, color = 'blue')
plt.xlabel('Age of participants')
plt.ylabel('Count')
plt.title('Age Distribution')
plt.show()


# In[ ]:


# Check for missing values in age
dataset['age'].isnull().sum()


# In[ ]:


# Lets create a utility function to fill the missing values in the column with the value provided

import math
def fill_missing_values(column_name, value):
    for i in range(len(dataset[column_name].values)):
        if math.isnan(dataset[column_name][i]):
            dataset[column_name][i] = value


# In[ ]:


import statistics
median = statistics.mode(dataset['age'].values)
fill_missing_values('age', median)


# In[ ]:


# Check if there are still any missing values
dataset['age'].isnull().sum()


# In[ ]:


plt.figure(figsize = (15,10))
sns.distplot(dataset['age'])


# In[ ]:


# Lets apply KMeans clustering algorithm between Gender and age
x = dataset[['gender', 'age']].values


# In[ ]:


# Lets create utility function to Get the Within Cluster Sum of Squares (WCSS) and plot the Elbow curve so that 
# we can decide optimum number of clusters

from sklearn.cluster import KMeans

def get_wcss(x):
    wcss = []
    for i in range(1,11):
        kmeans = KMeans(n_clusters = i, init = 'k-means++')
        kmeans.fit(x)
        wcss.append(kmeans.inertia_)
    return wcss

def plot_elbow_curve(wcss):
    plt.figure(figsize = (15,10))
    plt.plot(range(1,11), wcss, color = 'blue')
    plt.xlabel('Number of clusters')
    plt.ylabel('Within Cluster Sum of Squares')
    plt.title('Elbow Curve')
    plt.show()


# In[ ]:


# Get and plot the WCSS for Kmeans applied on Gender and Age
wcss = get_wcss(x)
plot_elbow_curve(wcss)


# In[ ]:


# Looks like 6 can be the optimum number of clusters for the extracted dataset
kmeans = KMeans(n_clusters = 6, init = 'k-means++')
y_clusters = kmeans.fit_predict(x)


# In[ ]:


plt.figure(figsize = (10,10))
plt.scatter(x[y_clusters == 0, 0], x[y_clusters == 0, 1], s = 100, c = 'red', label = 'Cluster-1')
plt.scatter(x[y_clusters == 1, 0], x[y_clusters == 1, 1], s = 100, c = 'blue', label = 'Cluster-2')
plt.scatter(x[y_clusters == 2, 0], x[y_clusters == 2, 1], s = 100, c = 'green', label = 'Cluster-3')
plt.scatter(x[y_clusters == 3, 0], x[y_clusters == 3, 1], s = 100, c = 'cyan', label = 'Cluster-4')
plt.scatter(x[y_clusters == 4, 0], x[y_clusters == 4, 1], s = 100, c = 'brown', label = 'Cluster-5')
plt.scatter(x[y_clusters == 5, 0], x[y_clusters == 5, 1], s = 100, c = 'pink', label = 'Cluster-6')

plt.xlabel('Gender - Male or Female')
plt.ylabel('Age of participant')
plt.title('Gender- Age Distribution of participants')
plt.show()


# [](http://)**Lets see the clustering between age of participant and age of partner!**

# In[ ]:


# Age distribution of the partners
plt.figure(figsize = (15,10))
sns.countplot(x = 'age_o', data= dataset, color = 'blue')
plt.xlabel('Age of partner')
plt.ylabel('Count')
plt.title('Age Distribution')
plt.show()


# In[ ]:


# Check for missing values in age of partner
dataset['age_o'].isnull().sum()


# In[ ]:


mode = statistics.mode(dataset['age_o'].values)
fill_missing_values('age_o', mode)


# In[ ]:


# Check if we still have missing values
dataset['age_o'].isnull().sum()


# In[ ]:


plt.figure(figsize = (15,10))
sns.distplot(dataset['age_o'])


# In[ ]:


# Lets apply K-Means between age of participant and age of partner
x = dataset[['age_o', 'age']].values


# In[ ]:


wcss = get_wcss(x)
plot_elbow_curve(wcss)


# In[ ]:


# Looks like 8 can be the optimum number of clusters for the extracted dataset
kmeans = KMeans(n_clusters = 8, init = 'k-means++')
y_clusters = kmeans.fit_predict(x)


# In[ ]:


# Scatter plot to show the distribution between the two parameters before clustering
plt.scatter(dataset['age_o'], dataset['age'])
plt.show()


# In[ ]:


# Plot the clusters obtained after applying K-Means
plt.figure(figsize = (10,10))
plt.scatter(x[y_clusters == 0, 0], x[y_clusters == 0, 1], s = 100, c = 'red', label = 'Cluster-1')
plt.scatter(x[y_clusters == 1, 0], x[y_clusters == 1, 1], s = 100, c = 'blue', label = 'Cluster-2')
plt.scatter(x[y_clusters == 2, 0], x[y_clusters == 2, 1], s = 100, c = 'green', label = 'Cluster-3')
plt.scatter(x[y_clusters == 3, 0], x[y_clusters == 3, 1], s = 100, c = 'cyan', label = 'Cluster-4')
plt.scatter(x[y_clusters == 4, 0], x[y_clusters == 4, 1], s = 100, c = 'brown', label = 'Cluster-5')
plt.scatter(x[y_clusters == 5, 0], x[y_clusters == 5, 1], s = 100, c = 'pink', label = 'Cluster-6')
plt.scatter(x[y_clusters == 6, 0], x[y_clusters == 6, 1], s = 100, c = 'brown', label = 'Cluster-7')
plt.scatter(x[y_clusters == 7, 0], x[y_clusters == 7, 1], s = 100, c = 'pink', label = 'Cluster-8')

plt.xlabel('Age of the partner')
plt.ylabel('Age of participant')
plt.title('Gender- Age Distribution of participants')
plt.show()

