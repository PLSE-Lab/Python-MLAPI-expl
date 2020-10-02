#!/usr/bin/env python
# coding: utf-8

# # Hierachical Clustering On Happines Report
# <div class="alert alert-block alert-info" style="margin-top: 20px">
# 1. [Introduction and Data Import](#0)<br>
# 2. [Pre-processing](#1)<br>
# 3. [Feature Selection and Normalizing](#2)
# 4. [Modeling](#3)
# 5. [Selecting Best K-Value](#4)
# <hr>

# # Introduction and Data Import <a id="0"></a>

# Introduction<br>
# There are many models for clustering out there. In this notebook, we will be presenting the model that is considered the one of the simplest model among them. Despite its simplicity, the K-means is vastly used for clustering in many data science applications, especially useful if you need to quickly discover insights from unlabeled data. In this notebook, you learn how to use k-Means for customer segmentation.
# <br><br>
# Some real-world applications of k-means:<br>
# 
# - Customer segmentation
# - Understand what the visitors of a website are trying to accomplish
# - Pattern recognition
# - Machine learning
# - Data compression<br><br>
# In this notebook we practice k-means clustering with 2 examples:<br><br>
# 
# - k-means on a random generated dataset
# - Using k-means for customer segmentation
# Import libraries
# Lets first import the required libraries. Also run %matplotlib inline since we will be plotting in this section.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random  
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans 
from sklearn.datasets.samples_generator import make_blobs 
get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


df=pd.read_csv("/kaggle/input/crime-data-from-2010-to-present/Crime_Data_from_2010_to_Present.csv")
df.head()


# # Pre-processing <a id="1"></a>

# In[ ]:


df.columns


# Just dropping out unnecessary columns

# In[ ]:


df.drop(columns=["DR Number","Date Reported","Date Occurred","Area Name","Crime Code Description","Weapon Description","Crime Code 1","Crime Code 2","Crime Code 3","Crime Code 4","Address","Cross Street","Premise Description","Weapon Used Code","Status Code","Location ","Status Description","MO Codes"],inplace=True)


# Filling in the null values of columns with most common index of the column.I am not able to use mean() beacuse the indexes are not numeric.

# In[ ]:


df["Victim Sex"].value_counts()


# In[ ]:


df["Victim Sex"].fillna("M",inplace=True)


# In[ ]:


df["Victim Descent"].value_counts()


# In[ ]:


df["Victim Descent"].fillna("H",inplace=True)


# Transforming all labels to numeric values to be able to use them in algorithm.

# In[ ]:


from sklearn import preprocessing
le_sex = preprocessing.LabelEncoder()
le_sex.fit(['M','F','X','H','N','-'])
df["Victim Sex"] = le_sex.transform(df["Victim Sex"].values) 


le_BP = preprocessing.LabelEncoder()
le_BP.fit([ 'H', 'W', 'B','X','A','K','F','C','I','J','P','U','V','Z','G','S','D','L','-','O'])
df["Victim Descent"] = le_BP.transform(df["Victim Descent"].values)



# # Feature Selection and Normalizing <a id="2"></a>

# In[ ]:


X = df[["Time Occurred","Area ID","Reporting District","Crime Code","Victim Age","Victim Sex","Victim Descent","Premise Code"]].head(1000).values
X[0:5]


# In[ ]:


from sklearn.preprocessing import StandardScaler
X = df.values[:,1:]
X = np.nan_to_num(X)
Clus_dataSet = StandardScaler().fit_transform(X)
Clus_dataSet


# # Modeling <a id="3"></a>

# In[ ]:


clusterNum = 3
k_means = KMeans(init = "k-means++", n_clusters = clusterNum, n_init = 12)
k_means.fit(X)
labels = k_means.labels_
print(labels)


# In[ ]:


df["Clus_km"] = labels
df.head(5)


# In[ ]:


df.groupby('Clus_km').mean()


# # Selecting Best K-Value <a id="4"></a>

# In[ ]:


cost =[] 
for i in range(1, 11): 
    KM = KMeans(n_clusters = i, max_iter = 500) 
    KM.fit(X) 
      
    # calculates squared error 
    # for the clustered points 
    cost.append(KM.inertia_)      
  
# plot the cost against K values 
plt.plot(range(1, 11), cost, color ='g', linewidth ='3') 
plt.xlabel("Value of K") 
plt.ylabel("Sqaured Error (Cost)") 
plt.show() # clear the plot 
  
# the point of the elbow is the  
# most optimal value for choosing k 


# Thank you for sharing your time to examine my kernel. If there is any questions please ask. If you think I should improve myself please comment.
