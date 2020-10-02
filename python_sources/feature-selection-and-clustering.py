#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


# read data
data = pd.read_csv('../input/ccdata/CC GENERAL.csv')
data.head()


# In[ ]:


data.describe()


# In[ ]:


#Count missing variable
data.isnull().sum().sort_values(ascending=False).head()


# In[ ]:


#Fill median missing variable
median = data['MINIMUM_PAYMENTS'].median()
data['MINIMUM_PAYMENTS'].fillna(median, inplace=True)


# In[ ]:


median2=data['CREDIT_LIMIT'].median()
data['CREDIT_LIMIT'].fillna(median2, inplace=True)


# In[ ]:


#Checking missing variable
data.isnull().sum().sort_values(ascending=False).head()


# In[ ]:


data.dtypes


# In[ ]:


#Dropping CUST_ID feature
data.drop(['CUST_ID'], axis=1, inplace=True)


# In[ ]:


import seaborn as sns # data visualization library  
plt.figure(figsize=(10,10))
sns.boxplot(data=data)
plt.xticks(rotation=90)


# In[ ]:


#Drop outliers according to z-score
from scipy import stats
z = np.abs(stats.zscore(data))
print(z)

threshold = 3
print(np.where(z > 3))

data_o = data[(z < 3).all(axis=1)]


# In[ ]:


data.shape


# In[ ]:


data_o.shape


# In[ ]:


#Normalize
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, normalize

# Get column names first
names = data_o.columns
# Create the Scaler object
scaler = preprocessing.StandardScaler()
# Fit your data on the scaler object
scaled_df = scaler.fit_transform(data_o)
scaled_df = pd.DataFrame(scaled_df, columns=names)
  
# Normalizing the Data 
normalized_df = normalize(scaled_df) 
  
# Converting the numpy array into a pandas DataFrame 
normalized_df = pd.DataFrame(normalized_df,columns=names) 


# In[ ]:


plt.figure(figsize=(10,10))
sns.boxplot(data=normalized_df)
plt.xticks(rotation=90)


# In[ ]:


#Using Pearson Correlation
plt.figure(figsize=(12,10))
cor = normalized_df.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()


# In[ ]:


#Correlation with output variable
cor_target = abs(cor["PURCHASES"])
#Selecting highly correlated features
relevant_features = cor_target[cor_target>0.5]
relevant_features


# In[ ]:


#Feature Selection using LassoCV

from sklearn.linear_model import LassoCV

#Feature Selection
X = normalized_df.drop("BALANCE",1)   #Feature Matrix
y = normalized_df["BALANCE"]          #Target Variable

reg = LassoCV()
reg.fit(X, y)
print("Best alpha using built-in LassoCV: %f" % reg.alpha_)
print("Best score using built-in LassoCV: %f" %reg.score(X,y))
coef = pd.Series(reg.coef_, index = X.columns)


# In[ ]:


print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  
      str(sum(coef == 0)) + " variables")


# In[ ]:


imp_coef = coef.sort_values()
import matplotlib
matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
imp_coef.plot(kind = "barh")
plt.title("Feature importance using Lasso Model")


# In[ ]:


X = X.drop("PURCHASES",1)   #Feature Matrix


# In[ ]:


#KMeans Clustering
#Defining WCSS Elbow point
from sklearn.cluster import KMeans

wcss=[]
for i in range (1,30):
    kmeans=KMeans(i)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
wcss


# In[ ]:


#Elbow Plot
plt.plot(range(1,30),wcss)
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()


# In[ ]:


#Another Technique to define n_cluster

# Import the KElbowVisualizer method 
from yellowbrick.cluster import KElbowVisualizer

# Instantiate a scikit-learn K-Means model
model = KMeans(random_state=0)

# Instantiate the KElbowVisualizer with the number of clusters and the metric 
visualizer = KElbowVisualizer(model, k=(2,30), metric='silhouette', timings=False)

# Fit the data and visualize
visualizer.fit(X)    
visualizer.poof()  


# In[ ]:


k_means_new=KMeans(6)
kmeans.fit(X)
cluster_new=X.copy()
cluster_new['cluster_pred']=k_means_new.fit_predict(X)
cluster_new.head()


# In[ ]:


# Visualize cluster shapes in 3d.

cluster1=cluster_new.loc[cluster_new['cluster_pred'] == 0]
cluster2=cluster_new.loc[cluster_new['cluster_pred'] == 1]
cluster3=cluster_new.loc[cluster_new['cluster_pred'] == 2]
cluster4=cluster_new.loc[cluster_new['cluster_pred'] == 3]
cluster5=cluster_new.loc[cluster_new['cluster_pred'] == 4]
cluster6=cluster_new.loc[cluster_new['cluster_pred'] == 5]


# In[ ]:


import seaborn as sns
#plot data with seaborn
facet = sns.lmplot(data=cluster_new, x='CREDIT_LIMIT', y='PAYMENTS',hue='cluster_pred', 
                   fit_reg=False, legend=True, legend_out=True)

