#!/usr/bin/env python
# coding: utf-8

# In this tutorial we will be doing customer segmentation using K means Algorithm.After that we will be using PCA to reduce the dimentionality of the problem.PAC + K Means will help us to improve the results.If you like the my work please do vote.

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


# **Importing Python Module**

# In[ ]:


import numpy as np
import pandas as pd
import scipy 
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# In[ ]:


df_segmentation = pd.read_csv('../input/retail-dataset-analysis/segmentation-data.csv',index_col =0)
df_segmentation.head()


# **Exploring data**

# In[ ]:


df_segmentation.describe()


# So we can see that we have obeservation for 2000 customers.Mean age of the dataset is 35.9 years.With an average income of $ 120954.

# **Correlation Estimate**

# In[ ]:


df_segmentation.corr()


# In[ ]:


plt.figure(figsize = (12,9))
s = sns.heatmap(df_segmentation.corr(),
               annot = True,
               cmap = 'RdBu',
               vmin = -1,
               vmax = 1)
s.set_yticklabels(s.get_yticklabels(),rotation = 0,fontsize = 12)
s.set_xticklabels(s.get_xticklabels(),rotation =90,fontsize =12)
plt.title('Correlation Heatmap')
plt.show()


# We can see a correlation of 0.57 between occupation and settlement size.This shows that a good occupation also means the size of the house would be bigger.
# Income and occupation has a correlation of 0.68 which shows once income highly depends on the type of occupation.

# **Vizualise Raw Data**

# In[ ]:


plt.figure(figsize = (12,9))
plt.scatter(df_segmentation.iloc[:,2],df_segmentation.iloc[:,4])
plt.xlabel('Age')
plt.ylabel('Income')
plt.title('Vizualization of raw data')


# There is no Linear Relation between the Age and Income.

# **Standardization**

# In[ ]:


scaler = StandardScaler()
segmentation_std = scaler.fit_transform(df_segmentation)


# Satndardization is done so that impact of all the parameters on the predicted value have same weightage.

# **K Means Clustering**

# In[ ]:


wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters =i,init ='k-means++',random_state=42)
    kmeans.fit(segmentation_std)
    wcss.append(kmeans.inertia_)


# In[ ]:


plt.figure(figsize = (10,8))
plt.plot(range(1,11),wcss,marker = 'o', linestyle = '--')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.title('K-means Clustering');


# From the Elbow Diagram we can see four would be the approprite number of clusters.

# In[ ]:


kmeans = KMeans(n_clusters = 4, init = 'k-means++',random_state = 42)
kmeans.fit(segmentation_std)


# **Results**

# In[ ]:


df_segm_kmeans = df_segmentation.copy()
df_segm_kmeans['Segment K-means'] = kmeans.labels_ 


# In[ ]:


df_segm_analysis = df_segm_kmeans.groupby(['Segment K-means']).mean()
df_segm_analysis 


# 0- has average age 56 income of $ 1.58 lac and 68 percent are married we call this call this class as well off.
# 
# 1-has average age 35 income of $ 0.97 lac and 4 percent are married we call this call this class as Fewer Opportunities
# 
# 2-has average age 36 income of $ 1.41 lac and 18 percent are married we call this call this class as Career Focused
# 
# 3-has average age 29 income of $ 1.05 lac and 88 percent are married we call this call this class as Standard
# 
# 
# 
# 

# In[ ]:


df_segm_analysis['N Obs'] = df_segm_kmeans[['Segment K-means','Sex']].groupby(['Segment K-means']).count()


# In[ ]:


df_segm_analysis['Prop Obs'] =df_segm_analysis['N Obs']/ df_segm_analysis['N Obs'].sum()


# In[ ]:


df_segm_analysis 


# In[ ]:


df_segm_analysis.rename({0:'Well Off',
                        1:'Fewer Opportunities',
                        2:'Standard',
                        3:'Career Focused'})


# So we can see that 31 % Fall in the Career Focused category
# 
# 33 % in Standard 
# 
# 21% in Fewer Opportunities 
# 
# 13.5 % in Well Off

# In[ ]:


df_segm_kmeans['Labels'] = df_segm_kmeans['Segment K-means'].map({0:'Well Off',
                        1:'Fewer Opportunities',
                        2:'Standard',
                        3:'Career Focused'}) 
#df_segm_kmeans


# In[ ]:


x_axis = df_segm_kmeans['Age']
y_axis = df_segm_kmeans['Income']
plt.figure(figsize = (10,8))
sns.scatterplot(x_axis,y_axis,hue = df_segm_kmeans['Labels'],palette =['g','r','c','m']);


# From the above graph we can clearly see that there is clear separation of only well off cluster.Other three cluster are not easily separable.Now we will try to use PCA to reduce the dimentionality of the model and therby improve the results and otain better segregation of clusters.

# **PCA**

# In[ ]:


from sklearn.decomposition import PCA
pca = PCA()


# In[ ]:


pca.fit(segmentation_std)


# In[ ]:


pca.explained_variance_ratio_


# In[ ]:


plt.figure(figsize = (12,9))
plt.plot(range(1,8),pca.explained_variance_ratio_.cumsum(),marker = 'o',linestyle = '--')
plt.title('Explained Variance by Components')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')


# We can see that three principle componets together explain more than 80% of the data.Generally the thumb rule is we consider principle components which can explain more than 80% of our result.

# **Fitting PCA with Four Principle Components**

# In[ ]:


pca = PCA(n_components =3)


# In[ ]:


pca.fit(segmentation_std)


# **PCA Results**

# In[ ]:


pca.components_


# We can see that PCA is 3x7 Array 

# In[ ]:


df_pca_comp = pd.DataFrame(data = pca.components_,
                          columns = df_segmentation.columns.values,
                          index = ['Component 1','Component 2','Component 3'])
df_pca_comp


# In[ ]:


plt.figure(figsize = (12,9))
sns.heatmap(df_pca_comp,
           vmin=-1,
           vmax=1,
           cmap='RdBu',
           annot=True)
plt.yticks([0,1,2],
          ['Component 1','Component 2','Component 3'],
           rotation =45,
          fontsize=9)


# In[ ]:


pca.transform(segmentation_std)


# In[ ]:


scores_pca = pca.transform(segmentation_std)


# **K Means clustering with PCA**

# In[ ]:


wcss = []
for i in range(1,11):
    kmeans_pca = KMeans(n_clusters =i,init ='k-means++',random_state=42)
    kmeans_pca.fit(scores_pca)
    wcss.append(kmeans_pca.inertia_)


# In[ ]:


plt.figure(figsize = (10,8))
plt.plot(range(1,11),wcss,marker = 'o', linestyle = '--')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.title('K-means Clustering with PCA')
plt.show()


# Looking at the elbow curve we can see that 4 would be the optimum number of cluster

# In[ ]:


kmeans_pca = KMeans(n_clusters = 4,init ='k-means++',random_state=42)


# In[ ]:


kmeans_pca.fit(scores_pca)


# **K-means clustering with PCA**

# In[ ]:


df_segm_pca_kmeans = pd.concat([df_segmentation.reset_index(drop=True),pd.DataFrame(scores_pca)],axis = 1)
df_segm_pca_kmeans.columns.values[-3:] = ['Component 1','Component 2','Component 3']
df_segm_pca_kmeans['Segment K-means PCA'] = kmeans_pca.labels_


# In[ ]:


#df_segm_pca_kmeans


# In[ ]:


df_segm_pca_kmeans_freq = df_segm_pca_kmeans.groupby(['Segment K-means PCA']).mean()
df_segm_pca_kmeans_freq


# 0 - Well Off 
# 
# 1 - Fewer Opportunities 
# 
# 2 - Standard 
# 
# 3 - Career Focused

# In[ ]:


df_segm_pca_kmeans_freq['N Obs'] = df_segm_pca_kmeans[['Segment K-means PCA','Sex']].groupby(['Segment K-means PCA']).count()
df_segm_pca_kmeans_freq['Prop Obs'] =df_segm_pca_kmeans_freq['N Obs']/ df_segm_pca_kmeans_freq['N Obs'].sum()
df_segm_pca_kmeans_freq = df_segm_pca_kmeans_freq.rename({0:'Well Off',
                                                         1:'Fewer Opportunities',
                                                         2:'Standard',
                                                         3:'Career Focused'})
df_segm_pca_kmeans_freq 


# In[ ]:


df_segm_pca_kmeans['Legend'] = df_segm_pca_kmeans['Segment K-means PCA'].map({0:'Well Off',
                                                         1:'Fewer Opportunities',
                                                         2:'Standard',
                                                         3:'Career Focused'})


# In[ ]:


x_axis = df_segm_pca_kmeans['Component 2']
y_axis = df_segm_pca_kmeans['Component 1']
plt.figure(figsize = (10,8))
sns.scatterplot(x_axis,y_axis,hue = df_segm_pca_kmeans['Legend'],palette = ['g','r','c','m'])


# From the above plot we can see that the customers are well segmented.So using PCA helped us to improve our customer segmentation results.
