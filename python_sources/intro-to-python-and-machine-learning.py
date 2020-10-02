#!/usr/bin/env python
# coding: utf-8

# # Machine Learning and Python Intership Training
# 
# #### Demo 1
# 
# 
# 
# 

# # Pandas
# 
# **PANDAS** probably is the most popular library for data analysis in Python programming language. This library is a high-level abstraction over low-level NumPy which is written in pure C.
# 
# #### Main Features:
# 
# * Tabular data with heterogeneously-typed columns, as in an SQL table or Excel spreadsheet.
# * Powerful, flexible group by functionality to perform split-apply-combine operations on data sets, for both aggregating and transforming data.
# * Intuitive merging and joining data sets, as in SQL
# 

# In[ ]:


from IPython.display import Image
from IPython.core.display import HTML 
Image(url="https://cdn-images-1.medium.com/max/800/1*pr2lbvx1jHw9aU1WsP73LA.gif")


# ## Object Creation
# 
# Data in Python is stored in Objects, such as lists, arrays dataframes etc.
# 
# 
# See the [Data Structure Intro section](https://pandas.pydata.org/pandas-docs/version/0.23.0/dsintro.html#dsintro)
# 

# #### Numpy Arrays
# 
# ndarrays are stored more efficiently than Python lists and allow mathematical operations to be vectorized, which results in significantly higher performance than with looping constructs in Python.

# In[ ]:


# import numpy library as np
import numpy as np

import matplotlib.pyplot as plt


# In[ ]:


arr = np.array([1,3,5,np.nan,6,8])
arr


# In[ ]:


arr[1]


# numpy arrays are very similar to python lists, but are stored more efficiently and support some matematical operations.

# #### Series
# They represent a one-dimensional labeled indexed array based on the NumPy ndarray.
# > [](http://) 
# Creating a [Series](https://pandas.pydata.org/pandas-docs/version/0.23.0/generated/pandas.Series.html#pandas.Series) by passing a list of values, letting pandas create a default integer index:

# to access the pandas library we prefix our function with **pd.**

# In[ ]:


# import pandas library as pd
import pandas as pd
# Create Pandas Series 
s = pd.Series([1,3,5,np.nan,6,8])
print(s)


# In[ ]:


series_index = pd.Series(np.array([10,20,30,40,50,60]), index=['a', 'b', 'c', 'd', 'e', 'f'])
series_index


# Pandas has a number of useful functions not only to manipulate data but also to generate data points.

# In[ ]:


#Generate dates
dates = pd.date_range('20130101', periods=6)
print(dates)


# Differences between **Numpy arrays** and **Pandas Series** 
# * Indexing - Custom indexing in Series, can contain number and letters; indexes in arrays are always numeric.
# * Series allow aligning data and matching indexes. 

# #### DataFrames

# Creating a [DataFrame](https://pandas.pydata.org/pandas-docs/version/0.23.0/generated/pandas.DataFrame.html#pandas.DataFrame) by passing a numpy array, with a datetime index and labeled columns:

# Generate a series of numbers between -20 and 40 in a 12 x 2 dimensional array as **temp_data** [randint Documentation](https://het.as.utexas.edu/HET/Software/Numpy/reference/generated/numpy.random.randint.html)

# In[ ]:


temp_data = pd.DataFrame(np.random.randint(low=-20, high=40, size=(12, 2)))
temp_data


# In[ ]:


df = pd.DataFrame(np.random.randn(6,4), index=[1,2,3,4,5,6], columns=list('ABCD'))
print(df)


# In[ ]:


# Generate new list of dates
dates = pd.date_range('20190101', periods=12)

# Generate dataframe with random temperatures
temp_data = pd.DataFrame(np.random.randint(low=-20, high=40, size=(12, 2)), index=dates, 
                             columns=['Forecast Avg', 'Actual Average'])
# Print temp data
temp_data


# Creating a DataFrame by passing a list of objects that can be converted to series-like.

# In[ ]:


df2 = pd.DataFrame({ 'A' : 1.,
                     'B' : pd.Timestamp('20130102'),
                     'C' : pd.Series(1,index=list(range(4)),dtype='float32'),
                     'D' : np.array([3] * 4,dtype='int32'),
                     'E' : pd.Categorical(["test","train","test","train"]),
                     'F' : 'foo' })

print(df2)
print("\n")
print(df2.dtypes)


# #### Exploring the dataframe
# .head(), tail(), .index, .values, .describe can be used to explorre the data 

# In[ ]:


temp_data.head()


# In[ ]:


temp_data.tail(3)


# In[ ]:


temp_data.index


# In[ ]:


temp_data.iloc[0]['Forecast Avg']


# #### Demo 2.
# #### K-Means Clustering
# 
# Cluster the data in df dataframe using K-means algorithm and visualize the results.
# 
# #### Step 1.
# *Load the dataframe*
# 

# In[ ]:


import pandas as pd
Data = {'x': [25,34,22,27,33,33,31,22,35,34,67,54,57,43,50,57,59,52,65,47,49,48,35,33,44,45,38,43,51,46],
        'y': [79,51,53,78,59,74,73,57,69,75,51,32,40,47,53,36,35,58,59,50,25,20,14,12,20,5,29,27,8,7]
       }
df = pd.DataFrame(Data,columns=['x','y'])


# #### Step 2.
# *Create data vectors from the dataframe*

# In[ ]:


# vectors = array_function( dataframe )
twoDvectors = np.array(df)


# #### Step 3.
# *Create and fit a Kmeans model*

# In[ ]:


from sklearn.cluster import KMeans

# model_name = Kmeans(n_Of_clusters).fit(vectors)
kmeans_model = KMeans(n_clusters = 3).fit(twoDvectors)

# get cluster centroids
# cluster_centroids = model_name.cluster_centers_
cluster_centroids = kmeans_model.cluster_centers_


# #### Step 4.
# *Visualize results*

# In[ ]:


import matplotlib.pyplot as plt

# plot feature points and cluster lables
plt.scatter(twoDvectors[:,0], twoDvectors[:,1], c= kmeans_model.labels_)

# plot cluster centroids 
plt.scatter(cluster_centroids[:,0], cluster_centroids[:,1], c='red', s=50)


# ### Main takeaways:
# * Clustering is powerful technique to understand your data
# * Distinguish between n - dimensional and 2-dimensional data
# * There is no right and wrong way in clustering data

# In[ ]:




