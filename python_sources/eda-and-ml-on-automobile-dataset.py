#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('pylab', 'inline')
import pandas as pd
from pandas import Series, DataFrame 


# In[ ]:


auto=pd.read_csv('../input/Automobile_data.csv')


# In[ ]:



auto.columns


# In[ ]:


auto.head()


# In[ ]:


auto.dtypes


# In[ ]:


auto.isnull().sum()


# In[ ]:


auto.describe().round(2)


# In[ ]:


plt.figure(figsize=(15,13))
sns.heatmap(auto.corr(),annot=True)


# In[ ]:


# Cleaning the Normalized losses field
auto[auto['normalized-losses']=='?'].count()
#normalized-losses    41


# In[ ]:


nl=auto['normalized-losses'].loc[auto['normalized-losses'] !='?'].count()
nmean=nl.astype(str).astype(int).mean()
auto['normalized-losses'] = auto['normalized-losses'].replace('?',nmean).astype(int)


# In[ ]:


# cleaning the price data
# Find out the number of values which are not numeric
auto['price'].str.isnumeric().value_counts()
# List out the values which are not numeric
auto['price'].loc[auto['price'].str.isnumeric() == False]
#Setting the missing value to mean of price and convert the datatype to integer
price = auto['price'].loc[auto['price'] != '?']
pmean = price.astype(str).astype(int).mean()
auto['price'] = auto['price'].replace('?',pmean).astype(int)
auto['price'].head()


# In[ ]:


# cleaning the horsepower
auto['horsepower'].str.isnumeric().value_counts()
horsepower = auto['horsepower'].loc[auto['horsepower'] != '?']
hpmean = horsepower.astype(str).astype(int).mean()
auto['horsepower'] = auto['horsepower'].replace('?',pmean).astype(int)


# In[ ]:


# cleaning the bore
# Find out the number of invalid value
auto['bore'].loc[auto['bore'] == '?']
# Replace the non-numeric value to null and conver the datatype
auto['bore'] = pd.to_numeric(auto['bore'],errors='coerce')
auto.dtypes


# In[ ]:


#cleaning stoke
auto['stroke'] = pd.to_numeric(auto['stroke'],errors='coerce')
auto.dtypes


# In[ ]:


#Cleaning the peak rpm data
# Convert the non-numeric data to null and convert the datatype
auto['peak-rpm'] = pd.to_numeric(auto['peak-rpm'],errors='coerce')
auto.dtypes


# In[ ]:


# cleaning the num-of-doors data
# remove the records which are having the value '?'
auto['num-of-doors'].loc[auto['num-of-doors'] == '?']
auto= auto[auto['num-of-doors'] != '?']
auto['num-of-doors'].loc[auto['num-of-doors'] == '?']


# In[ ]:


# Data Visulization
auto.columns


# In[ ]:


auto['make'].value_counts().plot(kind='bar')


# In[ ]:


#Insurance risk rating Histogram
auto.symboling.hist(bins=6)
plt.title("Insurance risk ratings of vehicles")
plt.ylabel('Number of vehicles')
plt.xlabel('Risk rating');


# In[ ]:


#Normalized losses histogram
auto['normalized-losses'].hist(bins=6,color='green',grid=False)
plt.title("Normalized losses of vehicles")
plt.ylabel('Number of vehicles')
plt.xlabel('Normalized losses');


# In[ ]:


#Fuel type chart
auto['fuel-type'].value_counts().plot(kind='barh',color='red')
plt.title("Fuel type frequence diagram")
plt.ylabel('Number of vehicles')
plt.xlabel('Fuel type');


# In[ ]:


#Scatter plot of price and engine size
#Findings: The more the engine size the costlier the price is

g = sns.lmplot('price',"engine-size", auto);


# In[ ]:


auto.groupby(['make','engine-size']).mean().round(2)


# In[ ]:


#Data Prepration for Machine Learning 


# In[ ]:


str_list=[] # empty list to contain columns with strings (words)
for colname,colvalue in auto.iteritems():
    if type(colvalue[1])== str:
        str_list.append(colname)
  # Get to the numeric columns by inversion
num_list=auto.columns.difference(str_list)


# In[ ]:


auto_num=auto[num_list]


# In[ ]:


auto_num.get_dtype_counts()


# In[ ]:


auto_num.dropna(inplace=True)


# In[ ]:


auto_num.astype(numpy.int64)


# In[ ]:


X=auto_num


# In[ ]:


plt.figure(figsize=( 12,10))
sns.heatmap(X.corr(),annot=True)


# In[ ]:


from sklearn.preprocessing import StandardScaler
X_std=StandardScaler().fit_transform(X)
#Dimentioanlity Reduction by PCA
from sklearn.decomposition import PCA
# Perform PCA
pca_auto= PCA(n_components=10)
pca_auto.fit(X_std)
pca_auto.explained_variance_ratio_.sum()


# In[ ]:


# Kmeans Clustering
from sklearn.cluster import KMeans
km_auto=KMeans(n_clusters=5,random_state=1)
km_auto.fit(X)


# In[ ]:





# In[ ]:


auto_num['cluster']=km_auto.labels_


# In[ ]:


auto_num

