#!/usr/bin/env python
# coding: utf-8

# # IRIS 

# ## Introduction
# 
# Hello **Kagglers** !!!
# 
# This is the very basic tutorial completely for beginners who just jump start into kaggle just like me.
# By now you should be having basic understanding of numpy, pandas, matplotlib, seaborn to kick start with the **exploratory data analysis** on the **IRIS** dataset from **UCI Machine Learning Repository**
# 
# I hope this notebook will help you as it is from scratch. **Please Vote**

# In[ ]:


# Importing the libraries

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# Importing the dataset

iris = pd.read_csv("../input/iris/Iris.csv")


# In[ ]:


# First 3 rows

iris.head(3)


# In[ ]:


# Last 3 rows

iris.tail(3)


# In[ ]:


# Size of the dataset

print('(Rows, Columns)=',iris.shape)


# In[ ]:


# Basic information about data and its datatypes

iris.info()


# In[ ]:


# Descibing the data by displaying the count, mean, standard deviation & percentile stats

iris.describe()


# In[ ]:


# fetching column names in the dataset

iris.columns


# In[ ]:


# Checking the no of null values in the dataset

iris.isnull().sum()


# In[ ]:


# Fetching columns having null values

iris.columns[iris.isnull().any()]


# In[ ]:


# Drop the unwanted columns.
# In dataset we have ID columns which represents the record number which can be removed

iris.drop(labels='Id', axis=1, inplace=True)


# In[ ]:


# Display the first 3 rows of dataset

iris.head(3)


# In[ ]:


# Checking the different categories and count of species

iris.Species.value_counts()


# In[ ]:


# Min values of different attributes of iris group by species

iris.groupby('Species').min()


# In[ ]:


# Cross verify above table with the below value for getting min value of Iris-sentosa species SepalLenthCm
iris[iris['Species']=='Iris-setosa']['SepalLengthCm'].min()


# In[ ]:


# Max values of different attributes of iris group by species

iris.groupby('Species').max()


# In[ ]:


# Mean values of different attributes of iris group by species

iris.groupby('Species').mean()


# In[ ]:


# Plotting graph with mean values according to the species

fig= plt.figure(figsize=(10,5))
for i in iris.columns[:-1]:
    plt.plot(iris.groupby('Species').mean()[i])
plt.legend()


# In[ ]:


# Distribution of SepalLength of Iris-setosa

sns.distplot(iris[iris['Species']=='Iris-setosa']['SepalLengthCm'], kde=False, bins=20)


# In[ ]:


# Distribution of PetalLengthCm of Iris-setosa

sns.distplot(iris[iris['Species']=='Iris-setosa']['PetalLengthCm'], kde=False, bins=15)


# In[ ]:


# Distribution of PetalWidthCm of Iris-setosa

print(sns.distplot(iris[iris['Species']=='Iris-setosa']['PetalWidthCm'], kde=False, bins=15))


# In[ ]:


# Pair plot displaying each and every attribute according to the species

sns.pairplot(iris, hue='Species')


# In[ ]:


# Scatter Plot (Sepal length Vs Sepal width)

fig=iris[iris.Species=='Iris-setosa'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='orange', label='Setosa')
iris[iris.Species=='Iris-versicolor'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='blue', label='versicolor', ax=fig)
iris[iris.Species=='Iris-virginica'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='green', label='virginica', ax=fig)
fig.set_title("Sepal length VS Sepal width")
fig.set_xlabel("sepal length")
fig.set_ylabel("sepal width")
fig=plt.gcf()
fig.set_size_inches(15,8)


# In[ ]:


# Scatter plot (Petal length VS Petal width)

fig=iris[iris.Species=='Iris-setosa'].plot(kind='scatter',x='PetalLengthCm',y='PetalWidthCm',color='orange', label='Setosa')
iris[iris.Species=='Iris-versicolor'].plot(kind='scatter',x='PetalLengthCm',y='PetalWidthCm',color='blue', label='versicolor', ax=fig)
iris[iris.Species=='Iris-virginica'].plot(kind='scatter',x='PetalLengthCm',y='PetalWidthCm',color='green', label='virginica', ax=fig)
fig.set_title("Petal Length VS Petal Width")
fig.set_xlabel("petal length")
fig.set_ylabel("petal width")
fig=plt.gcf()
fig.set_size_inches(15,8)

