#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#EDA Iris dataset

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

iris=pd.read_csv("../input/IRIS.csv")


#size of iris
iris.shape


# In[ ]:


#coloumns name
print(iris.columns) 


# In[ ]:


iris.head()


# In[ ]:


#how many datapoints for each classes are present

iris["species"].value_counts()

#this is a balanced dataset as each class has equal no of values or almost similar values is also balanced dataset(not imbalanced dataset)


# In[ ]:


#2-d scatter plot

iris.plot(kind='scatter',x='sepal_length',y='sepal_width')
plt.show()


# In[ ]:


import seaborn as sns

sns.set_style("whitegrid")
sns.FacetGrid(iris,hue="species",size=4)    .map(plt.scatter,"sepal_length","sepal_width")    .add_legend()
plt.show()


# In[ ]:


#plot on bases of petal


sns.set_style("whitegrid")
sns.FacetGrid(iris,hue="species",size=4)    .map(plt.scatter,"petal_length","petal_width")    .add_legend()
plt.show()


# In[ ]:


#Pair plot

plt.close();
sns.set_style("whitegrid")
sns.pairplot(iris,hue='species',size=3);
plt.show()


# Petal length and petal width are most important features.
# Setosa are easy differnetiable or linearly seperable but there is little overlap between Versicolor and verginica.
# We can create model using simple if-else condition.
# 
# Limitations of pair plot: can be used for only less features.. like max 6 feautures. as total plots will be like nc2 where n is no of features.
# 

# In[ ]:


#Histogram(1-D sctter plot kind of)

sns.FacetGrid(iris,hue="species",size=5)    .map(sns.distplot, "petal_length")    .add_legend();
plt.show();


# In[ ]:


#pdf and cdf

iris_setosa=iris[iris['species']=='Iris-setosa']
print(iris_setosa.head())
counts, bin_edges= np.histogram(iris_setosa['petal_length'],bins=10, density= True)
pdf=counts/(sum(counts))
print(pdf)

print(bin_edges)

cdf=np.cumsum(pdf)
print(cdf)

plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:],cdf)
plt.show();


# In[ ]:


#pdf and cdf

iris_setosa=iris[iris['species']=='Iris-setosa']
print(iris_setosa.head())
counts, bin_edges= np.histogram(iris_setosa['petal_length'],bins=10, density= True)
pdf=counts/(sum(counts))
print(pdf)

print(bin_edges)

cdf=np.cumsum(pdf)
print(cdf)

plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:],cdf)



iris_versicolor=iris[iris['species']=='Iris-versicolor']

counts_ve, bin_edges_ve= np.histogram(iris_versicolor['petal_length'],bins=10, density= True)
pdf_ve=counts_ve/(sum(counts_ve))
print(pdf_ve)

print(bin_edges_ve)

cdf_ve=np.cumsum(pdf_ve)
print(cdf_ve)

plt.plot(bin_edges_ve[1:],pdf_ve)
plt.plot(bin_edges_ve[1:],cdf_ve)


iris_virginica=iris[iris['species']=='Iris-virginica']

counts_vi, bin_edges_vi= np.histogram(iris_virginica['petal_length'],bins=10, density= True)
pdf_vi=counts_vi/(sum(counts_vi))
print(pdf_vi)

print(bin_edges_vi)

cdf_vi=np.cumsum(pdf_vi)
print(cdf_vi)

plt.plot(bin_edges_vi[1:],pdf_vi)
plt.plot(bin_edges_vi[1:],cdf_vi)


plt.show();


# In[ ]:


iris_virginica.describe()


# In[ ]:


sns.boxplot(x='species', y='petal_length',data=iris)
plt.show()


# In[ ]:


sns.violinplot(x='species', y='petal_length',data=iris)
plt.show()


# In[ ]:


#2D Density plot, contors-plot
sns.jointplot(x="petal_length", y="petal_width", data=iris_setosa, kind="kde");
plt.show();


# In[ ]:




