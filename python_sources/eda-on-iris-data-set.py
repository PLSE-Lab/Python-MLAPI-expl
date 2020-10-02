#!/usr/bin/env python
# coding: utf-8

# # Exploratory Data Analysis On Iris Dataset

# ## About Dataset:
# ### Attributes:
# 1. Id: Consequtive number of each data point/ Vector
# 2. SepalLength: Length of the sepal. A sepal ia part of the flower and often acts as support for the petals when blooming
# 3. SepalWidth:  Width of the sepal
# 4. PetalLength: Length of the petal. A petal is a colored part of the flower.
# 5. PetalWidth: Width of the petal
# 6. Species: There are 3 species of flower. Setosa, Virginica and Versicolor

# # Objective:
# 
# Given a new Flower, we have to classify it correctly into Setosa, Virginica or Versicolor using the dependent variables (Features are called dependent variables) Sepallength, Sepalwidth, Petallength, Petalwidth.

# In[ ]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# In[ ]:


#Reading the iris setosa data
data=pd.read_csv('../input/Iris.csv')


# In[ ]:


#checking the data
data.head(5)


# In[ ]:


#Shape of the data
data.shape


# ### Shape of data
# Number of features: 6
# <br>Number of datapoints: 150
# <br> You can see from the top 5 rows of the data that columns in the dataset are id, sepal length, sepal width, petal length, petal width and species. You can also print the columns by using data.columns

# In[ ]:


data.columns


# In[ ]:


data['Species'].value_counts()


# There are 50 points of each species present in the dataset. You can see clearly that the given dataset is balanced. Getting datasets like this in real-time scenarios is close to impossible.

# # 2-D Scatter Plot
# 
# You would get nc2 2-D scatter plots. So picking every combination of 2 features and drawing scatter plot would not be great. So instead we usually use pair plots. But for example, we can see how a scatter plot looks like.

# In[ ]:


sns.set_style('whitegrid')
sns.FacetGrid(data, hue='Species',size=5).map(plt.scatter,'SepalLengthCm','SepalWidthCm').add_legend()
plt.show()


# ## Observation
# We can see that Iris setosa is easily separated from virginica and versicolor. On the other hand there is a great deal of overlap between virginica and versicolor.
# <br> we can use sepallength and sepal width in our classification model.

# # Pair Plots
# Pair plots doesn't work if the dimensionality of the data is high even though they are good for low dimensionality data. The total number of plots we get are nc2 which is a really high number if we have a 100 dimensions. Going through those many plots and observing the structure of the data will take up a lot of time. This is drawback in using pair plots.

# In[ ]:


sns.set_style('whitegrid')
sns.pairplot(data, hue='Species',size=2).add_legend()
plt.show()


# # Observations
# We cannot consider Id while making observations because the data is balanced in such a way that 1-50 are setosa, next 50 are versicolor and next 50 are virginica. So considering other features we can tell from the plot that petal length and petal width are the most important features while making prediction.

# # UniVariate Analysis
# You can use pdf, cdf, boxplot and violin plots to do analysis on a single feature to find out which feature is essential for classification.
# <br> Analysis done on a single feature is called Univariate Analysis where as on two features is called Bivariate Analysis.
# <br> If you are doing analysis on more than two variables, it is called Multivariate analysis.

# ## PDF(Probability Density Function), CDF(Cumulative density function)
# PDF tells us what the probability of a new point falling in a particular range of values is.
# <br> CDF tell us what percentage of points are less than a particular value of feature.
# <br> useful scenario:
# <br> In companies which make deliveries we can tell what percenatge of customers are getting deliveries in 5 days or 2 days. By this we can analyze how to increase efficiency.
# <br> Integration of pdf is cdf and differentiation of cdf is pdf.

# In[ ]:


#Dividing the data into 3 species
setosa=data.loc[data['Species']=='Iris-setosa']
versicolor=data.loc[data['Species']=='Iris-versicolor']
virginica=data.loc[data['Species']=='Iris-virginica']


# In[ ]:


#pdf and cdf for petal lenghts of 3 species
counts,bin_edges=np.histogram(setosa['PetalLengthCm'],bins=10,density=True)
pdf=counts/sum(counts)
cdf=np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf,label='pdf_setosa')
plt.plot(bin_edges[1:],cdf,label='cdf_setosa')

counts,bin_edges=np.histogram(versicolor['PetalLengthCm'],bins=10,density=True)
pdf=counts/sum(counts)
cdf=np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf,label='pdf_versicolor')
plt.plot(bin_edges[1:],cdf,label='cdf_versicolor')

counts,bin_edges=np.histogram(virginica['PetalLengthCm'],bins=10,density=True)
pdf=counts/sum(counts)
cdf=np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf,label='pdf_virginica')
plt.plot(bin_edges[1:],cdf,label='cdf_virginica')

plt.xlabel('PetalLengthCm')
plt.legend(loc='center left',bbox_to_anchor=(1,0.5))
plt.show()


# # Observations:
# We can see that the petal length of 100% of points of setosa has petal length between 1 and 2 while there is a approximately 10% overlap between the points of versicolor and viginica.
# <br> we can write a simple if-else to separate setosa from the rest of the flowers

# In[ ]:


# pdf and cdf of petal width for 3 species
counts,bin_edges=np.histogram(setosa['PetalWidthCm'],bins=10,density=True)
pdf=counts/sum(counts)
cdf=np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf,label='pdf_setosa')
plt.plot(bin_edges[1:],cdf,label='cdf_setosa')

counts,bin_edges=np.histogram(versicolor['PetalWidthCm'],bins=10,density=True)
pdf=counts/sum(counts)
cdf=np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf,label='pdf_versicolor')
plt.plot(bin_edges[1:],cdf,label='cdf_versicolor')

counts,bin_edges=np.histogram(virginica['PetalWidthCm'],bins=10,density=True)
pdf=counts/sum(counts)
cdf=np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf,label='pdf_virginica')
plt.plot(bin_edges[1:],cdf,label='cdf_virginica')

plt.xlabel('PetalWidthCm')
plt.legend(loc='center left',bbox_to_anchor=(1,0.5))
plt.show()


# # Observations:
# It is fairly visible that the setosa are well seperated from virginica and versicolor and hence petal width is an important feature in classification.
# <br> If we observe the petal width for versicolor and virginica the overlap is approximately 20% which is greater than petal length.

# In[ ]:


# pdf and cdf of SepalLengthCm for 3 species
counts,bin_edges=np.histogram(setosa['SepalLengthCm'],bins=10,density=True)
pdf=counts/sum(counts)
cdf=np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf,label='pdf_setosa')
plt.plot(bin_edges[1:],cdf,label='cdf_setosa')

counts,bin_edges=np.histogram(versicolor['SepalLengthCm'],bins=10,density=True)
pdf=counts/sum(counts)
cdf=np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf,label='pdf_versicolor')
plt.plot(bin_edges[1:],cdf,label='cdf_versicolor')

counts,bin_edges=np.histogram(virginica['SepalLengthCm'],bins=10,density=True)
pdf=counts/sum(counts)
cdf=np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf,label='pdf_virginica')
plt.plot(bin_edges[1:],cdf,label='cdf_virginica')

plt.xlabel('SepalLengthCm')
plt.legend(loc='center left',bbox_to_anchor=(1,0.5))
plt.show()


# # Observations:
# We can see that there is a great deal of overlap between all the three species. Almost 30% overalp is there between setosa and versicolor. And there is an overlap of almost 90% between versicolor and virginica.
# <br> So, sepal length is of significantly lower importance than that of petal width.

# You can tell what features are important from bivariate analysis(pairplots). It would be enough to apply univariate analysis for the features that are useful.

# # Box-Plots:
# Box-Plots gives you Quantiles. The Boxplot contains 3 lines corresponding to 25th percentile, 50th percentile and 75th percentile. 25th percentile means 25% of the points have value less than the particular value. We can use it for an approximate calculation of error.

# In[ ]:


sns.boxplot(x='Species',y='PetalLengthCm',data=data)
plt.show()


# In[ ]:


sns.boxplot(x='Species',y='PetalWidthCm',data=data)
plt.show()


# In[ ]:


sns.boxplot(x='Species',y='SepalLengthCm',data=data)
plt.show()


# In[ ]:


sns.boxplot(x='Species',y='SepalWidthCm',data=data)
plt.show()


# #Observations:
# If you observe all the four plots you can easily observe the overlap. If you see the sepal length plot the there is an overlap at 75th percentile of versicolor and 25th percentile of virginica. But you can see no such overlap in petal length and petal width plot. BoxPlots and violin plots are the most used plots for univariate analysis.

# #Violin Plots:
# Viloin plots are a combination of boxplots and histograms. The outer region is the histogram and the inner blach region is the boxplot.

# In[ ]:


sns.violinplot(x='Species',y='PetalLengthCm',data=data)
plt.show()

sns.violinplot(x='Species',y='PetalWidthCm',data=data)
plt.show()

sns.violinplot(x='Species',y='SepalLengthCm',data=data)
plt.show()

sns.violinplot(x='Species',y='SepalWidthCm',data=data)
plt.show()


# # Summary
# <br>From all the above analysis we can tell that setosa is fairly well seperated from versicolor and virginica.
# <br>The importance of the features is in the order: petal length>petal width>>>sepal length>>>sepal width.
# <br>By using petal length and petal width, we can use simple if-else cases to build a model to classify setosa.
# <br> Setosa is linearly separable from versicolor and virginica but there is a fair amount of overlap between versicolor and virginca. We can safely assume that there will be atleast an error of 10% if we try to build a model using if-else to classify versivolor and virginica.
