#!/usr/bin/env python
# coding: utf-8

# ### Hello Dear Kagglers!
# This is a very basic tutorial for exploratory data analyis and data visualization using the '[Iris Data Set](http://archive.ics.uci.edu/ml/datasets/iris)'
# 
# The Iris data set is best known database to be found in the pattern recognition literature. Before moving to machine learning for pattern learning, we will perform EDA and visualize the data set.
# 
# The data set consists of 150 observations with 4 features - Sepal length, Sepal width, Petal length and Petal width, of three different species - Setosa, Versicolar and Virginica.
# 
# We'll use two libraries for this tutorial: [pandas](https://pandas.pydata.org/), [numpy](https://numpy.org/devdocs/), [matplotlib](https://matplotlib.org/), and [seaborn](https://seaborn.pydata.org/).
# 
# Press "Upvote" the notebook if you find it interesting and simple. You can also "Fork" at the top-right of this screen to run this notebook yourself and build each of the examples.

# In[ ]:


# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import warnings # current version of seaborn generates a bunch of warnings that we'll ignore
warnings.filterwarnings("ignore")

get_ipython().run_line_magic('matplotlib', 'inline')

iris = pd.read_csv('../input/iris/Iris.csv')
iris.shape


# In[ ]:


iris.head()


# In[ ]:


# Let's check if there in any inconsitency in the data set
iris = iris.drop('Id', axis=1)
iris.info()


# In[ ]:


# Let's see how many examples we have of each species
iris["Species"].value_counts()


# ### Visual Exploratory Data Analysis
# 
# #### Scatter Plots
# Scatter plots use a collection of points placed using Cartesian coordinates to display values from two variables.
# 
# By displaying a variable in each axis, we can detect if a relationship or correlation between the two variables exists. Scatter Plots are also great for observing the spread of the data as they retain the exact data values and sample size.

# In[ ]:


# Let's plot a scatter plot of the Iris features
fig = iris[iris.Species=='Iris-setosa'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='blue', label='Setosa',  figsize= (10,6))
iris[iris.Species=='Iris-versicolor'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='orange', label='Versicolor',ax=fig)
iris[iris.Species=='Iris-virginica'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='green', label='Virginica', ax=fig)
fig.set_xlabel("Sepal Length")
fig.set_ylabel("Sepal Width")
fig.set_title("Sepal Length vs Width")
plt.show()


# In[ ]:


# We can also use the seaborn library to make a similar plot
# A seaborn jointplot shows bivariate scatterplots and univariate histograms in the same figure

sns.jointplot(x="SepalLengthCm", y="SepalWidthCm", data=iris, size=5)
plt.show()


# In[ ]:


# Sepal Length using a Strip plot
sns.stripplot(y ='SepalLengthCm', x = 'Species', data =iris)
plt.show()


# In[ ]:


# We can look at an individual feature in Seaborn through a boxplot
sns.boxplot(x="Species", y="PetalLengthCm", data=iris)
plt.show()


# In[ ]:


# A violin plot combines the benefits of the previous two plots and simplifies them
# Denser regions of the data are fatter, and sparser thiner in a violin plot

sns.violinplot(x="Species", y="PetalLengthCm", data=iris, size=6)
plt.show()


# In[ ]:


# Let's see how are the length and width are distributed
iris.hist(edgecolor='black',bins = 25, figsize= (12,6))
plt.show()


# In[ ]:


# Another useful seaborn plot is the pairplot, which shows the bivariate relation between each pair of features

# From the pairplot, we'll see that the Iris-setosa species is separataed from the other two across all feature combinations

sns.pairplot(data = iris, hue="Species", size=3)
plt.show()


# **Heat Map** is used to find out the correlation between different features in the dataset. High positive or negative value shows that the features have high correlation.This helps us to select the parmeters for machine learning.

# In[ ]:


# Plotting heat map
sns.heatmap(iris.corr(), cmap="YlGnBu", annot=True, fmt="f")
plt.show()


# There is a high corelation between: Sepal Length & Petal Length, Sepal Length & Petal Width, and Petal Length & Petal Width. 

# ### Statistical Exploratory Data Analysis

# In[ ]:


iris.describe()


# In[ ]:


iris['Species'].unique()


# In[ ]:


# Filtering by species
indices = iris['Species'] == 'Iris-setosa'
setosa = iris.loc[indices,:]
indices = iris['Species'] == 'Iris-versicolor'
versicolor = iris.loc[indices,:]
indices = iris['Species'] == 'Iris-virginica'
virginica = iris.loc[indices,:]

# Delete the species column from each dataframe as same species are present

del setosa['Species'], versicolor['Species'], virginica['Species']


# In[ ]:


# Visual EDA for individual species

setosa.plot(kind = 'hist', bins =50, range = (0,8), alpha = 0.3)
plt.title('Setosa Data Set')
plt.xlabel('[cm]')

versicolor.plot(kind = 'hist', bins =50, range = (0,8), alpha = 0.3)
plt.title('Versicolor Data Set')
plt.xlabel('[cm]')

virginica.plot(kind = 'hist', bins =50, range = (0,8), alpha = 0.3)
plt.title('Virginica Data Set')
plt.xlabel('[cm]')

plt.show()


# In[ ]:


# ECDF
def ecdf(data):
    """Compute ECDF for a one-dimensional array of measurements."""
    n = len(data)

    x = np.sort(data)
    y = np.arange(1, n+1) / n

    return x, y

# Comparing ECDFs
x_set, y_set = ecdf(setosa['PetalLengthCm'])
x_vers, y_vers = ecdf(versicolor['PetalLengthCm'])
x_virg, y_virg = ecdf(virginica['PetalLengthCm'])


# Plot all ECDFs on the same plot
_ = plt.figure( figsize= (8,5))
_ = plt.plot(x_set, y_set, marker = '.', linestyle = 'none')
_ = plt.plot(x_vers, y_vers, marker = '.', linestyle = 'none')
_ = plt.plot(x_virg, y_virg, marker = '.', linestyle = 'none')

# Annotate the plot
plt.legend(('setosa', 'versicolor', 'virginica'), loc='lower right')
_ = plt.xlabel('petal length (cm)')
_ = plt.ylabel('ECDF')

# Display the plot
plt.show()


# The ECDFs expose clear differences among the species. Setosa is much shorter, also with less absolute variability in petal length than versicolor and virginica.
