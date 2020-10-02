#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# **Exploratory data analysis**
# 
# In this notebook we will take a look at basic data exploratory analysis using graphs and summary statistics.
# The primary aim is to try and make patters and inferences from the data.
# Multiple ways of the dong the same step is shown for ease of understanding. 
# 
# The data set being used is the famous Iris data set.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # Data frame manipulation
import seaborn as sns  # corelation plt
import matplotlib.pyplot as plt  # graphs


# In[ ]:


import pandas as pd
iris = pd.read_csv("../input/iris/Iris.csv")


# In[ ]:


iris.head()


# Checking for the names of column

# In[ ]:


iris.columns 


# Now lets look at the data types in the Iris data set

# In[ ]:


iris.info()


# Deleating a column 

# In[ ]:


iris.drop('Id', axis = 1, inplace = True)
# axis =1 is for deleating the column name, and inplace = true is for doing the change in the original data set rather than creating a new one.


# Les check if the column has been deleted.

# In[ ]:


iris.columns


# Lets look at the summary stat

# In[ ]:


iris.describe()


# We can also check for non-numeric data

# In[ ]:


iris.describe(include = 'object')


# All stats at the same time

# In[ ]:


iris.describe(include = 'all')


# Checking for the number of enteries for each species

# In[ ]:


iris['Species'].value_counts()  
# here since the data is evenly distributed so we have a balanced data set


# Doing the same thiing using a graph

# In[ ]:


sns.countplot('Species', data = iris). set_title('count for each species') 


# Ccreating a pie chart for the count proportion in the data set.
# Pie charts are usually not common in exploratory analysis and seaborn library does not support pie graphs yet

# In[ ]:


iris['Species'].value_counts().plot.pie().set_title('species count') 


# In[ ]:


sns.scatterplot(x= 'SepalLengthCm',
                y= 'SepalWidthCm',
                data= iris).set_title('scatter plot based on species')


# here we see that the scater plot is not very useful so lets try and color it based on species

# In[ ]:


sns.scatterplot(x= 'SepalLengthCm',
                y= 'SepalWidthCm',
                hue = 'Species',
                data= iris).set_title('scatter plot based on species')


# Here we notice that Iris sitosa can be seperated from the rest by using a straight line 
# but it is difficult to separate Iris-Versicolor and Iris-virginica. So lets try and make a few more plots and see how we can classify or group them together. (how to tell if a fower is one of the three)****

# Lets use pair plots to check for relationship between variables

# In[ ]:


sns.pairplot(iris, hue= 'Species')


# The graphs can be confusing, but we need to look at only half of the graph.
# Graphs on either side of diagonals are mirror images of each other so we need to look at just one triangle part.
# 
# 
# 1. As we can see from the diagonal columns that petal length and petal width can be used to separate the data. only verginica and versicolor are overlapping a bit in terms of petal width.
# 
# 2. Here we can use a simple if else statement to separate the data 
# 
# logic:
# 
# if petalLength < 2  && petalwidth <=1 then setosa
# else if petalLength < 5 && petalwidth <=2 then vericolor

# Now lets take a look at the bottom 4 graphs to check the relationship between sepals and petals

# In[ ]:


sns.pairplot(iris, hue= 'Species',
            x_vars = ['SepalLengthCm', 'SepalWidthCm'] ,
            y_vars = ['PetalLengthCm', 'PetalWidthCm'])


# Now lets try and answer what variables are useful in separating the flowers based on the species by using graphs.
# Do we still agree to the previous variables or do we discover some new relationship.

# In[ ]:


sns.FacetGrid(iris, col = 'Species').map(plt.hist, 'PetalLengthCm')


# OR we can use the below method that does not use the map function. If you only want to use the seaborn library for graphs

# In[ ]:


sns.catplot(x= 'PetalLengthCm' , kind= 'count', data = iris, col ='Species',col_wrap = 1,aspect =4,height =2.5)


# In[ ]:


sns.FacetGrid(iris, hue = 'Species').map(plt.hist, 'PetalLengthCm').add_legend()


# In[ ]:


sns.FacetGrid(iris, hue = 'Species').map(sns.distplot, 'PetalWidthCm')


# In[ ]:


sns.FacetGrid(iris, hue = 'Species').map(sns.distplot, 'PetalLengthCm').add_legend()


# In[ ]:


sns.FacetGrid(iris, hue = 'Species').map(sns.distplot, 'SepalLengthCm').add_legend()


# In[ ]:


sns.FacetGrid(iris, hue = 'Species').map(sns.distplot, 'SepalWidthCm')


# The farther these distributions are the better it is for us.
# so the first two graphs tell us that petal length and petal width are best for separation, but sepal length and width are not that useful This is exactly as the finding we get from the scatterplot

# These findings can easily be shown uing box plot and violin plots and regression plots

# In[ ]:


plt.figure(figsize = (15,10))
plt.subplot(2,2,1)
sns.boxplot(x = 'Species', y = 'SepalLengthCm', data = iris)

plt.subplot(2,2,2)
sns.boxplot(x = 'Species', y = 'SepalWidthCm', data = iris)

plt.subplot(2,2,3)
sns.boxplot(x = 'Species', y = 'PetalLengthCm', data = iris)

plt.subplot(2,2,4)
sns.boxplot(x = 'Species', y = 'PetalWidthCm', data = iris)


# In[ ]:


sns.pairplot(iris,
            kind = 'reg',
            )


# Generating Basic statistics using python

# In[ ]:


#   Mean sepal length per species
setosa = iris[iris.Species == 'Iris-setosa']
versicolor = iris[iris.Species == 'Iris-versicolor']
virginica = iris[iris.Species == 'Iris-virginica']
print(np.mean(setosa['SepalWidthCm']))
print(np.mean(versicolor['SepalWidthCm']))
print(np.mean(virginica['SepalWidthCm']))


# OR We can use groupby to do the same

# In[ ]:


iris.groupby('Species').mean()

