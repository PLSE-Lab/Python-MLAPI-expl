#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# ```
# In This Kernel, we are doing the easy to understand Data Visualizations using Iris Data Set
# Iris DataSet consists of 150 rows and is about the Iris Flower and its 3 Species or Varieties.
# The purpose of Data Visualization is to understand the nature of Data and to observe any interesting facts or Patterns.
# We can use Data Visualization to undersatnd:
# Distribution of Data
# Relation among Data
# Correlation among Data
# 
# To do this , we can use Matplotlib, Pandas and Seaborn Library with Seaborn the more fancier one.
# We can plot univariate(1 variable) or bivariate data(2 variable).
# We can bring in the effect of Categorical variables in terms of Hue in the plots to be able to visualize the effect in single Plot.
# ```

# ```
# To bring in some motivation,Please upvote :)
# ```

# In[ ]:


# Loading the Data
iris = pd.read_csv('../input/Iris.csv')


# In[ ]:


#Checking the head of Dataset
iris.head(5)


# In[ ]:


#Checking the column info
iris.info()


# ```
# Dataset iris has 6 Columns-
# Id-> It is just for numbering, so we should ignore this Column in making observations
# Species -> This is Target or Response Column, which we would Predict or making predictions based on other feature columns.
# SepalLengthCm - Sepal Length for the iris flower
# SepalWidthCm - Sepal Width for the iris flower
# PetalLengthCm - Petal Length for the iris flower
# PetalWidthCm - Petal Width for the iris flower
# ```

# In[ ]:


#to check the statisics for the iris dataset
iris.describe()


# In[ ]:


iris.groupby('Species').min()


# In[ ]:


iris.groupby('Species').max()


# In[ ]:


#Checking is there is any null data in iris dataset
iris.isnull().any()
# There is no missing Data.


# In[ ]:


iris.groupby('Species').count()


# In[ ]:


#using countplot to see number of each kind of iris species flower
sns.countplot(x=iris['Species'])


# ```
# To check the Distribution of Data, we can use Histogram. This can be used to see if the data has normal or skewed Distribution.
# If we need to check indivdual Column distribution, we can provide the column only
# ```

# In[ ]:


iris['PetalLengthCm'].hist(bins=20)


# ```
# To checxk for whole DataFrame , we can provide the Data Frame.
# But When the number of columns are more or we need the distribution for specific column,
# Then its better to check the histogram of invidual Column
# ```

# In[ ]:


iris.drop('Id',axis=1).hist(bins=20,figsize=(10,10))


# ```
# For checking for the distribution of PetalLengthCm and PetalWidthCm for different species in iris Flower Dataset and using more cosmetic effects for better understanding, we can follow the below approach.
# 
# ```

# In[ ]:


iris.groupby('Species').PetalLengthCm.plot.hist(alpha=0.4)
plt.xlabel('PetalLengthCm')
plt.suptitle('Histogram of PetalLengthCm for different Species')
plt.legend(loc=(0.69,0.75))
plt.grid()


# In[ ]:


iris.groupby('Species').PetalWidthCm.plot.hist(alpha=0.4)
plt.xlabel('PetalWidthCm')
plt.suptitle('Histogram of PetalWidthCm for different Species')
plt.legend(loc=(0.69,0.75))
plt.grid()


# In[ ]:


#Histogram using Matplotlib,Pandas and Seaborn.
plt.hist(data=iris,x='PetalLengthCm',bins=20)#Matplotlib
iris['PetalLengthCm'].hist(bins=20)# Pandas
sns.distplot(iris['PetalLengthCm'],bins=20) #Seaborn


# ```
# To check the relationship among Features, we can use scatter plot.
# We can check the separate scatterplot for different variables
# or we can use seaborn Pairplot to check the relationship of multiple variables at once.
# but it is not advisable to use Pairplot in cases where there are large number of features.
# In that case, a subset of features can be used for pairplot.
# ```

# In[ ]:


#to see the relationship between 2 variables.
sns.set_style('darkgrid')
sns.scatterplot(data=iris,x='PetalLengthCm',y='PetalWidthCm',hue='Species')


# In[ ]:


sns.scatterplot(data=iris,x='PetalLengthCm',y='SepalLengthCm',hue='Species')


# In[ ]:


#Since this data is not that big and has only 4 features, we can use pairplot 
#and check the relationshipo between 2 variables grouped by Species column.
sns.pairplot(data=iris.drop('Id',axis=1),hue='Species')


# In[ ]:


#Scatter Plot using Matplotlib,Pandas and Seaborn.
#sns.scatterplot(data=iris,x='PetalLengthCm',y='SepalLengthCm',hue='Species')#Seaborn.Hue can be used only here.
#plt.scatter(data=iris,x='PetalLengthCm',y='PetalWidthCm',c='green',marker='+')#matplotlib
#iris.plot.scatter(x='SepalLengthCm',y='SepalWidthCm',c='red',marker='+',s=10)#pandas
#iris.groupby('Species').plot.scatter(x='SepalLengthCm',y='SepalWidthCm',c='red',marker='+',s=10)#pandas
#but this will result in 3 separate plots. Hence using seaborn hue is easier and more visually appealing
# as it will plot the data on single axis.


# ```
# Further to check the Correlation among these variables, we can use corr() function and use
# Heatmap to visualize the correlation.
# ```

# In[ ]:


iris.drop('Id',axis=1).corr()


# ```
# We have noticed from the above correlation that Petal Length and Petal Width are most correlated features.Also, there is good correlation between petal Length and SepalLength.
# 
# ```

# In[ ]:


#Plotting the correlation using HeatMap
sns.heatmap(iris.drop('Id',axis=1).corr(),cmap='viridis',annot=True,)
plt.suptitle('Heatmap')


# ```
# correlation value of 0 refers to  No Correlation.
# +1 refers to Positive correlation
# -1 refers to negative correlation.
# ```

# ```
# Box plots can also be used for looking at the distribution.
# This provides more information than histogram.
# It shows the distribution , Min value, Max Value, First and Third Quartile, Median, IQR(Inter Quartile Range) and outliers.
# If needed to see the distribution along with any categorical variable,it is useful as well.
# Also,Violin Plots are similar to Box Plots.
# 
# ```

# In[ ]:


sns.boxplot(data=iris,y='SepalLengthCm')


# In[ ]:


sns.boxplot(data=iris,x='Species',y='SepalLengthCm')


# In[ ]:


sns.violinplot(data=iris,y='SepalWidthCm')


# In[ ]:


sns.violinplot(data=iris,x='Species',y='SepalLengthCm')


# ```
# Multi-plot grid for plotting conditional relationships:
# We can use Seaborn Facet Grid and Matplot lib Subplot to plot on mulitple grid on multiple axis derived by levels of Categorical data.
# ```

# In[ ]:


#Subplot
fig,axes = plt.subplots(ncols=2,nrows=1)# Creating the grid
axes[0].hist(iris['PetalLengthCm'])# Plotting on each axis
axes[0].set_title('PetalLengthCm')# Setting the Title
axes[1].scatter(iris['PetalLengthCm'],iris['PetalWidthCm'])# Plotting on each axis
axes[1].set_title('ScatterPlot')# Setting the Title


# In[ ]:


#Facet
g=sns.FacetGrid(data=iris,col='Species')# this creates the blank grid based on level of categorical variable.
g.map(plt.hist,'PetalLengthCm')# plotting using the grid created.
# grid can also be created using 2 Categorical Variables and span over rows.


# In[ ]:


h=sns.FacetGrid(data=iris,col='Species')
h.map(plt.scatter,'PetalLengthCm','PetalWidthCm',color='r')# for bivariate(2 variable) plotting


# ```
# Seaborn Facet Grid has more cosmetic effect and can be easily used to produce visibly appealing plots.
# For more documentation on Seaborn:
# https://seaborn.pydata.org/generated/seaborn.FacetGrid.html
# ```

# ```Please upvote :)```
