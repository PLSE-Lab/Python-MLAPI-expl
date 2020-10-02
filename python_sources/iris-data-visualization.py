#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings # current version of seaborn generates a bunch of warnings that we'll ignore
warnings.filterwarnings("ignore")
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style ='white',color_codes = True)

#Importing Input File

iris = pd.read_csv('../input/Iris.csv')
iris.head()


# In[ ]:


# Counting Examples in Each Category of Species

iris['Species'].value_counts()


# In[ ]:


# Scatter plot using Pandas Plot method

iris.plot(kind = 'scatter', x = 'SepalLengthCm', y = 'SepalWidthCm')


# In[ ]:


# Now let us do the scatter plot for Petal length and Petal width

iris.plot(kind = 'scatter', x = 'PetalLengthCm', y = 'PetalWidthCm')


# In[ ]:


# Now we will use SeaBorn Library method -->Joint Plot to make the same plot. Additional Advantage here is that 
#we will get Univariate Histograms along with Bivariate scatter Plot.

sns.jointplot(x="SepalLengthCm", y="SepalWidthCm", data=iris, height=8)


# In[ ]:


# In order to identify what data belong to what Species ..we need to color them to distinguish from one another. 
# Here we will use FacetGrid method of from SeaBorn Library. 


sns.FacetGrid(iris, hue='Species', height = 8).map(plt.scatter,"SepalLengthCm", "SepalWidthCm").add_legend()


# In[ ]:


# We will use Box Plot to visualize individual Features.

sns.boxplot(x="Species", y="PetalLengthCm", data=iris)


# In[ ]:


sns.boxplot(x="Species", y="PetalWidthCm", data=iris)


# In[ ]:


sns.boxplot(x="Species", y="SepalLengthCm", data=iris)


# In[ ]:


sns.boxplot(x="Species", y="SepalWidthCm", data=iris)


# In[ ]:


ax = sns.boxplot(x="Species", y="PetalLengthCm", data=iris)
ax = sns.stripplot(x="Species", y="PetalLengthCm", data=iris, jitter=True)  

#jitter is used to draw points on box plot --> saving axes as ax because we want both plots on top of one another


# In[ ]:


# Violin plot is helpful to combine both of above faetures and give more aesthetic look 

sns.violinplot(x="Species", y="PetalLengthCm", data=iris, height=8)


# In[ ]:


# Seaborn library's KdePlot is used to visualize univariate features 

sns.FacetGrid(iris, hue="Species", size=6)    .map(sns.kdeplot, "PetalLengthCm")    .add_legend()


# In[ ]:


# using PairPlot to do biVariate analysis 

sns.pairplot(iris.drop("Id", axis=1), hue = "Species", size=3)


# In[ ]:


#we can also update the kind of plot for diagonal 

sns.pairplot(iris.drop("Id", axis=1), hue="Species", size=3, diag_kind="hist")


# In[ ]:


# Andrews Curves involve using attributes of samples as coefficients for Fourier series
# and then plotting these

from pandas.tools.plotting import andrews_curves 

andrews_curves(iris.drop("Id",axis =1),"Species")


# In[ ]:


# Parallel coordinate System
#Multivariatre visualization
from pandas.tools.plotting import parallel_coordinates
parallel_coordinates(iris.drop("Id", axis=1), "Species")


# In[ ]:


from pandas.tools.plotting import radviz
radviz(iris.drop("Id", axis=1), "Species")


# In[ ]:





# In[ ]:




