#!/usr/bin/env python
# coding: utf-8

# ## This notebook demos Python data visualizations on the Iris dataset
# 
# This Python 3 environment comes with many helpful analytics libraries installed. It is defined by the [kaggle/python docker image](https://github.com/kaggle/docker-python)
# 
# We'll use three libraries for this tutorial: [pandas](http://pandas.pydata.org/), [matplotlib](http://matplotlib.org/), and [seaborn](http://stanford.edu/~mwaskom/software/seaborn/).
# 
# Press "Fork" at the top-right of this screen to run this notebook yourself and build each of the examples.

# In[ ]:


#  From sci-kit-learn get iris dataset
#  find keys in iris dataset
from sklearn import datasets
data = datasets.load_iris()
for keys in data.keys() :
    print(keys)


# In[ ]:


#  Get iris column names
data['feature_names']


# In[ ]:


# Reformat column names
import re
new_feature = []
for feature in data['feature_names']:
    new_feature.append(re.sub(r'(\w+) (\w+) \((\w+)\)',r'\1_\2_\3',feature))
print(new_feature)


# In[ ]:


# print first 10 data values of iris dataset
data['data'][:10]


# In[ ]:


# Covert list data to Dataframe
import pandas as pd
iris = pd.DataFrame(data['data'], columns=new_feature)
iris[:10]


# In[ ]:


# Iris species
data['target_names']


# In[ ]:


# Add species column to dataframe
import numpy as np
iris['species'] = np.nan
iris['species'][:50] = 'setosa'
iris['species'][50:100] = 'versicolor'
iris['species'][100:150] = 'virginica'


# In[ ]:


# Get first 10 data of iris dataframe
iris[:10]


# In[ ]:


# Get data info to check for missing value etc.
iris.info()


# In[ ]:


# Get number of datasets in each species 
iris['species'].value_counts()


# In[ ]:


# Scatter plot for length vs width 
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2,figsize=(10,8))
ax1.scatter(iris['sepal_length_cm'],iris['sepal_width_cm'])
ax1.set_xlabel('sepal_length_cm')
ax1.set_ylabel('sepal_width_cm')
ax2.scatter(iris['petal_length_cm'],iris['petal_width_cm'])
ax2.set_xlabel('petal_length_cm')
ax2.set_ylabel('petal_width_cm')
ax3.scatter(iris['sepal_length_cm'],iris['petal_length_cm'])
ax3.set_xlabel('sepal_length_cm')
ax3.set_ylabel('petal_length_cm')
ax4.scatter(iris['sepal_width_cm'],iris['petal_width_cm'])
ax4.set_xlabel('sepal_width_cm')
ax4.set_ylabel('petal_width_cm')


# In[ ]:


# get univariate hist plot with bivariate scatter pot

#fig = plt.figure(figsize=(8,5))
#ax1 = fig.add_subplot(121);
#ax2 = fig.add_subplot(122);
#fig, (ax1,ax2) = plt.subplots(2)
sns.jointplot(data=iris, x='sepal_length_cm',y='sepal_width_cm')
sns.jointplot(data=iris, x='petal_length_cm',y='petal_width_cm')


# In[ ]:


# Scatter plot by species
#fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2)
#fig = plt.figure()
#ax1 = fig.add_subplot(121)
#ax2 = fig.add_subplot(122)

sns.FacetGrid(data=iris, hue='species', size=5).map(plt.scatter,'sepal_length_cm','sepal_width_cm').add_legend()
sns.FacetGrid(data=iris, hue='species', size=5).map(plt.scatter,'petal_length_cm','petal_width_cm').add_legend()
sns.FacetGrid(data=iris, hue='species', size=5).map(plt.scatter,'sepal_length_cm','petal_length_cm').add_legend()
sns.FacetGrid(data=iris, hue='species', size=5).map(plt.scatter,'sepal_width_cm','petal_width_cm').add_legend()


# In[ ]:


# boxplot with jitter by species
sns.boxplot(data=iris, x='species', y='sepal_length_cm')
sns.stripplot(data=iris, x='species', y='sepal_length_cm', jitter=True, edgecolor='black')


# In[ ]:


# boxplot with jitter by species
sns.boxplot(data=iris, x='species', y='sepal_width_cm')
sns.stripplot(data=iris, x='species', y='sepal_width_cm', jitter=True, edgecolor='white')


# In[ ]:


# boxplot with jitter by species
sns.boxplot(data=iris, x='species', y='petal_length_cm')
sns.stripplot(data=iris, x='species', y='petal_length_cm', jitter=True, edgecolor='white')


# In[ ]:


# boxplot with jitter by species
sns.boxplot(data=iris, x='species', y='petal_width_cm')
sns.stripplot(data=iris, x='species', y='petal_width_cm', jitter=True, edgecolor='white')


# In[ ]:


# violin plot by species
sns.violinplot(data=iris, x='species', y='sepal_length_cm')


# In[ ]:


# violin plot by species
sns.violinplot(data=iris, x='species', y='sepal_width_cm')


# In[ ]:


# violin plot by species
sns.violinplot(data=iris, x='species', y='petal_length_cm')


# In[ ]:


# violin plot by species
sns.violinplot(data=iris, x='species', y='petal_width_cm')


# In[ ]:


# KDE plot by species
sns.FacetGrid(data=iris, hue='species').map(sns.kdeplot,'sepal_length_cm').add_legend()


# In[ ]:


# KDE plot by species
sns.FacetGrid(data=iris, hue='species').map(sns.kdeplot, 'sepal_width_cm').add_legend()


# In[ ]:


# KDE plot by species
sns.FacetGrid(data=iris, hue='species').map(sns.kdeplot, 'petal_length_cm')


# In[ ]:


# KDE plot by species
sns.FacetGrid(data=iris, hue='species').map(sns.kdeplot, 'petal_width_cm')


# In[ ]:


# pair plot with default diag_kind
sns.pairplot(data=iris, hue='species',)


# In[ ]:


# pair plot with KDE diag_kind
sns.pairplot(data=iris, hue='species', diag_kind='kde')


# In[ ]:


# boxplot by species
iris.boxplot(by='species', figsize=(20,10))


# In[ ]:


# Andrews curve by species
from pandas.tools.plotting import andrews_curves
andrews_curves(iris, 'species')


# In[ ]:


# parallel coordinates by species
from pandas.tools.plotting import parallel_coordinates
parallel_coordinates(iris,'species')


# In[ ]:


# radviz plot by species
from pandas.tools.plotting import radviz
radviz(iris,'species')

