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

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
#for visualization


# before doing anything read the description....
# 
# 
# Haberman data set for cancer survival
# Haverman data set is the result of the research conducted between 1958 and 1970 to examine the patient will survived for less than 5 years or grater than equal to five years after operation. The study was held at the University of Chicago's Billings Hospital.
# 
# **Content**
# It contains the three features and two classes
# all columns are numerical data
# 
# Number of Instances: 306
# Number of Attributes: 4 (including the class attribute)
# Attribute Information:
# Age of patient at time of operation (numerical)
# Patients year of operation (year - 1900, numerical)
# Number of positive axillary nodes detected (numerical)
# Survival status (class attribute) 1 = the patient survived 5 years or longer 
# 2 = the patient died within 5 year
# Missing Attribute Values: None

# In[ ]:


df = pd.read_csv('/kaggle/input/haberman/haberman.csv')


# In[ ]:


df.head()
#top-most values of the dataset


# In[ ]:


df.shape
#our dataset have 306 rows and 4 columns


# In[ ]:


df.columns
#columns of the dataset


# In[ ]:


df['status'].value_counts()


# In[ ]:


df.describe()


# In[ ]:


df.info()


# **#observations**
# * Only 225 patients survived 5 years or longer
# * And 81 the patient died within 5 year

# In[ ]:


sns.set_style("darkgrid");
sns.FacetGrid(df, hue='status', size=6).map(plt.scatter, "year", "nodes").add_legend();
plt.show();


# In[ ]:


sns.set_style('darkgrid');
sns.FacetGrid(df, hue='status' , size=6).map(plt.scatter, 'nodes', 'age').add_legend();
plt.show()


# **Histogram**

# In[ ]:


sns.FacetGrid(df, hue='status', size=5).map(sns.distplot,"year").add_legend()
plt.show()
#points are overlapping as we can see


# In[ ]:


sns.FacetGrid(df, hue='status', size=5).map(sns.distplot, 'age').add_legend()
plt.show()


# well the features are overlapping and we can't clearly make out anything but...
# * patients with age less than 35 and greater than 30  survived more than 5 years after operation
# * patients with age less than 83 and greater than 78 have survived not more than 5 years after operation
# * patients from age 35 to 78 we can't say anything as point are almost overlapping.

# In[ ]:


sns.FacetGrid(df, hue='status', size=5).map(sns.distplot, "nodes").add_legend()
plt.show()


# with large number of positive axillary nodes survival status decreases  

# **Box plot and Whiskers**

# In[ ]:


sns.boxplot(x='status', y='year', data=df)
plt.show()


# In[ ]:


sns.boxplot(x='status', y='age', data=df)
plt.show()


# In[ ]:


sns.boxplot(x='status', y='nodes', data=df)
plt.show()


# In[ ]:


#people most people who survived have zero positive axillary nodes


# **Violin plots**

# In[ ]:


sns.violinplot(x='status' , y='nodes', data=df, size=8)
plt.show()


# In[ ]:


sns.violinplot(x='status', y='age', data=df, size=8)
plt.show()


# In[ ]:


sns.violinplot(x='status', y='year', data=df, size=8)
plt.show()


# **3D scatter plot with Plotly Express**
# 
# [https://plot.ly/python/3d-scatter-plots/](http://)

# In[ ]:


import plotly.express as px
fig = px.scatter_3d(df, x='age', y='nodes', z='year', color='status')
fig.show()


# In[ ]:





# PDF and CDF
# 

# In[ ]:


#pdf cdf of year
counts, bin_edges = np.histogram(df['year'], bins=30 , density=True)
pdf = counts/sum(counts)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:], pdf)
plt.plot(bin_edges[1:], cdf)
plt.legend()

counts, bin_edges = np.histogram(df['year'], bins=30, density=True)
pdf = counts/(sum(counts))
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:], pdf)
plt.plot(bin_edges[1:], cdf)

plt.xlabel('Year')
plt.grid()
plt.show()


# #pdf cdf of positive_axillary_nodes

# In[ ]:


#pdf cdf of positive_axillary_nodes

counts,bin_edges = np.histogram(df['nodes'],bins = 30, density = True)
pdf = counts/(sum(counts))
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:],cdf)
plt.legend()

counts,bin_edges = np.histogram(df['nodes'],bins = 30, density = True)
pdf = counts/(sum(counts))
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:],cdf)

plt.xlabel('positive_axillary_nodes')
plt.grid()

plt.show()


# In[ ]:


#pdf cdf of Age

counts,bin_edges = np.histogram(df['age'],bins = 30, density = True)
pdf = counts/(sum(counts))
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:],cdf)
plt.legend()

counts,bin_edges = np.histogram(df['age'],bins = 30, density = True)
pdf = counts/(sum(counts))
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:],cdf)

plt.xlabel('Age')
plt.grid()

plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# **Pair-plot**

# In[ ]:





# In[ ]:


# pairwise scatter plot: Pair-Plot

plt.close();
sns.set_style("darkgrid");
sns.pairplot(df, hue='status', size=3)
plt.show()


# **Observations**
# * Positive_axillary_nodes is a useful feature to identify the survival_status of cancer patients
# * Age and Year of operation have overlapping curves so we can't have a suitable observation that can classify survival_status

# we have two classes 
# 1 = the patient survived 5 years or longer 
# 2 = the patient died within 5 year

# **#mean: average**

# In[ ]:


survived_patients = df[df['status']==1]
not_survived = df[df['status']==2]


# In[ ]:


print(np.mean(survived_patients))


# In[ ]:


print(np.mean(not_survived))


# **Observation**
# 
# * Mean age of patients who survived more than 5 years is 52 years and who didn't survive is 54 years
# * Those having more than 3 positive_axillary_nodes they have not survived more than 5 years
# * Those having less than 3 positive_axillary_nodes they have survived more than 5 years after the operation

# In[ ]:


sns.jointplot(x='age', y='nodes', data=df, kind='kde', heights=6)
plt.show()


# Final Conclusion
# Those having more than 3 positive_axillary_nodes they have not survived more than 5 years
# Those having less than 3 positive_axillary_nodes they have survived more than 5 years after the operation
# Positive_axillary_nodes is a useful feature to identify the survival_status of cancer patients
# Age and Year of operation have overlapping curves so we can't classify patients for their survival_status using age

# In[ ]:




