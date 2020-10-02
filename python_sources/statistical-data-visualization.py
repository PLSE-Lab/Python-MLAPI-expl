#!/usr/bin/env python
# coding: utf-8

# #  Seaborn: Statistical Data Visualization

# Seaborn is a Python data visualization library based on matplotlib. It provides a high-level interface for drawing attractive and informative statistical graphics.

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


# In[ ]:


cd = pd.read_csv("../input/CancerData.csv")


# In[ ]:


cd.head()


# In[ ]:


import seaborn as sns 
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')


# In[ ]:


dv = cd.diagnosis         # dependent variable                 # M or B 
list = ['id','diagnosis']
iv = cd.drop(list,axis = 1 ) # independent variable
iv.head()


# # Data Visualization : Cancer Data

# # Count plot ->
# A count plot can be thought of as a histogram across a categorical, instead of quantitative, variable. The basic API and options are identical to those for barplot(), so you can compare counts across nested variables.

# In[ ]:


ax = sns.countplot(dv,label="Count")
plt.grid(True,color='G')


# In[ ]:


B, M = dv.value_counts()
print('Number of Benign: ',B)
print('Number of Malignant : ',M)


# # Violin Plot ->
# A violin plot is a method of plotting numeric data. It is similar to a box plot, with the addition of a rotated kernel density plot on each side. Violin plots are similar to box plots, except that they also show the probability density of the data at different values, usually smoothed by a kernel density estimator.

# In[ ]:


y=dv
x=iv
data_n_2 = (iv - iv.mean()) / (iv.std())              # standardization
data = pd.concat([y,data_n_2.iloc[:,0:10]],axis=1)
data = pd.melt(data,id_vars="diagnosis",
                    var_name="features",
                    value_name='value')
plt.figure(figsize=(10,10))
sns.violinplot(x="features", y="value", hue="diagnosis", data=data,split=True, inner="quart", aspect=9)
plt.grid(True,color='G')
plt.xticks(rotation=90)


# # Cat Plot ->

# In[ ]:


sns.catplot(x="features", y="value", hue="diagnosis", aspect=.6,
            kind="swarm", data=data);


# In[ ]:


sns.catplot(x="features", y="value", hue="diagnosis", palette="ch:.25", data=data, aspect=3);


# In[ ]:


sns.catplot(x="features", y="value", hue="diagnosis", kind="bar", data=data, aspect=3);


# In[ ]:


sns.catplot(x="features", y="value", hue="diagnosis", jitter=False, data=data, aspect=3);


# In[ ]:


sns.catplot(x="features", y="value", data=data, legend=True, aspect=3);


# In[ ]:


sns.set(rc={'figure.figsize':(1001.7,8.27)})
sns.catplot(x="features", y="value", data=data,kind="box", legend=True , aspect=3);


# In[ ]:


sns.catplot(x="features", y="value", hue="diagnosis", kind="box", data=data, aspect=3);


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure(figsize=(20, 5))
sns.catplot(x="features", y="value", kind="boxen",
            data=data, aspect=3);


# # Factor Plot ->

# In[ ]:


g = sns.factorplot(x="features", y="value", hue="diagnosis",
        data=data, kind="box", aspect=3)


# In[ ]:


sns.set_style('ticks')
sns.violinplot(data=data, inner="points", ax=ax, aspect=3)    
sns.despine()


# # Box Plot ->
# The box plot is a standardized way of displaying the distribution of data based on the five number summary: minimum, first quartile, median, third quartile, and maximum.
# 

# In[ ]:


plt.figure(figsize=(10,10))
sns.boxplot(x="features", y="value", hue="diagnosis", data=data)
plt.xticks(rotation=90)


# In[ ]:


data.head()


# In[ ]:


x.head()


# # Joint Plot ->

# In[ ]:


g = sns.jointplot(x.loc[:,'texture_mean'], x.loc[:,'smoothness_mean'], data=cd, kind="hex", color="R")


# In[ ]:


sns.jointplot(x.loc[:,'texture_mean'], x.loc[:,'smoothness_mean'], kind="regg", color="G")


# # Joint Plot ->

# In[ ]:


sns.jointplot(x.loc[:,'concavity_worst'], x.loc[:,'concave points_worst'], kind="regg", color="#ce1414")


# In[ ]:


y.head()


# # Scatter Plot ->
# A scatter plot is a type of plot or mathematical diagram using Cartesian coordinates to display values for typically two variables for a set of data. If the points are coded, one additional variable can be displayed.

# In[ ]:


import plotly.express as px
fig = px.scatter(x, x=x.loc[:,'concavity_worst'], y=x.loc[:,'concave points_worst'],log_x=True, size_max=600)
fig.show()


# # Pair Grid Plot ->
# Subplot grid for plotting pairwise relationships in a dataset.
# 
# This class maps each variable in a dataset onto a column and row in a grid of multiple axes. Different axes-level plotting functions can be used to draw bivariate plots in the upper and lower triangles, and the the marginal distribution of each variable can be shown on the diagonal.

# In[ ]:


sns.set(style="white")
df = x.loc[:,['radius_worst','perimeter_worst','area_worst']]
g = sns.PairGrid(df, diag_sharey=False)
g.map_lower(sns.kdeplot, cmap="Blues_d")
g.map_upper(plt.scatter)
g.map_diag(sns.kdeplot, lw=3)
g = g.add_legend()


# ### Color the points using a categorical variable

# In[ ]:


df = cd.loc[:,['diagnosis','radius_worst','perimeter_worst','area_worst']]
g = sns.PairGrid(df, diag_sharey=False, hue="diagnosis")
g = g.map_offdiag(plt.scatter)
g.map_upper(plt.scatter)
g.map_diag(sns.kdeplot, lw=3)
g = g.add_legend()


# In[ ]:


df = cd.loc[:,['diagnosis','radius_worst','perimeter_worst','area_worst']]
g = sns.PairGrid(df, diag_sharey=False, hue="diagnosis")
g = g.map(sns.scatterplot, linewidths=1, edgecolor="w", s=40)
g = g.add_legend()


# # Swarm Plot ->

# In[ ]:


import time
from subprocess import check_output


# In[ ]:


sns.set(style="whitegrid", palette="muted")
data_dia = y
data = x
data_n_2 = (data - data.mean()) / (data.std())              # standardization
data = pd.concat([y,data_n_2.iloc[:,0:10]],axis=1)
data = pd.melt(data,id_vars="diagnosis",
                    var_name="features",
                    value_name='value')
plt.figure(figsize=(10,10))
tic = time.time()
sns.swarmplot(x="features", y="value", hue="diagnosis", data=data)

plt.xticks(rotation=90)


# # Correlation Map

# In[ ]:


f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(x.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)


# In[ ]:


fig, (axis1,axis2,axis3) = plt.subplots(1,3,figsize=(14,12))

sns.boxplot(x="features", y="value", hue="diagnosis", data = data, ax = axis1)
sns.violinplot(x="features", y="value", hue="diagnosis", data = data, split = True, ax = axis2)
sns.boxplot(x="features", y="value", hue="diagnosis", data = data, ax = axis3)


# In[ ]:


fig, saxis = plt.subplots(2, 3,figsize=(16,12))

sns.barplot(x="features", y="value", data=data, ax = saxis[0,0])
sns.barplot(x="features", y="value", order=[1,2,3], data=data, ax = saxis[0,1])
sns.barplot(x="features", y="value", order=[1,0], data=data, ax = saxis[0,2])

sns.pointplot(x="features", y="value",  data=data, ax = saxis[1,0])
sns.pointplot(x="features", y="value",  data=data, ax = saxis[1,1])
sns.pointplot(x="features", y="value", data=data, ax = saxis[1,2])

