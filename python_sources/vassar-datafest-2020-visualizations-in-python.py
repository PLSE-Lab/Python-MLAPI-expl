#!/usr/bin/env python
# coding: utf-8

# ### Import Libraries and Read in Data

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


# Read in Data
df = pd.read_csv("../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv")


# In[ ]:


df = df.sample(10000, random_state=42)


# In[ ]:


# Look at first 5 lines of data
df.tail()


# ### Univariate Viz

# #### Bar Plot

# What is the number of listings in each borough?

# In[ ]:


df.neighbourhood_group.value_counts().plot(kind='bar')
# or df.neighbourhood_group.value_counts().plot.bar()


# In[ ]:


# Horizontal Barplot
df.neighbourhood_group.value_counts().plot(kind='barh')


# In[ ]:


# In Seaborn

sns.barplot(x = 'index', y ='neighbourhood_group', data = df.neighbourhood_group.value_counts().to_frame().reset_index())


# #### Line Charts

# In[ ]:


df.neighbourhood_group.value_counts().plot(kind='line')


# #### Area Chart

# In[ ]:


df.neighbourhood_group.value_counts().plot(kind='area')


# #### Histogram / Density Plots

# look at the distribution of a certain numerical variable

# In[ ]:


df.number_of_reviews.hist()


# In[ ]:


plt.hist(df.minimum_nights, bins = 100)
plt.xlim(0,100)


# #### Boxplot (or similar types)

# In[ ]:


# Seaborn boxplot
sns.boxplot(y='number_of_reviews', data=df)


# In[ ]:


# Seaborn violinplot
sns.violinplot(y='number_of_reviews', data=df)


# #### Pie Graph

# In[ ]:


plt.pie(df['neighbourhood_group'].value_counts(), autopct='%1.1f%%', startangle = 100)
plt.axis('equal')


# ### Bivariate or Multivariate

# #### Scatterplot

# numerical var v.s. numerical var

# In[ ]:


# Simple scatterplot with matplotlib # of reviews v.s. # of reviews / month 
plt.scatter(df.reviews_per_month, df.number_of_reviews)


# In[ ]:


# Seaborn scatterplot # of reviews v.s. availability 
sns.scatterplot(x='availability_365',y='number_of_reviews', data=df, alpha=0.5)


# In[ ]:


# Seaborn scatterplot with color indicating different boroughs
sns.scatterplot(x='availability_365',y='number_of_reviews', hue = 'neighbourhood_group' ,data=df, alpha=0.7)


# #### Stacked Barplots

# In[ ]:


df.groupby(['neighbourhood_group','room_type']).number_of_reviews.mean().unstack().plot(kind='bar',stacked=True)


# #### pairplot

# Only numerical features are considered

# In[ ]:


sns.pairplot(df.sample(100))


# #### Scatter Matrix (similar to pairplot)

# In[ ]:


from pandas.plotting import scatter_matrix

pd.plotting.scatter_matrix(df.sample(100),figsize=(10,10))


# #### Multivariate Boxplots

# In[ ]:


plt.figure(figsize=(10,10))
sns.boxplot(x='neighbourhood_group', y='price',data=df)


# #### Bivariate KDE Plots

# KDE Plot described as Kernel Density Estimate is used for visualizing the Probability Density of a continuous variable

# In[ ]:


fig, ax = plt.subplots(1,1,figsize=(9,5))

sns.kdeplot(df[df['neighbourhood_group']=='Manhattan']['price'], ax=ax) 
sns.kdeplot(df[df['neighbourhood_group']=='Brooklyn']['price'], ax=ax)
plt.legend(['Manhattan', 'Brooklyn'])
plt.xlim(0,1300)


# ### Advanced Viz

# #### Plotly

# Plotly is another popular library for visualization but the sourcecode is made of javascript which gives the "interactivity" to Plotly

# In[ ]:


# Scatter Plot with Plotly

from sklearn.datasets import load_iris
import plotly.offline as py
import plotly.graph_objs as go
py.init_notebook_mode(connected=True)
from plotly import tools
import plotly.figure_factory as ff

iris = load_iris()
X = iris.data[:, :2]  # we only take the first two features
Y = iris.target

#x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
#y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5

trace = go.Scatter(x=X[:, 0],
                   y=X[:, 1],
                   mode='markers',
                   marker=dict(color=np.random.randn(150),
                               size=10,
                               colorscale='Viridis',
                               showscale=False))

layout = go.Layout(title='Training Points',
                   xaxis=dict(title='Sepal length',
                            showgrid=False),
                   yaxis=dict(title='Sepal width',
                            showgrid=False),
                  )
 
fig = go.Figure(data=[trace], layout=layout)
py.iplot(fig)


# #### Faceting

# In[ ]:


plt.figure(figsize=(8,8))
g = sns.FacetGrid(df, col='neighbourhood_group',col_wrap=3,col_order=['Manhattan','Brooklyn','Queens','Bronx','Staten Island'])
g.map(sns.boxplot, 'room_type','price',palette='Set2')
plt.tight_layout()

