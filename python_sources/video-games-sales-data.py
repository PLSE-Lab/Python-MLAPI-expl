#!/usr/bin/env python
# coding: utf-8

# # Visualization of Video Game Sales

# ## Import the required libraries.

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from mpl_toolkits import mplot3d


# ## Change the path of the Dataframe

# In[ ]:


print(os.listdir("../input"))


# ## Read the Dataframe

# In[ ]:


data=pd.read_csv("../input/vgsales.csv")


# ## To check the first 6 rows of data

# In[ ]:


data.head(6)


# ## To check statistics of data

# In[ ]:


data.describe()


# ## To get the info of the data, to check number of entirie, data type of each column etc

# In[ ]:


data.info()


# ## Using Groupby, Group average NA_Sales by Genre

# In[ ]:


AvgNASls= data.groupby('Genre')['NA_Sales'].mean()
AvgNASls


# ## Using Groupby, Group sum of NA_Sales by Genre

# In[ ]:


SumNASls= data.groupby('Genre')['NA_Sales'].sum()
SumNASls


# ## Count plot to see number of each Genre in the Dataframe

# In[ ]:


ax=sns.countplot(x = "Genre", data=data) 
plt.xticks(rotation=45)
ax.set_title(label='Number of Genres', fontsize=15)


# ## Bar Plots

# ### Barplot showing average of NA_Sales for each Genre

# In[ ]:


ax=sns.barplot(x='Genre',y='NA_Sales', data=data) 
plt.xticks(rotation=90)
ax.set_title(label='Average NA_Sales', fontsize=15)


# ### Barplot showing average of Global_Sales for each Genre

# In[ ]:


ax=sns.barplot(x='Genre',y='Global_Sales', data=data) 
plt.xticks(rotation=90)
ax.set_title(label='Average Global Sales', fontsize=15)


# ### Bar plot to show Other sales for each year

# In[ ]:


ax=sns.barplot(x="Year", y="Other_Sales", data=data)
plt.xticks(rotation=90)
ax.set_title(label='Average Other Sales', fontsize=15)


# ### Barplot showing average of Global_Sales for each year

# In[ ]:


sns.barplot(x='Year',y='Global_Sales', data=data) 
plt.xticks(rotation=90)


# ### Bar plot to show Other sales for each year, with Confidence Interval as 20

# In[ ]:


sns.barplot(x="Year", y="Other_Sales", data=data,ci=20)
plt.xticks(rotation=90)


# ### Bar plot to show Other sales for each year, with ci as 100

# In[ ]:


sns.barplot(x="Year", y="Other_Sales", data=data,ci=100)
plt.xticks(rotation=90)


# ## Scatter Plots

# ### Scatter plot to show how NA_Sales disrtibuted on each Rank

# In[ ]:


sns.scatterplot("Rank","NA_Sales",data=data)


# ### Scatter plot showing the EU_Sales for each year

# In[ ]:


sns.scatterplot(x="Year", y="EU_Sales", data=data);


# ### Scatter plot showing the EU_Sales for each year, subdeviding by Genre

# In[ ]:


sns.scatterplot(x="Year", y="EU_Sales",hue='Genre',data=data);


# ## Categorial plots

# ### Categorial plot to show NA_Sales for each Genre

# In[ ]:


sns.catplot(x="Genre", y="NA_Sales",data=data)
plt.xticks(rotation=90)


# ### Categorial plot to show NA_Sales for each Genre subdevided for each year

# In[ ]:


sns.catplot(x="Genre", y="NA_Sales",hue='Year',data=data)
plt.xticks(rotation=90)


# ### Catplot sales for each Year with Genre using boxplot

# In[ ]:


sns.catplot(x="Genre", y="Year", kind="box", data=data)
plt.xticks(rotation=45);


# ### Catplot to show NA_Sales for each year for each Genre

# In[ ]:


sns.catplot(x='Year',y='NA_Sales', col='Genre',col_wrap=3,data=data,kind='bar',height=4,aspect=2)


# ## Line Plots

# ### Line plot showing the Global_Sales for each year

# In[ ]:


ax=sns.lineplot(x="Year", y="Global_Sales",ci=None, data=data);
ax.set_title(label='Global sales/year', fontsize=15)


# ### Line plot showing the Global_Sales for each year,to represent the spread of the distribution at each Year by plotting the standard deviation

# In[ ]:


sns.lineplot(x="Year", y="Global_Sales",ci='sd', data=data);


# ### Line plot showing the Global_Sales for each year.Using Estimator

# In[ ]:


sns.lineplot(x="Year", y="Global_Sales",estimator=None, data=data);


# ### Line plot showing the Global_Sales for each year.Using Genre as hue parameter

# In[ ]:


sns.lineplot(x="Year", y="Global_Sales",hue='Genre', data=data);


# ### Line plot showing the NA_Sales for each Genre.Using Year as hue parameter

# In[ ]:


sns.lineplot(x="Genre", y="NA_Sales",hue='Year', data=data)
plt.xticks(rotation=45);


# ## Point Plots

# ### Point plot showing the NA_Sales for each Genre.

# In[ ]:


ax=sns.pointplot(x="Genre", y="NA_Sales", data=data)
plt.xticks(rotation=45);
ax.set_title(label='NA Sales', fontsize=15)


# ### Point plot showing the NA_Sales for each Genre. Using Time as hue parameter.

# In[ ]:


sns.pointplot(x="Genre", y="NA_Sales",hue='Year', data=data)
plt.xticks(rotation=45);


# ### Point plot showing the JP_Sales for each Year. 

# In[ ]:


sns.pointplot(x="Year", y="JP_Sales", data=data)
plt.xticks(rotation=90);


# ### Point plot showing the JP_Sales for each Year. Using Genre as hue parameter.

# In[ ]:


sns.pointplot(x="Year", y="JP_Sales",hue='Genre', data=data)
plt.xticks(rotation=90);


# ### Point plot showing the JP_Sales for each Year. Without joining

# In[ ]:


sns.pointplot(x="Year", y="JP_Sales",join=False, data=data)
plt.xticks(rotation=90);


# ### Point plot showing the JP_Sales for each Year. With estimator as median

# In[ ]:


from numpy import median
sns.pointplot(x="Year", y="JP_Sales", data=data,estimator=median)
plt.xticks(rotation=90);


# ### Point plot showing the JP_Sales for each Year. With confidence interval as standard deviation

# In[ ]:


sns.pointplot(x="Year", y="JP_Sales", data=data,ci='sd')
plt.xticks(rotation=90);


# ### Point plot showing the JP_Sales for each Year. Using the capsize

# In[ ]:


sns.pointplot(x="Year", y="JP_Sales", data=data,capsize=1)
plt.xticks(rotation=90);


# ## Histograms

# ### Histogram showing the NA_Sales

# In[ ]:


sns.distplot(data['NA_Sales'],hist=True,bins=5)


# ### Histogram showing the Global_Sales

# In[ ]:


sns.distplot(data['Global_Sales'],hist=True,bins=5)


# ### Histogram showing the Global_Sales using kde and rug

# In[ ]:


sns.distplot(data['Global_Sales'],kde=True,rug=True,bins=5)


# ## Pairplot

# ### Pairplot showing the comparison bewteen all numerical columns in the Dataframe

# In[ ]:


sns.pairplot(data)


# ## 3D Plot

# ### 3D plot to show NA Sales, JP Sales and EU Sales

# In[ ]:


fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.scatter(data['NA_Sales'], data['JP_Sales'],data['EU_Sales'])
ax.set_xlabel('NA_Sales')
ax.set_ylabel('JP_Sales')
ax.set_zlabel('EU_Sales')
ax.set_title(label='3D Plot', fontsize=15)
plt.show()


# ## Heat Plots

# In[ ]:


data.corr()
sns.heatmap(data.corr())


# In[ ]:




