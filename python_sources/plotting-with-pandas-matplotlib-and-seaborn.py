#!/usr/bin/env python
# coding: utf-8

# **A kind request, please upvote if this kernel helps. These scripts are inspired from [Kaggler learning](http://https://www.kaggle.com/learn/overview) posts. **
# 
# This section covers follwoing aspects:
# * Pandas basic plot functions,
# * Basic style functions
# * Subplots function
# * Pairplot
# * Heatplot

# In[ ]:


import pandas as pd
import numpy as np


# In[ ]:


df_original = pd.read_csv("../input/wine-reviews/winemag-data_first150k.csv")


# In[ ]:


df_original.head()


# In[ ]:


df_original.shape
#df_original.dim()


# In[ ]:


# bar plot
df_original['country'].value_counts().head(10).plot.bar()


# In[ ]:


df_original['points'].value_counts().sort_index().plot.bar()


# In[ ]:


#Plot Price
df_original['price'].value_counts().sort_index().plot.line()


# In[ ]:


df_original['points'].value_counts().sort_index().plot.area()


# In[ ]:


#Scatter Plot
df_original.plot.scatter(x='points', y='price')


# In[ ]:


df_original.plot.hexbin(x='price', y='points')


# In[ ]:


# Figsize argument
df_original['points'].value_counts().sort_index().plot.bar(
    figsize = (10,5),
    fontsize = 14,
    title = "Count of Wine-Points"
)


# In[ ]:


#Adjusting title fontsize using Matplotlib
import matplotlib as mlt

ax = df_original['points'].value_counts().sort_index().plot.bar(
    figsize = (12,6),
    fontsize = 14
)
ax.set_title("Count the Wine-Points", fontsize = 20)
ax.set_xlabel("Points", fontsize = 20)
ax.set_ylabel("Count", fontsize = 20)


# In[ ]:


import matplotlib.pyplot as plt
#Creating a frame of rows and columns to place the plots
fig, axi = plt.subplots(3,1, figsize = (12,12))
#plot1
df_original['points'].value_counts().plot.bar(
    ax = axi[0],
    fontsize = 14    
)
axi[0].set_title("Count points of Wine", fontsize = 20)
#Plot2
df_original['country'].value_counts().head(10).plot.bar(
    ax = axi[1],
    fontsize = 14
)
axi[1].set_title('Country', fontsize = 20)
#plot3
df_original['winery'].value_counts().head().plot.bar(
    ax = axi[2],
    fontsize = 14
)
axi[2].set_title("No of Wines from Winery")


# In[ ]:


df_original.isnull().count()


# # Exploring with *seaborn* package

# In[ ]:


import seaborn as sns


# In[ ]:


sns.countplot(df_original['country'])


# In[ ]:


sns.countplot(df_original['province'].head(20))


# In[ ]:


sns.kdeplot(df_original['price'])


# In[ ]:


ax = sns.kdeplot(df_original.query('price < 200').price)
ax.set_title("Price of Wine")


# In[ ]:


#KDE 2D plot
sns.kdeplot(df_original[df_original['price']< 200].loc[:,['price', 'points']].dropna().sample(5000))


# # Histogram plot in *Seaborn*
# 

# In[ ]:


ax = sns.distplot(df_original['points'], bins = 20, kde = False)


# In[ ]:


df1= df_original[df_original.variety.isin(df_original.variety.value_counts().head(5).index)]

sns.boxplot(
    x = 'variety',
    y = 'points',
    data = df1
)


# # Facet Grid

# In[ ]:


df_original.head()


# In[ ]:


#Extract wine scores from two countries US and France
df = df_original[df_original['country'].isin(['US','France', 'Canada', 'Spain'])]
g = sns.FacetGrid(df, col = "country", col_wrap = 2)


# In[ ]:


g = sns.FacetGrid(df, col = "country", col_wrap = 2)
g.map(sns.kdeplot, "price")


# In[ ]:


g = sns.FacetGrid(df, col = "country", col_wrap = 2)
g.map(sns.boxplot, "price")


# In[ ]:


g = sns.FacetGrid(df, col = "country", col_wrap = 2)
g.map(sns.kdeplot, "points")


# In[ ]:


import matplotlib.pyplot as plt
g = sns.FacetGrid(df, col = "country", col_wrap = 2)


# # Multi-Variant Plots

# In[ ]:


import re
import numpy as np


# In[ ]:


df_housing = pd.read_csv("../input/melbourne-housing-market/Melbourne_housing_FULL.csv")


# In[ ]:


df_housing.head(3)


# In[ ]:


df_tmp = df_housing.dropna(how='any',axis=0) 


# In[ ]:


df_tmp.head()


# In[ ]:


df_tmp2 = df_tmp[['Price', 'Distance', 'Rooms', 'Postcode', 'Bedroom2', 'Bathroom']]
df_tmp2.head()


# In[ ]:


sns.pairplot(df_tmp2)


# In[ ]:


#check the colnaems
df_tmp.columns


# In[ ]:


# Multivariant 
sns.lmplot(x='Distance', 
           y='Price',
          hue = 'Type',
          data = df_tmp)


# In[ ]:


# Multivariant 
sns.lmplot(x='Postcode', 
           y='Price',
          hue = 'Type',
          data = df_tmp)


# In[ ]:


# Multivariant 
sns.lmplot(x='Car', 
           y='Price',
          hue = 'Type',
          data = df_tmp)


# In[ ]:


# Multivariant 
sns.lmplot(x='Rooms', 
           y='Price',
          hue = 'Type',
          data = df_tmp)


# In[ ]:


# Multivariant 
sns.lmplot(x='Bathroom', 
           y='Price',
          hue = 'Type',
          data = df_tmp)


# # Grouped boxplot

# In[ ]:


sns.boxplot(x='Rooms',
           y ='Price',
           hue = 'Type',
           data = df_tmp)


# In[ ]:


sns.boxplot(x='Regionname',
           y ='Price',
           hue = 'Type',
           data = df_tmp,
           )


# # Heatplot
# 

# In[ ]:


r = df_tmp2.corr()
sns.heatmap(r)


# In[ ]:


sns.heatmap(r, annot = True)

