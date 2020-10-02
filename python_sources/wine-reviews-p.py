#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
reviews = pd.read_csv("../input/winemag-data_first150k.csv", index_col = 0)
reviews.head(3)


# In[ ]:


reviews['province'].value_counts().head(10).plot.bar(color='mediumblue')


#  California produces far more wine than any other province of the world! 

# In[ ]:


(reviews['province'].value_counts().head(10) / len(reviews)).plot.bar()


# In[ ]:


reviews['points'].value_counts().sort_index().plot.bar(color = 'mediumblue')


#  Most of the wines had an overall score of 87 allotted by Wine Magazine.

# In[ ]:


reviews['points'].value_counts().sort_index().plot.line()


# In[ ]:


reviews['points'].value_counts().sort_index().plot.area()


# In[ ]:


reviews[reviews['price'] < 200] ['price'].plot.hist();


# Wines less than 200$.

# In[ ]:


reviews['price'].plot.hist()


# In[ ]:


reviews[reviews['price'] > 1500].head(3)


# Wines cost more than 1500$.

# In[ ]:


reviews['points'].plot.hist()


# In[ ]:


reviews[reviews['price'] < 100].sample(100).plot.scatter(x = 'price', y = 'points')


# Scatter of points to price.

# In[ ]:


reviews[reviews['price'] < 100].plot.hexbin(x = 'price', y = 'points', gridsize = 15)


# With the hexplot he bottles of wine reviewed by Wine Magazine cluster around 87.5 points and around $20.

# In[ ]:


sns.countplot(reviews['points'])


# Bar plot of wines based on points.

# In[ ]:


sns.kdeplot(reviews.query('price < 200').price)


# Line plot of wines less than 200.

# In[ ]:


reviews[reviews['price'] < 200]['price'].value_counts().sort_index().plot.line()


# In[ ]:


Line plot of wines less than 200.


# In[ ]:


sns.kdeplot(reviews[reviews['price'] < 200].loc[:,['price', 'points']].dropna().sample(5000))


# In[ ]:


sns.distplot(reviews['points'], bins = 10, kde = False)


# Histogram of wines based on points.

# In[ ]:


sns.jointplot(x = 'price', y = 'points', data = reviews[reviews['price'] < 100])


# A correlation coefficient with histograms on the sides.

# In[ ]:


sns.jointplot(x = 'price', y = 'points', data = reviews[reviews['price'] < 100], kind = 'hex', gridsize = 20)


# A hex diagram with histograms on the sides.

# In[ ]:


df = reviews[reviews.variety.isin(reviews.variety.value_counts().head(5).index)]
sns.boxplot(x = 'variety', y = 'points', data = df)


# Boxplot showsthat although all five wines recieve broadly similar ratings, Bordeaux-style wines tend to be rated a little higher than a Chardonnay.

# In[ ]:


df = reviews[reviews.variety.isin(reviews.variety.value_counts()[:5].index)]
sns.violinplot(x = 'variety', y = 'points', data = df)


# The violin plot shows basically the same data as the boxplot, but is harder to misinterpret and much prettier than the utilitarian boxplot.

# In[ ]:


reviews['points'].value_counts().sort_index().plot.bar()


# Bar chart with reviews points and price.

# In[ ]:


reviews['points'].value_counts().sort_index().plot.bar(figsize=(12, 6))


# In[ ]:


reviews['points'].value_counts().sort_index().plot.bar(figsize=(12, 6), color='mediumblue',fontsize=18, 
                                                      title='Rankings given by the Wine magazine')


# A larger bar chart with reviews points and price.

# In[ ]:


import matplotlib.pyplot as plt

ax = reviews['points'].value_counts().sort_index().plot.bar(figsize=(12, 6), color='mediumblue',fontsize=18)
ax.set_title("Rankings given by the Wine magazine",fontsize=20)
sns.despine(bottom=True, left=True)


# Rankings given by wine.

# In[ ]:


import matplotlib.pyplot as plt
fig, axarr = plt.subplots(2, 1, figsize=(12, 8))


# Subplots. Initial sizes.

# In[ ]:


axarr


# In[ ]:


fig, axarr = plt.subplots(2, 1, figsize=(12, 8))
reviews['points'].value_counts().sort_index().plot.bar(ax = axarr[0], color = 'blue')
reviews['province'].value_counts().head(20).plot.bar(ax = axarr[1], color = 'blue')


# Subplot bar charts based on points and province.

# In[ ]:


fig, axarr = plt.subplots(2, 2, figsize=(12, 8))


# Initial two by  two subplots.

# In[ ]:


axarr


# In[ ]:


fig, axarr = plt.subplots(2, 2, figsize=(12, 8))
reviews['points'].value_counts().sort_index().plot.bar(ax = axarr[0][0], color = 'blue')
reviews['province'].value_counts().head(20).plot.bar(ax = axarr[1][1], color = 'blue')


# Subplot bar charts based on points and province.

# In[ ]:


fig, axarr = plt.subplots(2, 2, figsize = (12, 8))
reviews['points'].value_counts().sort_index().plot.bar(ax = axarr[0][0], fontsize = 12, color = 'mediumvioletred')
axarr[0][0].set_title("Wine Scores", fontsize = 18)
reviews['variety'].value_counts().head(20).plot.bar(ax = axarr[1][0], fontsize = 12, color = 'mediumvioletred')
axarr[1][0].set_title("Wine Varieties", fontsize = 18)
reviews['province'].value_counts().head(20).plot.bar(ax = axarr[1][1], fontsize = 12, color = 'mediumvioletred')
axarr[1][1].set_title("Wine Origin", fontsize = 18)
reviews['price'].value_counts().plot.hist(ax = axarr[0][1], fontsize = 12, color = 'mediumvioletred')
axarr[0][1].set_title("Wine Prices", fontsize = 18)
plt.subplots_adjust(hspace=.3)
import seaborn as sns
sns.despine()


# Subplot bar charts based on price, scores, variety and origin..

# In[ ]:


sns.countplot(reviews['points'], color = 'blue')


# Subplot bar chart based on points.

# In[ ]:


sns.kdeplot(reviews.query('price < 200').price)


# Line chart for wines less than 200.

# In[ ]:


reviews[reviews['price'] < 200]['price'].value_counts().sort_index().plot.line()


# Line chart for wines less than 200.

# In[ ]:


sns.kdeplot(reviews[reviews['price'] < 200].loc[:, ['price', 'points']].dropna().sample(5000))


# Bivariate plot.

# In[ ]:


sns.distplot(reviews['points'], bins = 10, kde = False)


# Custom bar chart for wine points.

# In[ ]:


sns.jointplot(x = 'price', y = 'points', data = reviews[reviews['price'] < 100])


# In[ ]:


sns.jointplot(x = 'price', y = 'points', data = reviews[reviews['price'] < 100], kind = 'hex', gridsize = 20)


# In[ ]:


df = reviews[reviews.variety.isin(reviews.variety.value_counts().head(5).index)]
sns.boxplot(x = 'variety', y = 'points', data = df)


# In[ ]:


sns.violinplot(x='variety', 
               y='points', 
               df=reviews[reviews.variety.isin(reviews.variety.value_counts()[:5].index)]
)


# In[ ]:


sns.violinplot(x='variety', 
               y='points', 
               data=reviews[reviews.variety.isin(reviews.variety.value_counts()[:5].index)]
)


# In[ ]:


reviews.head()


# In[1]:


import pandas as pd
reviews = pd.read_csv("../input/winemag-data-130k-v2.csv", index_col=0)
reviews.head()


# In[9]:


from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected = True)


# In[10]:


import plotly.graph_objs as go
iplot([go.Scatter(x = reviews.head(1000)['points'], y = reviews.head(1000)['price'], mode = 'markers')])


# In[11]:


iplot([go.Histogram2dContour(x = reviews.head(500)['points'], 
                            y = reviews.head(500)['price'],
                            contours = go.Contours(coloring = 'heatmap')),
      go.Scatter(x = reviews.head(1000)['points'], y = reviews.head(1000)['price'], mode = 'markers')])


# In[ ]:


df = reviews.assign(n = 0).groupby(['points', 'price'])['n'].count().reset_index()
df = df[df["price"] < 100]
v = df.pivot(index = 'price', columns = 'points', values = 'n').fillna(0).values.tolist()


# In[ ]:


iplot([go.Surface(z = v)])


# In[ ]:


df = reviews['country'].replace("US", "United States").value_counts()
iplot([go.Choropleth(
    locationmode = 'country names',
    locations = df.index.values,
    text = df.index,
    z = df.values
)])


# In[12]:


import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
reviews = pd.read_csv("../input/winemag-data-130k-v2.csv", index_col = 0)
reviews.head(3)


# In[2]:


from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)


# In[3]:


import plotly.graph_objs as go

iplot([go.Scatter(x=reviews.head(1000)['points'], y=reviews.head(1000)['price'], mode='markers')])


# This basic scatter plot is fully interactive.

# In[ ]:


iplot([go.Histogram2dContour(x=reviews.head(500)['points'],
                             y=reviews.head(500)['price'],
                             contours = go.Contours(coloring = 'heatmap')),
      go.Scatter(x=reviews.head(1000)['points'], y=reviews.head(1000)['price'], mode='markers')])


# KDE plot and scatter plot of the same data.

# In[ ]:


df = reviews.assign(n = 0).groupby(['points', 'price'])['n'].count().reset_index()
df = df[df['price'] < 100]
v = df.pivot(index = 'price', columns = 'points', values = 'n').fillna(0).values.tolist()


# In[ ]:


iplot([go.Surface(z=v)])


# iplot is the highest-level API is the most convenient one for general-purpose use.

# In[ ]:


df = reviews['country'].replace("US", "United States").value_counts()

iplot([go.Choropleth(
    locationmode = 'country names',
    locations = df.index.values,
    text = df.index,
    z = df.values
)])


# Plot of highest producers of wine.

# In[13]:


from plotnine import *


# In[14]:


top_wines = reviews[reviews['variety'].isin(reviews['variety'].value_counts().head(5).index)]


# In[15]:


df = top_wines.head(1000).dropna()

(ggplot(df)
 + aes(color = 'points')
+ aes('points', 'price')
+ geom_point()
+ stat_smooth())


# A simple scatter plot of amount of review points to price of wine and regression line.

# In[16]:


df = top_wines.head(1000).dropna()

(ggplot(df)
 + aes(color = 'points')
+ aes('points', 'price')
+ geom_point()
+ stat_smooth()
+ facet_wrap('~variety'))


# Faceting fits all the changes (color, regression line) in one place.

# In[17]:


(ggplot(df)
    + geom_point(aes('points', 'price'))
)


# Scatter plot with aes as a layer parameter.

# In[19]:


(ggplot(df, aes('points', 'price'))
   + geom_point()
)


#  aes as a parameter in the data.

# In[23]:


(ggplot(top_wines)
        + aes('points')
        + geom_bar()
       )


# Bar plot of count of reviews and price.

# In[26]:


(ggplot(top_wines)
    + aes('points', 'variety')
    + geom_bin2d(bins=20)
)


# Hexplot of points and variety.

# In[27]:


(ggplot(top_wines)
    + aes('points', 'variety')
    +geom_bin2d(bins=20)
    + coord_fixed(ratio=1)
    + ggtitle("Top Five Most Common Wine Variety points awarded")
             )


# Hexplot of the "Top Five Most Common Wine Variety points awarded."

# In[ ]:




