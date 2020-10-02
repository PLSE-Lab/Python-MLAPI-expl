#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

# sklearn preprocessing for dealing with categorical variables
from sklearn.preprocessing import LabelEncoder

# File system manangement
import os

# Suppress warnings 
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


df=pd.read_csv('../input/winemag-data-130k-v2.csv')
df.head()


# **Missing Value Analysis**

# In[ ]:


total = df.isnull().sum().sort_values(ascending=False)
percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
df2 = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
df2


# Region_2- These are more specific regions specified within a wine growing area (eg Rutherford inside the Napa Valley), but this value can be blank too.
# Designation- The vineyard within the winery where the grapes that made the wine are from. So we can't replace the missing values with any other value. If analysis is required in designation column, then the misiing values will have to be omitted.
# Similarly for taster_twitter_handle and taster_name , if analysis is required then missing values will have to be omitted

# **Handling duplicate data**

# In[ ]:


duplicate_bool=df.duplicated()
duplicate=df.loc[duplicate_bool == True]
print(duplicate)


# [](http://)No duplicate data is there

# **Describing the data**

# In[ ]:


df.info()


# **Data Analysis**

# **Univariate data analysis**

# Plotting the top 5 countries,winery,variety and province  with most wines

# In[ ]:


plt.figure(1,figsize=[8,8])
sns.countplot('country',data=df,order=pd.value_counts(df['country']).iloc[:5].index)
plt.figure(2,figsize=[8,8])
sns.countplot('province',data=df,order=pd.value_counts(df['province']).iloc[:5].index)
plt.figure(3,figsize=[8,8])
sns.countplot('winery',data=df,order=pd.value_counts(df['winery']).iloc[:5].index)
plt.figure(4,figsize=[8,8])
sns.countplot('variety',data=df,order=pd.value_counts(df['variety']).iloc[:5].index)


# USA and France have the most number of winery. In USA , California has the most number of wine shops. 
# Wines & winemakers and Testarossa produces highest number of wines and Pinot Noir is the most preferred wine variety.

# Plotting a word cloud of the description to find the most common words

# In[ ]:


from wordcloud import WordCloud, STOPWORDS
stopwords = set(STOPWORDS)

wordcloud = WordCloud(
                          background_color='white',
                          stopwords=stopwords,
                          max_words=200,
                          max_font_size=40, 
                          random_state=42
                         ).generate(str(df['description']))

print(wordcloud)
fig = plt.figure(figsize = (8, 8),facecolor = None)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()


# The top 5 words are aromas, flavour, oak , blackberry and wine. Mostly description is about type of flavours like berry, cherry and aroma and offers.

# Description of price and points(numerical data) and finding outliers

# In[ ]:


plt.figure(1)
sns.boxplot(x=df['points'],color="blue")
plt.figure(2)
sns.boxplot(x=df['price'],color="blue")


# As can be inferred, price has huge amount ofoutlier data becauze of the presence of some luxurious wines in france wineries.
# To handle outlier data in price, price values are kept  less than 200 for bivariate analysis.

# In[ ]:


plt.figure(1)
df['points'].value_counts().sort_index().plot.line()
plt.title("Distribution of points")
plt.xlabel("points")
plt.ylabel("Values")
plt.figure(2)
df['price'].value_counts().sort_index().plot.line()
plt.title("Distribution of price")
plt.xlabel("price")
plt.ylabel("Values")


# Points value are in the range of 80 to 100 whereas most prices are between 0 and 200 with few outlier data extending beyond 3000

# **Bivariate Data Analysis**

# Finding the association between points and price

# Since price has missing values as well as outlier data so we are taking price<200 and filling missing values with mean

# In[ ]:


a=df[df['price']<200]
a['price'].fillna(a['price'].mean())
a.head()


# In[ ]:


sns.jointplot(x="price", y="points", data=a)


# As seen, price betweeen 0 to 75 have points between 80 and 95. Higher price have mostly points between 87.5 and 97.5 with few values of 100

# Finding the points and price of the top 5 variety and winery

# In[ ]:


plt.figure(1)
sns.barplot(y="variety",x="points",data=df,order=pd.value_counts(df['variety']).iloc[:5].index)
plt.figure(2)
sns.barplot(y='winery',x='points',data=df,order=pd.value_counts(df['winery']).iloc[:5].index)
plt.figure(3)
sns.barplot(y="variety",x="price",data=df,order=pd.value_counts(df['variety']).iloc[:5].index)
plt.figure(4)
sns.barplot(y='winery',x='price',data=df,order=pd.value_counts(df['winery']).iloc[:5].index)


# Top 5 variety of wines have similar points greater than 90 whereas top 5 winery have different points. Testarossa and William Selyem have higher points as compared to the other 3 wineries.
# Price shows a huge variation in both winery and variety. Chardonnay and Red Blenad are comparatively lower priced as compared to other 3 varities. Louis Latour winery is priced much  higher than other wineries

# Creating a Facet grid of points and price of 4 countries: USA, France, Canada and Spain

# In[ ]:


b=df[df['country'].isin(['US','France','Canada','Spain'])]
b.head()


# In[ ]:


g=sns.FacetGrid(b, col="country")
g=g.map(sns.kdeplot, "price", color="r")


# Canada has the cheapest wines out of the 4 countries.Spain and Canada have a much lower price range and have wines less than 1000. USA and France show a much higher price range. France particularly have some luxurious wines with price greater than 3000

# In[ ]:


g = sns.FacetGrid(b, col = "country")
g.map(sns.kdeplot, "points", color="r")


# USA and France wineries have a wide range of points extending from 80 to 100. The distribution is also imperfect.
# Spain have a almost normal distribution of points from 80 to 100.In Canada, the points are till 97 only.

# Finding the number of designation in each winery

# In[ ]:


df1=df.groupby(['winery','designation']).size()
df1.sort_values(ascending=False)


# Stillwater Creek Vineyard in Novelty Hill has the most number of wines.

# Finding the points range of top 5 variety and winery

# In[ ]:


df2=df[df.variety.isin(df.variety.value_counts()[:5].index)]#top 5 variety values
df3=df[df.winery.isin(df.winery.value_counts()[:5].index)]#top 5 winery values
plt.figure(1)
sns.violinplot(x="points",y="variety",data=df2)
plt.figure(2)
sns.violinplot(x="points",y="winery",data=df3)


# As can be seen, the top 5 variety of wines have close points range with median values close to each other. Only Pinot Noir has median points greater than 90.
# On the other hand,  top 5 wineries have different point distributions with William Selyem having median points greater than 90 whereas DFJ Vinhos having median points close to 85

# Finding the mean price of each province in a country

# In[ ]:


#df.reset_index(inplace=True)
a=df.pivot_table(values='price',index=['country','province'])
a.sort_values(by=['price'],ascending=False)


# Colares in Portugal has the highest mean price of wines whereas Viile Timisului in Romania has the lowest.

# 
