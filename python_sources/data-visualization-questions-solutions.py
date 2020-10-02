#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Import pandas package as pd
import pandas as pd


# In[ ]:


#import matplotlib.pyplot package as plt 
import matplotlib.pyplot as plt 


# **Click on the "+Add Dataset" on the top right**
# * Search for **Wine Reviews**
# * Click on **Add** the Wine Reviews dataset which shows 130k wine reviews with variety, location, winery, price, and description
# * Explore the data and copy the path

# In[ ]:


#Import the CSV file winemag-data_first150k.csv into wine_data (do not include index column)
#wine_data = pd.read_csv("<Enter the full path here>", index_col=0)
wine_data = pd.read_csv("../input/wine-reviews/winemag-data_first150k.csv", index_col=0)


# In[ ]:


#Display first 5 lines of wine_data using head() function
wine_data.head()


# In[ ]:


#Find value_counts of province column
#__._____.value_counts()
wine_data.province.value_counts()


# In[ ]:


#Find value_counts() of province column and display first 10 rows using head() function
#__._____.value_counts().head(__)
wine_data.province.value_counts().head(10)


# In[ ]:


#Find value_counts() of province column and display first 10 rows using head() function and plot the BAR graph
#__._____.value_counts().head(__).plot.bar()
wine_data.province.value_counts().head(10).plot.bar()


# In[ ]:


#Plot the figure in the 18 x 6 inches grid
#plt.figure(figsize=(___,___))
plt.figure(figsize=(18,6))
wine_data.province.value_counts().head(10).plot.bar()


# In[ ]:


#Find value_counts() of province column and display first 10 rows, normalize the data and plot the BAR graph
#(__._____.value_counts().head(__)/len(wine_data).plot.bar()
plt.figure(figsize=(18,6))
(wine_data.province.value_counts().head(10)/len(wine_data)).plot.bar()


# In[ ]:


#Find value_counts of points column
#__._____.value_counts()
wine_data.points.value_counts()


# In[ ]:


#Find value_counts of points column and sort by index
#__._____.value_counts().sort_index()
wine_data.points.value_counts().sort_index()


# In[ ]:


#Find value_counts of points column and sort by index and plot the BAR graph
#__._____.value_counts().sort_index().plot.bar()
plt.figure(figsize=(18,6))
wine_data.points.value_counts().sort_index().plot.bar()


# In[ ]:


#Find value_counts of points column and sort by index and plot the LINE graph
#__._____.value_counts().sort_index().plot.line()
plt.figure(figsize=(18,6))
wine_data.points.value_counts().sort_index().plot.line()


# In[ ]:


#Find value_counts of points column and sort by index and plot the AREA graph
#__._____.value_counts().sort_index().plot.area()
plt.figure(figsize=(18,6))
wine_data.points.value_counts().sort_index().plot.area()


# In[ ]:


#Display the rows of wine_data where the price > 200
#__[________['________']>200]
wine_data[wine_data['price']>200]


# In[ ]:


#Display the rows of wine_data where the price > 200.  Display only price column
#__[________['________']>200]['------']
wine_data[wine_data['price']>200]['price']


# In[ ]:


#Display the rows of wine_data where the price > 200.  Display only price column and plot the histogram.
#__[________['________']>200]['------'].plot.hist()
plt.figure(figsize=(18,6))
wine_data[wine_data['price']>200]['price'].plot.hist()


# In[ ]:


#Display the rows of wine_data where the price > 200.  Plot the histogram of price with 100 bins
#__[________['________']>200]['------'].plot.hist(bins=100)
plt.figure(figsize=(18,6))
wine_data[wine_data['price']>200]['price'].plot.hist(bins=100)


# In[ ]:


#Display the rows of wine_data where the price > 1500.  Plot the histogram of price with 100 bins
#__[________['________']>1500]['------'].plot.hist(bins=100)
plt.figure(figsize=(18,6))
wine_data[wine_data['price']>1500]['price'].plot.hist(bins=100)


# In[ ]:


#Display value_counts() of country in wine_data. Display first 5 rows only.
#__._____.value_counts().head()
wine_data.country.value_counts().head()


# In[ ]:


#Display value_counts() of country in wine_data. Display first 5 rows only. Plot the pie chart
#__._____.value_counts().head().plot.pie()
plt.figure(figsize=(8,8))
wine_data.country.value_counts().head().plot.pie()


# In[ ]:


#Display the rows of wine_data where the price < 100.  
#_______[________['________']<100]
wine_data[wine_data['price'] < 100]


# In[ ]:


#Display the rows of wine_data where the price < 100.  Display 5 random samples
#_______[________['________']<100].sample(__)
wine_data[wine_data['price'] < 100].sample(50)


# In[ ]:


#Display the rows of wine_data where the price < 100.  Plot the scatter plot with 'price' on x axis, 'points' on y axis for 50 random samples
#_______[________['________']<100].sample(__).plot.scatter(x='------',y='------')
wine_data[wine_data['price'] < 100].sample(50).plot.scatter(x='price',y='points')


# In[ ]:


#Display the rows of wine_data where the price < 100.  Plot the scatter plot with 'price' on x axis, 'points' on y axis for all samples
#_______[________['________']<100].plot.scatter(x='------',y='------')
wine_data[wine_data['price'] < 100].plot.scatter(x='price',y='points')


# In[ ]:


#A hex plot aggregates points in space into hexagons, and then colors those hexagons based on the values within them
#Display the rows of wine_data where the price < 100.  Plot the hexplot with 'price' on x axis, 'points' on y axis and gridsize 15.
#wine_data[wine_data['price'] < 100].plot.hexbin(x='price', y='points', gridsize=15)
wine_data[wine_data['price'] < 100].plot.hexbin(x='price', y='points', gridsize=15)


# In[ ]:


#Display the rows of wine_data where the price > 100.  Plot the histogram of price with 100 bins
#__[________['________']>1500]['------'].plot.hist(bins=100)
wine_data[wine_data['price'] < 100].sample(100).plot.scatter(x='price', y='points')


# **Click on the "+Add Dataset" on the top right**
# * Search for **Most Common Wine Scores**
# * Click on **Add** the ost Common Wine Scores dataset which shows Review scores for five common wines
# * Explore the data and copy the path

# In[ ]:


#Import the CSV file winemag-data_first150k.csv into wine_data (do not include index column)
#wine_data = pd.read_csv("<Enter the full path here>", index_col=0)
wine_counts = pd.read_csv("../input/most-common-wine-scores/top-five-wine-score-counts.csv",index_col=0)


# In[ ]:


#Display first 5 rows of wine_counts
wine_counts.head()


# In[ ]:


#wine_counts counts the number of times each of the possible review scores was received by the five most commonly reviewed types of wines:
#Plot the stacked BAR chart of wine_counts
#_______.plot.bar(stacked=True)
wine_counts.plot.bar(stacked=True)


# In[ ]:


#Plot the stacked AREA chart of wine_counts
#_______.plot.area(stacked=True)
wine_counts.plot.area(stacked=True)


# In[ ]:


#Plot the stacked LINE chart of wine_counts
#_______.plot.line(stacked=True)
wine_counts.plot.line(stacked=True)


# **Click on the "+Add Dataset" on the top right**
# * Search for **Pokemon with stats**
# * Click on **Add** the "Pokemon with stats" dataset which shows "721 Pokemon with stats and types"
# * Explore the data and copy the path

# In[ ]:


#Import the Pokemon.csv" dataset
#pokemon = pd.read_csv("_______________________", index_col=0)
pokemon = pd.read_csv("../input/pokemon/Pokemon.csv", index_col=0)


# In[ ]:


#Display first 5 rows of pokemon dataset
#________.head()
pokemon.head()


# In[ ]:


#Plot scatter plot with 'Attack' on x-axis and 'Defence' on y-axis
#pokemon.plot.scatter(x = '_________', y = '_________')


# In[ ]:


#A hex plot aggregates points in space into hexagons, and then colors those hexagons based on the values within them
#Plot scatter plot with 'Attack' on x-axis and 'Defence' on y-axis with grid size of 15
#pokemon.plot.hexbin(x='__________', y='___________', gridsize=___)
pokemon.plot.hexbin(x='Attack', y='Defense', gridsize=15)


# In[ ]:


#Plot the bar chart of points column of wine_data
wine_data['points'].value_counts().sort_index().plot.bar(figsize=(12, 6))


# In[ ]:




