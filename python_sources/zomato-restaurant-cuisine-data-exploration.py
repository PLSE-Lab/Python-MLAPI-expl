#!/usr/bin/env python
# coding: utf-8

# ## You say Zomayto, I say Zomahto ##
# Zomato was founded in 2008 by Indian entrepreneurs. It launched in India but has expanded to many other countries, including the United States. 
# 
# It offers a lot of information about various restaurants, including menus, but this dataset is limited to mostly basic information such as location, price range, and ratings.
# 
# What I'm most interested in is what factors may impact a restaurant's ratings.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# begin by loading file and merging with country code
df = pd.read_csv('../input/zomato.csv',encoding="ISO-8859-1")
country = pd.read_excel('../input/Country-Code.xlsx')
zomato_df = pd.merge(df, country, on='Country Code')
zomato_df.head()


# In[ ]:


zomato_df.describe()


# Now that I've gotten an idea of what kind of data I'm working with, I'm going to check for problems with it. First of all, any NaNs?

# In[ ]:


nans = lambda df: df[df.isnull().any(axis=1)]
nans(zomato_df)


#  Just a handful of rows with NaN, all of them cuisines. That's one of the columns I want to analyze, so I think the best way to handle this is to simply drop them.

# In[ ]:


zomato_df = zomato_df.dropna()
nans(zomato_df)


# The next thing I notice is that the minimum value of the aggregate rating is zero, which doesn't seem right. Also, the average of the aggregate rating is less than 3, which seems really low. My guess is that the zeroes are dragging down the mean. How many of these suckers are there?

# In[ ]:


zomato_df.hist("Aggregate rating", bins=25)


# More than 2,000 strikes me as a lot.  But before I get rid of them, I will check the other rating fields to make sure there is really no ratings informations for these restaurants.

# In[ ]:


zeroes = zomato_df[zomato_df["Aggregate rating"] == 0]
zeroes["Rating color"].value_counts()


# In[ ]:


zeroes["Rating text"].value_counts()


# OK, we for sure do not have any ratings info for these restaurants. So I'll drop these rows from zomato_df, holding onto the zeroes dataframe in case I want to make predictions later.

# In[ ]:


zomato_df = zomato_df[zomato_df["Aggregate rating"] > 0]
zomato_df.describe()


# My next question is about the costs. First, because the minimum is zero, and second because the range is insane.

# In[ ]:


zomato_df[zomato_df["Average Cost for two"] == 0]


# In[ ]:


zomato_df[zomato_df["Price range"] == 1]["Average Cost for two"].hist()


# I am going to drop the rows with zeroes for average cost; there aren't that many of them and if price range is based on the average cost then that could be off as well.

# In[ ]:


zomato_df = zomato_df[zomato_df["Average Cost for two"] > 0]
zomato_df.describe()


# The last problem I see here is that the range for average cost  is enormous.  I assume this is based on currency.

# In[ ]:


zomato_df.groupby('Currency')['Average Cost for two'].mean()


# Yeah, it's pretty obvious that Indonesia is causing our pain here. Although Botswana and Sri Lanka are also obvious outliers, to a lesser degree, and for this field to be of any use we would have to standardize all of the fields to one currency. For my purposes today it probably  makes more sense to just stick to price range.
# 
# So now I'm going to drop the columns that don't seem useful and continue my analysis from there.

# In[ ]:


# I am not interested in all of these columns, so I am going to drop some
drop_columns = ["Restaurant ID", "Country Code", "Locality Verbose", "Latitude", "Longitude", "Rating color", "Currency", "Average Cost for two", "Address", "Locality", "Switch to order menu"]
zomato_df = zomato_df[zomato_df.columns.drop(drop_columns)]
zomato_df.head()


# Having pared down the columns, now I want to add one. Specifically I have noticed that many restaurants offer multiple cuisines. I'm curious about how having many cuisines affects a restaurant's ratings, so I'm going to add a column with a cuisine count.

# In[ ]:


zomato_df["Cuisine count"] = zomato_df["Cuisines"].apply(lambda x: x.count(",") + 1)
zomato_df.head()


# In[ ]:


zomato_df.describe()


# In[ ]:


zomato_df["Cuisine count"].hist(bins = 8) # 8 bins because the cuisine count ranges from one to eight


# I want to look at some tables and graphs to get a feel for how the data is broken down.

# In[ ]:


zomato_df["Country"].value_counts()


# The dominance of India is not surprising based on how Zomato started.
# 
# I'm going to add some more columns to support some of the breakdowns I'm interested in: which restaurants offer Indian food (alone or with other cuisines); which ones are in India and which aren't; which ones offer more than one cuisine; and which ones are above average vs. average or below.

# In[ ]:


zomato_df["Offers Indian food"] = zomato_df["Cuisines"].apply(lambda x: 'Yes' if x.find("Indian") >= 0 else 'No')
zomato_df["India"] = zomato_df["Country"].apply(lambda x: 'India' if x == "India" else 'outside India')
zomato_df["Multiple cuisines"] = zomato_df["Cuisine count"].apply(lambda x: 'Yes' if x > 1 else 'No')

avg_rating = zomato_df["Aggregate rating"].mean()
zomato_df["Relative rating"] = zomato_df["Aggregate rating"].apply(lambda x: 'Above average' if x > avg_rating else 'Average or below')
zomato_df.head()


# First thought: do restaurants that offer Indian food do better in the ratings?

# In[ ]:


sns.countplot(x = "Relative rating", hue = "Offers Indian food", data = zomato_df)


# If anything, restaurants that offer Indian food seem to do a little worse. I wonder how this interacts with price range.

# In[ ]:


sns.countplot(x = "Price range", hue = "Offers Indian food", data = zomato_df)


# In[ ]:


sns.countplot(x = "Price range", hue = "Relative rating", data = zomato_df)


# It might also be interesting to look at other cuisine t ypes -- like dessert, pizza, French, etc. -- but I will skip that for now.
# 
# A scatterplot for the price range and the aggregate rating may be more useful.

# In[ ]:


zomato_df.plot(x='Price range', y='Aggregate rating', kind='scatter', figsize=(12,6))
plt.title('Price range vs. Rating')


# I also want to  look at multiple cusines compared to the relative rating.

# In[ ]:


sns.countplot(x = "Relative rating", hue = "Multiple cuisines", data = zomato_df)


# The simple yes/no isn't telling me very much, so I'll look at it by cuisine count.

# In[ ]:


sns.countplot(x = "Relative rating", hue = "Cuisine count", data = zomato_df)


# The problem is, once we get more than three cuisines we don't have a lot of data. What if I put everything with more than three cuisines in the same bucket?

# In[ ]:


zomato_df["Multiple cuisines"] = zomato_df["Cuisine count"].apply(lambda x: str(x) if x <= 3 else '> 3')
cuisine_count_order = ['1', '2', '3', '> 3']
sns.countplot(x = "Relative rating", hue = "Multiple cuisines", data = zomato_df, hue_order = cuisine_count_order)


# The next thing I want to look at is the interaction of votes with the aggregate rating. Are the maximums and minimums being skewed by just a handful of votes?

# In[ ]:


zomato_df.plot(x='Votes', y='Aggregate rating', kind='scatter', figsize=(12,6))
plt.title('Votes vs. Rating')


# I'm going to finish up by looking at some of the columns I haven't examined yet just to see if they tell us anything interesting.

# In[ ]:


sns.countplot(x = "Relative rating", hue = "Has Table booking", data = zomato_df)


# In[ ]:


sns.countplot(x = "Relative rating", hue = "Has Online delivery", data = zomato_df)


# In[ ]:


sns.countplot(x = "Relative rating", hue = "Is delivering now", data = zomato_df)


# Based on this analysis, if I were going to try to predict aggregate ratings for the restaurants with no ratings, I would focus on price range and the number of cuisines offered. 
