#!/usr/bin/env python
# coding: utf-8

# Here, I will do a brief exploratory analysis of a dataset containing roughly 130k wine reviews. I will examine overall patterns in the price of wine for several countries, how a wine's price may be related to its vintage, and finally what are the common descriptors reviewers use for assessing a given wine. 

# First, the data is read in.

# In[24]:



import numpy as np 
import pandas as pd 
import seaborn as sns
import plotnine as pn
import matplotlib.pyplot as plt
reviews = pd.read_csv('../input/winemag-data-130k-v2.csv', index_col = 0)



# # First Steps

# Here are the first few rows of the data. 

# In[63]:


reviews.head()


# The vintage of the wine is often mentioned in its title. I'll attempt to extract it, and create a vintage variable. Vintages prior to 1990 have been removed because there are very few wines with vintages prior to 1990.

# In[27]:


years = reviews.title.str.extract('([1-2][0-9]{3})').astype('float64')

years[years < 1990] = None
reviews = reviews.assign(year = years)


# Certain countries are well known for their wines. Let's look at a few of these countries a little closer. 

# In[28]:


good_countries = reviews.loc[reviews.country.isin(['US','Italy','Portugal','Spain','France','Germany','Australia']),:]


# Below is graph of violin plots of price by countrty. As price variables are often highly right-skewed, I will plot the distribtuion of the log of the price of the wine. 

# In[46]:


plt.subplots(figsize=(10,10))

sns.violinplot(x = good_countries.country,y = np.log(good_countries.price), 
                          figure_size = [2,2]).set_title('Price by Country')


plt.xlabel("Country")
plt.ylabel("Log of Price")


# Also of interest is  how the average price of a bottle of wine varies with a given rating. Below are lowess plots of average price against rating, by country. Note that the relationship is nonlinear. 

# In[25]:



(pn.ggplot(good_countries,pn.aes(x = 'points', y = 'price', color = 'country')) 
 + pn.facet_wrap('~country', scales = 'free')+ pn.stat_smooth(method = 'lowess', span = .5)
)


# Turning our attention back to the entire data set of reviews. Let's look at how the average and maximum values of price and points vary by vintage. Intuitively, wines that were very young at the time a review was written will have lower reviews (because they have not yet matured), while older wines may be good or bad, depending on whether or not they are "too old". 

# In[30]:


yearly_price_mean = reviews.groupby('year').price.agg(['mean'])
yearly_price_max = reviews.groupby('year').price.agg(['max'])
yearly_point_mean = reviews.groupby('year').points.agg(['mean'])
yearly_point_max = reviews.groupby('year').points.agg(['max'])


# Looking at the number of wines in each vintage, we see that some of the vintages have very few specimens. Thus, a few of the years will not be considered. 

# In[14]:


reviews.year.value_counts().sort_index()


# In[41]:


fig, axarr = plt.subplots(2, 2, figsize=(16, 10))

(yearly_price_mean[yearly_point_mean.index >= 1994]
 .plot
 .line(title = 'Mean Price by Vintage',ax = axarr[0][0])
 .set(xlabel = 'Year',ylabel = 'Average Price')
)

(yearly_price_max[yearly_point_max.index >= 1994]
.plot
.line(title = 'Max Price by Vintage',ax = axarr[0][1])
.set(xlabel = 'Year',ylabel = 'Max Price'))

(yearly_point_mean[yearly_point_mean.index >= 1994]
.plot
.line(title = 'Mean Rating by Vintage',ax = axarr[1][0])
.set(xlabel = 'Year',ylabel = 'Average Rating'))

(yearly_point_max[yearly_point_max.index >= 1994]
.plot
.line(title = 'Max Rating by Vintage',ax = axarr[1][1])
.set(xlabel = 'Year',ylabel = 'Max Rating'))


# Here, we see that mean price and mean rating ted to change together. We also see that the oldest wines tend to command the highest prices on average (and ratings), while very recent wines tend to be less expensive (and are rated lower). 

# # A Basic Text Analysis

# There are certain words and characteristics that may be common when reviewing wine. For example, the review will often mention whether the wine is red or white, or will mention something about the taste of the wine (is it sweet, tart, etc.). 

# In[56]:


is_word_used = reviews.description.str.contains(pat = 'aroma|taste|color|grape|age')

sum(is_word_used)/len(is_word_used)


# Above, we see that the at least one of the words from the set {'aroma', 'taste','color',''grape", "age"} appears in roughly 51% of the reviews in this data set. This would suggest that these qualtities of a wine are often considered by reviewers when assessing a wine. Of course, these qualities may not be mentioned directly in a review. Instead, the reviewer may mention aspects of the wine that fit under one of these categories. For example, the wine may be described as "fruity" or perhaps the reviewer will list the specific fruits he or she can taste in the wine. In addition, references to how the wine is stored may be made. Below is an eximination of how reviewers may reference the qualities mentioned here. These words were chosen based on a prior belief of what flavors are commonly present in wine.

# In[62]:


is_word_used = reviews.description.str.contains(
    pat = 'fruit|crisp|clean|sweet|tart|red|white|wood|apple|pear|pineapple|lemon|pomegranate|wood|oak')

sum(is_word_used)/len(is_word_used)


# The words above take the first categories we considered, and break them down further. For example, here we search for specific qualities relating to taste, or specific colors of wine. These particular words appear in about 85% of the reviews in this data set, suggesting that these kinds of qualities are what reviewers notice, and consider when reviewing a wine. 
