#!/usr/bin/env python
# coding: utf-8

# Hi! I'm [Justin](http://justinkiggins.com), a data scientist and open data enthusiast.
# 
# In this notebook, we're going to explore some [](http://)data on Hass avocado prices using pandas and seaborn.
# 
# - How do avocado prices vary between cities in the US?
# - How have avocado prices fluctuated in the past few years?
# 
# First, we'll load the csv file using pandas & take a look at the first few rows.

# In[1]:


import pandas as pd
avocados = pd.read_csv(
    '../input/avocado.csv',
    index_col=0,
)
avocados.head()


# It looks like there is a "Date" column.... let's make sure that is a pandas Datetime column, so we can use it to it's full advantage when we need to.

# In[3]:


avocados['Date'] = pd.to_datetime(avocados['Date'])
avocados.head()


# There are a few columns we'll be focusing on for this analysis:
# 
# - `Date` - The date of the observation
# - `AveragePrice` - the average price of a single avocado
# - `type` - conventional or organic
# - `year` - the year
# - `Region` - the city or region of the observation
# 
# Next, let's import seaborn & set the style.

# In[5]:


import seaborn as sns
sns.set_style('white')


# Let's plot each region's average price & color by the year of the observation.

# In[20]:


mask = avocados['type']=='conventional'
g = sns.factorplot('AveragePrice','region',data=avocados[mask],
                   hue='year',
                   size=8,
                   aspect=0.6,
                   palette='Blues',
                   join=False,
              )


# Awesome! We can already see that there are major differences in Avocado prices between these cities.
# 
# Let's  reorder the plot into something sensible, though... say, the average price during 2018.
# 
# To do this, factorplot needs us to be explicit with the order, so we'll use pandas to groupby region and sort by price, then grab the indices.

# In[12]:


order = (
    avocados[mask & (avocados['year']==2018)]
    .groupby('region')['AveragePrice']
    .mean()
    .sort_values()
    .index
)


# In[14]:


g = sns.factorplot('AveragePrice','region',data=avocados[mask],
                   hue='year',
                   size=8,
                   aspect=0.6,
                   palette='Blues',
                   order=order,
                   join=False,
              )


# That's better. Looks like Phoenix & Tuscon have some of the cheapest avocados around.
# 
# Sorry, Chicago, your avocado toast is $$$.
# 
# Let's look more closely at how prices in these two cities fluctuate over the year. First, we'll filter down to only these two regions.

# In[38]:


regions = ['PhoenixTucson', 'Chicago']


# In[37]:


mask = (
    avocados['region'].isin(regions)
    & (avocados['type']=='conventional')
)


# Then we add a "Month" column, extracted from the "Date" column using pandas builting datetime attributes.

# In[39]:


avocados['Month'] = avocados['Date'].dt.month
avocados[mask].head()


# Now, let's plot prices over time.

# In[35]:


g = sns.factorplot('Month','AveragePrice',data=avocados[mask],
               hue='year',
               row='region',
               aspect=2,
               palette='Blues',
              )


# It looks like both of these areas suffered in [the great avocadopocalypse of 2017](http://http://www.latimes.com/business/la-fi-avocado-prices-20170815-story.html), but some of this appears to be due to expected seasonal variability.
# 
# Go ahead and fork this notebook so you can explore this data yourself! Here are some other questions you can ask:
# 
# - How has the popularity of avocados changed over time?
# - In which cities is it relatively expensive to buy organic avocados?
# 
# Enjoy!
# 
# Justin [@neuromusic](http://twitter.com/neuromusic)

# In[ ]:




