#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-whitegrid')

get_ipython().run_line_magic('matplotlib', 'inline')


# ## Loading, cleaning and organizing data

# In[ ]:


df = pd.read_csv('../input/FAO.csv', encoding = "ISO-8859-1")

df.head(5)


# We are only interested in looking at the data for the 12 South American states from 2000 to 2013.

# In[ ]:


new_names = {'Venezuela (Bolivarian Republic of)': 'Venezuela',
             'Bolivia (Plurinational State of)' : 'Bolivia'}
df.replace({'Area' : new_names}, inplace=True)

countries = ['Brazil', 'Colombia', 'Argentina', 
             'Peru', 'Venezuela', 'Chile',
             'Ecuador', 'Bolivia', 'Paraguay', 
             'Uruguay', 'Guyana', 'Suriname']
df = df[df['Area'].isin(countries)]

cols = ['Area', 'Item', 'Element', 
        'Y2000','Y2001','Y2002', 
        'Y2003', 'Y2004', 'Y2005', 
        'Y2006', 'Y2007', 'Y2008', 
        'Y2009', 'Y2010', 'Y2011', 
        'Y2012', 'Y2013']
df = df[cols]


df.columns = df.columns.str.replace(r'(Y)(\d{4})', r'\2') # Reformat column names for years: Y2001->2001

df['Sum 2000-2013'] = df.loc[:, '2000':'2013'].sum(axis=1)

df.head(5)


# ## Countrywise production

# Let's see how many tonnes of food have been produced by each South American state in the period comprised between 2000 and 2013. The result is expressed in 1000 tonnes.

# In[ ]:


prod = df.groupby('Area')['Sum 2000-2013'].sum().astype(int)
prod.sort_values(ascending=False, inplace=True)

prod


# Let's now visualize this data on a horizontal bar plot.

# In[ ]:


colors = ['#a6cee3', '#1f78b4', '#b2df8a',
          '#33a02c', '#fb9a99', '#e31a1c',
          '#fdbf6f', '#ff7f00', '#cab2d6',
          '#6a3d9a', '#ffff99', '#b15928']


# In[ ]:


ax = prod.sort_values().plot.barh(figsize=(10,7), fontsize=14,                  
                   title='Food production in South American in 2000-2013',
                   color=colors[::-1])

ax.set_xlabel('Food and feed produced (in 1000 tonnes)', fontsize=12)
ax.set_ylabel('Country', fontsize=12)
ax.title.set_size(20)

# write the value next to the bar
for x, y in zip(np.arange(len(prod)), prod.sort_values().values):
    plt.text(y+400000, x, str(y), fontsize=12,
            horizontalalignment='center')


plt.show()


# ## Countrywise production growth

# Now let's see how each country's food production evolved over the years. 

# In[ ]:


years = [str(year) for year in range(2000, 2014)]

annual = df.groupby('Area').sum()
annual.sort_values(by='Sum 2000-2013', ascending=False, inplace=True)
annual = annual[years]
annual


# Let's represent the production growth on a plot. Since Brazil's production is considerably higher than that of other countries, we'll draw a second plot, excluding Brazil.   

# In[ ]:


from cycler import cycler

# building two sets of colors to keep the coloring of countries consistent between the 2 plots.
cycle1 = cycler('color', colors)
cycle2 = cycler('color', colors[1:]) # 

fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(16,6))

ax1.set(xlim = (0, 13),
        ylabel='Food production in tonnes',
        prop_cycle= cycle1)
ax2.set(xlim = (0, 13),
        ylabel='Food production in tonnes',
        prop_cycle= cycle2)

ax1.set_title('Annual food production of South American countries in 2000-2013', fontsize= 14)
ax2.set_title('Annual food production of South American countries (w/o Brazil) in 2000-2013', fontsize= 14)

ax1.plot(annual.T)
ax2.plot(annual.iloc[1:].T)

ax1.legend(annual.index, ncol=2, loc=5)


plt.show()


# It may also be interesting to see the correlation between the growth rates of different countries.

# In[ ]:


fig, ax = plt.subplots(figsize=(10, 8))

ax = sns.heatmap(annual.T.corr())

ax.set_title('Correlation of food production growth', fontsize=18)

ax.set_xlabel('')
ax.set_ylabel('')

plt.show()


# There seems to be one country, Paraguay, that doesn't follow the general trend. While all the countries experienced a considerable or modest growth between 2004 and 2013, Paraguay experienced a production loss.

# ## Food vs. feed

# The next thing to explore is the proportion of feed in the overall food production countrywise.

# In[ ]:


feed = df.groupby(['Area','Element']).agg({'Sum 2000-2013': 'sum'}).unstack(level=1)['Sum 2000-2013']

feed = feed.div(feed.sum(axis=1), axis=0).multiply(100).round(0).astype(int) # convert values to percentage

feed.sort_values(by='Feed', inplace=True)

feed


# In[ ]:


ind = np.arange(len(feed))

fig = plt.figure(figsize=(10,6))

p1 = plt.bar(ind,feed['Feed'], color='r')
p2 = plt.bar(ind, feed['Food'], bottom=feed['Feed'], color='b')

plt.ylabel('Feed and food production (%)', fontsize=12)
plt.xticks(ind, feed.index, rotation=60, fontsize=11)
plt.title('Food vs feed', fontsize=18)
plt.legend(['Feed', 'Food'], fontsize=12,
          bbox_to_anchor=(0., 1.02, 1.2, -.1))

plt.show()


# Here again Paraguay stands out with almost 50 percent of its food production dedicated to feed.

# ## Food items produced by each country in 2013

# Looking at the data from 2013, let's find out how many types of food and feed are produced by each country.

# In[ ]:


items = df.groupby(['Area','Item']).sum().unstack(level=1)
items = items[['2013']]
items.fillna(0, inplace=True)
items = items.astype(int)

nitems = items[items.columns].gt(0).sum(axis=1)
nitems.sort_values(ascending=False, inplace=True)
nitems


# In[ ]:


ind = np.arange(len(nitems))

fig = plt.figure(figsize=(10,6))
plt.bar(ind, nitems, width=0.35, color='r')

plt.xticks(ind, nitems.index, rotation=60, fontsize=12)
plt.ylabel('# of food items', fontsize=12)
plt.ylim((40,100))
plt.title('Food items produced by South American countries', fontsize=18)

for x, y in zip(ind, nitems.values):
    plt.text(x, y+1, str(y), fontsize=12,
            horizontalalignment='center')


plt.show()


# ## Top 5 food items countrywise in 2013

# Let's find out which 5 items had the biggest production volume in 2013 in each state and draw a plot reflecting their weight in the overall country food production.

# In[ ]:


top5 = items.copy().T
top5.reset_index(level=0, drop=True, inplace=True)

# convert values to percentage
top5 = top5.div(top5.sum(axis=0), axis=1).multiply(100).round(2)

# convert the cells that are not in a country's top 5 to zero
for col in top5.columns:
    top = top5[col].nlargest(5)
    top5[col][~top5.index.isin(top.index)] = 0

# delete rows with only zeroes
top5 = top5.loc[(top5 != 0).any(axis=1), :]
top5.sort_values(by='Brazil', ascending=False, inplace=True)
top5 = top5.T

top5


# In[ ]:



top5.plot(kind='bar', stacked=True, 
               color=colors, figsize=(10,6))
plt.ylabel("Food produced (in %)", fontsize=12)
plt.xticks(fontsize=12)
plt.title('Top 5 food items produced by South American countries', fontsize=18)
plt.ylim(0,60)

plt.legend(bbox_to_anchor=(0., 0.72, 1.4, .1), fontsize=12)


plt.show()


# A total of 12 food items are found among the top 5 lists of the South American countries.
# 
# All countries have in common the presence of cereals and milk in their top 5 with the exception of Paraguay that doesn't include milk. Another special thing about Paraguay is that cassava, absent from the other countries' top 5, represents about 15% of its food production.

# In[ ]:





# In[ ]:





# In[ ]:




