#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.offline as py
import plotly.figure_factory as ff
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# ![](http://s1.1zoom.me/big0/129/Beer_Mug_514763_1280x839.jpg)
# 
# # Brewer's friend recipe exploration
# 
# This kernel has a play around with the recipe data from the [Brewer's Friend dataset](https://www.kaggle.com/jtrofe/beer-recipes/kernels) on Kaggle.
# 
# I look at the popularity of different beer types, how the style affects the alcohol levels and what brewing factors lead to what properties in the beer.

# ### Get the data
# 
# Quick summary of the abbreviated fields:
# 
# - **OG** (Original gravity): Used in brewing to control alcohol strength
# - **FG** (Final gravity): Also used in brewing to control alcohol strength
# - **ABV** (Alcohol by volume): Accepted measure of alcohol strength - can be seen as % that is alcohol
# - **IBU** (International bitterness units): Measures the bitterness of the beer, mainly influenced by how 'hoppy' the beer is

# In[2]:


df = pd.read_csv('../input/recipeData.csv', index_col='BeerID', encoding='latin1')
df.head()


# In[3]:


print('Number of recipes =\t\t{} \nNumber of beer styles =\t{}'.format(len(df), len(df['Style'].unique())))


# ... that's a lot of different kinds of beer. Let's see which ones are the most popular.

# ### Popularity of beer styles
# 
# Which styles have the most recipes?

# In[4]:


top_n_types = 15
recipe_popularity_as_perc = 100 * df['Style'].value_counts()[:top_n_types] / len(df)

pltly_data = [go.Bar(x=recipe_popularity_as_perc.index,
                     y=recipe_popularity_as_perc.values)]

layout = go.Layout(title='Most popular beer styles',
                   xaxis={'title': 'Style'},
                   yaxis={'title': 'Proportion of recipes (%)'},
                   margin=go.Margin(l=50, r=50, b=150, t=50, pad=4))

fig = go.Figure(data=pltly_data, layout=layout)
py.iplot(fig)


# American IPA pips American Pale Ales, although together they roughly a quarter of all recipes. Highlights the swing from lager to pale ales in the craft beer world.
# 
# How about broadening the categories to look at a higher level:

# In[ ]:





# In[5]:


broad_styles = ['Ale', 'IPA', 'Pale Ale', 'Lager', 'Stout', 'Bitter', 'Cider', 'Porter']
df['BroadStyle'] = 'Other'
df['Style'].fillna('Unknown', inplace=True)
for broad_style in broad_styles:
    df.loc[df['Style'].str.contains(broad_style), 'BroadStyle'] = broad_style


# In[6]:


style_popularity_as_perc = 100 * df['BroadStyle'].value_counts() / len(df)
style_popularity_as_perc.drop('Other', inplace=True)

pltly_data = [go.Bar(x=style_popularity_as_perc.index,
                     y=style_popularity_as_perc.values)]

layout = go.Layout(title='Most popular general styles',
                   xaxis={'title': 'Style'},
                   yaxis={'title': 'Proportion of recipes (%)'})

fig = go.Figure(data=pltly_data, layout=layout)
py.iplot(fig)


# Dark times for lager, fallen behind even Stouts. Business is booming however for IPAs.
# 
# This suggests a preference for stronger beers. Lets have a look at the [ABV](https://en.wikipedia.org/wiki/Alcohol_by_volume) values.
# 
# ### ABV
# 
# Firstly, whats the general picture like for ABV distribution:

# In[19]:


fig, ax = plt.subplots(1, 1, figsize=[12,5])
sns.distplot(df['ABV'], ax=ax)
ax.set_title('ABV distribution')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.get_yaxis().set_visible(False)
plt.show()


# Nice and normal, but with a suprising tail. Must mean there's some fiesty beers in there.
# 
# Also interesting is the peak around 6-7%, indicating craft beers are typically stronger than the mass stuff that usually sits around 3-5%.

# In[22]:


strengths = [10, 15, 20, 30, 50]
for abv in strengths:
    print('{} ({:.2f})%\tbeers stronger than {} ABV'.format(sum(df['ABV'] > abv), 100 * sum(df['ABV'] > abv) / len(df), abv))


# There *are* some fiesty beers in there. 7 of them unreasonably so!
# 
# I'm going to keep these out for the rest of the ABV analysis as they are clearly outliers.

# In[9]:


abv_df = df[df['ABV'] <= 15]


# Let's see how the ABV changes with the general beer type. I would expect IPAs to be strong (~7%) and lagers to be weak (~4%)

# In[24]:


fig, ax = plt.subplots(1, 1, figsize=[12, 5])
sns.violinplot(x='BroadStyle',
               y='ABV',
               data=abv_df,
               ax=ax)
ax.set_xlabel('General beer style')
ax.set_title('ABV by beer style')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.show()


# As usual, a lot going on in a violin plot. Some interesting points though:
# 
# - Ales and IPAs have very similar distributions, with IPAs shifted up in strength
# - Pale ales have the narrowest distribution so the majority must sit at 5%
# - Stouts are *strong*, bitters are the weakest
# - Cider is a lottery

# So ABV varies - but was is it about the brewing process that has the strongest effect on its final value?
# 
# Let's see by finding the correlation of our other features with ABV

# In[11]:


df['gravity_change'] = df['OG'] - df['FG']  # Suspect this may be correlated with ABV


# I'll need to encode the categorical features to get *some* degree of how they correlate - although not being linear relationships will weaken their influence as measured by correlations.
# 
# I also don't want a monster data set I'll drop them if they have a lot of unique values. I will drop 'Style' anyway as it isn't really part of the brewing method.

# In[12]:


df_for_corr = df.drop(['Style', 'BroadStyle', 'StyleID'], axis=1).copy()
categoricals = df_for_corr.columns[df_for_corr.dtypes == 'object']
for categorical in categoricals:
    print('{} has {} unique values'.format(categorical, len(df_for_corr[categorical].unique())))
    if len(df_for_corr[categorical].unique()) > 20:
           df_for_corr.drop(categorical, axis=1, inplace=True)


# In[13]:


encoded_df = pd.get_dummies(df_for_corr)
corr_mat = encoded_df.corr()
abv_corrs = corr_mat['ABV'].sort_values()
abv_corrs.drop(['ABV', 'Color', 'IBU'], inplace=True)  # Color and IBU are results rather than parts of the brewing process so drop.


# In[14]:


pltly_data = [go.Bar(y=abv_corrs.index,
                     x=abv_corrs.values,
                     orientation='h')]

layout = go.Layout(title='Linear correlations with ABV',
                   xaxis={'title': 'Correlation'},
                   margin=go.Margin(l=200, r=50, b=100, t=100, pad=4)
                  )

fig = go.Figure(data=pltly_data, layout=layout)
py.iplot(fig)


# So no particularly strong correlations, but it seems as though pitch rate, boil time and and temperature all increase ABV, whilst the mash thickness decreases it.
# 
# What about IBU and color - what brewing factors influence them? In fact just do a correlation plot to catch it all

# In[15]:


fig, ax = plt.subplots(1, 1, figsize=[10, 7])
sns.heatmap(corr_mat.mask(np.triu(np.ones(corr_mat.shape)).astype(np.bool)), ax=ax, center=0)
plt.show()


# So not a huge amount of correlation - but IBU, ABV and color are correlated to some degree. In other words, stronger beers are hoppier and darker.

# In[ ]:




