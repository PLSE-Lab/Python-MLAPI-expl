#!/usr/bin/env python
# coding: utf-8

# # "Coffee" and "Cafe" Search Engine Rankings All Over the World
# ### A quick analysis of the SERP rankings of "coffee" and "cafe" in all available countries on Google. 
# 
# This is an update on the recipe for visualizing a large number of SERPs as a quick SEO summary. 
# 
# There are two minor adjustments: 
# 
# - **Sorting:** results are now sorted by the number of appearances on SERPs, then by average position. This is not ideal, and will be discussed below.
# - **New summary numbers:** three numbers are added to make it easier to evaluate results:  
# 
# 
# 1. Total appearances: the number of times the domain appeared in SERPs (top 10)
# 2. Coverage: total appearances divided by the total queries (shown as a percentage)
# 3. Average position 
# 
# Here is how it looks like:
# ![](https://drive.google.com/uc?id=1LfD7M99G3IWoI7Ydqa0tfUTE9Vg476qY)
# 
# ## Some issues
# I'd like to start by mentioning some of the possible issues in the dataset: 
# 
# * Keywords: "coffee" doesn't say much about the intention of the user, and it is very generic. So is "cafe", although the intention is likely easier to figure out than "coffee".
# * Language: coffee is an English word, and cafe is used in many other languages. They requests were run in all available countries, many of which don't speak English, or the languages that have "cafe".  
# * Country weights: probably the most important issue. The visualization and counting assumes all countries are equal in value. They are not, whether in terms of population, GDP, coffe consumption etc. Ideally, you would have your own weights for countries, or whatever filtering parameters you are using and you can apply them for a better evaluation of the SERP ranks.
# 
# With the above issues in mind, let's take a look at how to produce the summary visualization shown above. 
# 
# ## Basic Setup

# In[ ]:


import advertools as adv # package used to import data from Google's API
import pandas as pd # pandas is pandas! 
pd.options.display.max_columns = None
import plotly.graph_objs as go
from plotly.offline import iplot, init_notebook_mode
init_notebook_mode()

cx = 'YOUR_GOOGLE_CSE_SEARCH_ENGINE_ID' # get it here https://cse.google.com/cse/
key = 'YOUR_GOOGLE_DEVELOPER_KEY' # get it here https://console.cloud.google.com/apis/library/customsearch.googleapis.com


# You will also have to [get credentials for the project](https://console.developers.google.com/apis/api/customsearch.googleapis.com/credentials) and 
# then [enable billing](https://console.cloud.google.com/billing/projects).
# 
# 

# The following code was used to generate the data. It is commented out just to show how the data was gathered, later we read them with `pandas.read_csv`. 
# 
# The `gl` parameter of the `adv.serp_goog` function stands for "geo-location". 
# The available `gl`'s are available as a dictionary as well, so you don't have to worry about which countries are available.

# In[ ]:


# coffee_df = adv.serp_goog(cx=cx, key=key, q='coffee',
#                           gl=adv.SERP_GOOG_VALID_VALS['gl'])

# cafe_df = adv.serp_goog(cx=cx, key=key, q='cafe',
#                         gl=adv.SERP_GOOG_VALID_VALS['gl'])


# In[ ]:


coffee_df = pd.read_csv('../input/coffee_serps.csv')
cafe_df = pd.read_csv('../input/cafe_serps.csv')

country_codes = sorted(adv.SERP_GOOG_VALID_VALS['gl'])
print('number of available locations:', len(country_codes))
print()
print('country codes:')
print(*[country_codes[x:x+15] for x in range(0, len(country_codes), 15)],
      sep='\n')


# ## Quick Overview of the Domains

# In[ ]:


print('Coffee domains:', coffee_df['displayLink'].nunique())
print('Cafe domains:', cafe_df['displayLink'].nunique())
common = set(coffee_df['displayLink']).intersection(cafe_df['displayLink'])
print('# Common domains:', len(common))
common


# As you can see, the number of domains ranking for "cafe" is almost 2.5 times that of "coffee". I think it makes sense, because the fomer is more of a local keyword, and the geo-location makes a difference. So Google is probably giving more local domains for each country. On the other hand "coffee" is a generic term about the plant/drink so the location doesn't play an important role. 
# 
# It's also interesting to see that only fifteen domain are common. If you remove the local versions of some of those domains, you will notice that they are even less than that. 
# 
# We now define some parameters to use in our visualization (later to be used in a function to create the vizualization).

# In[ ]:


num_domains = 15  # the number of domains to show in the chart
opacity = 0.02  # how opaque you want the circles to be
df = coffee_df  # which DataFrame you are using


# Getting the top domains is done by simply getting the ones that appeared the most in the dataset. As mentioned above this is not ideal, but once you see the top ten or fifteen side by side, you will get a good idea of who is performing the best. 
# 
# Feel free to create your own criteria for how you want to define the top domains. 

# In[ ]:


top_domains = df['displayLink'].value_counts()[:num_domains].index.tolist()
top_domains


# In[ ]:


top_df = df[df['displayLink'].isin(top_domains)]
top_df.head(3)


# `top_df` is basically the same DataFrame filtered by having only results that are in `top_domains`.  
# This is a summary of the number of appearances `rank_count` and the average position `rank_mean` for the domains that are in our `top_df`:

# In[ ]:


top_df_counts_means = (top_df
                       .groupby('displayLink', as_index=False)
                       .agg({'rank': ['count', 'mean']})
                       .set_axis(['displayLink', 'rank_count', 'rank_mean'],
                                 axis=1, inplace=False))
top_df_counts_means


# Now we merge the `top_df` with `top_df_counts_means`. Note that since the only common column name is `displayLink` I didn't specify the columns on which to merge. 

# In[ ]:


top_df = (pd.merge(top_df, top_df_counts_means)
          .sort_values(['rank_count', 'rank_mean'],
                       ascending=[False, True]))
top_df.iloc[:3, list(range(8))+ [-2, -1]]


# If you want to have a summary in a table format, you can do it as follows: (those numbers will eventually be added to the chart)

# In[ ]:


num_queries = df['queryTime'].nunique()

summary = (df
           .groupby(['displayLink'], as_index=False)
           .agg({'rank': ['count', 'mean']})
           .sort_values(('rank', 'count'), ascending=False)
           .assign(coverage=lambda df: (df[('rank', 'count')].div(num_queries))))
summary.columns = ['displayLink', 'count', 'avg_rank', 'coverage']
summary['displayLink'] = summary['displayLink'].str.replace('www.', '')
summary['avg_rank'] = summary['avg_rank'].round(1)
summary['coverage'] = (summary['coverage'].mul(100)
                       .round(1).astype(str).add('%'))
summary.head(10)


# If you want to quickly visually see how often each domain ranks per position on SERPs you can do so with the following code. 
# 
# What the code does is plot a circle for every time a domain appears on a certain rank/position. The more often the domain appears there, the more opaque the circle is, which helps to visually immediately spot how those ranks are distributed.
# 
# To make it more intuitive to view, I add the line `fig.layout.yaxis.autorange = 'reversed'`. It simply reverses the ranking so they appear in a similar way to actual SERPs, with rank one at the top, followed by two, all the way down to ten.

# In[ ]:


print('number of queries:', num_queries)
fig = go.Figure()
fig.add_scatter(x=top_df['displayLink'].str.replace('www.', ''),
                y=top_df['rank'], mode='markers',
                marker={'size': 35, 'opacity': opacity},
                showlegend=False)
fig.layout.height = 600
fig.layout.yaxis.autorange = 'reversed'
fig.layout.yaxis.zeroline = False
iplot(fig)


# The opacity of the circle makes it very easy to see the top spots. Yet, it is difficult to to know really how different they are, since there are many levels of opacity. For this we can add the actual numbers to the chart, and this way you can visually spot the top locations, and get the actual number of appearance per position. 
# We first create the DataFrame `rank_counts`, which shows how many times each domain ranked per position. 

# In[ ]:


rank_counts = (top_df
               .groupby(['displayLink', 'rank'])
               .agg({'rank': ['count']})
               .reset_index()
               .set_axis(['displayLink', 'rank', 'count'],
                         axis=1, inplace=False))
rank_counts[:15]


# Now we can easily add another layer to the chart, with the numbers mentioned: 

# In[ ]:


fig.add_scatter(x=rank_counts['displayLink'].str.replace('www.', ''),
                y=rank_counts['rank'], mode='text',
                marker={'color': '#000000'},
                text=rank_counts['count'], showlegend=False)
iplot(fig)


# In the chart above does having ncausa.org before wikipedia.org make sense?  
# It's clear that Wikipedia's average rank is higher, but what about the number of appearnces and coverage?  
# We add those data points for each domain with the following loops, so we can have a final number summarizing each of those metrics, and place them right below the domain on the chart. 

# In[ ]:


for domain in rank_counts['displayLink'].unique():
    rank_counts_subset = rank_counts[rank_counts['displayLink']==domain]
    fig.add_scatter(x=[domain.replace('www.', '')],
                    y=[11], mode='text',
                    marker={'size': 50},
                    text=str(rank_counts_subset['count'].sum()))
    fig.add_scatter(x=[domain.replace('www.', '')],
                    y=[12], mode='text',
                    text=format(rank_counts_subset['count'].sum() / num_queries, '.1%'))
    fig.add_scatter(x=[domain.replace('www.', '')],
                    y=[13], mode='text',
                    marker={'size': 50},
                    text=str(round(rank_counts_subset['rank']
                                   .mul(rank_counts_subset['count'])
                                   .sum() / rank_counts_subset['count']
                                   .sum(),2)))
fig.layout.title = ('Google Search Results Rankings<br>keyword(s): ' + 
                    ', '.join(list(df['searchTerms'].unique())) + 
                    ' | queries: ' + str(df['queryTime'].nunique()))
fig.layout.hovermode = False
fig.layout.yaxis.autorange = 'reversed'
fig.layout.yaxis.zeroline = False
fig.layout.yaxis.tickvals = list(range(1, 14))
fig.layout.yaxis.ticktext = list(range(1, 11)) + ['Total<br>appearances','Coverage', 'Avg. Pos.'] 
fig.layout.height = 700
fig.layout.width = 1200
fig.layout.yaxis.title = 'SERP Rank (number of appearances)'
fig.layout.showlegend = False
fig.layout.paper_bgcolor = '#eeeeee'
fig.layout.plot_bgcolor = '#eeeeee'
iplot(fig)


# Now we have a fuller picture about the domains.  
# You can immediately see which domains are performing the best. You can also see how many times they ranked for each position. Then you can see aggregates (Total appearances, Coverage, and Avg. Pos.)  
# 
# You will also notice that I added the number of queries dynamically to the chart's title, as well as the list of queries that were used  in the SERP DataFrame (in this case we only have one keyword).  
# 
# Now let's make it a function by putting all the code in sequence, and providing a few options to customize the chart:

# In[ ]:


def plot_serps(df, opacity=0.1, num_domains=10, width=1200, height=700):
    """
    df: a DataFrame resulting from running advertools.serp_goog
    opacity: the opacity of the markers [0, 1]
    num_domains: how many domains to plot
    """
    top_domains = df['displayLink'].value_counts()[:num_domains].index.tolist()
    top_df = df[df['displayLink'].isin(top_domains)]
    top_df_counts_means = (top_df
                       .groupby('displayLink', as_index=False)
                       .agg({'rank': ['count', 'mean']})
                       .set_axis(['displayLink', 'rank_count', 'rank_mean'],
                                 axis=1, inplace=False))
    top_df = (pd.merge(top_df, top_df_counts_means)
          .sort_values(['rank_count', 'rank_mean'],
                       ascending=[False, True]))
    rank_counts = (top_df
               .groupby(['displayLink', 'rank'])
               .agg({'rank': ['count']})
               .reset_index()
               .set_axis(['displayLink', 'rank', 'count'],
                         axis=1, inplace=False))
    num_queries = df['queryTime'].nunique()
    fig = go.Figure()
    fig.add_scatter(x=top_df['displayLink'].str.replace('www.', ''),
                    y=top_df['rank'], mode='markers',
                    marker={'size': 35, 'opacity': opacity},
                    showlegend=False)
    fig.layout.height = 600
    fig.layout.yaxis.autorange = 'reversed'
    fig.layout.yaxis.zeroline = False
    fig.add_scatter(x=rank_counts['displayLink'].str.replace('www.', ''),
                y=rank_counts['rank'], mode='text',
                marker={'color': '#000000'},
                text=rank_counts['count'], showlegend=False)
    for domain in rank_counts['displayLink'].unique():
        rank_counts_subset = rank_counts[rank_counts['displayLink']==domain]
        fig.add_scatter(x=[domain.replace('www.', '')],
                        y=[11], mode='text',
                        marker={'size': 50},
                        text=str(rank_counts_subset['count'].sum()))
        fig.add_scatter(x=[domain.replace('www.', '')],
                        y=[12], mode='text',
                        text=format(rank_counts_subset['count'].sum() / num_queries, '.1%'))
        fig.add_scatter(x=[domain.replace('www.', '')],
                        y=[13], mode='text',
                        marker={'size': 50},
                        text=str(round(rank_counts_subset['rank']
                                       .mul(rank_counts_subset['count'])
                                       .sum() / rank_counts_subset['count']
                                       .sum(),2)))
    fig.layout.title = ('Google Search Results Rankings<br>keyword(s): ' + 
                        ', '.join(list(df['searchTerms'].unique())) + 
                        ' | queries: ' + str(df['queryTime'].nunique()))
    fig.layout.hovermode = False
    fig.layout.yaxis.autorange = 'reversed'
    fig.layout.yaxis.zeroline = False
    fig.layout.yaxis.tickvals = list(range(1, 14))
    fig.layout.yaxis.ticktext = list(range(1, 11)) + ['Total<br>appearances','Coverage', 'Avg. Pos.'] 
    fig.layout.height = height
    fig.layout.width = width
    fig.layout.yaxis.title = 'SERP Rank (number of appearances)'
    fig.layout.showlegend = False
    fig.layout.paper_bgcolor = '#eeeeee'
    fig.layout.plot_bgcolor = '#eeeeee'
    iplot(fig)


# We can now try to reproduce the same vizualization using `coffee_df`:

# In[ ]:


plot_serps(coffee_df, opacity=0.07, num_domains=15)


# Now it's easy to get a summary for `cafe_df` as the function is ready. You can change `num_domains` if you want to have more or fewer domains on the chart. The `opacity` parameter becomes more important the more queries you run. The more domains, the lower you want to keep it, so you can better see the denser positions. 

# In[ ]:


plot_serps(cafe_df, opacity=0.07, num_domains=8)

