#!/usr/bin/env python
# coding: utf-8

# # Search engine results pages (SERPs) research with Python
# ![](https://storage.googleapis.com/kaggle-datasets-images/65508/129168/1e05588bb7a2735c017bff299c2eeb5f/data-original.png?t=2018-10-19-03-52-58)
# 
# ## How to import and merge multiple search engine results pages in one DataFrame (table / csv / excel sheet), with one function call

# When starting to do research for search engine optimization, you typically start by making a few manual searches, to see who is ranking the highest, how competitive the landscape might be, and so on.    
# Of course this is not enough, what you actually need is a data set summarizing each search results page in a tidy DataFrame, with keywords, ranks, titles, etc. each in its own column. Ideally you would have such a table for every keyword, for every product you have, and across countries, languages, or whatever search attributes you may have.  
# Something like the following table: 

# In[ ]:


get_ipython().system('pip install "advertools==0.7.4"')


# In[ ]:


import pandas as pd
(pd.read_csv('../input/pymarketing.csv').head())


# More importantly, you can generate such a dataset for multiple queries, countries, languages, etc. and on a large scale using one line of code. Typically, running a query manually and looking at the results is not that useful, because you only see results for that query and for your location. What can help in your analysis, is generating a large dataset for a query across fifty countries, or for a hundred queries in two countries, and in three languages. This is where you have results for hundreds of requests, and you can start seeing patterns and trends. Many other options exist, and these are just a few examples.
# 
# If this sounsd interesting, here is how you can do this.  

# ## Why Search? (skip if you do SEO)
# 
# 1. **You typically search alone:** Whether on your phone or laptop,  you are mostly by yourself. This means two things. 
#     - Alone means only for you: If two or more people were to decide on a movie to watch, or a restaurant to go to, they would have to find an option that caters for most people, something like the average. You end up compromising a little. Not with search, you say exactly what you want.
#     - Alone means private: If a group of friends are talking about something new to you, you might be embarassed to ask what that thing is. Not with Google, you simply search for "who is X?" or "what is Y?". No embarrasment. Sometimes you might be concerned about something, maybe a medical or psychological condition you might be having. Before alarming your loved ones, you safely go and ask The Google. 
# 2. **Search is active:** Compared to most other media, search is one of the few that only happen when you make a request. Listening to the radio, for example, you get messages that you didn't ask for during a show that you asked for.
# 3. **Perfect timing:** Because search is an active task, it happens at the perfect moment. Because you search when you are interested, ready, and receptive.  
# 4. **Expression of interest (not a selection from options):** When you go to a restaurant and are given a menu, you have to choose an option. There might not be exactly what you are craving at that moment, or what your body needs. Again, you will have to compromise a little and order something quite close. With search you say exactly what you want, no matter how ridiculously detailed that thing is that you are searching for. 
# 
# So, on search we express our exact desires, actively, in private, honestly, and when we are most receptive. So, I think it's important to know what content people are given at those moments.
# 
# 
# ### Approach
# 
# We will connect to Google's custom search engine API: This is mainly used to create a search engine on your website, and allows you to customize it. We will use it to get results for research purposes:
# 1. [Create a custom search engine](https://cse.google.com/cse/).  At first you might be asked to enter a site to search. Enter any domain, then go to the control panel and remove it. Make sure you enable "Search the entire web" and image search. You will also need to get your search engine ID, which you can find on the control panel page.  
# 2. [Enable the custom search API](https://console.cloud.google.com/apis/library/customsearch.googleapis.com). You wil need to create a project for this first.
# 3. [create credentials](https://console.developers.google.com/apis/api/customsearch.googleapis.com/credentials) for this project so you can get your key.
# 4. [enable billing for your project](https://console.cloud.google.com/billing/projects) If you want to run more than 100 queries per day. The first 100 queries are free, then for each 1,000 queries you pay $5.
# 
# Once this is set up (once you have your custom search engine ID, and your key), we will be ready to construct our queries and get the data. 
# 
# We will be using the `serp_goog` function from the [advertools](https://github.com/eliasdabbas/advertools) package. To install it:  
# `pip install advertools`
# 
# Start first be defining two variable as we will be using them in every request. I'm using the conventions used by Google's CSE:

# In[ ]:


cx = 'YOUR_CUSTOM_SEARCH_KEY'
key = 'YOUR_GOOGLE_DEV_KEY'

import advertools as adv
import pandas as pd
pd.options.display.max_columns = None
import plotly
import plotly.graph_objs as go

for p in [adv, pd, plotly]:
    print(p.__name__, p.__version__)


# The simplest request requires three parameters to be given; the query `q`, `cx`, and `key`, all others are optional. Here is a simple request:  
# 
# `yoga_mats = adv.serp_goog(q='yoga mats', cx=cx, key=key)`

# In[ ]:


yoga_mats = pd.read_csv('../input/yoga_mats.csv', parse_dates=['queryTime'])
print(yoga_mats.shape)
yoga_mats.head(3)


# Ten rows, as expected, with 148 columns. The columns are pretty much the same (up to the `pagemap` column, unless you supply extra parameters, in which case you will have additional columns each for the respective parameter.  
# 
# After that, the remaining columns (over 100), give some interesting meta data that you would see on the SERP, but are not the same for all domains, and not all domains would have the data, so you will see a lot NaNs in those columns. 
# 
# After taking a quick look, you decide to explore further and realize that there are  three other types of yoga mats that you are interested in; 'manduka yoga mats', 'thick yoga mats', and 'cheap yoga mats'. This is how you call the function for multiple keywords:  
# 
# `yoga_mats_multi = adv.serp_goog(q=['manduka yoga mats', 'thick yoga mats', 'cheap yoga mats'], cx=cx, key=key)`

# In[ ]:


yoga_mats_multi = pd.read_csv('../input/yoga_mats_multi.csv', parse_dates=['queryTime'])
print(yoga_mats_multi.shape)
yoga_mats_multi.groupby('searchTerms').head(2)


# As you can see, if you want to make multiple requests, you don't need to worry about looping and merging the response data. All you need to do is pass a list instead of a string, and now we have thirty rows for the new keywords that we requested.  
# Let's say you want to do the same research in another country that you are intested in. For example you operate in two English-speaking countries, Canada and Australia.  
# As with the last example, you simply need to pass a list of countries as you did with the keywords, and all combinations will be requested and handled: 
# 
# `yoga_mats = adv.serp_goog(q=['manduka yoga mats', 'thick yoga mats', 'cheap yoga mats'],   cx=cx, key=key,   gl=['ca', 'au'])`
#                            
# This will create 3 (keywords) x 2 (countries) = 6 (requests, and 60 rows therefore)
# 
# Keyword | Country
# --------|----------
# manduka yoga mats| Canada 
# thick yoga mats| Canada 
# cheap yoga mats| Canada
# manduka yoga mats| Australia
# thick yoga mats| Australia
# cheap yoga mats| Australia
# 
# Again, this is done with one request, and the resulting DataFrames get merged into one. 
# The same applies to any of the parameters of the function. Let's take a quick look at what's available. You can find the documentation for each of the parameters in the docstring of the function, as imported from [Google's documentation.](https://developers.google.com/custom-search/v1/cse/list) 

# In[ ]:


import inspect
print(*inspect.signature(adv.serp_goog).parameters.keys(), sep='\n')


# Parameter names are notPythonic and that is mainly to be consistent with Google's API, and to minimize any possible errors and bugs in the requests. Some of them are quite cryptic; `gl` for geo-location, `cr` for country-restrict for example. Again, you will find the full description in the doc string of the function if you need more details. 
# There are acceptable values for many of those parameters, and you can get those by checking the dictionary that has those values:

# In[ ]:


adv.SERP_GOOG_VALID_VALS.keys()


# In[ ]:


adv.SERP_GOOG_VALID_VALS['imgSize']


# In[ ]:


adv.SERP_GOOG_VALID_VALS['rights']


# For image search, you need to pass `searchType="image"`, or in case you want multiple results, both for web and image search, you can do it like this `searchType=["image", None]`. Together with the `searchType` parameter you can also make more specific image search queries by using any of the parameters that start with `img`.  
# 
# Now let's run a larger query, and add some options. I gathered the names of the some of the most popular cars, and created a template: 
# - **make model for sale** ("honda civic for sale", "toyota camry for sale",  etc.). 
# - **make model price** ( "honda civic price", "toyota camry price", etc.).  
# I also ran the query with `gl=['us', 'uk']`

# In[ ]:


make_model = ['Chevrolet Malibu','Hyundai Sonata','Ford Escape',
              'Hyundai Elantra','Kia Sportage','Nissan Sentra',
              'Hyundai Santa Fe Sport','Ford Fusion','Nissan Altima',
              'Nissan Rogue','GMC Terrain','Kia Sorento','Toyota Camry',
              'Volkswagen Passat','Kia Forte','Chevrolet Traverse',
              'Ford Mustang','Dodge Dart','Ford Focus','Chrysler 200',
              'Ford Explorer','Toyota Corolla','Mitsubishi Lancer',
              'Nissan Versa','Kia Sedona','Toyota Prius','Nissan Versa Note',
              'Buick Enclave','Jeep Patriot','Toyota RAV4','Chevrolet Tahoe',
              'Nissan Pathfinder','Toyota Yaris','Jeep Grand Cherokee',
              'Dodge Charger','Ford Edge','Jeep Compass','Nissan Frontier',
              'Hyundai Santa Fe','Chevrolet Malibu Limited','Nissan JUKE',
              'Volkswagen Beetle Coupe','Jeep Cherokee','Ford Fiesta',
              'INFINITI QX60','Ram 1500','INFINITI QX70','Hyundai Accent',
              'Buick Regal','Dodge Durango'
]


# In[ ]:


q_for_sale = [x + ' for sale' for x in make_model]
q_price = [x + ' price' for x in make_model]
queries = [q.lower() for q in q_for_sale + q_price]
print('Number of queries: 50 (make-model combinations) x 2 (keyword variations) x 2 (countries) = 200 queries\nSample:')
queries[:5] + queries[-5:]


# In[ ]:


# serp_cars = adv.serp_goog(q=queries, cx=cx, key=key, gl=['us', 'uk'])


# In[ ]:


cars4sale  = pd.read_csv('../input/cars_forsale_price_us_uk.csv', parse_dates=['queryTime'])


# Here are the top two results for five of the keywords:

# In[ ]:


print(cars4sale.shape)
cars4sale.groupby(['searchTerms']).head(2)[:10]


# Since this is a much larger dataset, we got 358 columns, so there must be some interesting information there.  
# 
# First, let's see who are the top domains in the US. One way to look at it, is to simply count for how many keywords a domain appears in the top ten results. Of course this assumes we are treating all keywords equally, which is not a good assumption in many cases. In this case it isn't a big problem because we are comparing top cars with other top cars. If you have more data about the importance of each keyword, you can apply a modifier to get a more meaningful set of numbers. 

# In[ ]:


(cars4sale[cars4sale['gl'] == 'us']
 .pivot_table('rank', 'displayLink', aggfunc=['mean', 'count'])
 .sort_values([('count', 'rank')], ascending=False)
 .assign(cumsum=lambda df: df[('count', 'rank')].cumsum(),
         cum_perc=lambda df: df['cumsum'].div(df[('count', 'rank')].sum()))
 .head(10)
 .style.format({('cum_perc',''): '{:.2%}', ('mean', 'rank'): '{:.1f}'})
 .set_caption('Top domains in USA'))


# The top ten domains seem to occupy 741 out of 1,000 results. Good luck trying to do SEO in the used car market in the US!  
# We can see the mean rank (avg. position) for each domain, so let's look at how those numbers are distributed.  
# We first filter the `cars4sale` DataFrame where `gl=="us"`, and where the `displayLink` is one of the top 10 domains.

# In[ ]:


top10domains_us = (cars4sale[cars4sale['gl'] == 'us']
                   ['displayLink'].value_counts().index[:10])
top10_df = (cars4sale[(cars4sale['gl'] == 'us') & 
                      (cars4sale['displayLink'].isin(top10domains_us))])
print(top10_df.shape)
top10_df.head(2)


# As expected, we get 741 rows.  
# Now to make things clearer on the plot when we visualize the data, we get the count of appearances per domain and for each position:

# In[ ]:


rank_counts = (top10_df
               .groupby(['displayLink', 'rank'])
               .agg({'rank': ['count']})
               .reset_index())
rank_counts.columns = ['displayLink', 'rank', 'count']
rank_counts.head()


# In this case, cars.usnews.com appeared as the first result four times, as the second result twenty two times, and so on. 

# In[ ]:


from plotly.offline import iplot, init_notebook_mode
init_notebook_mode()
fig = go.Figure()
fig.add_scatter(x=top10_df['displayLink'].str.replace('www.', ''),
                y=top10_df['rank'], mode='markers',
                marker={'size': 30, 
                        'opacity': 1/rank_counts['count'].max()})
fig.add_scatter(x=rank_counts['displayLink'].str.replace('www.', ''),
                y=rank_counts['rank'], mode='text',
                text=rank_counts['count'])

for domain in rank_counts['displayLink'].unique():
    rank_counts_subset = rank_counts[rank_counts['displayLink'] == domain]
    fig.add_scatter(x=[domain.replace('www.', '')],
                    y=[0], mode='text',
                    marker={'size': 50},
                    text=str(rank_counts_subset['count'].sum()))

    fig.add_scatter(x=[domain.replace('www.', '')],
                    y=[-1], mode='text',
                    text=format(rank_counts_subset['count'].sum() / top10_df['queryTime'].nunique(), '.1%'))
    fig.add_scatter(x=[domain.replace('www.', '')],
                    y=[-2], mode='text',
                    marker={'size': 50},
                    text=str(round(rank_counts_subset['rank']
                                   .mul(rank_counts_subset['count'])
                                   .sum() / rank_counts_subset['count']
                                   .sum(), 2)))

minrank, maxrank = min(top10_df['rank'].unique()), max(top10_df['rank'].unique())
fig.layout.yaxis.tickvals = [-2, -1, 0] + list(range(minrank, maxrank+1))
fig.layout.yaxis.ticktext = ['Avg. Pos.', 'Coverage', 'Total<br>appearances'] + list(range(minrank, maxrank+1))

fig.layout.height = 600 #max([600, 100 + ((maxrank - minrank) * 50)])
fig.layout.width = 1000
fig.layout.yaxis.title = 'SERP Rank (number of appearances)'
fig.layout.showlegend = False
fig.layout.paper_bgcolor = '#eeeeee'
fig.layout.plot_bgcolor = '#eeeeee'
fig.layout.autosize = False
fig.layout.margin.r = 2
fig.layout.margin.l = 120
fig.layout.margin.pad = 0
fig.layout.hovermode = False
fig.layout.yaxis.autorange = 'reversed'
fig.layout.yaxis.zeroline = False
fig.layout.template = 'none'
fig.layout.title = 'Top domains ranking for used car keywords in the US'
iplot(fig)


# I think it's clear how the top domains are distributed on SERPs. 
# carfax.com appeared on the first position thirty one times, on the second position nineteen times and so on. 
# Although we had the mean rank for each domain, this gives a clearer idea on how they are spread across the ranks.  We also see the average position, coverage, and the number of total appearance. Note that the coverage here is out of one thousand rows, because this is for the US only.  
# Let's see if the top domains have different content for the different keywords. We can take a random make-model combination and three of the top domains: 

# In[ ]:


(cars4sale
 [(cars4sale['displayLink'].isin(['www.cargurus.com', 'www.truecar.com', 'www.edmunds.com'])) & 
  (cars4sale['title'].str.contains('Ford Escape'))][['title','link']])


# In[ ]:


for position in [41, 45, 46, 50, 55, 1044, 1048, 1049, 1054]:
    print(cars4sale['searchTerms'][position])
    print('='*23)
    print(cars4sale['title'][position])
    print(cars4sale['link'][position])
    print('-' * 23, '\n')


# It seems ['make model price'](https://www.cargurus.com/Cars/2019-Ford-Escape-Price-c28499) has a completely different page from ['make model for sale'](https://www.cargurus.com/Cars/l-Used-Ford-Escape-d330) for cargurus.com.  
# Although the intention is very similar in the two keywords, some websites made the extra effort to create special content for each, and it's interesting actually. Some of those price pages offer some interesting charts on different models, similar cars, and so on.  
# Based on this it would be interesting to also explore other ideas and see how far / detailed those websites are going to cater for different types of content related to used cars. Maybe something related to mainenance, or spare parts? 
# 
# Now let's take a look at the additional meta data columns that we have. You will see some frequently appearing meta data like `og:` (Facebook's open graph) and `twitter:`

# In[ ]:


print(cars4sale.filter(regex='og:').shape)
cars4sale.filter(regex='og:').head()


# In[ ]:


print(cars4sale.filter(regex='twitter:').shape)
cars4sale.filter(regex='twitter:').dropna(how='all').sample(3)


# In[ ]:


print(cars4sale.filter(regex='al:').shape)
cars4sale.filter(regex='al:').dropna(how='all')


# In[ ]:


cars4sale.filter(regex='rating').dropna(how='all').sample(4)


# Feel free to check other patterns or insights by inspecting `cars4sale.columns`
# 
# Let's explore a few other options with the `serp_goog` function parameters. 
# 
# ## Number of Results
# 
# By default, the maximum number of results is ten. But sometimes you want to go futher than the first page, to explore futher. The `start` parameter determines which position to start the results from. So if you specfiy `start=7` for example, you will get the ten results starting from position seven.  
# You can further restrict the number of results you get with the `num` parameters. This determines the number of results to return, and can only be between one and ten, inclusive.  
# Because we can specify multiples parameters, we can use a simple trick of passing multiple `start` numbers.
# 
# `basketball = adv.serp_goog(q="basketball", cx=cx, key=key, start=[1, 11, 21]`  
# This means simply to make three requests, starting at position 1, 11, and 21.

# In[ ]:


# basketball = adv.serp_goog(q="basketball", cx=cx, key=key, start=[1, 11, 21])


# In[ ]:


basketball = pd.read_csv('../input/basketball.csv', parse_dates=['queryTime'])
print('Rows, columns:', basketball.shape)
print('\nRanks:', basketball['rank'].values, '\n')
basketball.head(2)


# ## Date Restriction
# 
# You can make a request while restricting the date of the content you are asking for, to be within the last X days, weeks, months, or years.  
# This is done through the `dateRstrict` parameter. It takes the following possible values: d, w, m, y, and possibly you can add an integer to specify how many days or months you want to restrict your query. 
# - `d`: results from the last day.
# - `d5`: results from the last five days.
# - `w` or `w5` for example.
# - `m`, `m3`, `y`, `y6` etc.
# 
# Let's see for example how results differ if we search for "Donald Trump" with different dates:
# 
# `trump =  adv.serp_goog(q='donald trump', cx=cx, key=key, dateRestrict=['d1', 'd30','m6'])`  
# 
# As you can guess, this will make three requests one asking for data from one day ago, from 30 days ago, and finally six months.
# 

# In[ ]:


# trump = adv.serp_goog(q='donald trump', cx=cx, key=key, dateRestrict=['d1', 'd30','m6'])


# In[ ]:


trump = pd.read_csv('../input/trump.csv', parse_dates=['queryTime'])
trump.groupby(['dateRestrict']).head()[['rank','dateRestrict','title', 'snippet', 'displayLink', 'queryTime']]


# Naturally, some sites like Wikipedia or Twitter would show the same URL, but the news ones show a different story based on the date option that we specified. 

# ## Including / Excluding Websites
# 
# Sometimes you want to restrict your search to a certain website, and sometimes you may want to search the web _excluding_ a certain domain. This can be done by the `siteSearch` and `siteSearchFilter` parameters. For example: 
# 
# `adv.serp_goog(q='iphone', cx=cx, key=key, siteSearch='www.apple.com', siteSearchFilter=['e', 'i'])`
#               
# As you can guess this will make two requests for the keyword 'iphone', one including `i` only www.apple.com and the other excluding `e` apple.com and searching the whole web. 
# 
# 
# This was a quick overview of how you can get SERP data easily, on a large scale, and consistently.  
# This can be used for exploration and ongoing work.
# - **Exploration:** at the beginning of a project where you still don't know about the main players, how diverse the landscape is, competition etc. You would typically make a few queries, analyze some data and decide on what to focus on. 
# - **Ongoing:** once you have a good idea, then you would probably want to run the same queries once a month, to keep track of how things are progressing, how positions are changing, and whether other players are making changes in their strategies. 
# 
# Happy to get any [feedback, bugs](https://github.com/eliasdabbas/advertools), or [suggestions](https://twitter.com/eliasdabbas).
# 
