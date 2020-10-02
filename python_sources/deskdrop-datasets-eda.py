#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from datetime import datetime
import re


# # Users interactions

# This section explores the dataset file containing users interactions on shared articles (**users_interactions.csv**).

# In[ ]:


interactions_df = pd.read_csv('../input/users_interactions.csv')
interactions_df.head(10)


# In[ ]:


def to_datetime(ts):
    return datetime.fromtimestamp(ts)

def to_datetime_str(ts):
    return to_datetime(ts).strftime('%Y-%m-%d %H:%M:%S')

print('First interaction: \t%s' % to_datetime_str(interactions_df['timestamp'].min()))
print('Last interaction: \t%s' % to_datetime_str(interactions_df['timestamp'].max()))


# In[ ]:


total_count = len(interactions_df)
android_count = len(interactions_df[interactions_df['userAgent'] == 'Android - Native Mobile App'])
ios_count = len(interactions_df[interactions_df['userAgent'] == 'iOS - Native Mobile App'])
print('# of interactions (total): \t%d' % total_count)
print('# of interactions (Android native app): \t%d' % android_count)
print('# of interactions (iOS native app): \t%d' % ios_count)


# In[ ]:


interactions_df['datetime'] = interactions_df['timestamp'].apply(lambda x: to_datetime(x))
interactions_df['month'] = interactions_df['datetime'].apply(lambda x: '{0}-{1:02}'.format(x.year, x.month))
interactions_df.groupby('month').size().plot(kind='bar', title='# interaction by month')


# Here is the description of event types:
# * VIEW: The user has opened the article. 
# * LIKE: The user has liked the article. 
# * COMMENT CREATED: The user created a comment in the article. 
# * FOLLOW: The user chose to be notified on any new comment in the article. 
# * BOOKMARK: The user has bookmarked the article for easy return in the future.

# In[ ]:


interactions_df.groupby('eventType').size().sort_values(ascending=False)


# In[ ]:


print('Distinct articles: \t%d' % len(interactions_df['contentId'].unique()))
print('Distinct users: \t%d' % len(interactions_df['personId'].unique()))
print('Distinct user sessions: \t%d' % len(interactions_df['sessionId'].unique()))


# The analysis of how many articles (items) a user has interacted with is important for recommender systems. Higher number of items consumed by users provides better modeling of users preference.  
# **We can observe that 50% of the users have interacted with 10 or more articles, making this dataset very suitable for collaborative filtering or content-based filtering methods.**

# In[ ]:


interactions_df.groupby('personId')['contentId'].size().describe()


# In[ ]:


interactions_df.groupby('personId')['contentId'].size().hist(bins=200, figsize=(15,5))


# Below we can see the countries and regions (states / districts) with more user interactions.

# In[ ]:


country_code_dict = {
    'BR': ('BRA', 'Brazil'),
    'US': ('USA', 'United States'),
    'KR': ('KOR', 'South Korea'),
    'CA': ('CAN', 'Canada'),
    'JP': ('JPN', 'Japan'),
    'AU': ('AUS', 'Australia'),
    'GB': ('GBR', 'United Kingdom'),
    'DE': ('DEU', 'Germany'),
    'DE': ('DEU', 'Germany'),
    'IE': ('IRL', 'Ireland'),
    'IS': ('ISL', 'Iceland'),
    'SG': ('SGP', 'Singapure'),
    'AR': ('ARG', 'Argentina'),
    'PT': ('PRT', 'Portugal'),
    'IN': ('IND', 'India'),
    'ES': ('ESP', 'Spain'),
    'IT': ('ITA', 'Italy'),
    'MY': ('MYS', 'Malaysia'),
    'CO': ('COL', 'Colombia'),
    'CN': ('CHN', 'China'),
    'CL': ('CHL', 'Chile'),
    'NL': ('NLD', 'Netherlands')
}

interactions_df['countryName'] = interactions_df['userCountry'].apply(lambda x: country_code_dict[x][1] if x in country_code_dict else None)
interactions_df[['userCountry','countryName']].head(10)


# In[ ]:


interactions_by_country_df = pd.DataFrame(interactions_df.groupby('countryName').size()                             .sort_values(ascending=False).reset_index())
interactions_by_country_df.columns = ['country', 'count']
interactions_by_country_df


# In[ ]:


import plotly.offline as py
py.offline.init_notebook_mode()

data = [ dict(
        type = 'choropleth',
        locations = interactions_by_country_df['country'],
        z = interactions_by_country_df['count'],
        locationmode = 'country names',
        text = interactions_by_country_df['country'],
        colorscale = [[0,"rgb(153, 241, 243)"],[0.005,"rgb(16, 64, 143)"],[1,"rgb(0, 0, 0)"]],
        autocolorscale = False,
        marker = dict(
            line = dict(color = 'rgb(58,100,69)', width = 0.6)),
            colorbar = dict(autotick = True, tickprefix = '', title = '# of Interactions')
            )
       ]

layout = dict(
    title = 'Total number of interactions by country',
    geo = dict(
        showframe = False,
        showcoastlines = True,
        projection = dict(
        type = 'equirectangular'
        ),
    margin = dict(b = 0, t = 0, l = 0, r = 0)
            )
    )

fig = dict(data=data, layout=layout)
py.iplot(fig, validate=False, filename='worldmap')


# In[ ]:


interactions_df['userCountryRegion'] = interactions_df['userCountry'] + '-' + interactions_df['userRegion']
interactions_df.groupby('userCountryRegion').size().sort_values(ascending=False).head(10)


# # Shared articles

# This section analyzes the articles shared in the platform.

# In[ ]:


articles_df = pd.read_csv('../input/shared_articles.csv')
articles_df.head(5)


# In[ ]:


print('First article sharing: \t%s' % to_datetime(articles_df['timestamp'].min()))
print('Last article sharing: \t%s' % to_datetime(articles_df['timestamp'].max()))


# In[ ]:


articles_df['datetime'] = articles_df['timestamp'].apply(lambda x: to_datetime(x))
articles_df['month'] = articles_df['datetime'].apply(lambda x: '{0}-{1:02}'.format(x.year, x.month))
articles_df[articles_df['eventType'] == 'CONTENT SHARED'].groupby('month').size()         .plot(kind='bar', title='# articles shared by month')


# In[ ]:


articles_df.groupby('eventType').size().sort_values(ascending=False)


# In[ ]:


print('Distinct articles: \t%d' % len(articles_df['contentId'].unique()))
print('Distinct sharers (users): \t%d' % len(articles_df['authorPersonId'].unique()))


# In[ ]:


articles_df.groupby('contentType').size().sort_values(ascending=False)


# In[ ]:


articles_df.groupby('lang').size().sort_values(ascending=False)


# In[ ]:


articles_df['urlDomain'] = articles_df['url'].apply(lambda x: re.sub(r'^http[s]*:\/\/', '', re.search(r'^http[s]*:\/\/[\w\.]*', x, re.IGNORECASE).group(0)))
articles_df[['urlDomain','url']].head()


# In[ ]:


main_domains_df = pd.DataFrame(articles_df[articles_df['lang'] == 'en'].groupby('urlDomain').size().sort_values(ascending=True))[-20:].reset_index()
main_domains_df.columns = ['urlDomain','count']
main_domains_df.plot(kind='barh', x='urlDomain', y='count', figsize=(10,10), title='Main domains on shared English articles')

