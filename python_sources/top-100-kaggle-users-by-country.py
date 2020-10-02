#!/usr/bin/env python
# coding: utf-8

# The purpose of this script is to show that it would be nice to have **country** and **city** data available in public Meta Kaggle database. To get such info for this notebook I did web scrapping on Kaggle for only top-100 Kaggle users. I hope Kaggle admins wouldn't be angry to me for that.
# 
# There are some ideas of how to use it:<br>
# 1. For each competition calculate distribution by country. e.g. recent one from TalkingData may involve more Data Scientist from China.<br>
# 2. Analyze Teams to understand why they are connected. Probably because they are living in one city.<br>
# 3. See what type of competitions involve people from which country.<br>
# and etc.
# 
# This is my first Python Notebook on Kaggle and there were some troubles to run my code from local Notebook because of libraries. If you know some best practice to improve code, it would be nice to share.
# 

# ## Load top-100 users to the DataFrame ##

# In[ ]:


import pandas as pd
import sqlite3

# Connect to the DB
con = sqlite3.connect('../input/database.sqlite')

# Exec select query
top100users = pd.read_sql_query("""
SELECT u.Id, u.UserName, u.DisplayName, cast(u.Ranking as INTEGER) as Ranking
FROM Users u
WHERE u.Ranking <= 100
ORDER BY u.Ranking""", con)

# Show Top-5
top100users.head(5)


# ## Web scraping code to get countries ##

# It doesn't work here, that is why I've commented it and would be using dictionary with preloaded data instead later. You could copy it and run locally.

# In[ ]:


#from bs4 import BeautifulSoup
#import re
#import requests
#from lxml import html

#geo_info = []
#for num in range(top100users.shape[0]):
#    response = requests.get('http://www.kaggle.com/{}'.format(top100users.loc[num, 'UserName']))
#    raw_text = str(BeautifulSoup(response.text, "lxml")).replace('"', '')
#    print re.findall('(country:[A-Za-z\s\n]+)', raw_text)[0][8:]
#    geo_info.append([top100users.loc[num, 'UserName'], 
#                     re.findall('(country:[A-Za-z\s\n]+)', raw_text)[0][8:]])
#top100users = pd.merge(top100users, pd.DataFrame(geo_info, columns=['UserName', 'Country']), how='left', on='UserName')
   


# Do some Country cleaning:<br>
#  1. Translate alpha-2 code to ordinal country names;<br>
#  2. Map some countries to the standard names;<br>
#  3. Capitalize letters.<br>
# <br>
# This is also commented because we don't have results from previous step.

# In[ ]:


#import pycountry as pc

## Generate dict to translate Alpha2 Code to Country Name
#alpha2_code = dict()
#for country in list(pc.countries):
#    alpha2_code[country.alpha2] = country.name

## Dict to mapping some non-standard names
#extraCountry = {'Russia':'Russian Federation',
#                'USA':'United States'}

## Apply mappings
#top100users.loc[top100users['Country'].isin(extraCountry.keys()), 'Country'] = map(lambda x: extraCountry[x], top100users.loc[top100users['Country'].isin(extraCountry.keys()), 'Country'])
#top100users.loc[top100users['Country'].isin(alpha2_code.keys()), 'Country'] = map(lambda x: alpha2_code[x], top100users.loc[top100users['Country'].isin(alpha2_code.keys()), 'Country'])
#top100users.loc[:, 'Country'] = map(lambda x: x.title(), top100users['Country'])
#top100users.head(5)


# Prepare dictionary to use it further instead of Kaggle scrapping.

# In[ ]:


#top100users[['Id', 'Country']].set_index('Id').to_dict()


# ## Define country dictionary for top-100 users ##

# In[ ]:


dict_country = {808: 'Russian Federation',  1455: 'Austria',  1483: 'Iran',  2036: 'Netherlands',  2242: 'United Kingdom',
  3090: 'Russian Federation',  3230: 'Japan',  4398: 'United States',  5309: 'Germany',  5635: 'Poland',  5642: 'Spain',
  6388: u'Japan',  6603: 'Japan',  7756: 'United States',  9766: 'United States',  9974: 'The Netherlands',
  10171: 'United States',  12260: 'Finland',  12584: 'United States',  16398: 'Null',  17379: 'Singapore',
  18102: 'Hungary',  18396: 'United States',  18785: 'United States',  19099: 'Russian Federation',  19298: u'Italy',
  19605: 'United States',  24266: 'Brazil',  27805: 'Russian Federation',  29346: 'Uae',  29756: u'Netherlands',
  31529: 'Croatia',  33467: 'Ukraine',  34304: 'Israel',  38113: 'Netherlands',  41959: 'India',  42188: 'Germany',
  42832: 'Null',  43621: 'United States',  48625: 'United States',  54836: u'Brazil',  56101: 'France',
  58838: 'South Korea',  59561: 'United States',  64274: u'United States',  64626: 'United States',  68889: u'Israel',
  70574: u'United States',  71388: u'India',  73703: 'Switzerland',  77226: 'United States',  85195: 'United States',
  90001: 'Singapore',  90646: 'Spain',  93420: 'United States',  94510: 'Turkey',  98575: 'United States',  99029: 'Turkey',
  100236: u'United States',  102203: u'China',  104698: u'United Kingdom',  105240: 'United States',  106249: 'Brazil',
  111066: 'Japan',  111640: u'Greece',  113389: 'India',  114032: 'United States',  114978: 'The Netherlands',
  116125: 'France',  147404: 'United States',  149229: 'Russian Federation',  149814: 'Null',  150865: 'United States',
  160106: u'Japan',  161151: u'Russian Federation',  163663: 'United States',  168767: 'Null',  170170: 'China',
  189197: 'Canada',  194108: 'United States',  200451: 'Russian Federation',  210078: u'Germany',  217312: u'Canada',
  218203: 'Japan',  221419: 'Null',  226693: u'Russian Federation',  254602: 'United States',  263583: u'Mexico',
  266958: 'Slovakia',  269623: 'Russian Federation',  275512: 'Israel',  275730: u'Germany',  278920: 'United States',
  300713: 'India',  312728: 'United States',  338701: 'Canada',  356943: u'India',  384014: 'Germany',
  405318: 'Lithuania',  582611: 'Belgium'}


# ## Create country list ##

# In[ ]:


# Merge Country dictionary with the top100users
top100usersCC = pd.merge(top100users, pd.DataFrame(list(dict_country.items()), columns=['Id','Country']), how='left', on='Id')
# Count countries
top100usersCC.loc[:, 'CountryCount'] = top100usersCC.groupby('Country')['Country'].transform('count')
# Create list with coounts ordered by desc
topCountries = top100usersCC[['Country', 'CountryCount']].drop_duplicates('Country', keep='first')                                                        .sort_values('CountryCount', ascending=False)                                                        .reset_index(drop=True)
topCountries


# ## Plot pie chart to show percentage ##

# In[ ]:


import matplotlib

matplotlib.style.use('ggplot')
get_ipython().run_line_magic('matplotlib', 'inline')

pieCountry = topCountries.set_index('Country')
pd.Series(pieCountry['CountryCount']).plot.pie(figsize=(13, 13), autopct='%0.1f')


# ## Plot countries on the Wolrd map ##

# In[ ]:


import plotly.offline as py
py.offline.init_notebook_mode()

data = [ dict(
        type = 'choropleth',
        locations = topCountries['Country'],
        z = topCountries['CountryCount'],
        locationmode = 'country names',
        text = topCountries['Country'],
        colorscale = [[0,"rgb(153, 241, 243)"],[0.2,"rgb(16, 64, 143)"],[1,"rgb(0, 0, 0)"]],
        autocolorscale = False,
        marker = dict(
            line = dict(color = 'rgb(58,100,69)', width = 0.6)),
            colorbar = dict(autotick = True, tickprefix = '', title = '# of Kagglers')
            )
       ]

layout = dict(
    title = 'Top100 Kagglers distributed by Country',
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
py.iplot(fig, validate=False, filename='d3-world-map')


# Please fill free to fork/copy/enhance this script.
