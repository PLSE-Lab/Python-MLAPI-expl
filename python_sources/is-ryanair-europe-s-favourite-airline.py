#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # Outline
# 
# 1. [Problem Definition](#Problem-Definition)
# 2. [Data Preparation](#Data-Preparation)
# 3. [Data Analysis](#Data-Analysis)
# 4. [Conclusions](#Conclusions)

# # Problem Definition

# Everything starts with this job posting that popped up one day in my LinkedIn feed.
# 
# <img src='https://media-exp1.licdn.com/media-proxy/ext?w=1134&h=223&f=pj&hash=xgLfZG3Im7edf6uwWMdCw9wsRnY%3D&ora=1%2CaFBCTXdkRmpGL2lvQUFBPQ%2CxAVta5g-0R6jnhodx1Ey9KGTqAGj6E5DQJHUA3L0CHH05IbfPWi7cMCNK-ako0AffigGjQA2ebq1EjTlGo7oKo7qKNx4iJ_kIJH5aRUPbhU4hGUB5sE-Pg&shareType=image' >
# 
# What stroke me was the subtitle, which is the rationale for this analysis and gave the title to this notebook:
# 
#    > Ryanair - **Europe's Favourite Airline**
#    
# This was very weird, because all the people I know (including myself) actually complain about Ryanair. Indeed, doing a bit of [research](https://corporate.ryanair.com/news/iata-confirms-ryanair-is-europes-favourite-airline/) I discovered that this claim was based on the fact that according to IATA Ryanair carried more international passengers than any other airline. 
# 
# According to me, this was a completely wrong and mischievous way to interpret data. Ryanair here is saying that carrying more passengers is equal to being European's favourite airline as if there was a direct causal relationship, but the reasons may be multiple and varied. The first three that come to my mind are:
# 1. **number** of connections
# 2. **monopoly** of the routes 
# 3. **airports control**.
# 
# Let's start with the former which is easier. The number of served customers grows as you open more shops. Similarly, Ryanair may be serving more customers simply because it has more flights.
# 
# As for the monopoly, think about this scenario: you and most of your friends where you live prefer BurgerKing over McDonald's, but in your city there is only McDonald's and you would have to drive 50 miles if you really would like one BurgerKing: most probably in one year you will end up having many more BigMacs than a Whoppers. Does this mean that McDonald's is your favourite because you bought more there? No, it happened simply as a result of a *monopoly*: the absence of alternatives forced you to buy at McDonald's. The feeling I have is that Ryanair is exactly like McDonald's in this example. It does not serve more customers because they actually prefer Ryanair, but simply because they do not have another option, they're victims of a monopoly.
# 
# Lastly, consider this other scenario, similar to the previous one: you live in a city with several restaurants, and from time to time you want to go out for dinner and try a new restaurant, choosing it somehow randomly. If most of these restaurants belong to the same chain, you will eat dinner in this chain's restaurants most of the times, although you have no special preference for it. It just happened because they had a bigger share of restaurants in the city. Here the city is the airport and going out for dinner to a restaurant is flying somewhere for leisure, a weekend abroad for example. You want to change place every time (it's improbable to visit the same place twice) and you will pick the city regardless of the company that operates (in the same way you choose the restaurant randomly, regardless of the owner). If in your airport there are almost exclusively Ryanair flights, you will end up flying with Ryanair more simply because there were more chances. This is also a form of monopoly.<br>
# Looking at it the other way round, does Ryanair choose to pick routes in busy or free airports?
# 
# The questions that need to be addressed are:
# 1.  how many routes does Ryanair have?
# 2.  how many "*exclusive*" routes (i.e. routes served by only one carrier) does Ryanair have?
# 3.  
#     3.a. in case of "*shared*" routes (i.e. routes served by multiple carriers), how many competitors does Ryanair have?<br>
#     3.b. in how many airports is Ryanair present? What's Ryanair presence (share of flights) in these airports?

# # Data Preparation

# ## Airlines

# In[ ]:


airlines_df = (
    pd.read_csv('/kaggle/input/airline-database/airlines.csv', 
                na_values=['\\N', '-', 'NAN'])
    .rename(columns=str.lower)
    .rename(columns=lambda col: '_'.join(col.split()))
    .dropna(subset=['iata'])
    [lambda df: df.active == 'Y']
    .drop(columns=['active', 'alias'])
    .assign(airline_id=lambda df: df.airline_id.map(str))
    .rename(columns={'airline_id': 'id'})
    .rename(columns=lambda col: 'airline_' + col)
)

airlines_df.head()


# ## European Airports

# In[ ]:


airports_df = (
    pd.read_csv('/kaggle/input/airports-train-stations-and-ferry-terminals/airports-extended.csv',
                header=None, names=['id', 'name', 'city', 'country', 'iata', 'icao', 'latitude', 
                                    'longitude', 'altitude', 'timezone', 'dst', 'tz_timezone', 'type', 'data_source'],
                na_values=['\\N', '-', 'NAN', 'unknown'])
    .set_index('id')
    [lambda df: df.type == 'airport']
    .reset_index(drop=True)
    .drop(columns=['type', 'timezone', 'tz_timezone', 'data_source'])
    .rename(columns=lambda col: 'airport_' + col)
)

eur_airports_iata_df = (
    pd.read_csv('/kaggle/input/european-airports-iata-codes/european_iatas_df.csv',
                header=None, names=['airport_city', 'airport_name', 'airport_iata'])
)

eur_airports_df = (
    airports_df[airports_df.airport_iata.isin(eur_airports_iata_df.airport_iata)].copy()
    [lambda df: ~df.airport_country.isin(['Russia', 'Turkey'])]
    .reset_index(drop=True))

# del airports_df, eur_airports_iata_df

print('All:', airports_df.shape)
print('Europe:', eur_airports_df.shape)
eur_airports_df.head()


# ## European Routes
# 
# A route is considered to be European if it connects to European airports.

# In[ ]:


routes_df = (
    pd.read_csv('/kaggle/input/flight-route-database/routes.csv', 
                na_values=['\\N', '-'])
    .rename(columns=str.strip)
    .drop(columns=['codeshare', 'equipment'])
    .rename(columns=str.lower)
    .rename(columns=lambda v: v.replace(' ', '_'))
    .rename(columns={'destination_apirport': 'dst_airport_iata',
                     'source_airport': 'src_airport_iata',
                     'source_airport_id': 'src_airport_id',
                     'destination_airport_id': 'dst_airport_id',
                     'airline': 'airline_iata'})
    .dropna(subset=['airline_id', 'src_airport_id', 'dst_airport_id'], how='any')
    .assign(airline_id=lambda df: df.airline_id.map(int).map(str),
            src_airport_id=lambda df: df.src_airport_id.map(int).map(str),
            dst_airport_id=lambda df: df.dst_airport_id.map(int).map(str),
            route_id=lambda df: df[['src_airport_iata', 'dst_airport_iata']].apply(sorted, axis=1).map('_'.join))
)

eur_routes_df = (
    routes_df[
        routes_df.src_airport_iata.isin(eur_airports_df.airport_iata)
        &
        routes_df.dst_airport_iata.isin(eur_airports_df.airport_iata)
    ].copy()
)

print('All:', routes_df.shape)
print('Europe:', eur_routes_df.shape)
eur_routes_df.head()


# Let's add more details about airline and airports of each route...

# In[ ]:


eur_routes_df2 = (
    eur_routes_df
    .pipe(pd.merge, 
          airlines_df[['airline_id', 'airline_iata', 'airline_name', 'airline_country']],
          how='left')
    .pipe(pd.merge, 
          eur_airports_df[['airport_iata', 'airport_country', 'airport_city', 'airport_latitude', 'airport_longitude']]
          .rename(columns=lambda col: 'src_' + col),
          how='left')
    .pipe(pd.merge, 
          eur_airports_df[['airport_iata', 'airport_country', 'airport_city', 'airport_latitude', 'airport_longitude']]
          .rename(columns=lambda col: 'dst_' + col),
          how='right')
    .assign(is_national_route=lambda df: df.src_airport_country == df.dst_airport_country)
    .assign(dist=lambda df: np.sqrt(
        (df.dst_airport_longitude - df.src_airport_longitude) ** 2 
        + (df.dst_airport_latitude - df.src_airport_latitude) ** 2))
)

eur_routes_df2.head()


# # Data Analysis

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

# ensure white background
plt.rcParams['figure.facecolor'] = 'w'


# ## Number of Connections

# In[ ]:


n_connections_srs = (
    eur_routes_df2
    .groupby('airline_name')
    .route_id
    .nunique()
    .sort_values()[::-1]
    .rename('# Unique Connections')
    .rename_axis('Airline')
)


# In[ ]:


fig, ax = plt.subplots(figsize=(10,6))

ax.barh(n_connections_srs.index[:25][::-1], n_connections_srs[:25][::-1])
ax.barh('Ryanair', n_connections_srs['Ryanair'], color='navy')

ax.set_xlabel('Unique Connections')

ax.set_title('Top 25 Airlines\nby Number of Unique Connections', weight='bold')
ax.grid(axis='x')
plt.show()


# ## Connections Monopoly (Exclusive Connections)

# In[ ]:


n_exclusive_connections_srs = (
    eur_routes_df2
    .groupby('route_id')
    .airline_id
    .nunique()
    [lambda srs: srs == 1]
    .reset_index()
    .pipe(pd.merge, eur_routes_df2[['route_id', 'airline_name']], how='left')
    .groupby('airline_name')
    .route_id
    .nunique()
    .sort_values()[::-1]
    .rename('# Exclusive Connections')
    .rename_axis('Airline')
)

fig, ax = plt.subplots(figsize=(10,6))

ax.barh(n_exclusive_connections_srs.index[:25][::-1], n_exclusive_connections_srs[:25][::-1])
ax.barh('Ryanair', n_exclusive_connections_srs.loc['Ryanair'], color='navy')
ax.set_xlabel('Exclusive Connections')
ax.set_title('Top 25 Airlines\nby Number of Exclusive Connections', weight='bold')
ax.grid(axis='x')

plt.show()


# In[ ]:


fig, ax = plt.subplots()

ax.pie(n_exclusive_connections_srs, autopct=lambda pct: ('%.2f' % pct) + '%' if pct > 4 else '',
       labels=[al if i < 5 else '' for i, al in enumerate(n_exclusive_connections_srs.index)])

plt.show()


# ## Airport Presence

# ### Number of Airports

# In[ ]:


n_airports_srs = (
    eur_routes_df2
    .groupby('airline_name')
    .src_airport_id
    .nunique()
    .sort_values()[::-1]
    .rename('# Airports')
    .rename_axis('Airline')
)

fig, ax = plt.subplots(figsize=(10,6))

ax.barh(n_airports_srs.index[:25][::-1], n_airports_srs[:25][::-1])
ax.barh('Ryanair', n_airports_srs.loc['Ryanair'], color='navy')
ax.set_xlabel('# Airports')
ax.set_title('Top 25 Airlines\nby Number of Base Airports', weight='bold')
ax.grid(axis='x')

plt.show()


# In[ ]:


n_airports_srs.head()


# ### Number of Countries

# In[ ]:


n_countries_srs = (
    eur_routes_df2
    .groupby('airline_name')
    .src_airport_country
    .nunique()
    .sort_values()[::-1]
    .rename('# Countries')
    .rename_axis('Airline')
)

fig, ax = plt.subplots(figsize=(10,6))

ax.barh(n_countries_srs.index[:25][::-1], n_countries_srs[:25][::-1])
ax.barh('Ryanair', n_countries_srs.loc['Ryanair'], color='navy')
ax.set_xlabel('# Countries')
ax.set_title('Top 25 Airlines\nby Number of Base Countries', weight='bold')
ax.grid(axis='x')

plt.show()


# ### Number of Competitors

# In[ ]:


n_competitors_df = (
    eur_routes_df2.groupby('src_airport_id')
    .airline_id.nunique().sub(1)
    .to_frame('n_competitors')
    .reset_index()
    .pipe(pd.merge, eur_routes_df2[['src_airport_id', 'airline_name']].drop_duplicates())
)

avg_competitors_srs = (
    n_competitors_df
    .groupby('airline_name')
    .n_competitors
    .mean()
    .sort_values()[::-1]
)

fig, ax = plt.subplots(figsize=(10, 6))

ax.barh(*list(zip(*list(avg_competitors_srs.reindex(index=n_airports_srs.index[:25]).sort_values().iteritems()))))
ax.barh('Ryanair', avg_competitors_srs.loc['Ryanair'], color='navy')
ax.set_xlabel('Avg Competitors per Airport')
ax.set_title('Avg Number of Competitors (per Airport)\namong Top 25 Airlines by Number of Airports', weight='bold')
ax.grid(axis='x')

plt.show()


# ### Presence

# In[ ]:


airport_presence_df = (
    eur_routes_df2
    .groupby('src_airport_id')
    .airline_name
    .apply(lambda srs: srs.value_counts(normalize=True).rename_axis('airline_name'))
    .rename('airport_presence')
    .reset_index()
)

avg_airport_presence_df = (
    airport_presence_df
    .groupby('airline_name')
    .airport_presence.mean()
    .sort_values()[::-1]
)

fig, (ax, ax1) = plt.subplots(1, 2, figsize=(16,6))

ax.barh(avg_airport_presence_df.index[:25][::-1], avg_airport_presence_df[:25][::-1])
ax.barh('Ryanair', avg_airport_presence_df.loc['Ryanair'], color='navy')
ax.set_xlabel('Avg Airport Presence (%)')
ax.set_title('Top 25 Airlines\nby Avg Airport Presence', weight='bold')
ax.grid(axis='x')

avg_presence_for_top25_airlines_by_airports = avg_airport_presence_df.reindex(index=n_airports_srs.index[:25]).sort_values()
ax1.barh(avg_presence_for_top25_airlines_by_airports.index, avg_presence_for_top25_airlines_by_airports)
ax1.barh('Ryanair', avg_airport_presence_df.loc['Ryanair'], color='navy')
ax1.set_xlabel('Avg Airport Presence (%)')
ax1.set_title('Avg Airport Presence\namong Top 25 Airlines by Number of Base Airports', weight='bold')
ax1.grid(axis='x')

plt.tight_layout()
plt.show()


# In[ ]:


avg_airport_presence_df.head()


# # Conclusions
# 
# All the hypothesis were confirmed:
# - Ryanair is indeed the airline with the highest number of connections, which implies carrying more passengers than other European airlines;
# - Ryanair is indeed the airline with the highest number of exclusive connections, thas is it has the monopoly on the highest number of routes (912 - 34% of all exclusive routes in Europe): all the passengers of these connections were Ryanair passengers for major cause;
# - Ryanair is present in the highest number of airports (156). Among the top 25 airlines in this chart, it also has the highest average presence per airport (43.55% - that is, on average it has the 43.55% of all the connections offered by the airport) and the lower number of competitors per airport (around 10).
# 
# The data tells us that there are at least other three valid and alternative explanations for Ryanair achievement which actually go against their proposed interpretations. The only thing they got right is that... they need a Senior Data Scientist.
