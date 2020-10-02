#!/usr/bin/env python
# coding: utf-8

# ## Hello,
# this is my first **serious** kernel. I wanted to make it as informative as possible preserving the simplicity.
# Hope you will find this code and visualizations useful.
# 
# For plots I use [plotly](https://plotly.com/python/). 
# 
# If you found this notebook interesting or useful please upvote it.
# 
# ### Some ideas...
# 1. Russia is a big country. But the most of it's population and economics is in european part of it. How strongly distance from Moscow and airport traffic are correlated? (can be a trap - both can be correlated with population) 
# 
# ### Links to parts
# 1. [Introduction and first important notices](#First-steps)
# 2. [Air_service EDA](#Air-Service)
# 3. [CARGO_AND_PARCELS EDA](#CARGO_AND_PARCELS)

# ### Importing libraries and getting path to files

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import plotly.express as px
import plotly.graph_objects as go

from scipy.stats import pearsonr

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from sklearn.manifold import TSNE

np.random.seed(42)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
os.chdir('/kaggle/input/russian-passenger-air-service-20072020/')


# # First steps

# ### Let's take a quick look at both files.

# In[ ]:


cargo = pd.read_csv('russian_air_service_CARGO_AND_PARCELS.csv')
air_service = pd.read_csv('russian_passenger_air_service_2.csv')


# In[ ]:


cargo.head()


# In[ ]:


air_service.head()


# In[ ]:


def lat(x):
    try:
        return float(x.replace('Decimal', '').replace('(', '').replace(')', '').replace('\'', '').split(',')[0])
    except:
        return np.nan
    
def long(x):
    try:
        return float(x.replace('Decimal', '').replace('(', '').replace(')', '').replace('\'', '').split(',')[1])
    except:
        return np.nan

air_service['Latitude'] = air_service['Airport coordinates'].apply(lat)
air_service['Longitude'] = air_service['Airport coordinates'].apply(long)

cargo['Latitude'] = cargo['Airport coordinates'].apply(lat)
cargo['Longitude'] = cargo['Airport coordinates'].apply(long)


# ### How many airports reported in 2020?

# In[ ]:


fig = go.Figure()
fig.add_trace(go.Bar(y=[(air_service[(air_service.Year==2020)].January>0).sum(), 
                            (air_service[(air_service.Year==2020)].February>0).sum(),
                            (air_service.loc[(air_service.Year==2020), ['January', 'February']].sum(axis=1)>0).sum()],
                         x=['January', 'February', 'Any']))
fig.update_layout(title='Reported airports (positive January/February)')
fig.show()


# #### We have data from a very little amount of airports in 2020. But what if not only in 2020?

# In[ ]:


print('There is', (air_service.groupby('Airport name')['Whole year'].sum()>0).sum(), '/', air_service['Airport name'].nunique(), 'airports with any positive traffic since 2007')


# ### Wow, almost 2/3 of airports are "dead". Let's delete them.

# # Air Service

# ### Let's take a deeper look at air_service file. 

# In[ ]:


monthes = air_service.columns[2:14]
alive = air_service.groupby('Airport name')['Whole year'].sum()>0
air_service = air_service[air_service['Airport name'].apply(lambda x: alive[x])]

fig = go.Figure()

for month in monthes:
    sum_traffic_per_year = air_service.groupby('Year')[month].sum().reset_index()
    sum_traffic_per_year = sum_traffic_per_year.rename(columns={month:'Traffic'})
    fig.add_trace(go.Scatter(y=sum_traffic_per_year['Traffic'], x=sum_traffic_per_year['Year'], name=month))
    
fig.update_layout(title='Traffic in every month grouped by year (sum)')
fig.show()


# ### What can we see here?
# 
# 1. Traffic increases from year to year. (not including December ;) )
# 2. The closer summer is - more people want to travel to somewhere. December is absolutely not popular time for traveling.
# 3. 2009 and 2016 showed decrease in traffic. Maybe because of financial crisis.
# 4. We have not enough information about 2020: it shows growth like there is no COVID-19. 

# ## Map

# In[ ]:


monthes = air_service.columns[2:14]

fig = go.Figure()

year_2019 = air_service.Year == 2019

fig.add_trace(go.Scattergeo(lat=air_service.loc[year_2019, 'Longitude'],
                           lon=air_service.loc[year_2019, 'Latitude'],
                           text=air_service.loc[year_2019, 'Airport name'].values,
                           marker={'size':np.clip(air_service.loc[year_2019, 'Whole year']/(2*10e4), 0, 30)}))

fig.update_layout(title='Location of airports and their year traffic in 2019',)
fig.show()


# ### What can we see here? 
# 1. Most of the traffic belongs to cities with the highest population.
# 2. Some popular places for vacation like Sochi also had a higher trafic in 2019.

# ### Most popular airports

# In[ ]:


average_year = air_service.groupby('Airport name')['Whole year'].mean().reset_index().sort_values(by='Whole year').iloc[-8:]
average_year = average_year.rename(columns={'Whole year': 'Average traffic'})

fig = go.Figure()

fig.add_trace(go.Bar(x=average_year['Airport name'].values, y=average_year['Average traffic'].values))
fig.update_layout(title='Average traffic of 8 most popular airports',)
fig.show()


# ### KMeans clusterization. 

# In[ ]:


cluster_number = 8

clusterizer = make_pipeline(StandardScaler(), KMeans(random_state=42, 
                                        n_clusters=cluster_number)).fit(air_service.loc[air_service.Year==2019, monthes])


# In[ ]:


fig = go.Figure()

for cluster in range(cluster_number):
    inCluster = clusterizer[1].labels_==cluster
    
    fig.add_trace(go.Scattergeo(lat=air_service.loc[year_2019].loc[inCluster, 'Longitude'],
                               lon=air_service.loc[year_2019].loc[inCluster, 'Latitude'],
                               text=air_service.loc[year_2019].loc[inCluster, 'Airport name'].values,
                               name=cluster,
                               marker={'size':np.log1p(air_service.loc[year_2019].loc[inCluster, 'Whole year'])}))

fig.update_layout(title='Location of airports and their year traffic (logarithmic) in 2019',)
fig.show()


# ### What can we see
# 1. Sochi, Krasnodar, Ekaterinburg are in the same cluster - these cities are the centers of tourism.
# 2. Moscow and Saint-Petersburg are the most popular cities and got their own clusters.

# ### Looking deeper into KMeans decisions...

# In[ ]:


tsne_decomposed = make_pipeline(StandardScaler(), TSNE(random_state=42))                .fit_transform(air_service.loc[air_service.Year==2019, monthes])

fig = go.Figure()
for cluster in range(cluster_number):
    fig.add_trace(go.Scatter(x=tsne_decomposed[clusterizer[1].labels_==cluster, 0],
                             y=tsne_decomposed[clusterizer[1].labels_==cluster, 1],
                             mode='markers', text=air_service.loc[air_service.Year==2019].\
                             loc[clusterizer[1].labels_==cluster, 'Airport name'], name=cluster))
fig.update_layout(title='TSNE decomposition (2019, monthes)')
fig.show()


# In[ ]:


fig = go.Figure()
for cluster in range(cluster_number):
    fig.add_trace(go.Scatter(x=monthes,
                             y=air_service.loc[air_service.Year==2019, monthes].loc[clusterizer[1].labels_==cluster].mean(axis=0),
                             mode='markers+lines', name=cluster))
fig.update_layout(title='Mean traffic per month for every cluster')
fig.show()


# ### The only difference between clusters is popularity of each airport. Trends are the same.

# 
# # CARGO_AND_PARCELS

# ### For quicker preview I'd like to use .head() again.

# In[ ]:


cargo.head()


# In[ ]:


print("Only", (cargo.groupby('Airport name')['Whole year'].sum()>0).sum(), 'airports are "alive"')


# In[ ]:


alive = cargo.groupby('Airport name')['Whole year'].sum()>0
cargo = cargo[cargo['Airport name'].apply(lambda x: alive[x])]


# In[ ]:


average_year = cargo.groupby('Airport name')['Whole year'].mean().reset_index().sort_values(by='Whole year').iloc[-8:]
average_year = average_year.rename(columns={'Whole year': 'Average cargo'})

fig = go.Figure()

fig.add_trace(go.Bar(x=average_year['Airport name'].values, y=average_year['Average cargo'].values))
fig.update_layout(title='Average cargo of 8 most popular airports',)
fig.show()


# ### What's here?
# 1. Khabarovsk and Vladivostok are new here, comparing to average traffic barplot.
# 2. Moscow and Saint Petersburg are still the top-4 airports.

# ### Since top have slightly changed, what can we expect from correlation between cargo and traffic?

# In[ ]:


same_airports = set(cargo['Airport name'].unique()).intersection(set(air_service['Airport name'].unique()))

cargo_pivot = pd.pivot_table(cargo[cargo['Airport name'].isin(same_airports)], values=monthes,
                             columns=['Year'], index=['Airport name']).fillna(0)
airser_pivot = pd.pivot_table(air_service[air_service['Airport name'].isin(same_airports)], values=monthes,
                              columns=['Year'], index=['Airport name']).fillna(0)

corr = pd.Series([pearsonr(cargo_pivot.values[a], airser_pivot.values[a])[0] for a in range(cargo_pivot.shape[0])],
          index=cargo_pivot.index)

corr = corr.reset_index().rename(columns={0:'Correlation'})

px.histogram(corr, x='Correlation', title='Distribution of correlation between cargo and traffic')


# In[ ]:


print('Skewness is', corr.Correlation.skew(), '; Mean is', corr.Correlation.mean())


# ### What can we see
# 1. Distribution is skewed. Skewness is negative, most of the correlations are positive
# 2. Mean is ~0.45. For some airports correlation is very high (>=0.7)

# ### But what affects correlation? Why some airports have a visisbly higher correlation?
# ### Why some have negative Pearson coefficient?
