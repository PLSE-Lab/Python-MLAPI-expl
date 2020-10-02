#!/usr/bin/env python
# coding: utf-8

# #                         Resturaunt Recommender Systems
# ## 1. *Recommendations based on Popularity*

# In[ ]:


import pandas as pd
import numpy as np


# These datasets are hosted on: https://archive.ics.uci.edu/ml/datasets/Restaurant+%26+consumer+data
# 
# They were originally published by: Blanca Vargas-Govea, Juan Gabriel González-Serna, Rafael Ponce-Medellín. Effects of relevant contextual features in the performance of a restaurant recommender system. In RecSys’11: Workshop on Context Aware Recommender Systems (CARS-2011), Chicago, IL, USA, October 23, 2011.

# In[ ]:


frame = pd.read_csv('../input/restaurant-data-with-consumer-ratings/rating_final.csv')
cuisine = pd.read_csv('../input/restaurant-data-with-consumer-ratings/chefmozcuisine.csv')
geodata = pd.read_csv('../input/restaurant-data-with-consumer-ratings/geoplaces2.csv')


# In[ ]:


frame.head()


# In[ ]:


cuisine.head()


# ## Recommendations based on counts

# In[ ]:


rating_count = pd.DataFrame(frame.groupby('placeID')['rating'].count())

rating_count.sort_values('rating', ascending=False).head()


# In[ ]:


most_rated_places = pd.DataFrame([135085, 132825, 135032, 135052, 132834], index=np.arange(5), columns=['placeID'])

summary = pd.merge(most_rated_places, cuisine, on='placeID')
summary


# In[ ]:


cuisine['Rcuisine'].describe()


# ## 2.  *Recommendations Based on Correlation*

# In[ ]:


frame.head()


# In[ ]:


geodata.head()


# In[ ]:


places =  geodata[['placeID', 'name']]
places.head()


# In[ ]:


cuisine.head()


# ## Grouping and Ranking Data

# In[ ]:


rating = pd.DataFrame(frame.groupby('placeID')['rating'].mean())
rating.head()


# In[ ]:


rating['rating_count'] = pd.DataFrame(frame.groupby('placeID')['rating'].count())
rating.head()


# In[ ]:


rating.describe()


# In[ ]:


rating.sort_values('rating_count', ascending=False).head()


# In[ ]:


places[places['placeID']==135085]


# In[ ]:


cuisine[cuisine['placeID']==135085]


# ## Preparing Data For Analysis

# In[ ]:


places_crosstab = pd.pivot_table(data=frame, values='rating', index='userID', columns='placeID')
places_crosstab.head()


# In[ ]:


Tortas_ratings = places_crosstab[135085]
Tortas_ratings[Tortas_ratings>=0]


# ## Evaluating Similarity Based on Correlation

# In[ ]:


similar_to_Tortas = places_crosstab.corrwith(Tortas_ratings)

corr_Tortas = pd.DataFrame(similar_to_Tortas, columns=['PearsonR'])
corr_Tortas.dropna(inplace=True)
corr_Tortas.head()


# In[ ]:


Tortas_corr_summary = corr_Tortas.join(rating['rating_count'])


# In[ ]:


Tortas_corr_summary[Tortas_corr_summary['rating_count']>=10].sort_values('PearsonR', ascending=False).head(10)


# In[ ]:


places_corr_Tortas = pd.DataFrame([135085, 132754, 135045, 135062, 135028, 135042, 135046], index = np.arange(7), columns=['placeID'])
summary = pd.merge(places_corr_Tortas, cuisine,on='placeID')
summary


# In[ ]:


places[places['placeID']==135046]


# In[ ]:


cuisine['Rcuisine'].describe()


# In[ ]:




