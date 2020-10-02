#!/usr/bin/env python
# coding: utf-8

# #  Fast Food Restaurants Across America

# ## Import Libraries

# In[3]:


import numpy as np
import pandas as pd
import gmplot
from fuzzywuzzy import process, fuzz


# ## Getting the Data

# In[4]:


data = pd.read_csv('../input/FastFoodRestaurants.csv')


# In[5]:


data = data[["address", "city", "country", "latitude", "longitude", "name", "postalCode", "province"]]


# In[6]:


data.head()


# In[ ]:


sorted(data.name.unique())


# In[ ]:


data['lowername'] = data['name'].apply(lambda x : x.lower().strip())


# In[ ]:


unique_names = sorted(data.lowername.unique())


# ## Brands with most restaurant chains

# In[ ]:


restaurants_counts = data.lowername.value_counts()
restaurants_counts = restaurants_counts[restaurants_counts>250]
restaurants_list = list(restaurants_counts.index)
restaurants_list


# ## Creating an Independent Function for replacement of the matches in the data.

# In[ ]:


def replace_name(data, column_name, brand, threshold_ratio = 90):
    query = data[column_name].unique()
    results = process.extract(brand, query, limit=10, scorer=fuzz.token_sort_ratio)
    string_matches = [results[0] for results in results if results[1] >= threshold_ratio]
    rows_with_matches = data[column_name].isin(string_matches) 
    data.loc[rows_with_matches, column_name] = brand
    return data.copy()


# ## Tokens

# In[ ]:


similar_name_list = list()
for restuarant in restaurants_list:
    query = data['lowername'].unique()
    results = process.extract(restuarant, query, limit=9, scorer=fuzz.token_sort_ratio)
    similar_name_list.append(results)
similar_name_list

##Check the token_sort_ratio upto the closed value.
#mcdonald's - 58
#burger king - 100
#wendy's - 77


# In[ ]:


tokens = [57, 100, 100, 76, 90, 37, 50, 88]
for token, restuarant in zip(tokens, restaurants_list):
    replace_name(data,'lowername',restuarant, token)


# In[ ]:


data = replace_name(data,'lowername','kentucky fried chicken', 90)
data.loc[data.lowername.str.startswith('kentucky fried chicken'), 'lowername'] = 'kfc'


# In[ ]:


sorted(data.lowername.unique())


# In[ ]:


data_top = data[data.lowername.isin(restaurants_list)]


# In[ ]:


data_top.lowername.value_counts()


# ## Heatmaps

# ### Heatmap of All Restaurants Data

# In[ ]:


gmap = gmplot.GoogleMapPlotter.from_geocode('US',5)
#Then generate a heatmap using the latitudes and longitudes
gmap.heatmap(data['latitude'], data['longitude'])


# In[ ]:


gmap.draw('full_list.html')


# ### Heatmap of Top Restaurants Data

# In[ ]:


gmap = gmplot.GoogleMapPlotter.from_geocode('US',5)
#Then generate a heatmap using the latitudes and longitudes
gmap.heatmap(data_top['latitude'], data_top['longitude'])


# In[ ]:


gmap.draw('top_list.html')


# #### Heatmap

# ![](files/Heatmap of Top Restaurants.png)

# ## Provinces with most number of top restaurants

# In[ ]:


data_top = data_top[["country", "province", "city", "lowername"]]


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
data_top_province = data_top.province.value_counts()


# In[ ]:


data_top_province_list = list(data_top_province[data_top_province>200].index)


# In[ ]:


data_top_province_top = data_top[data_top.province.isin(data_top_province_list)]


# In[ ]:


data_top_province_top_group = data_top_province_top.groupby(['province','lowername'])
data_top_province_top_group.size().unstack().plot(kind ='bar', figsize =(16,7))

