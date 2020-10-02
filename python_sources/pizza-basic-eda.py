#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
print(plt.style.available)
plt.style.use('Solarize_Light2')
df_pizza_restaurants = pd.read_csv('/kaggle/input/Datafiniti_Pizza_Restaurants_and_the_Pizza_They_Sell_May19.csv')
df_pizza_restaurants.info()


# In[ ]:


# https://geopandas.readthedocs.io/en/latest/gallery/create_geopandas_from_pandas.html
# import seaborn as sns
# sns.set(font_scale=1.5, style="white")

import geopandas
gdf = geopandas.GeoDataFrame(
    df_pizza_restaurants, geometry=geopandas.points_from_xy(df_pizza_restaurants.longitude, df_pizza_restaurants.latitude))
world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
fig, ax = plt.subplots(figsize=(16, 20))
world[world.continent == 'North America'].plot(
    color='white', edgecolor='black', ax=ax)
ax.set(title='Pizza Restaurant Location')
ax.set_axis_off()
gdf.plot(ax=ax, color='blue')

plt.show()


# In[ ]:


# fig, ax = plt.subplots(figsize=(16, 20))
# _=gdf.plot(ax=ax, color='blue')


# In[ ]:


df_pizza_restaurants.head(2).T


# In[ ]:


df_pizza_restaurants.columns


# In[ ]:


df_pizza_restaurants.isnull().sum()


# In[ ]:


for col in df_pizza_restaurants.columns:
    print(f"{col} {df_pizza_restaurants.nunique()}")


# In[ ]:


_=df_pizza_restaurants['city'].value_counts().nlargest(10).plot(kind='barh')


# In[ ]:


_=df_pizza_restaurants['province'].value_counts().nlargest(10).plot(kind='barh')


# In[ ]:


df_pizza_restaurants['primaryCategories'].value_counts()


# In[ ]:


_=df_pizza_restaurants.categories.value_counts().nlargest(10).plot(kind='barh')


# In[ ]:


_=df_pizza_restaurants['name'].value_counts().nlargest(10).plot(kind='barh')


# In[ ]:


_=df_pizza_restaurants['keys'].value_counts().nlargest(10).plot(kind='barh')


# In[ ]:


import seaborn as sns
sns.set(style="ticks")

sns.pairplot(df_pizza_restaurants, hue="priceRangeMin")

