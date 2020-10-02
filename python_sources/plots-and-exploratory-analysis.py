#!/usr/bin/env python
# coding: utf-8

# Exploratory analysis of beer dataset

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


beer=pd.read_csv("../input/beers.csv")
brewery=pd.read_csv("../input/breweries.csv")


# In[ ]:


beer.head()


# In[ ]:


brewery.head()


# In[ ]:


beer.shape


# In[ ]:


brewery.shape


# In[ ]:


beer.describe()


# In[ ]:


brewery.describe()


# Highest number of breweries by state wise

# In[ ]:


df=brewery.groupby('state').count()
df1=df.sort(columns='name',axis=0, ascending=False)
df1.head()


# Cities with highest number of breweries

# In[ ]:


cities=brewery.groupby('city').count()
cities1=cities.sort(columns='name', ascending=False)

cities1.head()


# Most popular craft beer in US

# In[ ]:


craft_beer=beer.groupby('style').count()
craft_beer1=craft_beer.sort(columns='id', ascending=False)
craft_beer1.head()


# Merge both beer and brewery dataset

# In[ ]:


brewery['brewery_id']=brewery.index
brewery.head()


# In[ ]:


new_data=beer.merge(brewery, on="brewery_id")
new_data.head()


# In[ ]:


new_data=new_data.rename(index=str, columns={"name_x":"beer_name", "name_y":"brewery_name"})
new_data.head()


# In[ ]:


new_data=new_data.drop(['Unnamed: 0_x', 'Unnamed: 0_y'], axis=1)
new_data.head()


# In[ ]:


plt.figure(figsize=(9,6))
plot1=new_data.state.value_counts().plot(kind='bar', title='Breweries by state')

plot1.set_xlabel('state')


# In[ ]:


plt.figure(figsize=(9,6))

plot1=new_data.groupby('city')['brewery_name'].count().nlargest(10).plot(kind='bar',title='Cities with most breweries')

plot1.set_ylabel('Number of breweries')


# In[ ]:


plt.figure(figsize=(9,6))
plot2=new_data.groupby('style')['beer_name'].count().nlargest(10).plot(kind='bar',title='Popular brewed beer styles')
plot2.set_ylabel('Number of Different Beers')


# In[ ]:




