#!/usr/bin/env python
# coding: utf-8

# # Wikiaves Exploratory Data Analysis
# 
# (Danilo Lessa Bernardineli - danilo.lessa@gmail.com)
# 
# On this notebook, I'll conduct an EDA on the Wikiaves data for getting familiarized with the characteristics and properties of it. Also, I'm going to conduct some specific explorations with the goal of getting insights about the Brazilian Birder community.

# ## Initial loading and cleaning

# In[ ]:


# Dependences
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as st

from IPython.display import HTML


# On next block, I'll perform some basic conversions for optimizing performance. It is always an good idea to assure that the analytic data is fully performing as expected and expressed in the correct data-types, as this ensures performance and predictibility.
# 
# Also, it is an good hint to try to embrace Pandas chaining properties, as this allows to express an clear data processing pipeline. More info in: https://tomaugspurger.github.io/method-chaining.html

# In[ ]:


data_filepath = '/kaggle/input/brazilian-bird-observation-metadata-from-wikiaves/wikiaves_metadata-2019-08-11.feather'

data = (pd.read_feather(data_filepath)
          .assign(registry_date=lambda df: pd.to_datetime(df['registry_date']))
          .assign(location_name=lambda df: pd.Categorical(df['location_name']))
          .assign(scientific_species_name=lambda df: pd.Categorical(df['scientific_species_name']))
          .assign(popular_species_name=lambda df: pd.Categorical(df['popular_species_name']))
          .assign(species_wiki_slug=lambda df: pd.Categorical(df['species_wiki_slug']))
          .assign(city=lambda df: pd.Categorical(df.location_name.apply(lambda row: row.split("/")[0])))
          .assign(state=lambda df: pd.Categorical(df.location_name.apply(lambda row: row.split("/")[1])))
       )


# ## Initial exploration

# The data then can be summarized into?

# In[ ]:


# data dtypes
data.dtypes


# In[ ]:


# the first three rows
data.head(3)


# By inspecting the above samples, we can assume that we have the following measurement (quantificable) variables: comment_count, like_count, views_count.
# 
# We have as attributes: is_flagged and registry_date, location_name, home_location_id and species_id and author_id.
# 
# Given the lack of quantificable variables, the best course of action is to aggregate rows through some attribute metric with the aim of generating further measurement variables. The first thing that we can do is to count the unique occurences of each property:

# In[ ]:


attributes = ['location_id', 'home_location_id', 'species_id', 'author_id', 'city']

data[attributes].nunique()


# ## Author-centric analysis

# In[ ]:


author_data = data.groupby("author_id")


# In[ ]:


# Author registry count distribution
author_count = author_data.author_id.count()
author_count.hist(log=True, bins=100)


# In[ ]:


# Top 10 observing authors
author_count.sort_values(ascending=False).head(10)


# In[ ]:


# Distinct species observed by author
author_species = author_data.species_id.nunique()
author_species.hist(log=True, bins=100)


# In[ ]:


# Top 10 distinctivess observations authors
author_species.sort_values(ascending=False).head(10)


# In[ ]:


# Author registry count versus distinct species number
sns.scatterplot(x=author_count, y=author_species)


# In[ ]:


# Same of above, but on the log-space and in an distribution visualization
sns.jointplot(x=author_count.map(np.log), y=author_species.map(np.log), kind='hex')


# In[ ]:


# Fraction of registries versus distinct species
(author_species / author_count).hist(log=False)


# ## Species-centric analysis

# In[ ]:


species_data = data.groupby("popular_species_name")


# In[ ]:


# Most registered species
species_count = species_data.species_id.count()
species_count.sort_values(ascending=False).head(10)


# In[ ]:


# Species sighted by the most authors
species_author = species_data.author_id.nunique()
species_author.sort_values(ascending=False).head(10)


# In[ ]:


sns.scatterplot(x=species_count, y=species_author)


# In[ ]:


sns.regplot(x=species_count, y=species_author, order=2, marker='x', scatter_kws={'s': 1}, ci=95)


# In[ ]:


# Degree-two polynomial regression
(a, b, c) = np.polyfit(species_count, species_author, deg=2)

n_registries = 1000
n_authors = a * n_registries ** 2 + b * n_registries + c
print(n_authors)


# ## Location-centric analysis

# In[ ]:


# Generate an dict for mapping Wikiaves location_id to an name
unique_locs = data.location_id.unique()
dataloc = data.set_index("location_id")
loc_map = {}
for unique_loc in unique_locs:
    val = dataloc.loc[unique_loc].location_name
    if type(val) == str:
        loc_map[unique_loc] = val
    else:
        loc_map[unique_loc] = val.iloc[0]


# In[ ]:


# Get the home location name, and lower the location name
data.loc[:, 'location_name'] = data.location_name.str.lower()
data.loc[:, "home_location_name"] = data.home_location_id.map(loc_map)


# In[ ]:


# Import another Kaggle dataset with variables about Brazilian cities
cities_filepath = '/kaggle/input/brazilian-cities/BRAZIL_CITIES.csv'
cities_data = (pd.read_csv(cities_filepath, delimiter=';')
                 .assign(location_name=lambda df: df.apply(lambda x:"{}/{}".format(x.CITY, x.STATE).lower(), axis=1))
                 .set_index("location_name")
              )


# In[ ]:


# Top 10 birder cities in terms of registry per capita
registries_per_location = data.groupby("location_name").author_id.count()
cities_data['registry_count'] = registries_per_location
author_density = (cities_data.registry_count / cities_data.IBGE_RES_POP).sort_values(ascending=False)
author_density.head(10)


# In[ ]:


# Top 10 birder cities in terms of unique authors per capita
authors_per_location = data.groupby("location_name").author_id.nunique()
cities_data['author_count'] = authors_per_location
author_density = (cities_data.author_count / cities_data.IBGE_RES_POP).sort_values(ascending=False)
author_density.head(10)


# ### Movie

# In[ ]:


dates = pd.date_range(start='2019-01-01', end='2019-07-01', freq='3m')

for i in range(len(dates) - 1):
    date1 = dates[i]
    date2 = dates[i + 1]
    ind = data.registry_date >= date1
    ind &= data.registry_date < date2
    #ind &= data.home_location_name == "recife/pe"
    location_data = data[ind].groupby("location_name")
    loc_count = pd.DataFrame(location_data.author_id.count()).join(cities_data)
    z_s = (loc_count.author_id)
    loc_count = pd.DataFrame(location_data.author_id.nunique()).join(cities_data)
    z_c = (loc_count.author_id / loc_count.IBGE_RES_POP)
    loc_count = pd.DataFrame(location_data.author_id.nunique()).join(cities_data)
    (loc_count.author_id / loc_count.IBGE_RES_POP).sort_values(ascending=False).head(10)
    x = loc_count.LONG
    y = loc_count.LAT
    fig = plt.figure(figsize=(15, 10), dpi=50)
    plt.scatter(x, y, c=np.log(z_c), s=np.sqrt(z_s), alpha=0.5, cmap='viridis')
    plt.title("{} until {}".format(date1, date2))
    plt.clim(-13, -4)
    plt.xlim([-75, -31])
    plt.ylim([-36, 7])
    plt.colorbar()
    plt.savefig("{}.png".format(i))
    plt.show()
print("done")


# !tar -czvf recife.tar.gz *.png

# In[ ]:


def create_download_link(title = "Download CSV file", filename = "data.csv"):  
    html = '<a href={filename}>{title}</a>'
    html = html.format(title=title,filename=filename)
    return HTML(html)

# create a link to download the dataframe which was saved with .to_csv method
create_download_link(filename='recife.tar.gz')

