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


sp_data = (data.where(lambda df: df.registry_date.dt.year.isin([2014, 2015, 2016, 2017, 2018]))
               .where(lambda df: df.state == "SP")
               .dropna(subset=["state", "registry_date"]))


# In[ ]:


out = []

for i, species_data in sp_data.groupby("species_wiki_slug"):
    x = species_data.registry_date.dt.month
    
    
    el = {"slug": i,
          "circmean": st.circmean(x, high=13, low=1),
          "circstd": st.circstd(x, high=13, low=1),
          "length": len(x)}
    
    out.append(el)


# In[ ]:


summary = (pd.DataFrame(out)
             .sort_values("circstd", ascending=True)
             .where(lambda df: df.length > 10)
             .dropna()
             .set_index("slug"))


# In[ ]:


summary.circstd.hist(bins=100)


# In[ ]:


summary.sort_values("circstd", ascending=True).assign(fator=lambda df: 3 / df.circstd).head(80).to_csv("migratory_birds.csv")


# In[ ]:


aa = sp_data.where(lambda df: df.species_wiki_slug == "tovacucu").dropna()
aa.groupby(aa.registry_date.dt.month).count().registry_date.plot()


# In[ ]:


summary.loc['suiriri']


# In[ ]:


aa = sp_data.where(lambda df: df.species_wiki_slug == "suiriri").dropna()
aa.groupby(aa.registry_date.dt.month).count().registry_date.plot()


# In[ ]:


aa = sp_data.where(lambda df: df.species_wiki_slug == "mariquita").dropna()
aa.groupby(aa.registry_date.dt.month).count().registry_date.plot()


# In[ ]:


summary


# In[ ]:


st.circmean(aa.registry_date.dt.month)


# In[ ]:


st.circstd(aa.registry_date.dt.month)


# In[ ]:




