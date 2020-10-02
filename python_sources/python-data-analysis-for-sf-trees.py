#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')

import os
print(os.listdir("../input"))


# In[ ]:


sftree = pd.read_csv("../input/san_francisco_street_trees.csv", header=0)
sftree.columns = ['Address', 'Care_assistant', 'Care_taker', 'dbh', 'Latitude', 'Legal_status', 'Location', 'Longitude', 'Permit_notes', 'Plant_date', 'Plant_type', 'Plot_size', 'Site_info', 'Site_order', 'Species', 'Tree_ID', 'X_coordinate', 'Y_coordinate']
sftree.head()


# In[ ]:


sftree.shape


# In[ ]:


# Examine data
print(len(sftree), " rows x ", len(sftree.columns), " columns")
print(sftree.columns)
sftree.info()
sftree.describe()


# In[ ]:


# Make a copy of the data to work with.
sf_trees = sftree.copy()


# In[ ]:


# Clean data
sf_trees = sf_trees.loc[sf_trees['Latitude'] < 38.000000] # For some reason this goes all the way up to Seattle??
sf_trees = sf_trees.loc[sf_trees['Species'] != 'Tree(s) ::'] # Clean this bit of data


# In[ ]:


# This looks much more reasonable
sf_trees.describe()


# In[ ]:


# Count trees by species
species_count = sf_trees.groupby('Species').Tree_ID.count()
print("There are", len(species_count), "species of tree in San Francisco.")


# In[ ]:


# Top 10 tree species in SF bar chart
top_ten_species = species_count.nlargest(10)
ax = top_ten_species.plot.bar()


# In[ ]:


# Top 10 tree species in SF pie chart
ax = top_ten_species.plot.pie()


# In[ ]:


# Get dataframe where only popular trees are shown
popular_species = top_ten_species.index.tolist()
popular_species_df = sf_trees.loc[sf_trees['Species'].isin(popular_species)]
# display(popular_species_df)


# In[ ]:


# Basic scatterplot of the top ten most popular trees in SF.
x, y = popular_species_df.Latitude.tolist(), popular_species_df.Longitude.tolist()
plt.scatter(x,y)


# In[ ]:


sns.lmplot(x="Latitude", y="Longitude", col="Species", data=popular_species_df,
          col_wrap=2, lowess=True)

