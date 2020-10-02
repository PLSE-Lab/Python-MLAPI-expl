#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[3]:


loansDF = pd.read_csv('../input/kiva_loans.csv')
loansDF.head()


# In[4]:


loansDF.shape


# Any NaN's, besides in 'tags' ?

# In[5]:


loansDF[loansDF.drop('tags', axis=1).isnull().any(axis=1)]


# Looks like some regions are NaNs:

# In[6]:


#entries without region name:
loansDF[loansDF["region"].isnull()].shape


# In[7]:


loansDF.drop(loansDF[loansDF["region"].isnull()].index).shape


# ## MPI Regions 

# In[8]:


mpiregionsDF = pd.read_csv("../input/kiva_mpi_region_locations.csv")
mpiregionsDF.sort_values("ISO").head()


# In[9]:


mpiregionsDF[["country","region"]].describe()


# Why are there 1008 country entries but only 984 regions?   
# I guess, that some region names repeat in different countries, or are (country,region) combinations not unique?

# In[10]:


#Let's look at nans:
mpiregionsDF[mpiregionsDF.isnull().any(axis=1)]


# There are lots of nans in the lat/long clumns, so those won't be helpful

# In[11]:


mpiClean = mpiregionsDF.drop(mpiregionsDF[mpiregionsDF[["lat","lon"]].isnull().any(axis=1)].index)


# In[12]:


# check
mpiClean[mpiClean.isnull().any(axis=1)]


# In[13]:


mpiClean.shape


# In[14]:


mpiClean[["country","region"]].describe()


# In[15]:


mpiClean.head()


# ## Merge loans and MPI regions
# Merge the two data sets by using country and region as keys

# In[16]:


#match geo data with country and region names
loansgeoDF = pd.merge(loansDF,mpiClean, on=["country","region"], how="inner", sort=False, validate="many_to_one")
#loansgeoDF.head()


# Here we validated that key combinations in mpiClean are unique. By choosing "inner", we dropped key combinations (e.g. those with region=nan) that are not represented in the right dataset

# In[17]:


print(loansgeoDF.shape)


# It looks like out of 671205 rows in loansDF, only 50953 remain after merging, i.e. only that many (country,region) key combinations are found in the MPI regions database.   
# What are the country,region combintions in the lons database that are not in MPI regions?

# In[18]:


loansgeoAll = pd.merge(loansDF,mpiClean, on=["country","region"], how="left", sort=False, validate="many_to_one")
print(loansgeoAll.shape)


# In[19]:


#show those with lat, long = nan:
loansgeoAll[["country","region","lender_count"]][loansgeoAll[["lat","lon"]].isnull().any(axis=1)].groupby(["country","region"]).aggregate(sum)


# These (country,region) combinations, i.e. most of those in the loans database, are not included in the MPI region information.
# 

# ## Visualizing regional MPI and corresponding loans
# For now we focus only on the 50953 loan entries with full MPI information

# In[20]:


from bokeh.io import output_file, show, output_notebook
from bokeh.models import GeoJSONDataSource
from bokeh.plotting import figure
from bokeh.tile_providers import STAMEN_TONER, STAMEN_TERRAIN_RETINA
from pyproj import Proj, transform
output_notebook()


# In[21]:


bound = 20000000 #meters
fig = figure(tools='pan, wheel_zoom', x_range=(-bound, bound), y_range=(-bound/5, bound/5))
fig.axis.visible = False
fig.add_tile(STAMEN_TERRAIN_RETINA)

lon =  loansgeoDF["lon"].values
lat =  loansgeoDF["lat"].values
lenders = loansgeoDF["lender_count"]

from_proj = Proj(init="epsg:4326")
to_proj = Proj(init="epsg:3857")

x, y = transform(from_proj, to_proj, lon, lat)

fig.circle(x, y, size=lenders/20., alpha=0.6,
          color= "blue",   #;  {'field': 'rate', 'transform': color_mapper},
       ) #fill_alpha=0.7, line_color="white", line_width=0.5)

#output_file("stamen_toner_plot.html")
show(fig)


# In[ ]:





# In[ ]:





# In[ ]:





# Good, checked that _merge_ is always "both", i.e. perfect matching between tables
