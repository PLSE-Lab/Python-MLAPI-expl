#!/usr/bin/env python
# coding: utf-8

# This Kernel is under construction.

# # 1. Retrieving the Data
# ## 1.1 Loading libraries

# In[14]:


# Loading libraries

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.basemap import Basemap


# ## 1.2 Reading the data

# In[7]:


kiva_loans_data = pd.read_csv("../input/kiva_loans.csv")
kiva_mpi_locations_data = pd.read_csv("../input/kiva_mpi_region_locations.csv")
loan_theme_ids_data = pd.read_csv("../input/loan_theme_ids.csv")
loan_themes_by_region_data = pd.read_csv("../input/loan_themes_by_region.csv")


# # 2. Data review
# ## 2.1 Tables overview
# ### kiva_loans.csv

# In[ ]:


kiva_loans_data.head()


# ### kiva_mpi_region_locations.csv

# In[ ]:


kiva_mpi_locations_data.head()


# ### loan_theme_ids.csv

# In[ ]:


loan_theme_ids_data.head()


# ### loan_themes_by_region.csv

# In[ ]:


loan_themes_by_region_data.head()


# ## 2.2 Statistical Data overview
# ### kiva_loans.csv

# In[ ]:


kiva_loans_data.drop(['id'], axis=1).describe()


# In[ ]:


kiva_loans_data.drop(['id'], axis=1).describe(include=['O'])


# ### kiva_mpi_region_locations.csv

# In[ ]:


kiva_mpi_locations_data.drop(['geo'], axis=1).describe(include=['O'])


# ### loan_themes_by_region.csv

# In[ ]:


loan_themes_by_region_data.drop(['Loan Theme ID', 'geocode_old', 'geocode', 'geo', 'mpi_geo'], axis=1).describe(include=['O'])


# ## 2.3 Top sectors of loans

# In[11]:


plt.figure(figsize=(13,9))
sectors = kiva_loans_data['sector'].value_counts()
sns.barplot(y=sectors.index, x=sectors.values)
plt.xlabel('Number of loans', fontsize=20)
plt.ylabel("Sectors", fontsize=20)
plt.title("Number of loans per sector", size=30)
plt.show()


# # 3. Geographical visualization of data
# ## 3.1 Locations of loans' provision
# 

# In[15]:


map = Basemap()

lat = kiva_mpi_locations_data["lat"].tolist()
lon = kiva_mpi_locations_data["lon"].tolist()

x,y = map(lon,lat)

plt.figure(figsize=(15,8))
map.plot(x,y,"go",color ="orange",markersize =6,alpha=.6)
map.shadedrelief()

