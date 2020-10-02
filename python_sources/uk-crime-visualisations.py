#!/usr/bin/env python
# coding: utf-8

# # Introduction
# In this notebook, we will explore the [Recorded Crime Data at the Police Force Area Level](
# https://www.kaggle.com/r3w0p4/recorded-crime-data-at-police-force-area-level) dataset.
# This dataset contains criminal offences that occurred in the United Kingdom and are grouped into rolling 12-month
# totals.
# 
# We will follow the [Data Science Methodology](https://www.ibmbigdatahub.com/blog/why-we-need-methodology-data-science)
# as the process for our analysis.
# We will skip the first four steps (i.e. business understanding, analytic approach, data requirements, data collection)
# because they are not applicable to our analysis.

# # Data Understanding
# We will be using data from the [Office for National Statistics](
# https://www.ons.gov.uk/), the United Kingdom's largest independent producer of official statistics.
# Our first step is to try to understand the data, and to ensure that data is correctly formatted and of good quality.
# 
# We will start by importing dependencies and reading the data from the `.csv` file to a dataframe.

# In[ ]:


import matplotlib
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import pandas as pd
import numpy as np
import folium
import requests
import json
import os


# In[ ]:


path_data = os.path.join('..', 'input', 'recorded-crime-data-at-police-force-area-level', 'rec-crime-pfa.csv')
path_geo = os.path.join('..', 'input', 'pfasgeo2017', 'pfas-geo-2017.json')

df = pd.read_csv(path_data)
df.head()


# ## Overview
# 
# We will now check the dataset size and whether the data types are being interpreted correctly.

# In[ ]:


print(df.shape)
print(df.dtypes)


# We see that `12 months ending` is being interpreted as type `object`.
# However, this is a date type, so we will change its data type to `datetime64[ns]` and indicate that the date is in the
# format `%d/%m/%Y`.

# In[ ]:


df['12 months ending'] = pd.to_datetime(df['12 months ending'], format='%d/%m/%Y')

df.head()


# In fact, we only really need the year from `12 months ending`, so we will create a new column `year` of type `int64`
# that contains only the year that the offences were committed and drop `12 months ending`.
# We do not need `Region` either, so we will drop that column as well.

# In[ ]:


df['year'] = pd.DatetimeIndex(df['12 months ending']).year
df.drop(['12 months ending'], inplace=True, axis=1)
df.drop(['Region'], inplace=True, axis=1)

df.head()


# Now let's rename the column headers so that they are short and consistent.

# In[ ]:


df.rename(inplace=True, columns={
    'PFA': 'pfa',
    'Offence': 'offence',
    'Rolling year total number of offences': 'total'
})

df.head()


# We will now use `.describe()` to get quick insights into our data.

# In[ ]:


df.describe(include='all')


# ## Column Checking
# 

# We will go through the data of each column and fix any erroneous data that we find.

# In[ ]:


def quick_look(col_name):
    colunique = np.sort(df[col_name].unique())
    colnull = df[col_name].isnull().values.sum()
    
    print(colunique)
    print('Count unique:', len(colunique))
    print('Count null:', colnull)


# ### pfas

# In[ ]:


quick_look('pfa')


# For our assessment, we will only consider *territorial* PFAs that are associated with one-or-more counties of England
# and Wales.

# In[ ]:


rows_before = df.shape[0]

df = df[df['pfa'] != 'Action Fraud']
df = df[df['pfa'] != 'British Transport Police']
df = df[df['pfa'] != 'CIFAS']
df = df[df['pfa'] != 'UK Finance']

rows_after = df.shape[0]

quick_look('pfa')
print('Rows from:', rows_before, 'to:', rows_after, 'dropped:', (rows_before - rows_after))


# We now have data for all [43 territorial PFAs](https://www.police.uk/forces/) in England and Wales.

# ### offence

# In[ ]:


quick_look('offence')


# Some of these crimes look very similar (e.g. `Domestic burglary`, `Non-domestic burglary`).
# To simplify our analysis, we will combine similar offences.

# In[ ]:


df.loc[df['offence'] == 'Domestic burglary', 'offence'] = 'Burglary'
df.loc[df['offence'] == 'Non-domestic burglary', 'offence'] = 'Burglary'
df.loc[df['offence'] == 'Non-residential burglary', 'offence'] = 'Burglary'
df.loc[df['offence'] == 'Residential burglary', 'offence'] = 'Burglary'

df.loc[df['offence'] == 'Bicycle theft', 'offence'] = 'Theft'
df.loc[df['offence'] == 'Shoplifting', 'offence'] = 'Theft'
df.loc[df['offence'] == 'Theft from the person', 'offence'] = 'Theft'
df.loc[df['offence'] == 'All other theft offences', 'offence'] = 'Theft'

df.loc[df['offence'] == 'Violence with injury', 'offence'] = 'Violence'
df.loc[df['offence'] == 'Violence without injury', 'offence'] = 'Violence'

df = df.groupby(['pfa', 'offence', 'year']).sum().reset_index()

quick_look('offence')


# ### total

# In[ ]:


quick_look('total')


# The total crime count should be 0 or more.
# Let's remove all rows with an invalid crime count.

# In[ ]:


rows_before = df.shape[0]
df = df[df['total'] >= 0]
rows_after = df.shape[0]

quick_look('total')
print('Rows from:', rows_before, 'to:', rows_after, 'dropped:', (rows_before - rows_after))


# We will show a quick visualisation of the distribution of total crimes.

# In[ ]:


df['total'].plot(kind='hist', color='red')
plt.show()


# ### year

# In[ ]:


quick_look('year')


# We see that all of the years are within the expected range.
# Data were collected differently before 2007, and so we will only consider data between 2007 and 2018.

# In[ ]:


rows_before = df.shape[0]
df = df[df['year'] >= 2007]
rows_after = df.shape[0]

quick_look('year')
print('Rows from:', rows_before, 'to:', rows_after, 'dropped:', (rows_before - rows_after))


# ## Initial Data Insights

# We want to map our crime data to the PFAs of England and Wales, so we will use the GeoJSON file from [here](https://ckan.publishing.service.gov.uk/dataset/police-force-areas-december-2017-ew-bgc/resource/446cea20-c981-498e-aad2-f52965a26c0e) to do this.
# Our Notebook already has this downloaded, but you can find the direct `.geojson` file [here](http://geoportal1-ons.opendata.arcgis.com/datasets/bb12117b37134a03874c55175cf7f4bc_2.geojson).

# In[ ]:


pfageo = os.path.abspath(path_geo)
print(pfageo)


# We will make a convenience function `map_britain` to easily map data.
# 
# Note: there is a known issue with rendering complex maps in Chrome (see [here](https://github.com/python-visualization/folium/issues/812)) that might result in blank outputs.
# If this occurs, try another browser e.g. Firefox.

# In[ ]:


def map_britain(data):
    britain = folium.Map(location=[52.355518, -1.174320],
                         zoom_start=6,
                         tiles='cartodbpositron')
    folium.Choropleth(
        geo_data=pfageo,
        name='choropleth',
        data=data,
        columns=['pfa', 'total'],
        key_on='feature.properties.pfa17nm',
        fill_color='YlGn',
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name='# Crimes'
    ).add_to(britain)

    folium.LayerControl().add_to(britain)
    return britain


# ### Weapon Possession Offences
# 
# Let's extract weapon possession offence data.

# In[ ]:


label_weapon = 'Possession of weapons offences'
df_weapon = df.loc[df['offence'] == label_weapon]


# We will map the total number of offences in 2018.

# In[ ]:


map_britain(df_weapon.loc[df_weapon['year'] == 2018])


# PFAs that include major cities (e.g. London, Manchester, Birmingham, Leeds) appear to suffer from weapon possession offences more than most other places.
# Let's compare the trend of offences for the PFAs over the years.

# In[ ]:


labels_weapon_high = ['Metropolitan Police', 'Greater Manchester', 'West Midlands', 'West Yorkshire']
df_weapon_high = df_weapon.loc[df_weapon['pfa'].isin(labels_weapon_high)]

sns.lineplot(data=df_weapon_high, x='year', y='total', hue='pfa')


# We see that offences were on a steep decline in London (i.e. `Metropolitan Police`) from `2008` but started to increase again from approximately `2013` for all PFAs.

# # TO BE CONTINUED

# In[ ]:




