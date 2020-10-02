#!/usr/bin/env python
# coding: utf-8

# In this notebook I will have a closer look at the ocean-related variables.
# 
# First, let's create an object for accessing the dataset

# In[ ]:


from google.cloud import bigquery
client = bigquery.Client()
dataset_ref = client.dataset('noaa_icoads', project='bigquery-public-data')
dset = client.get_dataset(dataset_ref)


# Let's look one more time at the tables present in it

# In[ ]:


[i.table_id for i in client.list_tables(dset)]


# Also let's have a look at the field names and types

# In[ ]:


icoads_core_2017 = client.get_table(dset.table('icoads_core_2017'))
[i.name+", type: "+i.field_type for i in icoads_core_2017.schema]


# And, to finish inspectation, let's have a look at some data

# In[ ]:


schema_subset = [col for col in icoads_core_2017.schema if col.name in ('year', 'month', 'day', 'hour', 'latitude', 'longitude', 'sea_level_pressure', 'sea_surface_temp', 'present_weather')]
results = [x for x in client.list_rows(icoads_core_2017, start_index=100, selected_fields=schema_subset, max_results=10)]


# In[ ]:


for i in results:
    print(dict(i))


# As we have a 5TB per month quota, it is a good practice to analyze what amount of data the SQL query will scan (not return!!!)
# 
# First function returns same information as second, but in human-readable format

# In[ ]:


def estimate_gigabytes_scanned_h(query, bq_client):
    # see https://cloud.google.com/bigquery/docs/reference/rest/v2/jobs#configuration.dryRun
    my_job_config = bigquery.job.QueryJobConfig()
    my_job_config.dry_run = True
    my_job = bq_client.query(query, job_config=my_job_config)
    BYTES_PER_GB = 2**30
    print("This query takes "+str(round(my_job.total_bytes_processed / BYTES_PER_GB, 2))+" GB of quota.")
    
def estimate_gigabytes_scanned(query, bq_client):
    # see https://cloud.google.com/bigquery/docs/reference/rest/v2/jobs#configuration.dryRun
    my_job_config = bigquery.job.QueryJobConfig()
    my_job_config.dry_run = True
    my_job = bq_client.query(query, job_config=my_job_config)
    BYTES_PER_GB = 2**30
    return my_job.total_bytes_processed / BYTES_PER_GB


# In[ ]:


estimate_gigabytes_scanned_h("SELECT sea_level_pressure FROM `bigquery-public-data.noaa_icoads.icoads_core_1662_2000`", client)
estimate_gigabytes_scanned("SELECT sea_level_pressure FROM `bigquery-public-data.noaa_icoads.icoads_core_1662_2000`", client)


# Let's have a look at the Gulf Stream region

# In[ ]:


QUERY = """
        SELECT latitude, longitude, sea_surface_temp, wind_direction_true, amt_pressure_tend,  air_temperature, sea_level_pressure, wave_direction, wave_height, timestamp
        FROM `bigquery-public-data.noaa_icoads.icoads_core_2017`
        WHERE longitude > -74 AND longitude <= -44 AND latitude > 36 AND latitude <= 65 AND wind_direction_true <= 360
        """


# In[ ]:


estimate_gigabytes_scanned_h(QUERY, client)


# In[ ]:


import pandas as pd


# Let's execute our QUERY and put result into Pandas dataframe.
# (**uncomment for execution**)

# In[ ]:


df = client.query(QUERY).to_dataframe()


# In[ ]:


df.size


# In[ ]:


df.head(10)


# In[ ]:


print(df.latitude.size, df.longitude.size)


# Let's draw coordinates according to [instruction](https://matplotlib.org/basemap/users/examples.html)

# In[ ]:


from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np
# set up orthographic map projection with
# perspective of satellite looking down at 50N, 100W.
# use low resolution coastlines.
map = Basemap(projection='ortho',lat_0=45,lon_0=-60,resolution='l')
# draw coastlines, country boundaries, fill continents.
map.drawcoastlines(linewidth=0.25)
map.drawcountries(linewidth=0.25)
map.fillcontinents(color='coral',lake_color='aqua')
# draw the edge of the map projection region (the projection limb)
map.drawmapboundary(fill_color='aqua')
# draw lat/lon grid lines every 30 degrees.
map.drawmeridians(np.arange(0,360,30))
map.drawparallels(np.arange(-90,90,30))
# make up some data on a regular lat/lon grid.
lats = df['latitude'].values
lons = df['longitude'].values
x, y = map(lons, lats)
# contour data over the map.
cs = map.scatter(x,y)
plt.title('contour lines over filled continent background')
plt.show()


# Let's count how much [NaN](https://en.wikipedia.org/wiki/NaN) values we have in each column:

# In[ ]:


df_nans = df.isnull().sum(axis = 0)
print ("Number of NaN's by column:\n\n", df_nans, sep = "") # \n means newline, sep = "" removes space between elements of print command
# Now I want to know the percentage of NaN's in each column, so I need to get values of df_nans and divide by number of recordings in df. Let's look what type df_nans has
print ("\nType of df_nans:\n\n", type(df_nans), sep = "")
# After looking at the output of dir() command excluding built-in and private elements I understood that values of df_nans are accessable with .values element
# (uncomment next command to look at dir() output), also full dir() available if you want to look at built-in and private elements
# print ([f for f in dir(df_nans) if not f.startswith('_')]) # print (dir(df_nans))
# Let's divide df_nans by df row vount and output it with per cent sign
df_nans_perc = 100*df_nans/len(df.index)
pd.options.display.float_format = '{:,.2f}%'.format
print("\nPer cent of NaN's by column:\n\n", df_nans_perc, sep = "")
pd.options.display.float_format = '{:,.2f}'.format


# In[ ]:


#Let's create separate dataset with latitude, longitude, sea_surface_temp, wind_direction_true, air_temperature, sea_level_pressure, timestamp:
df_no_nans =df[['latitude', 'longitude', 'sea_surface_temp', 'wind_direction_true', 'air_temperature', 'sea_level_pressure', 'timestamp']]
#And now let's remove all rows with NaN elements:
print("Rows in df_no_nans:\nBefore dropping NaN's:", len(df_no_nans.index))
df_no_nans = df_no_nans.dropna()
print("After:                ", len(df_no_nans.index))


# In[ ]:


list(df_no_nans)


# In[ ]:


import seaborn as sns
sns.heatmap(df_no_nans.corr(), 
        xticklabels=df_no_nans.corr().columns,
        yticklabels=df_no_nans.corr().columns)


# Obviously, air temperature correlates with sea surface temperature.
# 
# As little region was chosen, a correlation between coordinates also has place and there is a little anticorrelation between latitude and temperature of sea surface and air.
