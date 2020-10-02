#!/usr/bin/env python
# coding: utf-8

# In[23]:


import warnings
warnings.filterwarnings('ignore')
from google.cloud import bigquery
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
matplotlib.style.use('ggplot')
matplotlib.rcParams['figure.figsize'] = (16, 20)
import pandas_profiling
import cartopy.crs as ccrs
import pandas as pd
# https://github.com/SohierDane/BigQuery_Helper
from bq_helper import BigQueryHelper


# 
# 
# bq_helper requires the creation of one BigQueryHelper object per dataset. Let's make one now. We'll need to pass it two arguments:
# 
#   -  The name of the BigQuery project, which on Kaggle should always be bigquery-public-data
#   -  The name of the dataset, which can be found in the dataset description.
# 
# 

# In[24]:


# Use  bq_helper to create a BigQueryHelper object
noaa_gsod = BigQueryHelper(active_project= "bigquery-public-data", 
                              dataset_name= "noaa_gsod")


# The first thing I like to do with a dataset is to list all of the tables. 

# In[25]:


noaa_gsod.list_tables()


# ## Stations
# 
# I will start my basic EDA with the `stations` table.  Let's get some details about its columns by viewing the table schema. 

# In[26]:


noaa_gsod.table_schema("stations")


# In[27]:


get_ipython().run_cell_magic('time', '', 'noaa_gsod.head("stations", num_rows=20)')


# ## Checking the size of a query before running it

# Now that we have an idea of what data in `stations` looks like, we are ready to write a simple query. We should check how much memory it will scan. It is a good habit to get into for when you are working with large datasets hosted on BigQuery. 

# In[28]:


QUERY = """SELECT name, country, lat, lon, elev, begin, end
            FROM `bigquery-public-data.noaa_gsod.stations` """


# In[29]:


noaa_gsod.estimate_query_size(QUERY)


# The error above is due to the fact that `end` is a reversed keyword and one of columns in the table is named `end`. Because the renaming of the column is not possible, we will need to wrap the offending identifier in backticks. **

# In[30]:


QUERY = """SELECT name, country, lat, lon, elev, begin, `end`
            FROM `bigquery-public-data.noaa_gsod.stations` """
noaa_gsod.estimate_query_size(QUERY)


# Running this query will take around `1.80 MB`. (The query size is returned in gigabytes.)
# > Important: When you're writing your query, make sure that the name of the table (next to FROM) is in back ticks (`), not single quotes ('). The reason for this is that the names of BigQuery tables contain periods in them, which in SQL are special characters. Putting the table name in back ticks protects the table name, so it's treated as a single string instead of being run as code.

# ## Running the query

# Now that we've made sure that we are not scanning several terabytes of data, we are ready to actually run our query. 
# 
# We have two methods available to help you do this:
# 
#    -  `BigQueryHelper.query_to_pandas(query)`: This method takes a query and returns a Pandas dataframe.
#    - `BigQueryHelper.query_to_pandas_safe(query, max_gb_scanned=1)`: This method takes a query and returns a Pandas dataframe only if the size of the query is less than the upperSizeLimit (1 gigabyte by default).
# 

# In[31]:


stations = noaa_gsod.query_to_pandas_safe(QUERY, max_gb_scanned=0.1)


# In[32]:


stations.head()


# Now that we have a Pandas dataframe, it is time to dive in our data analysis.
# Let's Generate descriptive statistics that summarize our dataframe.

# In[33]:


stations.info()


# In[34]:


stations.describe(include=["O"])


# In[35]:


stations.describe()


# Let's use [pandas-profiling package](https://github.com/pandas-profiling/pandas-profiling) to generate profile reports from stations dataframe. 

# In[36]:


pandas_profiling.ProfileReport(stations)


# 
# ## Maps with Layers
# 
# - Geopandas provides a high-level interface to the matplotlib library for making maps.
# - Mapping shapes is as easy as using the plot() method on a GeoSeries or GeoDataFrame.
# 
# Let's overlay stations over world GeoDataFrame to better visualize theier geographical locations.
# 
# 
#  Before combining maps, however, remember to always ensure they share a common CRS (so they will allign).

# In[37]:


from shapely.geometry import Point
import geopandas as gpd


# In[38]:


# Geopandas = pandas + (shapely * projection)
gdf = gpd.GeoDataFrame(stations, geometry=None)         .set_geometry([Point(r.lon, r.lat) for _, r in stations.iterrows()],
                       crs={"init": "EPSG:4326"})


# In[39]:


gdf.head()


# In[63]:


world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
world.plot();


# In[64]:


# Now we can overlay stations over country outlines
base = world.plot(color='white')
plt.title('9000 Weather Stations')
gdf.plot(ax=base, marker='*', color='royalblue', markersize=5, alpha=0.5);


# **And Yes, there is quite a lot of stations in the United States!**
