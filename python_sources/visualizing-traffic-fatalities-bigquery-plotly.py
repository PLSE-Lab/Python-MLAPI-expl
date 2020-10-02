#!/usr/bin/env python
# coding: utf-8

# Some necessary import statements.  Access the US Traffic Fatality Records BigQuery database stored on Kaggle and uses plotly for it's easy geographical plotting.

# In[ ]:


import pandas as pd
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.figure_factory as ff
import plotly.graph_objs as go
from google.cloud import bigquery
from bq_helper import BigQueryHelper
import matplotlib.cm as cm
import numpy as np


# We start by forming our query.  This is a string which contains the SQL for the query.  This is a fairly basic SQL query, just to illustrate the basics and how to access from a database.  Much more complex queries are possible, but require more SQL knowledge.
# 
# Our query is designed to get the total number of fatal accidents in each county.
# 
# The main parts of this SQL query are:
#     
#     SELECT - This says what columns we want.  Notice we can also get computed columns.  Here we tell SQL to SUM the number_of_fatalities and the AS says to return that as a column named fatalities. How do we tell it what we want to sum over (e.g. how does it know we don't just want the total number of fatal accidents reported)? -- This is what GROUP BY (see below) is used for
#     
#     FROM - This says what table we are selecting from.  Many databases will have a lot of tables (this one included), so SQL needs to know which table you want to pull from.
#     
#     WHERE - This says to only select records where some conditions are met.
#     
#     GROUP BY - This is used with aggregate functions (like SUM), to specify what we wish to SUM over.   All of the records  with the same entries for each of the columns listed in GROUP BY will be grouped together, and the aggregate function will be performed over each group.  All columns not being aggregated must be specified in the GROUP BY clause.
#     
#     ORDER BY - This just tells how the results should be ordered, it gives a column name and ASC or DESC for ascending and descending, respecitvely.  Additional column names can be given with them separated by columns (to first order by one column, then by another).

# In[ ]:


QUERY = """SELECT
  state_name,
  state_number,
  county,
  SUM(number_of_fatalities) AS fatalities
FROM
  `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
WHERE
  county < 997
GROUP BY
  county, state_number, state_name
ORDER BY
  fatalities DESC"""


# In[ ]:


bq_assistant = BigQueryHelper("bigquery-public-data", "nhtsa_traffic_fatalities")


# Before running, we should check the size of our query (in many cases you pay by the size or have a limited amount you can query in a set period of time).

# In[ ]:


print(bq_assistant.estimate_query_size(QUERY))


# This is about 1MB, so there should be no issue running many variations.  (Kaggle limits to 5TB of queries over a 30 day period)

# In[ ]:


df_2015 = bq_assistant.query_to_pandas(QUERY)


# Let's look at what dataframe we get from this query.

# In[ ]:


df_2015.head()


# The 'state_number' and 'county' columns are together used to identify the county within the entire US.  The US (and the plotly library) uses a 5 digit FIPS code (2 for state number and 3 for county number), where state_numbers like 6 for California are treated as 06 (and likewise for counties with less than 3 digits).  So, for the first county with 651 fatalities, the identifying code is 06037.  The below adds these formatted codes as a column to the dataframe.

# In[ ]:


df_2015_mod = df_2015.copy()
df_2015_mod['state_number'] = df_2015_mod['state_number'].apply(lambda x: str(x).zfill(2))
df_2015_mod['county'] = df_2015_mod['county'].apply(lambda x: str(x).zfill(3))
df_2015_mod['FIPS'] = df_2015_mod['state_number'] + df_2015_mod['county']

df_2015_mod.head()


# One way to visualize this information is via a heat map.  The following uses plotly to create a choropleth where we visualize the fatal accidents by county.

# In[ ]:


fips = df_2015_mod['FIPS'].tolist()
values = df_2015_mod['fatalities'].tolist()

fig = ff.create_choropleth(
    fips = fips, values = values, scope = ['usa'],
    show_state_data = False,
    show_hover = True, centroid_marker = {
        'opacity': 0
    },
    asp = 2.9,
    title = 'USA by Traffic Fatalities',
    legend_title = '# Traffic Fatalities'
)
iplot(fig, filename = 'choropleth_full_usa')


# Make this prettier by specifying bins and a colorscale (to prevent every unique value from appearing differently in the colorbar).

# In[ ]:


cmap = cm.get_cmap('Greens')
num_bins = 12

colorscale = cmap(np.linspace(0,1,num_bins+1))[:,:-1]
colorscale = ['rgb' + str(tuple(np.round(colorscale[i,:]*255).astype(int))) for i in range(len(colorscale))]
endpts = [1,10,25,40,60,80,120,170,230,300,400,600]

fig = ff.create_choropleth(
    fips = fips, values = values, scope = ['usa'],
    show_state_data = False,
    binning_endpoints = endpts,
    colorscale = colorscale,
    show_hover = True, centroid_marker = {
        'opacity': 0
    },
    asp = 2.9,
    title = 'USA by Traffic Fatalities',
    legend_title = '# Traffic Fatalities'
)
iplot(fig, filename = 'choropleth_full_usa_bins')


# 
# 
