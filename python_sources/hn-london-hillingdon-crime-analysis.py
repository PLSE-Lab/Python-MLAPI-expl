#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import bq_helper
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


london = bq_helper.BigQueryHelper(active_project="bigquery-public-data", dataset_name="london_crime")


# In[ ]:


# Query to select Hillingdons's crime stats by year and month
hillingdon_crime_query = """
SELECT year, month, sum(value) AS `total_crime`
FROM `bigquery-public-data.london_crime.crime_by_lsoa`
WHERE borough = 'Hillingdon'
GROUP BY year, month;
        """

# Perform and store the query results 
hillingdon_crime = london.query_to_pandas_safe(hillingdon_crime_query)
hillingdon_crime.head()


# In[ ]:


hillingdon_crime.describe().total_crime


# We have 108 months of data here (9 years), with a mean of 1,941 recorded crimes per month, but a range of over 1000. 

# In[ ]:


hillingdon_crime.index


# In[ ]:


hillingdon_crime['date'] = pd.to_datetime(hillingdon_crime.year.map(str) + '-' + hillingdon_crime.month.map(str), format = '%Y-%m')
hillingdon_crime.set_index('date', inplace=True)


# In[ ]:


hillingdon_crime.total_crime.plot(figsize=(15, 6))
plt.title('Total crime in Camden')
plt.ylabel('Total crime per month')
plt.xlabel('')
plt.show()


# It appears as though crime was roughly on the decline between 2008 and 2011, and then dropped steeply after 2012, before re-establishing an upward trend post-2014. It also appears to fluctuate wildly throughout the months, so I imagine there is some seasonal impact here.

# In[ ]:


period_1417 = hillingdon_crime.loc['2014':'2017',]


# In[ ]:


period_1417.total_crime.plot(figsize=(15, 6))
plt.title('Total crime in Hillingdon between 2014-2017')
plt.ylabel('Total crime per month')
plt.show()


# In[ ]:


hillingdon_major_query = """
SELECT year, month, major_category, sum(value) AS `total_crime`
FROM `bigquery-public-data.london_crime.crime_by_lsoa`
WHERE borough = 'Hillingdon'
GROUP BY year, month, major_category
ORDER BY year, month;
        """
# Perform and store the query results
hillingdon_major = london.query_to_pandas_safe(hillingdon_major_query)
hillingdon_major.head()


# In[ ]:


hillingdon_major['date'] = pd.to_datetime(hillingdon_major.year.map(str) + '-' + hillingdon_major.month.map(str), format = '%Y-%m')
hillingdon_major.drop(columns = ['year', 'month'], inplace = True)
hillingdon_major.head()


# In[ ]:


hillingdon_major_pivot = hillingdon_major.pivot(index = 'date', columns = 'major_category', values = 'total_crime')


# In[ ]:


hillingdon_major_pivot.plot(subplots = True, figsize=(15, 15))
plt.show()


# In[ ]:




