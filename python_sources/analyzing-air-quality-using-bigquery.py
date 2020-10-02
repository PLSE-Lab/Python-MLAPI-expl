#!/usr/bin/env python
# coding: utf-8

# ****WORK IN PROGRESS, *Any Feedback welcome*.**
# **This is my very first kernel and I wanted to experiment with Big Query. I decided to go with this data set because I think it's fascinating and given the levels of pollution we may get to see very interesting insights**
# 
# Abstract:
# 

# In[ ]:


import numpy as np
import pandas as pd
from google.cloud import bigquery
from bq_helper import BigQueryHelper
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='ticks')


# **Questions I'd Like to explore:
# 1. What do each of the files mean?
# 2. How much of a speedup do I get with BigQuery?
# 3. Do I reach the data limit for BigQuery?
# 4. Does Air Quality get worst as years go by?
# 5. Do Industrial cities have a much worst average quality?
# 6. Seasonal Air Quality Trends
# 7. Day-Night Air Quality Trends
# 8.  Land locked vs Coastal Air Quality
# 9. Can we find Error/Noisy Data?**

# In[ ]:


bq_assistant = BigQueryHelper("bigquery-public-data", "epa_historical_air_quality")
bq_assistant.head("pm25_frm_daily_summary", num_rows=3)


# In[ ]:


QUERY = """
    SELECT
       city_name, extract(Year from date_local) as year,
        AVG(aqi) as average_aqi,MAX(aqi) as max_aqi
    FROM
      `bigquery-public-data.epa_historical_air_quality.pm25_frm_daily_summary`
    WHERE
      lower(city_name) in ("new york","boston","chicago","san francisco")
      AND sample_duration = "24 HOUR"
      AND poc = 1
      group by 1,2
    ORDER BY year
        """
df = bq_assistant.query_to_pandas(QUERY)



# In[ ]:


print("Changing datatype of Year")
print('')
print(df.dtypes)
print('')
df.year = pd.to_datetime(df.year, format='%Y')
print(df.dtypes)


# In[ ]:


#df.plot(x='day_of_year', y='aqi', style='.')
sns.set(font_scale=1)
fg = sns.FacetGrid(data=df, hue='city_name', aspect=4,size=2,palette="Set1")
fg.map(plt.scatter, 'year', 'average_aqi').add_legend().set_axis_labels('year','Average Aqi').set_titles("Average AQI in Major Cities")


# In[ ]:


QUERY1 = """
    SELECT
       city_name, state_name,
        AVG(aqi) as average_aqi,MAX(aqi) as max_aqi
    FROM
      `bigquery-public-data.epa_historical_air_quality.pm25_frm_daily_summary`
    WHERE
    sample_duration = "24 HOUR"
      AND poc = 1
      group by 1,2
    ORDER BY 3 desc
    limit 100
        """
df1 = bq_assistant.query_to_pandas(QUERY1)


# In[ ]:


print("Seems like California consistently has the worst air quality index")
df1.head(n=10)


# In[ ]:


print("""Looking over at Column metrics in the data ,we see that California may actually have more number of stations where measurements are taken hence to get a more accurate number on worst AQI states, we need to normalize it
""")
df1['state_name'].value_counts()


# In[ ]:





# 
