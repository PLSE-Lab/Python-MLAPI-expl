#!/usr/bin/env python
# coding: utf-8

# After SQL Scanvenger Hunt by Rachael Tatman I take OpenAQ air quality bigquery data source for data exploratory analysis ( and cleaning !), some feature engineering (converting units of measurement), and practice graphs and plots with Python packages.

# In[ ]:


import numpy as np
import pandas as pd
from bq_helper import BigQueryHelper
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
from wordcloud import WordCloud
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# I am going to use Sohier Dane [bq_helper library](https://github.com/SohierDane/BigQuery_Helper).

# In[ ]:


bq_assistant = BigQueryHelper('bigquery-public-data', 'openaq')
bq_assistant.list_tables()


# The Data source has only one table. Let's see how many rows are in there...

# In[ ]:


query = """SELECT COUNT(*) AS total_rows 
           FROM `bigquery-public-data.openaq.global_air_quality`"""
total_rows = bq_assistant.query_to_pandas_safe(query)
print(total_rows)


# Okay, not that much. It's safe to query without scanning huge amounts of data.

# Obviously the "value" field will be very important. I will check and make some data cleaning first. And later I should convert units of measurement if I am going to compare locations using different units.

# In[ ]:


query = """SELECT value 
           FROM `bigquery-public-data.openaq.global_air_quality`
           WHERE unit = 'ppm'"""
ppm_values = bq_assistant.query_to_pandas_safe(query)
ppm_values.describe()


# Well, there are negative values, that should be discarded. Let's do that and plot a histogram to detect suspicious outliers.

# In[ ]:


query = """SELECT value 
           FROM `bigquery-public-data.openaq.global_air_quality`
           WHERE unit = 'ppm' AND value >= 0"""
ppm_values = bq_assistant.query_to_pandas_safe(query)
plt.figure(figsize=(12,4))
bins = np.linspace(0,400,10,dtype='i')
plt.hist( ppm_values.value, bins=bins,color='green')


# In[ ]:


ppm_values.describe()


# It seems ok... just one outlier with a value near 400 ppm, way out of the mean, median or quartiles. I will discard it. 

# In[ ]:


query = """SELECT * 
           FROM `bigquery-public-data.openaq.global_air_quality`
           WHERE unit = 'ppm' AND value >= 350"""
ppm_values = bq_assistant.query_to_pandas_safe(query)
# save outlier value to filter in next queries
ppm_value_outlier = ppm_values.value[0]
ppm_values


# Now the same analysis with values in ug/m3 unit.

# In[ ]:


query = """SELECT value 
           FROM `bigquery-public-data.openaq.global_air_quality`
           WHERE unit != 'ppm'"""
ugm3_values = bq_assistant.query_to_pandas_safe(query)
ugm3_values.describe()


# Ouch ! There's a lot to check here... Let's discard negative values and those greater than, say, a million, and make some histograms. Surely some outliers (error data) shouldn't be included in the analysis.

# In[ ]:


query = """SELECT value 
           FROM `bigquery-public-data.openaq.global_air_quality`
           WHERE unit != 'ppm' AND value >= 0 AND value < 1000000"""
ugm3_values = bq_assistant.query_to_pandas_safe(query)
plt.figure(figsize=(12,4))
plt.hist( ugm3_values.value, bins=np.linspace(0,1000000,10))


# In[ ]:


ugm3_values.describe()


# In[ ]:


# We can zoom in at values less than 20,000
query = """SELECT value
           FROM `bigquery-public-data.openaq.global_air_quality`
           WHERE unit != 'ppm' AND value >= 0 AND value < 20000"""
ugm3_values = bq_assistant.query_to_pandas_safe(query)
plt.figure(figsize=(12,4))
plt.hist( ugm3_values.value, bins=np.linspace(0,20000,10),log=True)


# It seems we can discard outliers having values greater then 20,000. Max values in ppm units where way below this, we can filter in queries..

# Now, I want to see all the measurement locations at a world map. 

# In[ ]:


query = """SELECT longitude, latitude
           FROM `bigquery-public-data.openaq.global_air_quality`
           -- cleaning invalid data
           WHERE value >= 0 AND value < 20000 
           """
df = bq_assistant.query_to_pandas_safe(query)


# In[ ]:


# Background image got at the internet
# https://vignette.wikia.nocookie.net/pixar/images/b/b0/20100625011514%21World_Map_flat_Mercator.png/revision/latest?cb=20120823094025&format=original
img = mpimg.imread("../input/World_Map_flat_Mercator.png")
plt.figure(figsize=(18,9))
imgplot = plt.imshow(img,zorder=1)
# Have to scale latitudes and longitudes to fit world map at background
plt.scatter((df.longitude+167)*1468/360, (df.latitude*-1+126)*1006/218, c=sns.color_palette("autumn"), s=1, alpha=0.5,zorder=2)
plt.grid(True)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Geographic Distribution of Measurement Locations")
plt.show()


# We can see locations mainly at North America and Europe. Some more in Asia and South America, and a few at Oceania and Africa.

# I obtained the Country coding used at OpenAQ. Let's create a DataFrame  that I can use later for code translation.

# In[ ]:


# I misnamed the csv file as "cities", sorry, it's "countries"
data = pd.read_csv("../input/OpenAQ_cities.csv", header=0)
country_codes = pd.DataFrame(data,columns=["Country","Code"])
country_codes.set_index('Code');


# For the next graphs I want to compare/rank location, cities, etc. so I need to convert measured values to only one unit.
# - I will use this formula to convert units in ppm to ug/m3: 
#            1 ppm = 1266 ug/m3 
# - To get that I used https://www.lenntech.com/calculators/ppm/converter-parts-per-million.htm applying the air constitution of 80% of nitrogen and 20% of oxygen (it is a reasonable approximation, other gases can be ignored).

# In[ ]:


# Obtain totals per location using normalized values to ppm units
query = """SELECT * FROM (
            WITH norm_values AS 
            (
                SELECT location, city, country, 
                      (value / IF((rtrim(ltrim(unit)) != 'ppm'),1266,1)) as nvalue
                FROM `bigquery-public-data.openaq.global_air_quality`
                -- cleaning invalid data
                WHERE value >= 0 AND value < 20000 
            )
            SELECT location, city, country, SUM(nvalue) AS total_pollution
            FROM norm_values
            GROUP BY location, city, country
            ORDER BY total_pollution DESC
            LIMIT 20)
           ORDER BY total_pollution, country, city, location
            """

df = bq_assistant.query_to_pandas_safe(query)
# Filter that ppm outlier detected previously and get top 10 locations to plot
df = df[df.total_pollution < ppm_value_outlier][:10]


# In[ ]:


plt.figure(figsize=(12,4))
plt.xlabel("Pollution in ppm")
plt.title("Total pollution: Top 10 Worst Locations")
plt.yticks(np.arange(10),df.location,rotation=0) 
# Using a log scale for better visualization
plt.barh(np.arange(10),df.total_pollution,align='center', tick_label=df.country+'-'+df.city+'-'+df.location,log=False, color=sns.light_palette('purple',10, reverse=False))
plt.show()


# In[ ]:


# Obtain totals per location using normalized values to ppm units
query = """SELECT * FROM (
            WITH norm_values AS 
            (
                SELECT city, country, 
                      (value / IF((rtrim(ltrim(unit)) != 'ppm'),1266,1)) as nvalue
                FROM `bigquery-public-data.openaq.global_air_quality`
                -- cleaning invalid data
                WHERE value >= 0 AND value < 20000 
            )
            SELECT city, country, SUM(nvalue) AS total_pollution
            FROM norm_values
            GROUP BY city, country
            ORDER BY total_pollution
            LIMIT 100)
            """

df = bq_assistant.query_to_pandas_safe(query)
# Filter that ppm outlier detected previously
df = df[df.total_pollution < ppm_value_outlier]
df[:10]


# In[ ]:


# Chart countries having best AQ, translate country codes with country names
txt = ''
for n in df.country:
    ctry = country_codes[country_codes['Code'] == n].Country.values[0]
    ctry = ctry.replace( ' ', '')
    txt = txt + ctry + ' '
plt.figure(figsize=(6,6))
wc = WordCloud(background_color='gray', max_font_size=200,
                            width=600,
                            height=400,
                            max_words=25,
                            relative_scaling=.3).generate(txt)
plt.imshow(wc)
plt.title("Countries with more Best AQ Cities", fontsize=14)
plt.axis("off");


# Let's see how locations update measures and plot a distribution.

# In[ ]:


query = """SELECT DATETIME_DIFF(now,past,DAY) AS days_update FROM (
            SELECT DATETIME(CURRENT_TIMESTAMP()) AS now, DATETIME(timestamp) AS past           
            FROM `bigquery-public-data.openaq.global_air_quality`
            -- cleaning invalid data
            WHERE value >= 0 AND value < 20000)
           ORDER BY days_update
           """
df = bq_assistant.query_to_pandas_safe(query)
df = df[ df.days_update >= 0]
# save some statistic data for the plot
days_update_mean = df.days_update.mean()
days_update_std = df.days_update.std()


# In[ ]:


plt.figure(figsize=(12,4))
plt.ylabel("Locations updated")
plt.xlabel("Update delay (days)")
plt.title("Update delay at Locations (days)")
plt.hist( df.days_update, color='y', bins=np.linspace(0,days_update_mean+(2*days_update_std),20));

