#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from google.cloud import bigquery
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import bq_helper
from tabulate import tabulate
client = bigquery.Client()
my_colors =       ['#ff8000', # Indian saffron
                   '#0000ff',  # Indian Blue
                   '#008000', # Indian Green
                    '#78C850',  # Grass
                    '#6890F0',  # Water
                    '#A8B820',  # Bug
                    '#A8A878',  # Normal
                    '#A040A0',  # Poison
                    '#F8D030',  # Electric
                    '#E0C068',  # Ground
                    '#EE99AC',  # Fairy
                    '#C03028',  # Fighting
                    '#F85888',  # Psychic
                    '#B8A038',  # Rock
                    '#705898',  # Ghost
                    '#98D8D8',  # Ice
                    '#7038F8',  # Dragon
                    '#F08030',  # Fire
                   ]


# In[ ]:


queryIoT = """
SELECT *
FROM `bigquery-public-data.hacker_news.full` 
WHERE 
(REGEXP_CONTAINS(title, r"(?i)(internet of things)") 
OR REGEXP_CONTAINS(title, r"( IOT )") 
OR REGEXP_CONTAINS(title, r"(^IOT )") 
OR REGEXP_CONTAINS(title, r"(^IOT$)") 
OR REGEXP_CONTAINS(text, r"(?i)(internet of things)") 
OR REGEXP_CONTAINS(text, r"( IOT )") 
OR REGEXP_CONTAINS(text, r"(^IOT )") 
OR REGEXP_CONTAINS(text, r"(^IOT$)"))
AND (type="story" OR type="comment")
ORDER BY time
"""
query_job = client.query(queryIoT)
iterator = query_job.result()
rows = list(iterator)
df_iot = pd.DataFrame(data=[list(x.values()) for x in rows], columns=list(rows[0].keys()))
df_iot['date'] = pd.to_datetime(df_iot.timestamp)
df_iot['year'] = df_iot.date.dt.year.astype('category')


# In[ ]:


sns.set(style="whitegrid", color_codes=True)
plt.figure(figsize=(12,8))
sns.countplot(x="year",palette=my_colors, data=df_iot);


# In[ ]:


len(df_iot)


# In[ ]:


df_iot.head()


# In[ ]:


import json
with open('iothackernews.json', 'w') as outfile:
    json.dump(json.loads(df_iot.to_json(orient='records')), outfile)


# In[ ]:


queryBigData = """
SELECT *
FROM `bigquery-public-data.hacker_news.full` 
WHERE 
(
REGEXP_CONTAINS(title, r"(?i)(big data)") 
OR REGEXP_CONTAINS(title, r"(?i)(BigData)")
OR REGEXP_CONTAINS(text, r"(?i)(big data)") 
OR REGEXP_CONTAINS(text, r"(?i)(BigData)")
)
AND (type="story" OR type="comment")
ORDER BY time
"""
query_job = client.query(queryBigData)
iterator = query_job.result()
rows = list(iterator)


# In[ ]:


df_bigdata = pd.DataFrame(data=[list(x.values()) for x in rows], columns=list(rows[0].keys()))
df_bigdata['date'] = pd.to_datetime(df_bigdata.timestamp)
df_bigdata['year'] = df_bigdata.date.dt.year.astype('category')


# In[ ]:


sns.set(style="whitegrid", color_codes=True)
plt.figure(figsize=(12,8))
sns.countplot(x="year",palette=my_colors, data=df_bigdata);


# In[ ]:


len(df_bigdata)


# In[ ]:


df_bigdata.head(10)


# In[ ]:


#Remove comments without parents
df_bigdata = df_bigdata[(df_bigdata['type']!='comment') | (df_bigdata['parent'].isin(df_bigdata['id']))]


# In[ ]:


#get things with parent
df_bigdata.head(10)


# In[ ]:


len(df_bigdata)


# In[ ]:


#filter data from 2011 only
import time
import datetime
s = "01/01/2011"
start = time.mktime(datetime.datetime.strptime(s, "%d/%m/%Y").timetuple())


# In[ ]:


start


# In[ ]:


df_bigdata = df_bigdata[df_bigdata['time'] >= start]


# In[ ]:


df_bigdata.head(10)


# In[ ]:


len(df_bigdata)


# In[ ]:


import json
with open('bigdatahackernews.json', 'w') as outfile:
    json.dump(json.loads(df_bigdata.to_json(orient='records')), outfile)


# In[ ]:


queryCPS = """
SELECT *
FROM `bigquery-public-data.hacker_news.full` 
WHERE 
(REGEXP_CONTAINS(title, r"(?i)(cyber[ \-]physical systems)") 
OR REGEXP_CONTAINS(title, r"( CPS )") 
OR REGEXP_CONTAINS(title, r"(^CPS )") 
OR REGEXP_CONTAINS(title, r"(^CPS$)") 
OR REGEXP_CONTAINS(title, r"(?i)(cyber[ \-]physical systems)") 
OR REGEXP_CONTAINS(text, r"( CPS )") 
OR REGEXP_CONTAINS(text, r"(^CPS )") 
OR REGEXP_CONTAINS(text, r"(^CPS$)"))
AND (type="story" OR type="comment")
ORDER BY time
"""
query_job = client.query(queryCPS)
iterator = query_job.result()
rows = list(iterator)
df_cps = pd.DataFrame(data=[list(x.values()) for x in rows], columns=list(rows[0].keys()))
df_cps['date'] = pd.to_datetime(df_cps.timestamp)
df_cps['year'] = df_cps.date.dt.year.astype('category')


# In[ ]:


df_cps.head()


# In[ ]:


df_cps = pd.DataFrame(data=[list(x.values()) for x in rows], columns=list(rows[0].keys()))
df_cps['date'] = pd.to_datetime(df_cps.timestamp)
df_cps['year'] = df_cps.date.dt.year.astype('category')


# In[ ]:


#Remov e data from 2011
df_cps = df_cps[df_cps['time'] >= start]


# In[ ]:


sns.set(style="whitegrid", color_codes=True)
plt.figure(figsize=(12,8))
sns.countplot(x="year",palette=my_colors, data=df_cps);


# In[ ]:


import json
with open('cpshackernews.json', 'w') as outfile:
    json.dump(json.loads(df_cps.to_json(orient='records')), outfile)


# In[ ]:




