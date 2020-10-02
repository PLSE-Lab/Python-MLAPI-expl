#!/usr/bin/env python
# coding: utf-8

# Copied from "No.1 popular female names in each decade" (https://www.kaggle.com/amanemisa/no-1-popular-female-names-in-each-dacade)
# 

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sqlite3
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[ ]:


con = sqlite3.connect('../input/database.sqlite')
NationalNames = pd.read_csv('../input/NationalNames.csv')


# In[ ]:


popular_male_dacade = pd.read_sql_query("""
WITH name_dacade AS (
SELECT 
CASE WHEN year like '188%' THEN '1880-1889'
     WHEN year like '189%' THEN '1890-1899'
     WHEN year like '190%' THEN '1900-1909'
     WHEN year like '191%' THEN '1910-1919'
     WHEN year like '192%' THEN '1920-1929'
     WHEN year like '193%' THEN '1930-1939'
     WHEN year like '194%' THEN '1940-1949'
     WHEN year like '195%' THEN '1950-1959'
     WHEN year like '196%' THEN '1960-1969'
     WHEN year like '197%' THEN '1970-1979'
     WHEN year like '198%' THEN '1980-1989'
     WHEN year like '199%' THEN '1990-1999'
     WHEN year like '200%' THEN '2000-2009'
     WHEN year like '201%' THEN '2010-2019'
END AS dacade,
Name, SUM(Count) AS Total_Count
FROM NationalNames
WHERE Gender = 'M'
GROUP BY dacade, Name)
SELECT dacade, Name, MAX(Total_Count) AS Total_Count
FROM name_dacade
GROUP BY dacade""", con)
popular_male_dacade


# In[ ]:


John_Robert_James_Michael_Jacob_year = pd.read_sql_query("""
SELECT year,Name, Count
FROM NationalNames
WHERE Gender = 'M' AND Name IN ('John','Robert','James','Michael','Jacob')
""", con)


# In[ ]:


fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(1,1,1)
John_Robert_James_Michael_Jacob_year[John_Robert_James_Michael_Jacob_year['Name']=='John'].plot(x='Year', y='Count',color = 'red',ax=ax,label='John')
John_Robert_James_Michael_Jacob_year[John_Robert_James_Michael_Jacob_year['Name']=='Robert'].plot(x='Year', y='Count',color = 'green',ax=ax,label='Robert')
John_Robert_James_Michael_Jacob_year[John_Robert_James_Michael_Jacob_year['Name']=='James'].plot(x='Year', y='Count',color = 'blue',ax=ax,label='James')
John_Robert_James_Michael_Jacob_year[John_Robert_James_Michael_Jacob_year['Name']=='Michael'].plot(x='Year', y='Count', color ='orange',ax=ax,label = 'Michael')
John_Robert_James_Michael_Jacob_year[John_Robert_James_Michael_Jacob_year['Name']=='Jacob'].plot(x='Year', y='Count', color ='black',ax=ax,label = 'Jacob')
fig.suptitle('Name trends from 1880 to 2014', fontsize=12)

