#!/usr/bin/env python
# coding: utf-8

# # Exploratory Data Analysis with SQL
# 
# The purpose here is to use some queries to explore the data.
# 

# In[ ]:


# Sqlite is a library that implements a SQL database engine. 
import sqlite3
import numpy as np
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


#connection
conn = sqlite3.connect('../input/world-development-indicators/database.sqlite')


# In[ ]:


#sqlite_master is a table with database schema
pd.read_sql(""" SELECT *
                FROM sqlite_master
                WHERE type='table';""",
           conn)


# ## <span style='color:DarkGoldenrod'> Let's explore the Indicators table.  </span>

# # 1- Selecting

# In[ ]:


# check the head
pd.read_sql("""SELECT *
               FROM Indicators
               LIMIT 3;""",
           conn)


# In[ ]:


# how many rows?
pd.read_sql("""SELECT COUNT(*)
               FROM Indicators;""",
           conn)


# In[ ]:


# checking for missing values in one column
pd.read_sql("""SELECT COUNT(*)
               FROM Indicators
               WHERE IndicatorName IS NULL;""",
           conn)


# In[ ]:


# if I need filter not null
pd.read_sql("""SELECT *
               FROM Indicators
               WHERE IndicatorName IS NOT NULL;""",
           conn)


# In[ ]:


# checking for missing values in all columns
pd.read_sql("""SELECT COUNT(*) - COUNT(CountryName) AS missing_CountryName,
                      COUNT(*) - COUNT(CountryCode) AS missing_CountryCode,
                      COUNT(*) - COUNT(IndicatorName) AS missing_IndicatorName,
                      COUNT(*) - COUNT(IndicatorCode) AS missing_IndicatorCode, 
                      COUNT(*) - COUNT(Year) AS missing_Year, 
                      COUNT(*) - COUNT(Value) AS missing_Value
                FROM Indicators;""",
           conn)


# In[ ]:


# how many indicators?
pd.read_sql("""SELECT COUNT (DISTINCT IndicatorName)
                FROM Indicators;""",
           conn)


# In[ ]:


# selecting distinct indicators
pd.read_sql("""SELECT DISTINCT IndicatorName
                FROM Indicators;""",
           conn)


# # 2- Filtering

# In[ ]:


# I wanna search for some indicator about GDP
pd.read_sql("""SELECT DISTINCT IndicatorName
               FROM Indicators
               WHERE IndicatorName LIKE 'GDP%';""",
           conn)


# In[ ]:


# how about GDP per capita of Brazil in last years ?
pd.read_sql(""" SELECT *
                FROM Indicators
                WHERE IndicatorName ='GDP per capita (current US$)'
                AND CountryName = "Brazil"
                AND Year>=2010;""",
           conn)


# In[ ]:


# let's  compare with China
pd.read_sql("""SELECT *
               FROM Indicators
               WHERE IndicatorName='GDP per capita (current US$)'
               AND (CountryName= 'Brazil' OR CountryName= 'China')
               AND Year>=2010;""",
           conn)


# In[ ]:


# let's check the 90's in Brazil
pd.read_sql("""SELECT * 
               FROM Indicators
               WHERE IndicatorName='GDP per capita (current US$)'
               AND CountryName='Brazil'
               AND Year BETWEEN 1990 AND 1999;""",
           conn)


# In[ ]:


# let's check other countries in 2014
pd.read_sql("""SELECT *
               FROM Indicators
               WHERE IndicatorName='GDP per capita (current US$)'
               AND CountryName IN ("Brazil", "China", "India")
               AND Year=2014;""",
            conn)


# # 3- Aggregating and Summarizing

# In[ ]:


# let's search for the highest GDP per capita in 2014
pd.read_sql("""SELECT CountryName, MAX (Value)
               FROM Indicators
               WHERE IndicatorName = 'GDP per capita (current US$)'
               AND Year = 2014;""",
            conn)


# In[ ]:


#Let's order (and reafirm the highest GDP per capita)
pd.read_sql("""SELECT * 
               FROM Indicators
               WHERE IndicatorName='GDP per capita (current US$)'
               AND Year= 2014
               ORDER BY Value DESC
               LIMIT 3;""",
           conn)


# In[ ]:


# Let's compare the averages
pd.read_sql(""" SELECT CountryName, AVG(Value)
                FROM Indicators
                WHERE IndicatorName= 'GDP per capita (current US$)'
                AND CountryName IN ('Brazil', 'China', 'India', 'Angola')
                AND Year>2010
                GROUP BY CountryName;""",
           conn)


# In[ ]:


#how many measures during this time?
pd.read_sql(""" SELECT CountryName, count (*) AS n_measures
                FROM Indicators
                WHERE IndicatorName= 'GDP per capita (current US$)'
                AND CountryName IN ('Brazil', 'China', 'India', 'Angola')
                AND Year>2010
                GROUP BY CountryName
                ORDER BY n_measures
                LIMIT 10;""",
           conn)


# In[ ]:


# important to know that Angola has less measures during this time
pd.read_sql(""" SELECT *
                FROM Indicators
                WHERE IndicatorName= 'GDP per capita (current US$)'
                AND CountryName = 'Angola'
                AND Year>2010;""",
           conn)


# # 4- JOINs

# In[ ]:


# let's make a join to get the information about indicators GDP related measured in 2014 in Brazil
pd.read_sql(""" SELECT Indicators.*, Series.LongDefinition
                FROM Indicators
                LEFT JOIN Series 
                ON Indicators.IndicatorName  = Series.IndicatorName
                WHERE Indicators.IndicatorName LIKE 'GDP%'
                AND CountryName ='Brazil'
                AND Year=2014;""",
            conn)


# # 5- Using CASE WHEN
# 
# CASE statements are like "IF this THEN that".  
# Here I'm going to use CASE statements for a custom discretization.

# In[ ]:


# I'm using arbitrary values, this is an exercise with didactic purposes: cuts on 10000 and 80000
df=pd.read_sql(""" SELECT *,
                   CASE WHEN Value < 10000 THEN 'Low'
                   WHEN Value > 80000 THEN 'High'
                   ELSE 'Medium' END AS Category
                   FROM Indicators
                   WHERE IndicatorName='GDP per capita (current US$)'
                   AND Year=2014;""",
              conn)


# In[ ]:


fig, axes= plt.subplots(1,2, figsize=(8,4),sharey=True)

ax1= sns.distplot(df.Value, bins=40, hist_kws={'edgecolor':'k'}, color='mediumseagreen',kde=False,ax=axes[0])
ax1.set_title('Histogram')
ax1.set(xlabel="GDP per capita (current US$) - Value")
ax1= sns.despine()

ax2= sns.countplot(x='Category', data=df, palette="Greens", ax=axes[1])
ax2.set_title('Frequency Count')
ax2.set(xlabel="GDP per capita (current US$) - Category")
ax2= sns.despine();


# In[ ]:


fig.savefig('eda_sql.png', transparent=True)


# # 6 - Window Function
# With a window function, we can make calculations across rows "in a window" and return a value for each row.
# This window can be, for example, grouped sets based on another column or even an ordered set.

# In[ ]:


# adding the mean of last 3 years using window function

pd.read_sql(""" SELECT *,
                ROUND(AVG(Value) OVER (PARTITION BY CountryCode),0) AS AVG_3Y
                FROM Indicators
                WHERE IndicatorName='GDP per capita (current US$)'
                AND Year IN (2012,2013,2014);""",
           conn)


# # 7 - Common Table Expression (CTE)
# A CTE will save results of a query temporary. It can help simplifying some queries and, for example, can also help filtering based on ranking results. 

# In[ ]:


# the first place from each year
pd.read_sql(""" WITH GDP_3 AS 
                (SELECT *,
                RANK () OVER (PARTITION BY Year ORDER BY Value DESC) AS myrank
                FROM Indicators
                WHERE IndicatorName='GDP per capita (current US$)'
                AND Year IN (2012,2013,2014))
                
                SELECT *
                FROM GDP_3
                WHERE myrank = 1;""",
           conn)

