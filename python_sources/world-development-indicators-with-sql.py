#!/usr/bin/env python
# coding: utf-8

#  # The purpose of this project was to practice SQL

# **SQL** stands for **Structured Query Language** and it is an ANSI standard computer language for accessing and manipulating database systems. It is used for managing data in relational database management system which stores data in the form of tables and relationship between data is also stored in the form of tables. **SQL** statements are used to retrieve and update data in a database

# <img src="https://i.imgur.com/1riy1gs.jpg.jpg" width="800">

# In[ ]:


import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


conn = sqlite3.connect('../input/world-development-indicators/database.sqlite')


# # SQL Retrieving Data from Tables

# In[ ]:


pd.read_sql(""" SELECT *
                FROM sqlite_master
                WHERE type='table';""",
           conn)


# In[ ]:


pd.read_sql("""SELECT *
               FROM Indicators
               LIMIT 10;""",
           conn)


# In[ ]:


#Displaying 3 numbers in 3 columns
pd.read_sql("""SELECT 5, 10, 15;""",
           conn)


# In[ ]:


#The sum of 2 numbers
pd.read_sql("""SELECT 55+90;""",
           conn)


# In[ ]:


#The result of an arithmetic expression
pd.read_sql("""SELECT 10+15-5*2;""",
           conn)


# In[ ]:


#Displaying Year and Value
pd.read_sql("""SELECT Year, Value
               FROM Indicators;""",
           conn)


# In[ ]:


pd.read_sql("""SELECT CountryName, CountryCode, IndicatorName
               FROM Indicators;""",
           conn)


# In[ ]:


#Retrieving the values of CountryName for all countries
pd.read_sql("""SELECT DISTINCT CountryName
               FROM Indicators;""",
           conn)


# # SQL Filtering and Sorting

# In[ ]:


#Grouping and counting records by region
pd.read_sql("""SELECT Region, COUNT(*) AS [Count]
               FROM Country 
               GROUP BY Region
               ORDER BY 2 DESC;""",
            conn)


# In[ ]:


#Information about Bolivia
pd.read_sql("""SELECT *
               FROM Indicators
               WHERE CountryName = "Bolivia";""",
           conn)


# In[ ]:


#Displaying CountryName and Value which belong to the country of Sweden
pd.read_sql("""SELECT CountryName, Value
               FROM Indicators
               WHERE CountryName = "Sweden";""",
           conn)


# In[ ]:


#Information for the countries from the year of 2000
pd.read_sql("""SELECT *
               FROM Indicators
               WHERE Year = "2000";""",
           conn)


# In[ ]:


#IndicatorName and IndicatorCode for the country that holds the CountryCode "ZWE"
pd.read_sql("""SELECT IndicatorName, IndicatorCode
               FROM Indicators
               WHERE CountryCode = "ZWE";""",
           conn)


# In[ ]:


#Values for Russian Federation since 1990
pd.read_sql("""SELECT Value
               FROM Indicators
               WHERE Year >=1990
               AND CountryName = "Russian Federation";""",
           conn)


# In[ ]:


#Values for the United States between 1975 and 1986 inclusive
pd.read_sql("""SELECT Value
               FROM Indicators
               WHERE CountryName = "United States"
               AND Year>=1975 AND Year<=1986;""",
           conn)


# In[ ]:


#All the details of Central African Republic and Rwanda after 1999
pd.read_sql("""SELECT *
               FROM Indicators
               WHERE Year >1999
               AND CountryName IN ('Central African Republic',
               'Rwanda');""",
           conn)


# In[ ]:


#Information about GDP
pd.read_sql("""SELECT *
               FROM Indicators
               WHERE IndicatorName LIKE 'GDP%';""",
           conn)


# In[ ]:


#Details for Chile of 2010 and Peru of 2011 together
pd.read_sql("""SELECT *
               FROM Indicators
               WHERE (CountryName = "Chile" AND Year=2010)
               UNION SELECT *
               FROM Indicators
               WHERE (CountryName = "Peru" AND Year=2011);""",
           conn)


# In[ ]:


#Showing information about the countries in the year 2015 except Arab World
pd.read_sql("""SELECT *
               FROM Indicators
               WHERE Year=2015
               AND CountryName NOT IN ('Arab World');""",
           conn)


# In[ ]:


#IndicatorName for the countries not started with the letter 'P' and arranged the list as the most recent comes first, then by name in order
pd.read_sql("""SELECT *
               FROM Indicators
               WHERE CountryName NOT LIKE 'P%'
               ORDER BY YEAR DESC, IndicatorName;""",
           conn)


# In[ ]:


#Countries with the Value between 100 and 500
pd.read_sql("""SELECT *
               FROM Indicators
               WHERE Value BETWEEN 100 AND 500;""",
           conn)


# # SQL Aggregate Functions

# In[ ]:


#Calculating the average value of urban population
pd.read_sql("""SELECT AVG(Value)
               FROM Indicators
               WHERE IndicatorName = 'Urban population';""",
           conn)


# In[ ]:


#The lowest GDP per capita in 2013
pd.read_sql("""SELECT CountryName, MIN (Value)
               FROM Indicators
               WHERE IndicatorName = 'GDP per capita (current US$)'
               AND Year = 2013;""",
            conn)


# In[ ]:


#Country Name and Country Code
pd.read_sql("""SELECT CountryName as "Country Name", CountryCode as "Country Code"
               FROM Indicators;""",
            conn)


# In[ ]:


#Displaying the countries with the highest GDP per capita in 2009
pd.read_sql("""SELECT * 
               FROM Indicators
               WHERE IndicatorName='GDP per capita (current US$)'
               AND Year= 2009
               ORDER BY Value DESC
               LIMIT 10;""",
           conn)


# In[ ]:


#Comparing Life expectancy at birth max values for Russian Federation, Bolivia, United States, Nigeria and India from 2012 inclusive
pd.read_sql(""" SELECT CountryName, MAX(Value)
                FROM Indicators
                WHERE IndicatorName= 'Life expectancy at birth, total (years)'
                AND CountryName IN ('Russian Federation', 'Bolivia',
                'United States', 'Nigeria', 'India')
                AND Year>=2012
                GROUP BY CountryName;""",
           conn)


# In[ ]:


#Death rate in Latin America
pd.read_sql("""SELECT *
               FROM Indicators
               WHERE IndicatorName='Death rate, crude (per 1,000 people)'
               AND CountryName LIKE 'Latin America%'
               ORDER BY Value ASC;""",
           conn)


# In[ ]:


#Null values
pd.read_sql("""SELECT COUNT(*)
               FROM Indicators
               WHERE IndicatorName IS NULL;""",
           conn)


# In[ ]:


#The sum of hospital beds
pd.read_sql("""SELECT SUM(Value)
               FROM Indicators
               WHERE IndicatorName = 'Hospital beds (per 1,000 people)';""",
           conn)


# # SQL Join

# In[ ]:


#Fertility rate in Bolivia
pd.read_sql(""" SELECT Indicators.*, Series.LongDefinition
                FROM Indicators
                LEFT JOIN Series 
                ON Indicators.IndicatorName  = Series.IndicatorName
                WHERE Indicators.IndicatorName LIKE 'Fertility rate%'
                AND CountryName ='Bolivia';""",
            conn)


# In[ ]:


#CO2 emissions in the world
pd.read_sql(""" SELECT Indicators.*, Series.LongDefinition
                FROM Indicators
                LEFT JOIN Series 
                ON Indicators.IndicatorName  = Series.IndicatorName
                WHERE Indicators.IndicatorName LIKE 'CO2%'
                AND CountryName ='World'
                ORDER BY Year DESC
                LIMIT 10;""",
            conn)


# # Hope, this is useful! :)
