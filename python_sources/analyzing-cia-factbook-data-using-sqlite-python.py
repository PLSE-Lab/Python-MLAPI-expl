#!/usr/bin/env python
# coding: utf-8

# ## Analyzing CIA Factbook Data Using SQLite and Python
# 
# The CIA World Factbook is a compendium of statistics about all of the countries on Earth. The Factbook contains demographic information like:
# 
# - `population` - The population as of 2015 .
# - `population_growth` - The annual population growth rate, as a percentage.
# - `area` - The total land and water area.
# 
# The data `factbook.db` can be gotten from this [link](https://github.com/makozi/Analyzing-CIA-Factbook-Data-Using-SQLite-and-Python/blob/master/factbook.db)

# In[ ]:


import sqlite3
import pandas as pd
conn= sqlite3.connect('../input/cia-factbook-data/factbook-data/factbook.db')
q= "SELECT  * FROM sqlite_master WHERE type='table';"
pd.read_sql_query(q,conn)


# In[ ]:


cursor= conn.cursor()
cursor.execute(q).fetchall()


#  #### Let's run another query that returns the first 5 rows of the facts table in the database.

# In[ ]:


q1= "SELECT * FROM facts limit 5"
pd.read_sql_query(q1,conn)


# Here are the descriptions for some of the columns:
# 
# - `name` - The name of the country.
# - `area` - The total land and sea area of the country.
# - `population` - The country's population.
# - `population_growth` - The country's population growth as a percentage.
# - `birth_rate` - The country's birth rate, or the number of births a year per 1,000 people.
# - `death_rate` - The country's death rate, or the number of death a year per 1,000 people.
# - `area `- The country's total area (both land and water).
# - `area_land` - The country's land area in square kilometers.
# - `area_water` - The country's waterarea in square kilometers.
# 
# 
# 
# 
# 
# ## Summary Statistics
# 
# Writing a single query that returns the following:
# - Minimum population
# - Maximum population
# - Minimum population growth
# - Maximum population growth

# In[ ]:


q2= '''
    select min(population) min_pop, max(population) max_pop, min(population_growth) min_pop_growth, max(population_growth) max_pop_growth from facts
'''
pd.read_sql_query(q2, conn)


# From the table above:
# - The Minimum population is 0
# - Maximum population is 7256490011 (or more than 7.2 billion people)
# - Minimum population growth is 0.0
# - Maximum population growth is  4.02
# 
# 
# 
# 
# ## Outliers
# 
# 
# - Writng  a query that returns the countrie(s) with a population of 7256490011 .
# 

# In[ ]:


q3= '''
    SELECT * FROM facts WHERE population==(SELECT max(population) FROM facts)
'''

pd.read_sql_query(q3,conn)


# From the data above, the population of the `World` is 7256490011 (or more than 7.2 billion people)
# 
# 
# 
# 
# 
# 
# - Writing a query that returns the countrie(s) with a population of 0:
# 
# 
# 

# In[ ]:


q4= ''' 

select * from facts where population==(select min(population) from facts)
'''

pd.read_sql_query(q4,conn)


# From the data above, Antarctica has the minimum population which is 0

# ## Histogram

# Moving on to generating histograms for the rest of the countries in the table, ignoring these 2 rows. I will write a query that returns all of the values in the columns I want to visualize.
# 

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)

q5 = '''
select population, population_growth, birth_rate, death_rate
from facts
where population != (select max(population) from facts)
and population != (select min(population) from facts);
'''
pd.read_sql_query(q5, conn).hist(ax=ax)


# ## Countries with the highest population density

# In[ ]:


q6='''
select name, cast(population as float)/cast(area as float) density from facts order by density desc limit 20
'''
pd.read_sql_query(q6, conn)


# Macau has the highest population density which is accurate when compared with the result from [Wikipedia](https://en.wikipedia.org/wiki/List_of_countries_and_dependencies_by_population_density).

# In[ ]:


q7 = '''select population, population_growth, birth_rate, death_rate
from facts
where population != (select max(population) from facts)
and population != (select min(population) from facts);
'''
pd.read_sql_query(q7, conn)

