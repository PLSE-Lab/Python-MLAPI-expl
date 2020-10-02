#!/usr/bin/env python
# coding: utf-8

# # Introduction 
# 
# SQL or Structured Query Language is the de-facto language used for interacting with ("querying" per industry speak) relational databases. A relational database consists of tables, where each table contains records, or rows of data organised by fields or columns. On the topic of relational databases,  there are many different flavours and forms of relational database management systems (RDMS) - SQL Server, MySQL, PostgreSQL etc. 
# 
# In this Kaggle dataset, the database that we are given to work with is a SQLite  database. SQLite is not your "classical" database in the sense that it is a self-contained, disk-based database that gets embedded in the application that uses it and hence does not require a separate server process.
# 
# There seems to be very few notebooks on Kaggle talking about integrating Python with SQL and therefore this notebook aims to try to bridge this gap. 
# 
# Let's go.

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import sqlite3
import os
from bokeh.plotting import figure, show
from bokeh.charts import Bar
from bokeh.io import output_notebook
output_notebook()
get_ipython().run_line_magic('matplotlib', 'inline')
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import warnings
warnings.filterwarnings('ignore')


# Unlike my other notebooks, instead of reading data into a pandas dataframe from a csv (comma-separated value) file type, we will query the database via SQL. I will also show how we can store any relevant queries into a Pandas dataframe. To start off, we have to create a connection to the sqlite3 database as such:

# In[ ]:


conn = sqlite3.connect('../input/database.sqlite')


# Once we have our connection setup in python, we can create a **Cursor** object from which we can call the execute( ) method and to perform SQL statements and queries. 

# In[ ]:


c = conn.cursor()


# Having created our cursor object, we can now execute our SQL statement. If you are not too familiar with the following syntax, please bear with me until the following section where I will explain in detail what each SQL command does.
# 
# You can distinguish SQL commands in my code (from Python) as they will be embedded within a triple quotation mark """

# In[ ]:


for row in c.execute(
                    # SQL statement 
                    """
                        SELECT   * 
                        FROM     Country 
                        LIMIT    2
                        
                     """ ):
    print(row)


# Now that's all and good that we have managed to print out the first two rows of the data from our Sqlite database. However I still have a niggling feeling that the current method is inconvenient in the sense that we have to use a for loop just to execute our SQL statement. 
# 
# Thankfully for us, the Pandas library comes with methods that allow one to interact with and query SQL databases and we will explore this in the upcoming section.

# ### Basics of SQL Queries
# 
# Before we continue on with the notebook, I will list here the important SQL statements that are most widely used  
# 
# **SELECT** : Statement used to select rows and columns from a database. 
# 
# **FROM** :  Specifies which table in the database you want to direct your query to
# 
# **WHERE**: clause for filtering for a specified value(s)
# 
# **GROUP BY**: Aggregating data. Needs to be used in conjunction with SQL aggregating functions like SUM and COUNT.
# 
# **ORDER BY**: Sorting columns in the database 

# # 1. SQL and Pandas Equivalent statements
# 
# In this section I shall be comparing a particular SQL statement to its Pandas equivalent in the hope that if you are familiar with the Pandas syntax but not so much SQL, this may allow you to have a familiar reference point with which to familiarise yourself with.
# 
# First let us read in the **Country** table in our 

# In[ ]:


# Store Country data in a pandas dataframe via a SQL query
Country = pd.read_sql(
                       """
                       
                        SELECT  * 
                        FROM    Country
                        
                       """, con=conn)


# **A.) SELECT, LIMIT and head**
# 
# The SELECT statement in SQL is probably the most ubiquitous statement as one will need this statement to select records from a database. Normally you will see this being used very often in conjunction with the asterisk symbol : **SELECT *** .  What this does is to select all rows and columns with in the database. However if one wants to select only a certain number of rows, this is where LIMIT comes in
# 
# I think it is rather safe to assume that most Kagglers understand the use of invoking the head( ) call on a dataframe. It essentially returns the top (user-specified) number of rows in your data. Equivalently, one can also do the same thing via a SQL query with the use of the LIMIT statement as follows:

# In[ ]:


# Pandas code
Country.head(3)


# In[ ]:


# SQL query 
pd.read_sql(
            """
                SELECT   * 
                FROM     Country 
                LIMIT    3 
                
            """, con=conn)


# **B.) WHERE and Boolean Indexing**
# 
# The SQL WHERE clause is mainly used for filtering records of interest. Therefore if the records fulfill the conditions as laid out by the WHERE clause, then that record will be returned. The equivalent of this in Python and Pandas is that of Boolean Indexing - a.k.a passing into the DataFrame another DataFrame in a comparison statement as follows:

# In[ ]:


# Pandas Boolean Indexing
Country[Country['CountryCode'] == 'AFG']


# In[ ]:


# SQL WHERE clause
pd.read_sql(
        """ 
            SELECT   * 
            FROM     Country 
            WHERE    CountryCode = 'AFG'
            
        """, con=conn)


# **C.) GROUP BY and dataframe aggregation**
# 
# The GROUP BY clause is very useful when aggregations are required to be generated. When I say aggregations, these are taken to mean things (in SQL speak) such as COUNT, MAX, MIN, SUM etc. 
# 
# In the following example, I shall perform an aggregation on the Country dataset by counting (COUNT function) the number of records that belong to a certain Region. As a rule of thumb, to know what we have to add to our GROUP BY statement is simply the column that we want to aggregate on (Region in our case).

# In[ ]:


# SQL GROUP BY Clause
pd.read_sql(
        """ 
            SELECT      Region
                        ,COUNT(*) AS [Count]
            FROM        Country 
            GROUP BY    Region
            ORDER BY    2 DESC
            
        """, con=conn)


# I snuck in an ORDER BY statement and what this does is to sort the data in descending order (DESC keyword). Anyway, we can see that this GROUP BY does counts all the records (aggregate) that belong to a particular region and and then outputs the result in a ordered tabular format. 
# 
# Particularly interesting is the fact that we have an empty string as one of our categories in Region and there are 33 records in the database that can be attributed to this. Perhaps this could be brought up as a data quality issue and definitely warrants further investigation.

# **D.) SQL JOIN**

# **E) UNION, INTERSECT and EXCEPT**
# 
# SQL also comes with a handful of useful Set operations, namely that of UNION, INTERSECT and the EXCEPT statements. These statements perform exactly as their name suggests (from set theory).

# In[ ]:





# # 2. Data Analysis and Visualisations
# 
# Having discussed at some length to basic SQL statements and how we can interact and query SQL databases through Python let us now carry on with our World Developmental analysis. To start off, I shall create a dataframe via a query of the **Indicator** table with a handful of manually chosen indicators (as the full table contains too many indicators for this notebook)
# 
# A quick description of the indicators are as follows:
# 
# **AG.LND.PRCP.MM** :  Average precipitation in depth (mm per year)
# 
# **EG.ELC.ACCS.ZS** :  Access to electricity (% of population)
# 
# **EG.ELC.FOSL.ZS** :  Electricity production from oil, gas and coal sources (% of total)
# 
# 
# 

# In[ ]:


Indicators = pd.read_sql(""" SELECT   * 
                             FROM     Indicators 
                             WHERE    IndicatorCode IN 
                                      (  'AG.LND.PRCP.MM, AG.LND.FRST.K2'
                                       , 'EG.ELC.ACCS.ZS', 'EG.ELC.FOSL.ZS'
                                       , 'EN.POP.DNST', 'SG.VAW.REAS.ZS'
                                       , 'SM.POP.NETM', 'SP.POP.65UP.TO.ZS'
                                       , 'FI.RES.TOTL.DT.ZS', 'GC.DOD.TOTL.GD.ZS'
                                       , 'MS.MIL.XPND.GD.ZS','SI.POV.GINI'
                                       , 'IP.JRN.ARTC.SC', 'SE.ADT.1524.LT.ZS'
                                      )  
                        """, con=conn)


# ### 2A. GINI Index analysis
# 
# To start off with our analysis, let us take a look at the GINI index of some of the countries we have in our dataset. As a quick primer the GINI index (in its normalised form) is a statistical measure used 

# In[ ]:


#Regions = ['ARB', 'EUU', 'LCN' , 'NAC',  'EAS', 'SSF', 'World']
gini = Indicators[Indicators['IndicatorCode']== 'SI.POV.GINI']


# In[ ]:


# Plotting a Subplot of the Seaborn regplot
f, ((ax1, ax2, ax3), (ax4,ax5,ax6), (ax7, ax8, ax9)) = plt.subplots(3,3,figsize=(12,10))

points = ax1.scatter(gini[gini['CountryCode'] == 'CHN']["Year"], gini[gini['CountryCode'] == 'CHN']["Value"],
                     c=gini[gini['CountryCode'] == 'CHN']["Value"], s=100, cmap="viridis")
sns.regplot("Year", "Value", data=gini[gini['CountryCode'] == 'CHN'], ax=ax1)
ax1.set_title("GINI Index of China")

points = ax2.scatter(gini[gini['CountryCode'] == 'ARG']["Year"], gini[gini['CountryCode'] == 'ARG']["Value"],
                     c=gini[gini['CountryCode'] == 'ARG']["Value"], s=85, cmap="viridis")
sns.regplot("Year", "Value", data=gini[gini['CountryCode'] == 'ARG'], ax=ax2)
ax2.set_title("GINI Index of Argentina")

points = ax3.scatter(gini[gini['CountryCode'] == 'IND']["Year"], gini[gini['CountryCode'] == 'IND']["Value"],
                     c=gini[gini['CountryCode'] == 'IND']["Value"], s=100, cmap="afmhot")
sns.regplot("Year", "Value", data=gini[gini['CountryCode'] == 'IND'], ax=ax3)
ax3.set_title("GINI Index of India")

points = ax4.scatter(gini[gini['CountryCode'] == 'USA']["Year"], gini[gini['CountryCode'] == 'USA']["Value"],
                     c=gini[gini['CountryCode'] == 'USA']["Value"], s=100, cmap="Purples_r")
sns.regplot("Year", "Value", data=gini[gini['CountryCode'] == 'USA'], ax=ax4)
ax4.set_title("GINI Index of USA")

points = ax5.scatter(gini[gini['CountryCode'] == 'COL']["Year"], gini[gini['CountryCode'] == 'COL']["Value"],
                     c=gini[gini['CountryCode'] == 'COL']["Value"], s=100, cmap="YlOrBr")
sns.regplot("Year", "Value", data=gini[gini['CountryCode'] == 'COL'], ax=ax5)
ax5.set_title("GINI Index of Colombia")

points = ax6.scatter(gini[gini['CountryCode'] == 'AUS']["Year"], gini[gini['CountryCode'] == 'AUS']["Value"],
                     c=gini[gini['CountryCode'] == 'AUS']["Value"], s=100, cmap="Blues")
sns.regplot("Year", "Value", data=gini[gini['CountryCode'] == 'AUS'], ax=ax6)
ax6.set_title("GINI Index of Australia")

points = ax7.scatter(gini[gini['CountryCode'] == 'KEN']["Year"], gini[gini['CountryCode'] == 'KEN']["Value"],
                     c=gini[gini['CountryCode'] == 'KEN']["Value"], s=100, cmap="Blues")
sns.regplot("Year", "Value", data=gini[gini['CountryCode'] == 'KEN'], ax=ax7)
ax7.set_title("GINI Index of Kenya")

points = ax8.scatter(gini[gini['CountryCode'] == 'CAF']["Year"], gini[gini['CountryCode'] == 'CAF']["Value"],
                     c=gini[gini['CountryCode'] == 'CAF']["Value"], s=100, cmap="winter")
sns.regplot("Year", "Value", data=gini[gini['CountryCode'] == 'CAF'], ax=ax8)
ax8.set_title("GINI Index of Central African Republic")

points = ax9.scatter(gini[gini['CountryCode'] == 'IDN']["Year"], gini[gini['CountryCode'] == 'IDN']["Value"],
                     c=gini[gini['CountryCode'] == 'IDN']["Value"], s=100, cmap="magma")
sns.regplot("Year", "Value", data=gini[gini['CountryCode'] == 'IDN'], ax=ax9)
ax9.set_title("GINI Index of Indonesia")
#sns.set_style(style="white")
plt.tight_layout()


# ### 2B. Youth Literacy Rate (% of population)
# 
# Onto our next indicator, which has an indicator code of **SE.ADT.1524.LT.ZS** : Youth Literacy rates. Previously on one of my other Kernels, I had published a piece of analysis on global Youth unemployment and it seemed to hit quite a chord with readers. Delving deeper into 

# In[ ]:


data = Indicators[Indicators['IndicatorCode'] == 'SE.ADT.1524.LT.ZS'][Indicators['Year'] == 2010]
x, y = (list(x) for x in zip(*sorted(zip(data['Value'].values, data['CountryName'].values), 
                                                            reverse = False)))

# Plotting using Plotly 
trace2 = go.Bar(
    x=x ,
    y=y,
    marker=dict(
        color=x,
        colorscale = 'Portland',
        reversescale = True
    ),
    name='Percentage of Youth Literacy Rate',
    orientation='h',
)

layout = dict(
    title='Barplot of Youth Literacy Rate',
     width = 600, height = 1200,
    yaxis=dict(
        showgrid=False,
        showline=False,
        showticklabels=True,
#         domain=[0, 0.85],
    ))

fig1 = go.Figure(data=[trace2])
fig1['layout'].update(layout)
py.iplot(fig1, filename='plots')


# ### 2C. Access to Electricity
# 
# Let's now inspect another very important indicator and that would be one of a country's access to electricity.

# In[ ]:


data = Indicators[Indicators['IndicatorCode'] == 'EG.ELC.ACCS.ZS'][Indicators['Year'] == 2012]
x, y = (list(x) for x in zip(*sorted(zip(data['Value'].values, data['CountryName'].values), 
                                                            reverse = False)))

# Plotting using Plotly 
trace2 = go.Bar(
    x=x ,
    y=y,
    marker=dict(
        color=x,
        colorscale = 'Portland',
        reversescale = True
    ),
    name='Percentage of Countries with Access to Electriciy',
    orientation='h',
)

layout = dict(
    title='Barplot of Countries with Access to Electricity',
     width = 700, height = 1600,
    yaxis=dict(
        showgrid=False,
        showline=False,
        showticklabels=True,
#         domain=[0, 0.85],
    ))

fig1 = go.Figure(data=[trace2])
fig1['layout'].update(layout)
py.iplot(fig1, filename='plots')


# ### *TO BE CONTINUED. NEED TO SLEEP* 

# In[ ]:




