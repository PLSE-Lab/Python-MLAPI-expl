#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from functools import reduce
import plotly
import plotly.plotly as py
import cufflinks as cf
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.offline as offline
offline.init_notebook_mode()
from plotly import tools
import plotly.tools as tls


# In[ ]:


get_ipython().run_line_magic('load_ext', 'sql')


# In[ ]:


get_ipython().run_line_magic('sql', 'mysql+pymysql://root:@fe512_mysql/fe512db')


# In[ ]:


get_ipython().run_cell_magic('sql', '', "SET sql_mode = '';")


# # Discovery of Barcelona Dataset
# ### Group #10
# ### Artun Asaduroglu - Haihan Hi

# ## The datasource used for this project is Govermntal Data created by the municipality of Barcelona. 
# 
# #### Data Description
# Open Data BCN, a project that was born in 2010, implementing the portal in 2011, has evolved and is now part of the Barcelona Ciutat Digital strategy, fostering a pluralistic digital economy and developing a new model of urban innovation based on the transformation and digital innovation of the public sector and the implication among companies, administrations, the academic world, organizations, communities and people, with a clear public and citizen leadership.
# 
# #### Data Dictionary
# 
# The data set consists of 17 tables (5-15 columns each), the range of records  is between 48 - 560,000. The average number of record per table is about 80,000 and the total number of record is 1,295,958.
# 
# Data can be categorized in four main category which are Demography, Accidents, Enviroment, Transportation
# 
#    ##### Demography
# 
#     Births -> Births by nationalities and by neighbourhoods of the city of Barcelona (2013-2017).
# 
#     Deaths -> Deaths by quinquennial ages and by neighbourhoods of the city of Barcelona (2015-2017).
# 
#     Population -> Population by neighbourhood, by quinquennial ages and by genre of the city of Barcelona (2013-2017). Reading registers of inhabitants.
# 
#     Unemployment -> Registered unemployement by neighbourhood and genre in the city of Barcelona (2013-2017).
# 
#     Immigrants by Nationality -> Immigrants by nationality and by neighbourhoods of the city of Barcelona (2015-2017).
# 
#     Immigrants Emigrants by Age -> Immigrants and emigrants by quinquennial ages and by neighbourhood of the city    of Barcelona (2015-2017).
# 
#     Immigrants Emigrants by Destination -> Immigrants and emigrants by place of origin and destination, respectively (2017).
# 
#     Immigrants Emigrants by Destination2 -> Immigrants and emigrants by place of origin and destination, respectively, and by neighbourhoods of the city of Barcelona (2017).
# 
#     Immigrants Emigrants by Sex -> Immigrants and emigrants by sex by neighbourhoods of the city of Barcelona (2013-2017).
# 
#     Most Frequent Baby Names -> 25 Most common baby names in Barcelona, disaggregated by sex. Years 1996-2016.
# 
#     Most Frequent Names -> 50 Most common names of the inhabitants of Barcelona, disaggregated by decade of birth and sex.
# 
#    ##### Accidents
# 
#     Accidents(2017) -> List of accidents handled by the local police in the city of Barcelona. Incorporates the  number of injuries by severity, the number of vehicles and the point of impact.
# 
#    ##### Environment
# 
#     Air Quality(Nov2017) -> Air quality information of the city of Barcelona. Mesure data are showed of O3 (tropospheric Ozone), NO2 (Nitrogen dioxide), PM10 (Suspended particles).
# 
#     Air Stations(Nov2017) -> Air quality measure stations of the city of Barcelona. Main characteristics of each one are detailed.
# 
#    #####  Transportation
# 
#     Bus Stops -> Bus stops, day bus stops, night bus stops, airport bus stops of the city of Barcelona.
# 
#     Transports -> Public transports (underground, Renfe, FGC, funicular, cable car, tramcar, etc) of the city of    Barcelona.
# 
# 
# 
# #### ER Diagram 
# To create the ER diagram MYSQL workbench is used. After completing the primary and foreign key relation in the dataset, using the "Reverse Engineer" tool the ER diagram is constructed.
# 
# #### Visulization 
# For visulization Python(Plotly&Seaborn) and Tableau is used. 
# 
# Python is directly used through Jupyter notebook using sql tools. Instead of creating new tables for visuzilation the result of queries is assigned directly to a dataframe.
# 
# Tableau is used to create geogrephica maps. Longtitude and Latitutde information which are in some tables is used to create interactive maps. However, connecting Tableau and Jupyter is not possible on Stevens account, but possible to link using internal server.
# 
# 
# #### Data Source:
# 
# Below the original data source where the data(Municipality of Barcelona) is published separately and the data source(Kaggle) where all the excel files are aggregated.
# 
# Original Data Source : https://opendata-ajuntament.barcelona.cat/en 
# 
# Data downloaded from : https://www.kaggle.com/xvivancos/barcelona-data-sets
# 
# #### Referenrences:
# 
# https://towardsdatascience.com/jupyter-magics-with-sql-921370099589
# 
# 
# http://www.bigendiandata.com/2017-06-27-Mapping_in_Jupyter/
# 
# https://plot.ly/python/bar-charts/

# In[ ]:


get_ipython().run_cell_magic('sql', '', 'CREATE DATABASE barcelona;')


# In[ ]:


get_ipython().run_cell_magic('sql', '', 'SHOW DATABASES;')


# In[ ]:


get_ipython().run_line_magic('sql', 'USE barcelona;')


# In[ ]:


get_ipython().run_cell_magic('sql', '', 'SET FOREIGN_KEY_CHECKS=0;')


# In[ ]:


get_ipython().run_cell_magic('sql', '', 'show tables')


# ###### ---------------------------------------------------------------------------------------------------
# ### 1 TABLE CREATION & DATA UPLOAD

# In[ ]:


get_ipython().run_cell_magic('sql', '', 'CREATE TABLE IF NOT EXISTS barcelona.accidents (\n    accident_id VARCHAR(15) NULL,\n    district_name VARCHAR(20) NULL,\n    neighborhood_name VARCHAR(45) NULL,\n    street VARCHAR(54) NULL,\n    weekday VARCHAR(9) NULL,\n    month_data VARCHAR(9) NULL,\n    day_data INT NULL,\n    hour_data INT NULL,\n    partoftheday VARCHAR(9) NULL,\n    mildinjuries INT NULL,\n    seriousinjuries INT NULL,\n    victims INT NULL,\n    vehiclesinvolved INT NULL,\n    longitude FLOAT NULL,\n    latitude FLOAT NULL\n);')


# In[ ]:


get_ipython().run_cell_magic('sql', '', 'LOAD DATA INFILE \'/home/data/barcelona/accidents_2017.csv\' INTO TABLE barcelona.accidents\n    FIELDS \n        TERMINATED BY \',\'\n    OPTIONALLY\n        ENCLOSED BY \'"\'\n    LINES\n        TERMINATED BY \'\\r\\n\'\n        IGNORE 1 LINES ;')


# In[ ]:


get_ipython().run_cell_magic('sql', '', 'CREATE TABLE IF NOT EXISTS barcelona.air_quality (\n    station VARCHAR(24) NULL,\n    air_quality VARCHAR(8) NULL,\n    longitude FLOAT NULL,\n    latitude FLOAT NULL,\n    o3_hour VARCHAR(3) NULL,\n    o3_quality VARCHAR(4) NULL,\n    o3_value VARCHAR(3) NULL,\n    n_o2_hour VARCHAR(3) NULL,\n    n_o2_quality VARCHAR(8) NULL,\n    n_o2_value VARCHAR(3) NULL,\n    p_m10_hour VARCHAR(3) NULL,\n    p_m10_quality VARCHAR(8) NULL,\n    p_m10_value VARCHAR(2) NULL,\n    generated1 VARCHAR(16) NULL,\n    date_time INT NULL\n);')


# In[ ]:


get_ipython().run_cell_magic('sql', '', 'LOAD DATA INFILE \'/home/data/barcelona/air_quality_Nov2017.csv\' INTO TABLE barcelona.air_quality\n    FIELDS\n        TERMINATED BY \',\'\n    OPTIONALLY\n        ENCLOSED BY \'"\'\n    LINES \n        TERMINATED BY \'\\r\\n\'\n        IGNORE 1 LINES ;')


# In[ ]:


get_ipython().run_cell_magic('sql', '', 'CREATE TABLE IF NOT EXISTS barcelona.air_stations(\n    station VARCHAR(49) NULL,\n    longitude FLOAT NULL,\n    latitude FLOAT NULL,\n    ubication VARCHAR(64) NULL,\n    district_name VARCHAR(20) NULL,\n    neighborhood_name VARCHAR(37) NULL\n);')


# In[ ]:


get_ipython().run_cell_magic('sql', '', 'LOAD DATA INFILE \'/home/data/barcelona/air_stations_Nov2017.csv\' INTO TABLE barcelona.air_stations\n    FIELDS \n        TERMINATED BY \',\'\n    OPTIONALLY\n        ENCLOSED BY \'"\'\n    LINES TERMINATED BY \'\\r\\n\'\n        IGNORE 1 LINES ;')


# In[ ]:


get_ipython().run_cell_magic('sql', '', "INSERT INTO barcelona.air_stations(station)\nVALUES ('Barcelona - Observ Fabra');")


# In[ ]:


get_ipython().run_cell_magic('sql', '', 'CREATE TABLE IF NOT EXISTS barcelona.births(\n    year_data INT NULL,\n    district_code INT NULL,\n    district_name VARCHAR(20) NULL,\n    neighborhood_code INT NULL,\n    neighborhood_name VARCHAR(45) NULL,\n    gender VARCHAR(6) NULL,\n    number_value INT NULL\n);')


# In[ ]:


get_ipython().run_cell_magic('sql', '', 'LOAD DATA  INFILE \'/home/data/barcelona/births.csv\' INTO TABLE barcelona.births\n    FIELDS \n        TERMINATED BY \',\'\n    OPTIONALLY\n        ENCLOSED BY \'"\'\n    LINES\n        TERMINATED BY \'\\r\\n\'\n        IGNORE 1 LINES ;')


# In[ ]:


get_ipython().run_cell_magic('sql', '', 'CREATE TABLE IF NOT EXISTS barcelona.bus_stops (\n    code_info VARCHAR(4) NULL,\n    transport VARCHAR(16) NULL,\n    longitude FLOAT NULL,\n    latitude FLOAT NULL,\n    bus_stop VARCHAR(73) NULL,\n    district_name VARCHAR(20) NULL,\n    neighborhood_name VARCHAR(45) NULL\n);')

