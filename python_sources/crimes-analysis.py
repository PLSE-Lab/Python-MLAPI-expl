#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set(style="ticks", color_codes=True)

# Importing and connecting the datasets
data1 = pd.read_csv('../input/dt1.csv')
data2 = pd.read_csv('../input/dt2.csv')
frames = [data1, data2]
data_all = pd.concat(frames, ignore_index = True)

# Removing Total from column DISTRICT
data_all = data_all[data_all['DISTRICT']!='TOTAL']
# Checking if there are null cells and describing data
data_all.info()
data_all.describe(include='all')


# First Observations:
# - Cruelty by husband or relatives is the most common crime, the rarest - importation
# - Most of criminal districts are in state Uttar Pradesh
# - We need more general data at the beginning

# In[ ]:


# Let's skip now different districts and show total number of crimes in states
data = data_all.drop("DISTRICT", axis=1)
data = data.groupby(["STATE/UT"], as_index=False).agg({'Rape':sum, 'Kidnapping and Abduction':sum, 
                   'Dowry Deaths':sum, 
                   'Assault on women with intent to outrage her modesty':sum,	
                   'Insult to modesty of Women':sum, 
                   'Cruelty by Husband or his Relatives':sum,	
                   'Importation of Girls':sum})

#adding new column with sum of crimes for each state
data['total'] = data[['Rape', 'Kidnapping and Abduction', 'Dowry Deaths',	
    'Assault on women with intent to outrage her modesty', 
    'Insult to modesty of Women','Cruelty by Husband or his Relatives',	
    'Importation of Girls']].sum(axis=1)


data.sort_values(by='total', ascending=False).head()


# In[ ]:


pieState = data.sort_values(by='total', ascending=False).set_index('STATE/UT')[:30]
pd.Series(pieState['total']).plot.pie(figsize=(13, 13), autopct='%0.1f')


# We know that most of crimes appeared in 'ANDHRA PRADESH', 'UTTAR PRADESH', 'WEST BENGAL', 'RAJASTHAN', 'MADHYA PRADESH'. Those states will be considerd in the future. Now we want to observate how many of each crimes appeared in those states (2 the most popular) in years 2001-2013

# In[ ]:


# adding back "Year" column to our data
data = data_all.drop("DISTRICT", axis=1)
data = data.groupby(["STATE/UT", "Year"], as_index=False).agg({'Rape':sum, 'Kidnapping and Abduction':sum, 
                   'Dowry Deaths':sum, 
                   'Assault on women with intent to outrage her modesty':sum,	
                   'Insult to modesty of Women':sum, 
                   'Cruelty by Husband or his Relatives':sum,	
                   'Importation of Girls':sum})

# creating 'Crime' column instead of existing columns with various kinds of crimes
data_long = pd.melt(data, id_vars = ["STATE/UT", "Year"], value_vars = ['Rape','Kidnapping and Abduction',	'Dowry Deaths',	
                                        'Assault on women with intent to outrage her modesty',	'Insult to modesty of Women',	
                                        'Cruelty by Husband or his Relatives',	'Importation of Girls'], var_name = 'Crime')
data_long.head()


# In[ ]:


#Number of each crimes dependent on year for each states altogether
sns.pointplot(data=data_long, x = 'Year',  y = 'value', hue = 'Crime')


# Cruelty by husband or his relatives is the most often crime and increases among years. Assault and kidnapping have raising tendency in last 2 years too. Importation, dowry deaths and insult are keeping constant, smaller values. Rapid increase of crimes amount is observed in last years.

# In[ ]:


#Number of each crimes changing in years 2001-2013 for two the most dangerous states
data_choosen = data_long.loc[(data_long['STATE/UT'] == 'ANDHRA PRADESH') | (data_long['STATE/UT'] == 'UTTAR PRADESH')]
grid = sns.FacetGrid(data_choosen, col= 'STATE/UT', row = 'Crime')
grid.map(sns.pointplot, 'Year', 'value')
grid.add_legend()


# Number of most of crimes in Andhra and Uttar are comparable, as we see on graphs above. 
# It would be interesting in which districts most of crimes happened in last year 2013.

# In[ ]:


data_st_dist = data_all.loc[data_all['Year']==2012]
data_st_dist = data_st_dist.drop("Year", axis = 1)
data_st_dist = data_st_dist.loc[data_st_dist['STATE/UT'] == 'ANDHRA PRADESH']

# adding "total" column again
data_st_dist['total'] = data_st_dist[['Rape','Kidnapping and Abduction', 'Dowry Deaths',
    'Assault on women with intent to outrage her modesty', 
    'Insult to modesty of Women','Cruelty by Husband or his Relatives',
    'Importation of Girls']].sum(axis=1)


data_st_dist.sort_values(by='total', ascending=False).head()


# VIJAYAWADA CITY, HYDERABAD CITY, CYBERABAD and KRISHNA in ANDHRA PRADESH are the most dangerous districts for women to live nowadays. UTTAR PRADESH is the second dangerous state.
# 
