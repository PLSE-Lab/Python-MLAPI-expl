#!/usr/bin/env python
# coding: utf-8

# # Exploratory data analysis & pandas DataFrame tutorial:Crimes in Boston
# ##### Name: Wei Zhang  ; email: zhan151@usc.edu

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # 1. Data attribute
# #### a). Use .read_csv() to load the csv file ,and name the dataframe as crimes. Display the first 5 rows using .head() to have a feeling about data

# In[ ]:


crimes = pd.read_csv("/kaggle/input/crimes-in-boston/crime.csv",header=0,encoding = 'unicode_escape')
crimes.head(5)


# #### b).Use .shape[ ] to answer question: The data includes how many incidents? Hint: Since each row represents one incident, so how many rows in total?

# In[ ]:


crimes.shape[0]


# we have 310973 incidents recorded.

# # 2. Street with highest number of crime incidents
# #### a). apply .groupby() to dataframe to organize the dataframe by street name, get the total number of incidents corresponding to each street
# 

# In[ ]:


streets = crimes.groupby("STREET").size()
streets


# #### b) sort by incident number, so we have the street where crimes are most frequent on top. Show the top 3 dangerous street with their corresponding incidents number

# In[ ]:


streets.sort_values(ascending = False).head(3)


# #### c) return the street name with greatest number of incidents using .idxmax(). Hint: the index is the street name

# In[ ]:


streets.idxmax()


# ##### Observation : Residents should be careful in "WASHINGTON ST". Incidents happened most frequently in "WASHINGTON ST". Government should assign more police force in this area.

# #### d). Select all the incidents that happened in "WASHINGTON ST" and form a new dataframe called Washington. Display first five rows

# In[ ]:


#Washington = crimes[crimes.STREET == "WASHINGTON ST"]
Washington = crimes.query('STREET =="WASHINGTON ST"')
Washington.head(5)


# Explaination: Both of the methods worked to select incidents revelant to "Washington ST". Be careful with the notation inside .query()

# #### e) the police wants to browse the reason of incidents in Washington Street, but the table displays too many irrelevant attributes. Can you show him the column of OFFENSE_CODE_GROUP for each row only?

# In[ ]:


reasons = Washington.loc[:,"OFFENSE_CODE_GROUP"]
reasons

>Explaination: In the .loc[] function, ":" means we want information from every single row, "OFFENSE_CODE_GROUP" helps leave the column named "OFFENSE_CODE_GROUP" only
# #### f): Find the most frequent reason behind incidents. Hint: you can use .value_counts () in this case since the above data is a series. Be careful value_counts() does not work for dataframe

# In[ ]:


reasons.value_counts().sort_values(ascending = False).head(3)


# ##### Observation: After investigation, we realized larceny, drug violation, motor vehicle accident are three most frequent cause of the incidents happened in Washington Street. Therefore, the police can tackle and prevent the crimes specifically. 

# # 3. Avgerage annual incidents number
# #### a). group by the crimes dataframe by 'YEAR', obatin the total incidents number for each year, draw a bar plot with x-axis as the year, and y-axis the frequency of incidents

# In[ ]:


years = crimes.groupby("YEAR").size()
years.plot.bar()


# ##### Obervation: The crimes rate culminated during 2016 and 2017.But in overall, the graph shows a decreasing trend of crime rate in Boston. Boston has an improved city safety. 

# #### b). get the annual average

# In[ ]:


years.mean()

