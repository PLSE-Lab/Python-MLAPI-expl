#!/usr/bin/env python
# coding: utf-8

# 
# **Parsing Nested JSON with Pandas**
# 
# Nested JSON files can be painful to flatten and load into Pandas.

# In[ ]:


import json 
import pandas as pd 
from pandas.io.json import json_normalize #package for flattening json in pandas df

#load json object
with open('../input/raw_nyc_phil.json',"r") as file:
    d = json.load(file)

print("Type", type(d))


# In[ ]:


#lets put the data into a pandas df
#clicking on raw_nyc_phil.json under "Input Files"
#tells us parent node is 'programs'
nycphil = json_normalize(d['programs'])
print(type(nycphil))
nycphil.head(3)


# We see (at least) two nested columns, ```concerts``` and ```works```. Json_normalize [docs](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.io.json.json_normalize.html) give us some hints how to flatten semi-structured data further. Let's unpack the ```works``` column into a  standalone dataframe. We'll also  grab the flat columns so we can do analysis. The parameters here are a bit unorthodox, see if you can understand what is happening.
# 

# In[ ]:


works_data = json_normalize(data=d['programs'], record_path='works', 
                            meta=['id', 'orchestra','programID', 'season'])
works_data.head(3)


# Great! We:
# 
# 1. passed the json object data path ```d[programs]```
# 
# 2. passed the record path within the object we wanted to parse ```works```
# 
# 3. passed the parent metadata we wanted to append
# 
# Your turn: can you unpack the ```concerts``` data?
# 

# In[ ]:


#flatten concerts column here
concerts_data = json_normalize(data = d["programs"], record_path="concerts", meta=['id', 'orchestra','programID', 'season'])
concerts_data.head(3)


# **Deeply Nested Data**
# 
# So what if you run into a nested array inside your nested array? If you go back and look at the flattened ```works_data```, you can see a *second* nested column, ```soloists```. Luckily, json_normalize docs show that you can pass in a list of columns, rather than a single column, to the record path to directly unflatten deeply nested json.
# 
# Let's flatten the ```'soloists'``` data here by passing a list. Since ```soloists``` is nested in ```works```, we can pass that as:

# In[ ]:


soloist_data = json_normalize(data=d['programs'], record_path=['works', 'soloists'], 
                              meta=['id'])
soloist_data.head(3)


# Hope you enjoyed this quick tutorial and JSON parsing is a bit less daunting. 
# 
# **Fork this notebook** to complete the ```concerts``` dataframe, and try playing around with changing the different parameters.

# But what if I want to flatten concerts and works together?
# 

# In[ ]:


intermediate_level1_1 = json_normalize(data = d["programs"], record_path = "works", meta = ['id', 'orchestra', 'season', 'programID'])
intermediate_level1_1.head(3)


# In[ ]:


intermediate_level2 = json_normalize(data=d["programs"], record_path=["works", "soloists"], meta=['id'])
intermediate_level2.head(20)


# In[ ]:


intermediate_level1_combined = intermediate_level1_1.merge(intermediate_level2, how="inner", on=['id'])
intermediate_level1_combined.head(6)


# In[ ]:


intermediate_level1_2 = json_normalize(data=d['programs'], record_path='concerts', meta=['id'])
intermediate_level1_2.head(20)


# In[ ]:


total = intermediate_level1_combined.merge(intermediate_level1_2, how="inner", on="id")
total.head(10)


# ## This final created total dataframe can now be written into a csv or saved in a SQL table
