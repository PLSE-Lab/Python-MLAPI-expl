#!/usr/bin/env python
# coding: utf-8

# This tutorial will cover how to work with the new format for the [CDC Mortality dataset](https://www.kaggle.com/cdc/mortality). 
# 
# Compared to v1 the sql files have all been dropped, the data is now in one single CSV per year, and all of the lookup tables are bundled into a single json per year. You'll find some new lookup tables, such as the `358_cause_recode`. This reflects a fundamental change in how the data was prepared. I'm parsing the [pdf manuals for each year](https://www.cdc.gov/nchs/nvss/mortality_public_use_data.htm) with [Tabula](http://tabula.technology/), then using that data to unpack [the raw fixed width files provided by the CDC](https://www.cdc.gov/nchs/data_access/vitalstatsonline.htm#Mortality_Multiple). This allows makes it possible to work with this dataset with a minimum of manual data entry. 
# 
# If you're curious, you can review my data preparation code [here](https://gist.github.com/SohierDane/0f2cf7a8538ca35431dd7575ac38e7ca).
# 
# You'll note that this method generates lookup tables that vary year from year. In recent years with high quality pdfs these reflect actual tweaks to the manuals. Some years changed a single cause grouping code, though it's unclear if this was intentional or a typo in the source. The further back in time we go the more the pdfs resemble raw scans and the more likely it is that a change is simply a character recognition error. I don't expect this will be a problem until we get into the 1990's, but if you see an unexpected result please check the source manual and let us know if you find an error. If you need to check a large batch of data, try validating what you're seeing with the [CDC's Wonder research tool](https://wonder.cdc.gov/). 

# In[1]:


import json
import pandas as pd


# We'll just read in small slice of the data so that everything runs quickly.
# 
# The codes are strings but often look number-like, such as '012', so I'll enforce the object datatype to prevent pandas from dropping leading zeroes.

# In[2]:


data_2015 = pd.read_csv('../input/2015_data.csv', nrows=10**4, dtype=object)
with open('../input//2015_codes.json', 'r') as f_open:
    code_maps_2015 = json.load(f_open)


# Next, we'll take a look at the most common causes of death in our sample.

# In[3]:


print(data_2015['358_cause_recode'].mode()[0])


# That's not terribly informative. Let's decode that value and try again.

# In[4]:


data_2015['decoded_358_cause'] = data_2015['358_cause_recode'].apply(
    lambda x: code_maps_2015['358_cause_recode'][x])


# In[5]:


print(data_2015['decoded_358_cause'].mode()[0])


# Better, but still not ideal. We've run into one of the main known issues with the current data processing method. ICD cause groupings have both summary and detail entries but the information about these linkages is lost. Please post in the forums if this is causing problems for you; I can probably rebuild these links from the precise ICD code ranges (like `(I26-I51)`) if there is interest.
# 
# If we scan through the cause recodes, it becomes clear that this code is a subset of `090` and the group is a type of `malignant neoplasm`. In plain 
# english, a cancer.

# In[6]:


code_maps_2015['358_cause_recode']


# This was a trivial example, but the same methods can be used to read and decode data for all of the fields for all years. As you can see below, there's plenty of information here to work with Happy hacking!

# In[7]:


data_2015.info()

