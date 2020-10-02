#!/usr/bin/env python
# coding: utf-8

# # How to crack the code of: 
# ## THE BUREAU OF LABOR STATISTICS
# ### A dataset full of stuff that looks useless
# ---
# 
# Here's a series_id from one of the datafiles.
# `CEU0500000001`
# 
# To the naked eye this string doesn't look like much. It's kind of ugly to look at and it reeks of beaurocratic procedure. Looking at a zillion of these makes me feel like I'm waiting in line at some government office and there's a clerk in the background saying "Well, in accordance with BLS-13903-2, chapter 12, secton 11, subsection 3a, 1 niner..."
# 
# Yikes.  
# <br><br>
# Series ID's actually contain a lot of info, so let's break them out into something useful.
# 
# ![Hooray I'm useful!](https://i.imgur.com/zO3l2LV.gif)
# <br><br>

# In[ ]:


import csv
import numpy as np
import pandas as pd
import os
from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
infile = '../input/all.data.combined.csv'


# In[ ]:


df = pd.read_csv(infile)
df.head()


# I'm going to "expand" the Series IDs into multiple columns so that tit's more interesting. I learned all of this from reading the documentation included in the dataset. It's actually not that bad to read and it's pretty good at explaining this stuff. 
# 
# Here's an example of what we're doing for every series id:

# In[ ]:


# We'll start with this series id
seriesid = 'CEU0500000001'

# Series ids contain these bits of info:

# Survey types
survey_abbreviation = seriesid[:2]
# Seasonal codes
seasonal_code = seriesid[2]
# Industry codes
industry_code = seriesid [3:11]
# Data type codes
data_type_code = seriesid[11:]

print("\n".join(["seriesid:", seriesid]))
series = [survey_abbreviation, seasonal_code, industry_code, data_type_code]
for thing in series:
    print(thing)


# Now I'll expand the components of the Series IDs into their own columns. I'll use a random smple of 20 rows to show what I'm doing on a small scale.

# In[ ]:


# Random sample of the data
sampl = df.sample(n=20)
sampl = pd.DataFrame(sampl.series_id)
sampl.columns = ["series_id"]
sampl.head()


# In[ ]:


# Expand out the Series IDs
sampl["survey_abbrevations"] = sampl.series_id.str[:2]
sampl["seasonal_code"] = sampl.series_id.str[2]
sampl["industry_code"] = pd.to_numeric(sampl.series_id.str[3:11])
sampl["data_type_code"] = pd.to_numeric(sampl.series_id.str[11:])

sampl


# Now I am going to replace the data_type_code column with the actual descriptions

# In[ ]:


# Setting up the mapping dictionary thing
file = '../input/ce.datatype.csv'
datatype = pd.read_csv(file)

# "zip" the two columns into a dictionary
datatypemapper = dict(zip(datatype.data_type_code, datatype.data_type_text))

# Replace the codes with the descriptions
sampl = sampl.replace({"data_type_code": datatypemapper})

# Rename the column
sampl = sampl.rename(columns = {"data_type_code": "data_type"})

sampl


# Next will be the the industry codes. One thing to note here is that the industry codes are also made of multiple components. If you take a look at the ce.industry.csv mapping file you will see all the stuff you can glean from the industry code. For this kernel I'm going to change the industry code into the industry name.

# In[ ]:


file = '../input/ce.industry.csv'
industry = pd.read_csv(file)
industry.head()


# In[ ]:


# "zip" the two columns into a dictionary
industrymapper = dict(zip(industry.industry_code, industry.industry_name))

# Replace the codes with the descriptions
sampl = sampl.replace({"industry_code": industrymapper})

# Rename the column
sampl = sampl.rename(columns = {"industry_code": "industry_name"})

sampl


# 
