#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# # I want to know 
# > the trending of crashes by year ( number of crashes, number of fatalities)

# **Let's load the file and look at some rows**

# In[ ]:


# read csv file
file_path = "../input/planecrashinfo_20181121001952.csv"
data = pd.read_csv(file_path)
data.head(5)


# # About the data:
# * 5783 rows, 13 columns
# * From 1908 to 2018 (Nov 19)
# * non-null object
# * all column dtype:  object

# In[ ]:


data.shape


# In[ ]:


data.info()


# # Columns I will use
# * "date" to extract year
# * "fatalities" and "ground" for  calculating number of crashes, number of fatalities
# 
# So let's remove the others columns

# In[ ]:


data = data[["date", "fatalities", "ground"]]
data.head(5)


# # Let's look each column closer

# * **date**
#  * No missing value
#  * Data format  %B %d, %Y (September 09, 1913)- also need to strip() the value
#  * dtype: object
#  
# * **fatalities**
#   * No missing value
#   * Data format:   ex: 14   (passengers:?  crew:?)
#   * dtype: object
#   
# * **ground**
#   *  Missing value: 52
#   * Data format  %B %d, %Y (September 09, 1913)- also need to strip() the value
#   * dtype: object

# In[ ]:


data.info()


# In[ ]:


data["date"].describe()


# In[ ]:


date_missing_value = data["date"][data["date"]=="?"]
len(date_missing_value)


# In[ ]:


data["fatalities"].describe()


# In[ ]:


fatalities_missing_value = data["fatalities"][data["fatalities"]=="?"]
len(fatalities_missing_value)


# In[ ]:


data["ground"].describe()


# In[ ]:


ground_missing_value = data["ground"][data["ground"]=="?"]
len(ground_missing_value)


# # Clean data
# * Extract "year" from "date" column
# * Extract total_killed from "fatalities" and "ground" columns

# In[ ]:


data["year"] = data["date"].str.rsplit(",", n = 1, expand=True)[1].str.strip()
data["year"] = pd.to_numeric(data["year"], errors="coerce")
data["year"].describe()


# In[ ]:


# extract number of person in columns(aboard, fatalities, ground)
data["fatalities_num"] = data["fatalities"].str.split("(", n = 1, expand=True)[0].str.strip()
data["fatalities_num"] = pd.to_numeric(data["fatalities_num"], errors="coerce")
data["fatalities_num"].describe()


# In[ ]:


# extract number of person in columns(fatalities, ground)
data["ground_num"] = pd.to_numeric(data["ground"].str.strip(), errors="coerce")
data["ground_num"].describe()


# In[ ]:


data["total_killed"] = data["ground_num"] + data["fatalities_num"]
data["total_killed"].describe()


# In[ ]:


data.head(5)


# # Group information by year
# 
# 
# 
# 

# In[ ]:


total_killed= data[["year",  "total_killed"]].groupby("year").sum()
total_crash= data["year"].value_counts().sort_index(ascending=True).rename_axis('year').to_frame('total_crashes')


# # Plot

# In[ ]:


ax = total_crash.plot(figsize=(16,4))
total_killed.plot(ax=ax, secondary_y=True)


# 

# 
