#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


get_ipython().system('git clone "https://github.com/pursh2002/analyzing_NYC_high_school_data.git"')


# In[ ]:


import os
os.chdir('/kaggle/working/analyzing_NYC_high_school_data/')


# In[ ]:


ls


# - For the purposes of this project, we'll be using data about New York City public schools, which can be found here.https://data.cityofnewyork.us/browse?category=Education
# 

# - Reading in the Data

# In[ ]:


import pandas as pd 
data_files = [
    "ap_2010.csv",
    "class_size.csv",
    "demographics.csv",
    "graduation.csv",
    "hs_directory.csv",
    "sat_results.csv"
]
data = {}
for f in data_files:
    d = pd.read_csv("schools/{0}".format(f))
    key_name = f.replace(".csv","")
    data[key_name] = d


# - Exploring the SAT Data**

# In[ ]:


data['sat_results'].head()


# - Exploring the Remaining Data
# -Given these observations, let's explore the other data sets to see if we can gain any insight into how to combine them.

# In[ ]:


for k in data:
    print(data[k].head())


# * Reading in the Survey Data
# - The files are tab delimited and encoded with Windows-1252 encoding. An encoding defines how a computer stores the contents of a file in binary. The most common encodings are UTF-8 and ASCII. Windows-1252
# - After we read in the survey data, we'll want to combine it into a single dataframe. We can do this by calling the pandas.concat() function: z = pd.concat([x,y], axis=0)

# In[ ]:


all_survey = pd.read_csv('schools/survey_all.txt',delimiter="\t",encoding="windows-1252")
d75_survey = pd.read_csv('schools/survey_d75.txt',delimiter="\t",encoding="windows-1252")

survey = pd.concat([all_survey,d75_survey],axis=0)

survey.head()


# Cleaning Up the Surveys

# - There are two immediate facts that we can see in the data:
# 
# - There are over 2000 columns, nearly all of which we don't need. We'll have to filter the data to remove the unnecessary ones. Working with fewer columns will make it easier to print the dataframe out and find correlations within it.
# The survey data has a dbn column that we'll want to convert to uppercase (DBN). The conversion will make the column name consistent with the other data sets.
# - https://data.cityofnewyork.us/Education/2011-NYC-School-Survey/mnz3-dyi8

# In[ ]:


survey["DBN"] = survey["dbn"]
survey_fields = [
    "DBN", 
    "rr_s", 
    "rr_t", 
    "rr_p", 
    "N_s", 
    "N_t", 
    "N_p", 
    "saf_p_11", 
    "com_p_11", 
    "eng_p_11", 
    "aca_p_11", 
    "saf_t_11", 
    "com_t_11", 
    "eng_t_11", 
    "aca_t_11", 
    "saf_s_11", 
    "com_s_11", 
    "eng_s_11", 
    "aca_s_11", 
    "saf_tot_11", 
    "com_tot_11", 
    "eng_tot_11", 
    "aca_tot_11",
]
# Filter survey so it only contains the columns we listed above. You can do this using pandas.DataFrame.loc[].
survey = survey.loc[:,survey_fields]
# Assign the dataframe survey to the key survey in the dictionary data.
data["survey"] = survey

print(survey.head())


# - Inserting DBN Fields

# - When we explored all of the data sets, we noticed that some of them, like class_size and hs_directory, don't have a DBN column. hs_directory does have a dbn column, though, so we can just rename it.

# In[ ]:


data['class_size'].head()


# In[ ]:


data['sat_results'].head()


# - From looking at these rows, we can tell that the DBN in the sat_results data is just a combination of the CSD and SCHOOL CODE columns in the class_size data. The main difference is that the DBN is padded, so that the CSD portion of it always consists of two digits. That means we'll need to add a leading 0 to the CSD if the CSD is less than two digits long. Here's a diagram illustrating what we need to do:
# - As you can see, whenever the CSD is less than two digits long, we need to add a leading 0. We can accomplish this using the pandas.Series.apply() method, along with a custom function that:
# 1. Takes in a number.
# 1. Converts the number to a string using the str() function.
# 1. Check the length of the string using the len() function.
# 1. If the string is two digits long, returns the string.
# 1. If the string is one digit long, adds a 0 to the front of the string, then returns it.
# 1. You can use the string method zfill() to do this.
# 

# - Once we've padded the CSD, we can use the addition operator (+) to combine the values in the CSD and SCHOOL CODE columns. Here's an example of how we would do this:
# 
# dataframe["new_column"] = dataframe["column_one"] + dataframe["column_two"]

# In[ ]:


# copy the dbn column in hs_directory into a new column called DBN.
data['hs_directory']['DBN']= data['hs_directory']['dbn']

# zfill() pads string on the left with zeros to fill width
def pad_csd(num):
    return str(num).zfill(2)

# Create a new column called padded_csd in the class_size data set
# Use the pandas.Series.apply() method along with a custom function to generate this column.
data['class_size']['padded_csd'] = data["class_size"]["CSD"].apply(pad_csd)
# Use the addition operator (+) along with the padded_csd and SCHOOL CODE columns of class_size, then assign the result to the DBN column of class_size
data["class_size"]["DBN"] = data["class_size"]["padded_csd"] + data["class_size"]["SCHOOL CODE"]
print(data["class_size"].head())


# - Combining the SAT Scores

# In[ ]:




