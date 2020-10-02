#!/usr/bin/env python
# coding: utf-8

# In this notebook, we will explore

# 1) [Explore the raw data](#Explore+Data)  
# 2) [Reading the data from CSV into Pandas](#Read+CSV+Into+Dataframe)  
# 3) [How to interpret JSON format and convert it into data frame](#Working+on+JSON+data)  
# 4) [Reading JSON into data frames](#Load+data+as+JSON)  
# 5) [Putting learning into action by creating function that reads CSV data and flattens JSON](#Putting+Together)  
# 6) [Reduce the size of the data without losing data significance](#Reduce+Data+Size)  
# 7) [Save data](#Save+Data)

# ## Explore+Data
# ### 1. Lets look at the size of each file

# In[ ]:


import os
idir = "../input/"
for x in os.listdir(idir):
        f = idir + x
        s = os.stat(f)
        num_lines = sum(1 for line in open(f))
        print(x + ":" + str(round(s.st_size / (1024 * 1024)) ) + " MB : " + str(num_lines) + " lines")


# ### 2. Lets review  first 5 lines of the file

# In[ ]:


rows = 0
for line in open(f): 
    print(line)
    rows += 1
    if rows > 4:
        break


# ## Read+CSV+Into+Dataframe

# In[ ]:


import numpy as np 
import pandas as pd 
ifile = idir + "train.csv"
num_rows = 3
df = pd.read_csv(ifile, nrows = num_rows, converters={'fullVisitorId': str})
df.head()


# In[ ]:


df.info()


# ## Working+on+JSON+data
# ### 1. Convert strings into JSON data  
# In the above data frame, columns with JSON data are treated as strings.  We need to convert them to JSON.<br />
# Below code converts JSON String to JSON object

# In[ ]:


import json
j = json.loads(df["totals"][0])
j


# ### 2. Parse json into data frame  
# The below code will help convert json object into a dataframe.

# In[ ]:


from pandas.io.json import json_normalize
json_normalize(j)


# ## Load+data+as+JSON
# With the above snippets, lets load the CSV file with JSON columns in JSON format
# 
# ### 1. Read the dataframe with JSON converters
# 

# In[ ]:


json_cols = ['device', 'geoNetwork', 'totals', 'trafficSource'] # List columns where data is stored in JSON format

# Apply converter to convert JSON format data into JSON object
json_conv = {col: json.loads for col in (json_cols)}

# Read the CSV with the new converter
df = pd.read_csv(idir + "train.csv", 
    dtype={'fullVisitorId': str},
    converters=json_conv, 
    nrows=num_rows)

df.head()


# ### 2. Convert the json data into columns  
# The below code takes the json code and converts it into columns

# In[ ]:


tdf = json_normalize(df["totals"])
tdf.head()


# ### 3. Drop original column
# We dont need the original json column and dropping it will reduce RAM

# In[ ]:


df = df.drop(columns = ["totals"])
df.head()


# We can see the data frame is created with all possible columns.  NaN value in  columns where this column doesnt exist in json 

# ### 4. Adding the columns into original df  
# We now want to add these  columns into the original dataframe.  Lets create a column name by appending the parent column name
# 

# In[ ]:


tdf.columns = ["totals_" + col for col in tdf.columns]
df = df.merge(tdf, left_index=True, right_index=True)
df.head()


# ## Putting+Together

# In[ ]:


def ld_dn_df(csv_file, json_cols, rows_to_load = 100 ): 

    # Apply converter to convert JSON format data into JSON object
    json_conv = {col: json.loads for col in (json_cols)}

    # Read the CSV with the new converter
    df = pd.read_csv(csv_file, 
        dtype={'fullVisitorId': str},
        converters=json_conv, 
        nrows=rows_to_load, 
        low_memory = False
        )
    
    for jcol in json_cols: 
        tdf = json_normalize(df[jcol])
        tdf.columns = [jcol + "_" + col for col in tdf.columns]
        df = df.merge(tdf, left_index=True, right_index=True)
        
    df = df.drop(columns = json_cols)
    return df


# In[ ]:


get_ipython().run_cell_magic('time', '', 'rows_to_load = 1000000\njson_cols = ["totals", "device", "geoNetwork", "trafficSource"]\ntrain_df =  ld_dn_df("../input/train.csv", json_cols, rows_to_load)')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'test_df =  ld_dn_df("../input/test.csv", json_cols, rows_to_load)')


# ## Reduce+Data+Size
# We can take the below steps: <br />
# 1) Reduce the long default values to simple strings<br />
# 2) Download 64 bit numbers to 32 bit where possible<br />
# 3) Create categorical variables from strings<br />
# The below code addresses 1st step

# In[ ]:


def replace_def_vals(df):
    df.replace({'(not set)': np.nan,
               'not available in demo dataset': np.nan,
               '(not provided)': np.nan,
               'unknown.unknown': np.nan,
               '(none)':np.nan,
               '/':np.nan,
               'Not Socially Engaged':np.nan},
              inplace=True)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'replace_def_vals(train_df)')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'replace_def_vals(test_df)')


# ## Save+Data
# The below code saves the files and checks the number of records and the size.

# In[ ]:


f = './train_flat.csv'
train_df.to_csv(f, index = False)
s = os.stat(f)
num_lines = sum(1 for line in open(f))
print(x + ":" + str(round(s.st_size / (1024 * 1024)) ) + " MB : " + str(num_lines) + " lines")


# In[ ]:


f = './test_flat.csv'
test_df.to_csv(f, index = False)
s = os.stat(f)
num_lines = sum(1 for line in open(f))
print(x + ":" + str(round(s.st_size / (1024 * 1024)) ) + " MB : " + str(num_lines) + " lines")


# ### We reduced the size of the data to ~15% of original size!!!  
# Train CSV has gone down from **1434 MB** to **213 MB**

# <b>References</b><br /><br />
# 
# https://www.kaggle.com/julian3833/1-quick-start-read-csv-and-flatten-json-fields<br />
# http://pandas.pydata.org/pandas-docs/stable/merging.html<br />
# https://stackoverflow.com/questions/13293810/import-pandas-dataframe-column-as-string-not-int <br />
# https://docs.python.org/2/library/json.html <br />
# https://stackoverflow.com/questions/845058/how-to-get-line-count-cheaply-in-python <br />
# https://pandas.pydata.org/pandas-docs/stable/generated/pandas.io.json.json_normalize.html <br />
# 
