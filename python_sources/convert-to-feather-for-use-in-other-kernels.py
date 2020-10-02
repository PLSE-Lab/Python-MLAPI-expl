#!/usr/bin/env python
# coding: utf-8

# All files apart from the NGS files are quite small. The NGS files however, each can contain up to 9 million rows. In that case we need to reduce the amount of memory they take up. In this kernel I will show you how to do this in an easy way.
# 
# The NGS datasets contains player position, speed and direction data for each player during the entire course of the play. The NGS dataset is the only dataset that contains Timeas a variable. 

# ## Import Libraries

# In[ ]:


import numpy as np 
import pandas as pd
import os
import tqdm
import gc
import feather


# In[ ]:


PATH = '../input/'


# ## What data are available

# In[ ]:


print(os.listdir("../input"))


# Let's focus only on the NGS files. Below are the files.

# * NGS-2016-pre.csv
# * NGS-2016-reg-wk1-6.csv
# * NGS-2016-reg-wk7-12.csv
# * NGS-2016-reg-wk13-17.csv
# * NGS-2016-post.csv
# 
# 
# * NGS-2017-pre.csv
# * NGS-2017-reg-wk1-6.csv
# * NGS-2017-reg-wk7-12.csv
# * NGS-2017-reg-wk13-17.csv
# * NGS-2017-post.csv

# ## How many rows are there per file

# In[ ]:


get_ipython().run_cell_magic('time', '', "\nwith open(f'{PATH}NGS-2016-pre.csv') as file:\n    n_rows = len(file.readlines())\nprint('2016 pre rows:', n_rows)\n\nwith open(f'{PATH}NGS-2016-reg-wk1-6.csv') as file:\n    n_rows = len(file.readlines())\nprint('2016 wk1-6 rows:', n_rows)\n\nwith open(f'{PATH}NGS-2016-reg-wk7-12.csv') as file:\n    n_rows = len(file.readlines())\nprint('2016 wk7-12 rows:', n_rows)\n\nwith open(f'{PATH}NGS-2016-reg-wk13-17.csv') as file:\n    n_rows = len(file.readlines())\nprint('2016 wk13-17 rows:', n_rows)\n\nwith open(f'{PATH}NGS-2016-post.csv') as file:\n    n_rows = len(file.readlines())\nprint('2016 post rows:', n_rows)\n\nwith open(f'{PATH}NGS-2017-pre.csv') as file:\n    n_rows = len(file.readlines())\nprint('2017 pre rows:', n_rows)\n\nwith open(f'{PATH}NGS-2017-reg-wk1-6.csv') as file:\n    n_rows = len(file.readlines())\nprint('2017 wk1-6 rows:', n_rows)\n\nwith open(f'{PATH}NGS-2017-reg-wk7-12.csv') as file:\n    n_rows = len(file.readlines())\nprint('2017 wk7-12 rows:', n_rows)\n\nwith open(f'{PATH}NGS-2017-reg-wk13-17.csv') as file:\n    n_rows = len(file.readlines())\nprint('2017 wk13-17 rows:', n_rows)\n\nwith open(f'{PATH}NGS-2017-post.csv') as file:\n    n_rows = len(file.readlines())\nprint('2017 post rows:', n_rows)")


# ## Load the Data and Define Data Types

# All NGS file have the same structure, which means we can load them individually and concatenate them afterwards. Before doing that we can have a sneak peak at what the data look like and then define the data types before loading the csv files. We coud obviously do this once we have loaded the files but this way saves time and memory right away. 

# In[ ]:


# Only load the first 5 rows to get an idea of what the data look like
df_temp = pd.read_csv(f'{PATH}NGS-2016-pre.csv', nrows=5)
df_temp.head()


# In[ ]:


# Get information on the datatypes
df_temp.info()


# In[ ]:


# Find out the smallest data type possible for each numeric feature
float_cols = df_temp.select_dtypes(include=['float'])
int_cols = df_temp.select_dtypes(include=['int'])

for cols in float_cols.columns:
    df_temp[cols] = pd.to_numeric(df_temp[cols], downcast='float')
    
for cols in int_cols.columns:
    df_temp[cols] = pd.to_numeric(df_temp[cols], downcast='integer')

print(df_temp.info())


# By 'downcasting' the numeric features we have almost halved the memory consumption. However, not all data types are correct yet. 
# 1. From the data disctionary we know that **Event** is text and should be data type 'object'. 
# 2. And **Time** is obviously datetime. It's faster though importing it as 'string' and then converting it.
# 3. We can also go a step further with **SeasonYear**. The feature only contains two values: 2016 and 2017. If we changed that to 0 and 1, then we can reduce the data type further from int16 to int8, saving more memory. 
# 4. **GSISID** is only an integer and not a float. However, there are NANs in the data, which prevent us from converting this straight away.
# 
# Now let's define the data types and then upload the NGS files.

# In[ ]:


dtypes = {'Season_Year': 'int16',
         'GameKey': 'int16',
         'PlayID': 'int16',
         'GSISID': 'float32',
         'Time': 'str',
         'x': 'float32',
         'y': 'float32',
         'dis': 'float32',
         'o': 'float32',
         'dir': 'float32',
         'Event': 'str'}

col_names = list(dtypes.keys())


# In[ ]:


ngs_files = ['NGS-2016-pre.csv',
             'NGS-2016-reg-wk1-6.csv',
             'NGS-2016-reg-wk7-12.csv',
             'NGS-2016-reg-wk13-17.csv',
             'NGS-2016-post.csv',
             'NGS-2017-pre.csv',
             'NGS-2017-reg-wk1-6.csv',
             'NGS-2017-reg-wk7-12.csv',
             'NGS-2017-reg-wk13-17.csv',
             'NGS-2017-post.csv']


# In[ ]:


# Load each ngs file and append it to a list. 
# We will turn this into a DataFrame in the next step

df_list = []

for i in tqdm.tqdm(ngs_files):
    df = pd.read_csv(f'{PATH}'+i, usecols=col_names,dtype=dtypes)
    
    df_list.append(df)


# In[ ]:


# Merge all dataframes into one dataframe
ngs = pd.concat(df_list)

# Delete the dataframe list to release memory
del df_list
gc.collect()

# Convert Time to datetime
ngs['Time'] = pd.to_datetime(ngs['Time'], format='%Y-%m-%d %H:%M:%S')

# See what we have loaded
ngs.info()


# ## Final Steps
# 
# 1. We can turn Season_Year into a category with the values 0 and 1 to save more memory. The feature only contains the values 2016 and 2017.
# 2. We can delete the NANs in GSISD and then turn it into an integer. 
# 3. We can save this as a feather file, which is much faster to load. But we might have to define the data types again after loading.

# In[ ]:


# Turn Saeson_Year into a category and ultimately into an integer (this doesn't seem necessary)
if False:
    ngs['Season_Year'] = ngs['Season_Year'].astype('category').cat.codes


# In[ ]:


# There are 2536 out of 66,492,490 cases where GSISID is NAN. Let's drop those to convert the data type
ngs = ngs[~ngs['GSISID'].isna()]


# In[ ]:


# Convert GSISID to integer
ngs['GSISID'] = ngs['GSISID'].astype('int32')


# In[ ]:


# Save to feather so we can use it in other kernels
ngs.reset_index(drop=True).to_feather(f'ngs.feather')


# This concludes this tutorial. I hope you found it useful and I'm keen to hear any feedback and how to improve things.

# In[ ]:


get_ipython().system('ls -lh')


# In[ ]:




