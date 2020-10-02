#!/usr/bin/env python
# coding: utf-8

# ## Introduction
# 
# I've forked the automatically-generated kernel from Kaggle bot to better explore some of the data from LexML.
# Please do continue adding more useful functionnality to it by clicking the blue "Fork Notebook" button at the top of this kernel to begin editing.

# ## Exploratory Analysis
# 
# The default kaggle bot code wasn't plotting any of the data, due to its path location and use of json files.

# In[ ]:


#from mpl_toolkits.mplot3d import Axes3D
#from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# In[ ]:


print(sorted(os.listdir('../input')))


# In[ ]:


len(os.listdir('../input/2019/data/json/2019'))


# In[ ]:


print(sorted(os.listdir('../input/2019/data/json/2019')))


# In[ ]:


# Read contents of one of the json files: 1_2019.json
df = pd.read_json("../input/2019/data/json/2019/1_2019.json")


# In[ ]:


df.columns


# In[ ]:


df.describe(include="object")


# In[ ]:


df.dtypes


# In[ ]:


# Column `data` should be of type date:
df['data2'] = pd.to_datetime(df['data'])


# In[ ]:


# Number of documents per month
df['data2'].groupby(df["data2"].dt.month).count().plot(kind="bar")


# It seems that each of the 136 json files from 2019 contains 1000 references to legal documents each.
# 
# Let's do some basic exploring for columns 'autoridade', 'localidade' and 'tipoDocumento':

# In[ ]:


df['autoridade'].value_counts()


# In[ ]:


df['facet-localidade'].value_counts()


# In[ ]:


df['facet-tipoDocumento'].value_counts()


# ## Ideas for further analysis
# 
# 1. Read all the json files from a single year (or even the whole dataset) into a single dataframe for ease of use in getting aggregated statistics
# 2. Create functions to access the text content from a referenced document by its url from LexML
# 3. Publish some interesting results!

# ## Conclusion
# This concludes your starter analysis! To go forward from here, click the blue "Fork Notebook" button at the top of this kernel. This will create a copy of the code and environment for you to edit. Delete, modify, and add code as you please. Happy Kaggling!
