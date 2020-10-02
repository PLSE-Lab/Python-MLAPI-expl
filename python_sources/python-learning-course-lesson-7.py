#!/usr/bin/env python
# coding: utf-8

# # Python Learning Course - Lesson 7
# ## Introduction to Pandas

# In[1]:


import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))


# In[2]:


fao_db = pd.read_csv("../input/FAO.csv", encoding='latin1') # read csv file from directory and input data into a dataframe


# In[3]:


fao_db.head(3) # return the first 3 entries to ensure file was imported


# In[4]:


fao_db.shape # get information about the dataframe shape


# In[5]:


fao_db.index # get information regarding the indexing of the dataframe


# In[6]:


fao_db.columns # get information about the dataframe columns


# In[7]:


fao_db.info() # get information about the value types each columns contains


# In[8]:


fao_db.count() # get information about the number of data in each column - notice different amounts in the years' columns
               # indicates countries leaving or entering production. also indicates new countries or merging of countries.


# In[9]:


fao_db[['Area', 'Y2000']].head(5) # get only the Area and Year 2000 columns


# In[12]:


new_db = fao_db[['Area', 'Y2000']].copy()
new_db.head(5)


# In[30]:


fao_db.iloc[:,2:].head(7) # get all columns after Area column


# In[27]:


fao_db.iloc[4:7,4:8]


# In[31]:


fao_db[fao_db.columns[10:]].head(5) # get only Years


# In[39]:


fao_db[['Area']].head(5) # get only the Area column


# In[40]:


fao_db['Area'].head(5)


# In[35]:


fao_db['Area'].unique() # get unique countries


# In[41]:


len(fao_db['Area'].unique()) # get number of unique countries


# In[42]:


fao_db['Item'].unique()


# In[44]:


len(fao_db['Item'].unique())


# In[45]:


fao_db.loc[fao_db['Area'] == 'Greece']


# In[46]:


fao_db.loc[fao_db['Area'] == 'Greece'].describe()


# In[47]:


fao_db.loc[fao_db['Area'] == 'Greece'].Y1961.describe()


# In[48]:


fao_db.Y1961


# In[49]:


fao_db.Y1961.describe()

