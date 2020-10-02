#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import json

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Any results you write to the current directory are saved as output.


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


print(os.listdir("../input"))


# In[ ]:


COLLISIONS_CSV = '../input/nypd-motor-vehicle-collisions.csv'
METADATA_JSON = '../input/socrata_metadata.json'
DATADICT = '../input/Collision_DataDictionary.xlsx'


# In[ ]:


collisions = pd.read_csv(COLLISIONS_CSV)


# In[ ]:


collisions.head()


# In[ ]:


collisions.tail()


# In[ ]:


collisions.info()


# In[ ]:


print(f'Number of columns: {len(collisions.columns)}')


# In[ ]:


collisions.describe()


# In[ ]:


# Assuming that the DATE and Time columns are sorted (meaning row 0 is latest date and last row represents the earliest date)
latest_date = collisions.iloc[0].DATE
earliest_date = collisions.iloc[-1].DATE


# In[ ]:


latest_date


# In[ ]:


earliest_date


# In[ ]:


# Filter the dataframes into number of injured and number of deaths
injured_df = collisions[collisions['NUMBER OF PERSONS INJURED'] > 0]
killed_df = collisions[collisions['NUMBER OF PERSONS KILLED'] > 0]


# In[ ]:


# Now plot these dataframes per borough
fig, ax = plt.subplots(1, figsize=(12,8))
injured_df.BOROUGH.value_counts().plot(kind='bar', color='blue', position=0, ax=ax)
killed_df.BOROUGH.value_counts().plot(kind='bar', color='red', position=1, ax=ax)
ax.set_title('Number of persons injured and killed per borough', fontsize=20, fontweight='bold')
ax.set_ylabel('Number of persons injured/killed')
plt.show()


# **As can be seen, number of deaths compared to injured is so small its hard to see. So re-plotting it on two axes. Maybe this is confusing, maybe a box plot or something else may be more appropriate******

# In[ ]:


fig = plt.figure(figsize=(12,8))

bx = fig.add_subplot(111)
bx2 = bx.twinx() # Create another axes that shares the same x-axis as bx.
injured_df.BOROUGH.value_counts().plot(kind='bar', color='blue', width=0.4, position=0, ax=bx)
killed_df.BOROUGH.value_counts().plot(kind='bar', color='red', width=0.4, position=1, ax=bx2)
bx.set_title('Number of persons injured and killed per borough', fontsize=20, fontweight='bold')
bx.set_ylabel('Number of persons injured')
bx2.set_ylabel('Number of persons killed')

plt.show()


# ***As can be seen, number of persons killed is way less that injuries, which is encouraging***

# In[ ]:




