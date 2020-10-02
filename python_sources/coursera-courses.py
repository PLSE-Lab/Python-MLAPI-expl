#!/usr/bin/env python
# coding: utf-8

# # **COURSERA DATA**

# ## Import libraries

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# <br>
# <hr>
# <br>

# ## Import data

# In[ ]:


col_name = ["id", "title", "organizer", "type", "rating", "difficulty", "students"]
df = pd.read_csv("../input/coursera-course-dataset/coursea_data.csv", header=0, names=col_name, index_col="id").sort_values("id")
df.head()


# #### Convert students to integer

# In[ ]:


# Split df['student'] into number and value
s = pd.DataFrame()
s['number'] = pd.to_numeric(df['students'].str[:-1])
s['value'] = df['students'].str[-1]
s.head()


# In[ ]:


# Which type of values do we have?
s['value'].value_counts()


# In[ ]:


# Ok, simply convert 'k' to thousand and 'm' to million
s.loc[s['value']=='k', 'value'] = 1000 
s.loc[s['value']=='m', 'value'] = 1000000
s.head()


# In[ ]:


# Multiply number and value, convert to integer and assign it back to df['students']
df['students'] = pd.to_numeric(s['number']*s['value'], downcast='integer')
df['students'].head()


# #### Info and description of DataFrame

# In[ ]:


df.info()


# In[ ]:


df.describe()


# <br>
# <hr>
# <br>

# ## Top organizers

# In[ ]:


mask = df.organizer.value_counts() >= 10
top_organizers = df.organizer.value_counts()[mask]
top_organizers


# In[ ]:


top_organizers.plot(kind='barh', figsize=(14,6), title="Top Organizers")


# #### Courses of a particular organizer

# In[ ]:


particular_organizer = "Google Cloud"
mask = df["organizer"] == particular_organizer
df[mask].sort_values(by='rating', ascending=False)


# <br>
# <hr>
# <br>

# ## Certificate types

# In[ ]:


cert_types = df.type.value_counts()
cert_types


# It is worth noting that on coursera platform, Professional certificate are made up of courses and/or specialization. Each specialization is also made up of courses. However, not all courses are inside a specialization or professional certificate, some courses are independent and stand alone.

# In[ ]:


mask= df["type"] == "PROFESSIONAL CERTIFICATE"
df[mask].sort_values(by='students', ascending=False)


# <br>
# <hr>
# <br>

# ## Difficulty

# In[ ]:


df.difficulty.value_counts()


# #### Advanced courses

# In[ ]:


mask = df["difficulty"] == "Advanced"
df[mask].sort_values(by='title', ascending=True)


# <br>
# <hr>
# <br>

# ## Top rated courses

# In[ ]:


# 5 star
mask = df['rating']==5.0
df[mask]


# <br>
# <hr>
# <br>

# ## Most popular courses

# In[ ]:


mask = df['students']>=500000
df[mask].sort_values(by='students', ascending=False)


# <br>
# <hr>
# <br>

# ## Search for a keyword in title

# In[ ]:


keyword = 'Data Science'
mask = df['title'].str.find(keyword) != -1
df[mask]


# <br>
# <hr>
# <br>

# In[ ]:




