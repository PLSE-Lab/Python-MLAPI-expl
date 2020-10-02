#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


gamedata=pd.read_csv('../input/ign.csv')


# In[3]:


gamedata.head()


# In[4]:


#lets delete some messy column of array from this data frame
del gamedata['Unnamed: 0']
del gamedata['title']
del gamedata['url']
gamedata.head()


# In[5]:


gamedata.describe()


# **SCORE_PHRASE ANALYSIS :**

# In[6]:


gamedata["score_phrase"].value_counts()


# In[7]:


gamedata.score_phrase.value_counts()[:6].plot.pie(figsize=(10,10))#top six scorephrase


# In[8]:


positivescorephrase = ["Great", "Good", "Masterpiece", "Amazing"]
neither_positive_nor_negativescorephrase = ["Medicore", "Okay"]
negativescorephrase = ["Bad","Disaster","Awful","Painful","Unbearable"]
def mapping(item):
    if item in positivescorephrase:
        return "positivescorephrase"
    if item in neither_positive_nor_negativescorephrase:
        return "neither_positive_nor_negativescorephrase"
    if item in negativescorephrase:
        return "negativescorephrase"

gamedata["scorephrasetype"] = gamedata["score_phrase"].map(mapping)


# In[9]:


gamedata["scorephrasetype"].value_counts()


# In[10]:


gamedata.scorephrasetype.value_counts().plot.pie(figsize=(10,10))


# **PLATFORM ANALYSIS :**

# In[11]:


gamedata["platform"].value_counts()


# In[12]:


gamedata.platform.value_counts()[:6].plot.pie(figsize=(10,10))#top six platform


# In[13]:


Computer = ["PC", "Macintosh", "Linux", "Commodore 64/128", "Windows Surface", "SteamOS"]
Console = ["PlayStation 2", "Xbox 360", "Wii", "PlayStation 3", "Nintendo DS", "PlayStation", "Xbox",
           "GameCube", "Nintendo 64", "Dreamcast", "PlayStation 4", "Xbox One", "Wii U", "Genesis",
           "NES", "TurboGrafx-16", "Super NES", "Sega 32X", "Master System", "Nintendo 64DD", "Saturn",
           "Atari 2600", "Atari 5200", "TurboGrafx-CD", "Ouya"]
Portable = ["Nintendo DSi", "PlayStation Portable", "Game Boy Advance", "Game Boy Color", "Nintendo 3DS",
            "PlayStation Vita", "Lynx", "NeoGeo Pocket Color", "Game Boy", "N-Gage", "WonderSwan",
            "New Nintendo 3DS", "WonderSwan Color", "dreamcast VMU"]
Mobile = ["iPhone", "iPad", "Android", "Windows Phone", "iPod", "Pocket PC"]
Arcade = ["Arcade", "NeoGeo", "Vectrex"]

def mapping(item):
    if item in Computer:
        return "Computer"
    if item in Console:
        return "Console"
    if item in Portable:
        return "Portable"
    if item in Mobile:
        return "Mobile"
    if item in Arcade:
        return "Arcade"
    return "Other"

gamedata["platformtype"] = gamedata["platform"].map(mapping)


# In[14]:


gamedata["platformtype"].value_counts()


# In[15]:


gamedata.platformtype.value_counts().plot.pie(figsize=(10,10))


# **SCORE ANALYSIS :**

# In[16]:


scores=gamedata.score
scores.describe()


# In[17]:


gamedata.score.value_counts()[:6.0].plot.pie(figsize=(10,10))#TOP firstclass SCORE


# **GENRE ANALYSIS :**

# In[18]:


gamedata["genre"].value_counts()


# In[19]:


gamedata.genre.value_counts()[:6].plot.pie(figsize=(10,10))#top six genre


# **EDITORS_CHOICE ANALYSIS :**

# In[20]:


gamedata["editors_choice"].value_counts()


# In[21]:


gamedata.editors_choice.value_counts().plot.pie(figsize=(10,10))


# **RELEASE YEAR ANALYSIS :**

# In[22]:


gamedata["release_year"].value_counts()


# In[23]:


gamedata.release_year.value_counts().plot.barh(figsize=(10,10))


# **RELEASE MONTH ANALYSIS :**

# In[24]:


gamedata.release_month. value_counts()


# In[25]:


gamedata.release_month.value_counts().plot.barh(figsize=(10,10))


# **RELEASE DATE ANALYSIS :**

# In[26]:


gamedata.release_day.value_counts()


# In[27]:


gamedata.release_day.value_counts().plot. barh(figsize=(10,10))

