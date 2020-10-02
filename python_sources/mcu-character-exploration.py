#!/usr/bin/env python
# coding: utf-8

# # Marvel Cinematic Universe Character Exploration

# In[ ]:


# Import Library
import pandas as pd


# In[ ]:


# Import Dataset
df = pd.read_csv("../input/mcu-characters/mcu_characters.csv")
df.info()


# In[ ]:


df.head()


# In[ ]:


# Show Missing Data
df.isnull().sum()

