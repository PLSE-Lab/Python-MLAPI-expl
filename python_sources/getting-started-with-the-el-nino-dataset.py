#!/usr/bin/env python
# coding: utf-8

# # First, let's take a look at what our dataset looks like.

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("../input/elnino.csv")

# Remove extra space in columns
df.columns = [col.strip() for col in df.columns]

df.head()


# In[ ]:


# Air Temp summary statistics
df['Air Temp'] = pd.to_numeric(df['Air Temp'], errors='coerce')
df['Air Temp'].describe()


# In[ ]:


# Sea Surface Temp summary statistics
df['Sea Surface Temp'] = pd.to_numeric(df['Sea Surface Temp'], errors='coerce')
df['Sea Surface Temp'].describe()


# # It's often easier to see relationships visually. Let's see if there's any relationship between Air Temp and Sea Surface Temp using a Seaborn joint scatterplot. 

# In[ ]:


sns.jointplot(x="Air Temp", y="Sea Surface Temp", data=df, size=7)


# # Other ideas to explore:  
# * How do the variables relate to each other?
# * Which variables have a greater effect on the climate variations?
# * Does the amount of movement of the buoy effect the reliability of the data?
