#!/usr/bin/env python
# coding: utf-8

# A work in progress. Is it possible to predict whether someone will be stopped/arrested just by their race and crime?

# In[ ]:


import numpy as np 
import pandas as pd 

from matplotlib import pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
get_ipython().run_line_magic('matplotlib', 'inline')

import os
print(os.listdir("../input"))

import warnings
warnings.filterwarnings("ignore")


# In[ ]:


data = pd.read_csv("../input/sdpd-data/vehicle_stops_2016_datasd.csv")
race_codes = pd.read_csv("../input/sdpd-race-codes/vehicle_stops_race_codes.csv")


# In[ ]:


data.head()


# In[ ]:


pd.isnull(data).sum()


# In[ ]:


race_codes


# In[ ]:


race_codes_dictionary = {}

for i in race_codes["Race Code"]:
    race_codes_dictionary[i] = race_codes.loc[race_codes["Race Code"] == i, "Description"].iloc[0]

race_codes_dictionary


# In[ ]:


set(data["subject_race"])


# In[ ]:


data = data.dropna(subset=["subject_race"])

data = data.replace("X", "O")

races = list(data["subject_race"])
subject_race = []

for i in races:
    subject_race.append(race_codes_dictionary[i])

data["Subject Race"] = subject_race


# In[ ]:


set(data["subject_sex"])


# In[ ]:


set(data["arrested"])


# In[ ]:


data.shape


# In[ ]:


print(data.keys())


# In[ ]:


data.describe()


# In[ ]:


set(data["stop_cause"])


# In[ ]:


set(data["subject_age"])


# In[ ]:


subject_ages = []

data = data.dropna(subset=["subject_age"])

for i in data["subject_age"]:
    if i == "No Age":
        subject_ages.append(-1)
    else:
        subject_ages.append(int(i))

data["subject_age"] = subject_ages

plt.hist(data["subject_age"])


# In[ ]:


set(data["subject_age"])


# In[ ]:




