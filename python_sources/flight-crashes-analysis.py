#!/usr/bin/env python
# coding: utf-8

# Setup the environment and load data file

# In[ ]:


import pandas as pd

pd.set_option('max_rows', 5)
crashes = pd.read_csv("../input/planecrashinfo_20181121001952.csv")
# add year 
crashes['CrashYear'] = pd.DatetimeIndex(pd.to_datetime(crashes['date'])).year
crashes['CrashMonth'] = pd.DatetimeIndex(pd.to_datetime(crashes['date'])).month
crashes['CrashDay'] = pd.DatetimeIndex(pd.to_datetime(crashes['date'])).day
crashes


# Year distribution for the crashes

# In[ ]:


crashes.groupby("CrashYear").size().plot(figsize=(25, 6))


# Month :) distribution for crashes

# In[ ]:


crashes.groupby("CrashMonth").size().plot.bar(figsize=(25, 6))


# Let's find out what is the "top 50" operators

# In[ ]:


#Top 50 operators
operators = crashes.groupby("operator").size()
ordered_operators = operators.sort_values(ascending = False)
ordered_operators = ordered_operators.iloc[:50]
ordered_operators.plot.bar(figsize=(25, 6))


# And now check the "top 50" plane models

# In[ ]:


#Top 50 models
models = crashes.groupby("ac_type").size()
ordered_models = models.sort_values(ascending = False)
ordered_models = ordered_models.iloc[:50]
ordered_models.plot.bar(figsize=(25, 6))


# It looks like "Aeroflot" is the "leader" in the crashes. Let's find out the planes statistic for this operator

# In[ ]:


aeroflot = crashes[crashes["operator"] == "Aeroflot"]
aeroflot_ordered_models = aeroflot.groupby("ac_type").size().sort_values(ascending = False)
aeroflot_ordered_models.plot.bar(figsize=(25, 6))


# Next would be to check how the top 50 model statistic looks without the "Aeroflot" crashes

# In[ ]:


not_aeroflot_models = crashes[crashes["operator"] != "Aeroflot"]
not_aeroflot_ordered_models = not_aeroflot_models.groupby("ac_type").size().sort_values(ascending = False)
not_aeroflot_ordered_models.iloc[:50].plot.bar(figsize=(25, 6))


# Let's find the time distribution for "Aeroflot" crashes

# In[ ]:


aeroflot.groupby("CrashYear").size().plot(figsize=(25, 6))


# Let's find who was operating the "Douglas DC-3"

# In[ ]:


douglas_dc3 = crashes[crashes["ac_type"] == "Douglas DC-3"]
douglas_dc3_ordered = douglas_dc3.groupby("operator").size().sort_values(ascending = False)
douglas_dc3_ordered.iloc[:50].plot.bar(figsize=(25, 6))


# Let's find the time distribution for "Douglas DC-3" crashes

# In[ ]:


douglas_dc3.groupby("CrashYear").size().plot(figsize=(25, 6))

