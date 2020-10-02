#!/usr/bin/env python
# coding: utf-8

# # Import libraries

# In[ ]:


import pandas as pd
import numpy as np
from datetime import datetime, timezone
import folium
from folium import plugins
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

tqdm.pandas()
# %load_ext nb_black


# # Load data

# In[ ]:


taxi = pd.read_csv(
    "../input/taxi-trajectory-data-extended/train_extended.csv.zip",
    sep=",",
    compression="zip",
    low_memory=False,
)


# # Trips over time

# ## Number of trips per day of week and hour

# In[ ]:


sns.set(rc={"figure.figsize": (16, 6)})
data = taxi.pivot_table(
    index="HOUR", columns="WEEKDAY", values="TRIP_ID", aggfunc="count"
)
data.columns = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

ax = sns.heatmap(data, cmap="coolwarm")


# There's clearly a big difference between working days and weekend days.
# 
# On working days, the activity is concentrated between 8 h and 18 h, with a peak at first hour on Mondays.
# 
# Also there's big activity in friday and saturday night, specially this last one.
# 
# Friday evening also shows more activity, compared to the other working days.
# 
# It can be deduced that there are two main sources of activity: work and leisure. The first one take place on working days and working hours. The second one, in Friday evening and night and Saturday night.
# 
# It can not be ignored that weekday and hour have and sound correlation with activity.

# ## Number of trips per day of week and hour faceted per call type

# In[ ]:


def draw_heatmap(*args, **kwargs):
    data = kwargs.pop("data")
    d = data.pivot_table(
        index="HOUR", columns="WEEKDAY", values="TRIP_ID", aggfunc="count"
    )
    d.columns = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    sns.heatmap(d, **kwargs)


data = taxi[["TRIP_ID", "CALL_TYPE", "WEEKDAY", "HOUR"]].copy()
data["CALL_TYPE"] = data.CALL_TYPE.astype("category")
data.CALL_TYPE.cat.rename_categories(
    {"A": "CENTRAL", "B": "STAND", "C": "OTHER"}, inplace=True
)

g = sns.FacetGrid(data, col="CALL_TYPE", height=6, aspect=1)
g = g.map_dataframe(draw_heatmap, "WEEKDAY", "HOUR", cmap="coolwarm")


# The way a trip is demanded, when analised in a timely dimension, is a factor that could help to classify different behaviours.
# 
# The most self-evident observation, is that the other call type meets at a very specific point in time: the Friday and Saturday nights. Possibly it's because the taxi drivers go to leisure areas to pick up clients, rather than wait at the stand or wait for a phone call.
# 
# Trips dispatched from the central have a big contrast between the first hour in the morning and the rest of the day. And a mild peak at Friday and Saturday evenings.
# 
# Trips demanded in a stand, has a flatter behavior, with specific levels and times of activity.
# 
# Call type has, certainly, a big correlation with activity.

# ## Number of trips throught the year

# In[ ]:


# Drop last day (30.06.14) as it is overlapping (week 27 / weekday 0)
# with the first day previous year (01.07.13) and trips of both days are aggregated
taxi = taxi[(taxi.YEAR != 2014) | (taxi.MONTH != 6) | (taxi.DAY != 30)]

taxi["WEEK_NUMBER"] = taxi.TIMESTAMP.apply(
    lambda x: datetime.fromtimestamp(x).isocalendar()[1]
)

data = taxi.pivot_table(
    index="WEEKDAY", columns="WEEK_NUMBER", values="TRIP_ID", aggfunc="count"
)

data.set_index(
    pd.Series(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]), inplace=True
)

# Reverse weekdays
data = data.iloc[::-1]

sns.set(rc={"figure.figsize": (16, 2)})
ax = sns.heatmap(data, cmap="coolwarm")


# Relevant sigle days:
# 
#  * 1 January: New Year's Eve
#  * 26 November: Strong citizen protests (not repetitive)
#  * 6 June: ???
#  * 25 December: Christmas Day
#  
#  Relevant multiple consecutive days:
#  
#  * 3 to 10 May: ???
#  * 5 to 8 June: ???
#  * August: holidays

# # Trip distance and duration over time

# ## Trip distance and time per day of week and hour

# In[ ]:


sns.set(rc={"figure.figsize": (16, 6)})

# Drop extreme values as the could spoil averages
taxi = taxi[(taxi.TRIP_DISTANCE < taxi.TRIP_DISTANCE.quantile(0.99))]

distance = taxi.pivot_table(
    index="HOUR", columns="WEEKDAY", values=["TRIP_DISTANCE"], aggfunc=np.mean
)
distance.columns = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

# Drop extreme values as the could spoil averages
taxi = taxi[(taxi.TRIP_TIME < taxi.TRIP_TIME.quantile(0.99))]

time = taxi.pivot_table(
    index="HOUR", columns="WEEKDAY", values=["TRIP_TIME"], aggfunc=np.mean
)
time.columns = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

fig, axes = plt.subplots(1, 2)

sns.heatmap(distance, cmap="coolwarm", ax=axes[0])
sns.heatmap(time, cmap="coolwarm", ax=axes[1])


# Although distance and time are highly correlated, the means are not. 
# 
# In distance, there's a big group of trips longer than the rest. Those are trips very early at night on working days. Extremely accused, on Monday. Maybe there are trips to airport to get the firsts flights.
# 
# In time, by night and weekends, the trip time is below the average. But at evenings, specially on Fridays, the trips take longer, maybe caused by dense traffic.

# ## Trip distance per day of week and hour faceted per call type

# In[ ]:


def draw_heatmap(*args, **kwargs):
    data = kwargs.pop("data")
    d = data.pivot_table(
        index=args[0], columns=args[1], values=args[2], aggfunc=np.mean
    )
    d.columns = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    sns.heatmap(d, **kwargs)
#     print(d.values.min())
#     print(d.values.max())


data = taxi[
    ["CALL_TYPE", "WEEKDAY", "HOUR", "TRIP_DISTANCE", "TRIP_TIME"]
].copy()
data["CALL_TYPE"] = data.CALL_TYPE.astype("category")
data.CALL_TYPE.cat.rename_categories(
    {"A": "CENTRAL", "B": "STAND", "C": "OTHER"}, inplace=True
)

g = sns.FacetGrid(data, col="CALL_TYPE", height=6, aspect=1)
g = g.map_dataframe(
    draw_heatmap,
    "HOUR",
    "WEEKDAY",
    "TRIP_DISTANCE",
    vmin=5.2,
    vmax=12.3,
    cmap="coolwarm",
)


# 

# ## Trip time per day of week and hour faceted per call type

# In[ ]:


g = sns.FacetGrid(data, col="CALL_TYPE", height=6, aspect=1)
g = g.map_dataframe(
    draw_heatmap,
    "HOUR",
    "WEEKDAY",
    "TRIP_TIME",
    vmin=8.6,
    vmax=15.6,
    cmap="coolwarm",
)


# 

# In[ ]:


## Trips distance throught the year


# In[ ]:


# Drop last day (30.06.14) as it is overlapping (week 27 / weekday 0)
# with the first day previous year (01.07.13) and trips of both days are aggregated
taxi = taxi[(taxi.YEAR != 2014) | (taxi.MONTH != 6) | (taxi.DAY != 30)]

taxi["WEEK_NUMBER"] = taxi.TIMESTAMP.apply(
    lambda x: datetime.fromtimestamp(x).isocalendar()[1]
)

data = taxi.pivot_table(
    index="WEEKDAY",
    columns="WEEK_NUMBER",
    values="TRIP_DISTANCE",
    aggfunc=np.mean,
)

data.set_index(
    pd.Series(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]), inplace=True
)

# Reverse weekdays
data = data.iloc[::-1]

sns.set(rc={"figure.figsize": (16, 2)})
ax = sns.heatmap(data, cmap="coolwarm")


# In[ ]:


# Drop last day (30.06.14) as it is overlapping (week 27 / weekday 0)
# with the first day previous year (01.07.13) and trips of both days are aggregated
taxi = taxi[(taxi.YEAR != 2014) | (taxi.MONTH != 6) | (taxi.DAY != 30)]

taxi["WEEK_NUMBER"] = taxi.TIMESTAMP.apply(
    lambda x: datetime.fromtimestamp(x).isocalendar()[1]
)

data = taxi.pivot_table(
    index="WEEKDAY", columns="WEEK_NUMBER", values="TRIP_TIME", aggfunc=np.mean
)

data.set_index(
    pd.Series(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]), inplace=True
)

# Reverse weekdays
data = data.iloc[::-1]

sns.set(rc={"figure.figsize": (16, 2)})
ax = sns.heatmap(data, cmap="coolwarm")

