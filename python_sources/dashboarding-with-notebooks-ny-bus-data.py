#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt


# Let's start by quickly importing our data, and having a look at what it look like.

# In[ ]:


file_path = "../input/bus-breakdown-and-delays.csv"

df = pd.read_csv(file_path)
df.head()


# There are a couple of interesting things we could look at. Two of them are:
# 
# - How many accumulated minutes are lost for students, per day ?
# - How many breakdowns happen per borough ?
# 
# These are the ones we will focus on in this notebook.
# 
# First, we are going to select recent data, from the last `N_OF_DAYS` days, and create a new column, `datetime`, corresponding to event occurence dates as `datetime` objects.

# In[ ]:


N_OF_DAYS = 20

today = datetime.date.today()
first_day = today - datetime.timedelta(N_OF_DAYS)

df['datetime'] = pd.to_datetime(df['Occurred_On'])
recent_data = df[(df.datetime >= pd.Timestamp(first_day)) & (df.datetime <= pd.Timestamp(today))]


# Now, let's compute how many minutes are lost per incident.
# 
# We are going to define a function taking a row as its input, doing some quick data cleaning, and returning the number of minutes lost by students. This function will then be applied to the data we just selected, and stored in a new column (`lost_minutes`).

# In[ ]:


def compute_lost_minutes(row) -> int:
    try:
        n_students = int(row['Number_Of_Students_On_The_Bus'])
    except (ValueError, TypeError):
        print('n_students', row['Number_Of_Students_On_The_Bus'])
        n_students = 0
    try :
        if isinstance(row['How_Long_Delayed'], str):
            n_delay = int(''.join([c for c in row['How_Long_Delayed'] if c in '0123456789']))
        else :
            n_delay = row['How_Long_Delayed']
    except (ValueError, TypeError):
        print('n_delay', row['How_Long_Delayed'])
        n_delay = 0
    return n_students * n_delay

recent_data['lost_minutes'] = recent_data.apply(compute_lost_minutes, axis = 1)


# We can now compute lost minutes per day.

# In[ ]:


minutes_per_day = []
for n_day in range(1, N_OF_DAYS):
    sub_df = recent_data[
                (recent_data.datetime >= pd.Timestamp(today - datetime.timedelta(n_day))) &
                (recent_data.datetime <= pd.Timestamp(today - datetime.timedelta(n_day - 1)))
            ]
    minutes_per_day.append([str(today - datetime.timedelta(n_day)), sub_df['lost_minutes'].sum()])

minutes_per_day = list(reversed([m for m in minutes_per_day if m[1]]))
    
plt.bar([m[0] for m in minutes_per_day], [m[1] for m in minutes_per_day])
plt.xticks([m[0] for m in minutes_per_day], rotation='vertical')
plt.show()


# And plot the #Breakdown per borough !

# In[ ]:


borough = recent_data[recent_data.Breakdown_or_Running_Late == 'Breakdown']['Boro'].value_counts()

borough.plot(kind = 'bar')


# In[ ]:




