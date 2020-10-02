#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import seaborn as sns; sns.set()

plt.rcParams['figure.figsize'] = (13, 16)


# In[ ]:


crimes = pd.read_csv('../input/Crime_Data_2010_2017.csv')


# In[ ]:


crimes.tail()


# Some conversions...

# In[ ]:


crimes['Date Occurred'] = pd.to_datetime(crimes['Date Occurred'], format="%m/%d/%Y")
crimes['Date Reported'] = pd.to_datetime(crimes['Date Reported'], format="%m/%d/%Y")


# We extract the **Hour** of the occurrence of the crime

# In[ ]:


crimes['Time Occurred'] = crimes['Time Occurred'].astype(str).str.zfill(4)
crimes['Hour Occurred'] = crimes['Time Occurred'].apply(lambda t: int(t[:2]))


# Also, we compute the delta, in days, between the crime date and its reporting date

# In[ ]:


crimes['Delta of Report'] = (crimes['Date Reported'] - crimes['Date Occurred']).dt.days


# We analyze crimes from 2015

# In[ ]:


crimes_from_15 = crimes[(crimes['Date Occurred'] >= '01/01/2015')]
print(crimes_from_15.shape)
crimes_from_15.tail()


# In[ ]:


gr_count = crimes_from_15.groupby(['Crime Code Description'], as_index=['Crime Code Description']).count().ix[:, 1]
gr_count


# We select the *most frequent crimes*

# In[ ]:


selected_crimes_from_15 = gr_count[gr_count > 20000]


# In[ ]:


selected_names = selected_crimes_from_15.index
print("\n".join(selected_names))


# Plotting crimes per **hour of the day**

# In[ ]:


g = sns.FacetGrid(crimes_from_15, 
                  row="Crime Code Description", 
                  row_order=selected_names,
                  size=1.9, aspect=4, 
                  sharex=True,
                  sharey=False)

g.map(sns.distplot, "Hour Occurred", bins=24, kde=False, rug=False)


# It's interesting to notice how **Assaults** and **Vehicle Thefts** happen mostly in the evening, while **Petty thefts** happen around noon. **Burglaries** tend to happen when people are away from home morning to late afternoon

# Let's analyze the occurrences of crimes during the cosidered period of time (2015 - Current)

# In[ ]:


crimes_time_series = crimes_from_15.groupby(['Crime Code Description', 'Date Occurred'], as_index=['Crime Code Description', 'Date Occurred']).count().ix[:,1].unstack(level=0).unstack(level=0).fillna(0)


# In[ ]:


for i, col in zip(range(1, len(selected_names) + 1), selected_names):
    plt.subplot(len(selected_names), 1, i)
    plt.title(col)
    crimes_time_series[col].rolling(window=20, min_periods=20).mean().plot()


# There's a clear upward trend of **Robberies** and **Vandalism**. Also, note how all crimes tend to decrease in the end, probably due to the fact that not all the recent crimes have been recorded in the dataset

# Is there any correlation among these crimes?

# In[ ]:


correlation_matrix = crimes_time_series.unstack(0)[selected_names].corr()

sns.heatmap(correlation_matrix)


# It seems there's a correlation among similarly named types of crimes. (**Assaults**) and (**Burglary**, **Petty Thefts**), (**Vehicle Stolen**, **Burglary from Vehicle**). It could be that:
# 
# 1. Crimes are not recorded 100% correctly and there's a certain overlap
# 2. There are external causes that drive similar crimes (weather, general mood, ... don't know)
# 3. In the case of Vehicle crimes, some criminals may work efficiently by stealing things from the vehicles or the vehicles themselves, if they manage to, during certain days and doing nothing in others.

# Finally, let's see which crimes have a median difference of at least 3 days between the date of occurrence and the date of reporting. We require at least 50 cases.

# In[ ]:


deltas = crimes.groupby(['Crime Code Description'])['Delta of Report'].describe()
deltas[(deltas["50%"] >= 3) & (deltas["count"] > 50)]


# Some crimes may be hard to spot immediately, like Unauthorized Computer Access (think of Equifax hacking, reported after months), Credit cards frauds (if no alert on transactions has been set) and insurance frauds. Other may leave a certain trauma to be overcome before reporting them.
