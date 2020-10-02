#!/usr/bin/env python
# coding: utf-8

# I'm new to data science. I don't want to deal with the more complex geographic information, so i'll start with the other features.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# # car accidents in NY
# I'll try to get some insights from the non-geographic features of this dataset:
# 
#  - date, time
#  - \# injured
#  - \# killed
#  - vehicle type
#  - vehicle factor (reason for accident)

# In[ ]:


df = pd.read_csv('../input/database.csv', parse_dates=[['DATE','TIME']])
df.columns


# # Add severity of the accident as a number: 
#  - 0=minor (no injuries) 
#  - 1=medium (injuries, non-fatal) 
#  - 2=fatal

# In[ ]:


injured = [x for x in df.columns if 'INJURED' in x]
killed = [x for x in df.columns if 'KILLED' in x]
df['severity'] = 'No injuries' 
df.loc[df[killed].sum(axis=1)>0,'severity'] = 'Fatal'
df.loc[(df[killed].sum(axis=1)==0)&(df[injured].sum(axis=1)>0),'severity'] = 'Injured'
df.severity.value_counts()


# ## Major causes for accidents

# The following chart shows all possible causes for accidents, and their numbers in the data

# In[ ]:


factors = [x for x in df.columns if 'FACTOR' in x]
fig, ax = plt.subplots(figsize=(4,8))
d = df.copy()
d = pd.DataFrame([d.loc[:,x].value_counts() for x in factors]).fillna(0).sum().sort_values(ascending=True)
d.plot(kind='barh',ax=ax)
ax.set_title('Causes of vehicle accidents in 2015, NYC')
ax.set_xlabel('# reported')


# We can see that most of the reasons are quite rare, and there are around 10 significant reasons for accidents. Lets see if the major causes differ when taking the severity of the accident into account (i'll drop the unspecified cases)

# In[ ]:


factors = [x for x in df.columns if 'FACTOR' in x]
d = df[factors].join(df.severity)
s0 = pd.DataFrame([d.loc[d.severity=='No injuries',x].value_counts() for x in factors]).fillna(0).sum().sort_values(ascending=False)
s1 = pd.DataFrame([d.loc[d.severity=='Injured',x].value_counts() for x in factors]).fillna(0).sum().sort_values(ascending=False)
s2 = pd.DataFrame([d.loc[d.severity=='Fatal',x].value_counts() for x in factors]).fillna(0).sum().sort_values(ascending=False)

fig = plt.figure(figsize=(9,3))
ax = fig.add_subplot(131 )
ax.set_title('No injuries')
s0[1:8].plot(kind='bar',ax=ax)
ax = fig.add_subplot(132 )
ax.set_title('Non fatal injuries')
s1[1:8].plot(kind='bar',ax=ax)
ax = fig.add_subplot(133 )
ax.set_title('Fatal Accidents')
s2[1:8].plot(kind='bar',ax=ax)


# As we can see, the number 1 **specified** cause for all accident types, is Driver Inattention/Distraction. However, it is interesting to note that the causes for fatal accidents are more "active" on the driver side: Disregarding traffic controls, not keeping distance, not yielding, etc., while for the less harmful accidents the causes seem more passive: Driver tired/drowsy, pavement is slippery, etc.
# Also, as Waleed Alhaider writes in the comments, most of the accidents' cause is unknown, which makes the conclusions that I have here ledd
# This already might suggest campaigns for safe driving, or police punishment policy, some direction: Focus on the active disobedience of drivers rather than passive problems such as road structure, tiredness, mechanical failures, etc.

# # When are we Tired or Drowsy?

# In[ ]:


factors = [x for x in df.columns if 'FACTOR' in x]
d = (df[factors].join(df.severity).join(df.DATE_TIME)).set_index('DATE_TIME').copy()
d['HOURS'] = d.index.map(lambda t: t.hour)
hour_fatigue = d.loc[(d[factors]=='Fatigued/Drowsy').any(axis=1),:].groupby('HOURS').count().loc[:,factors].sum(axis=1)
# normalize by total accidents at that hour
hour_data = 100.0* hour_fatigue / d.groupby('HOURS').size()
fig, ax = plt.subplots(figsize=(6,4))
hour_data.plot(kind='bar',ax=ax)
ax.set_xlabel('Time of day')
ax.set_ylabel('% of tiredness related accidents')
ax.set_ylim(5)
ax.set_title('Fraction of Tiredness related accidents along the day')


# Well, for me this is a surprise. Unless I have some mistake, the conclusion is that **one is twice more likely to be involved in a tiredness related accident in the afternoon**, compared to the middle of the night.

# # Vehicle types and accidents

# In[ ]:


vehicle_types = [x for x in df.columns if 'TYPE' in x]
fig, ax = plt.subplots(figsize=(4,8))
d = df.copy()
d = pd.DataFrame([d.loc[:,x].value_counts() for x in vehicle_types]).fillna(0).sum().sort_values(ascending=True)
d.plot(kind='barh',ax=ax)
ax.set_title('Involvement of vehicle types in accidents, 2015, NYC')
ax.set_xlabel('# reported')


# So we can see that most of the accidents happen to normal vehicles, followed by sports cars, taxis and vans. 
# What is more interesting, though, is the normalized frequency with respect to the total amount of each type of vehicle. 
# 
# We'll focus on taxis: are taxis involved in more accidents, considering the ratio between taxis and normal vehicles? 
# The ratio between the number of taxis and the number of normal vehicles is given by:

# In[ ]:


vehicle_types = [x for x in df.columns if 'TYPE' in x]
d = df.copy()
d = pd.DataFrame([d.loc[:,x].value_counts() for x in vehicle_types]).fillna(0).sum().sort_values(ascending=True)
print("Ratio of taxis to normal cars involved in accidents: 1:{} ".format(int(d['PASSENGER VEHICLE']/d.TAXI)))


# In order to know whether taxis are more prone to accidents, we need to know the ratio between taxis and normal cars in NYC. However, that information is not available in the given dataset. Searching the web, I found the following non-accurate source:
# 1:40 ([source][1])
# 
# This ratio result suggests that taxis are around 4 times more likely to be involved in an accident in NYC compared to normal vehicles. This is not really accurate, however, since taxis are spending much more time on the roads of NYC compared to normal cars. I would **guess** a factor of 10:1 and that makes a taxi 2.5 time **less** likely to be involved in an accident in NYC. 
# 
# So if you visit NY, go for a cab!
# 
# 
# 
#   [1]: https://www.google.co.il/url?sa=t&rct=j&q=&esrc=s&source=web&cd=1&cad=rja&uact=8&ved=0ahUKEwjxjLLvqOXRAhUDIJoKHXXKCGsQFggYMAA&url=https%3A%2F%2Fwww.quora.com%2FWhat-is-the-ratio-of-cabs-to-other-cars-in-Manhattan&usg=AFQjCNEgdrdle7-f26Fk8Z_wTVlvx-2Mog&sig2=IW5WG87jhzxT5aY-HMBkmw&bvm=bv.145822982,d.bGs
#   [2]: http://www.nyc.gov/html/tlc/downloads/pdf/2014_taxicab_fact_book.pdf

# In[ ]:




