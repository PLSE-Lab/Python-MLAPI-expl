#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import json as json


# The recent Florida high school massacre has thrust the topic of school shootings and the related debate regarding gun control back into the forefront of public discourse (sometimes productive, often not so much).  I wanted to better understand the data and trends related to this topic, especially the relative risk of being killed or injured in a school shooting, so...
# 
# (Note: school populations datasets come from the US Census.  School shooting dataset us a compilation based on a mashup of the Pah/Amaral/Hagan research on school shootings with the Wikipedia article on school shootings (heavily footnoted) from 1990 to present.)

# # School Enrollment

# First let's look at some background data on the school popuation - how many children go to school and how do they break down across the different school tiers.

# In[ ]:


# import census data
df_s = pd.read_csv('../input/cps_01_formatted.csv')


# In[ ]:


df_s.head()


# In[ ]:


# add a column for total grade school students (K-12)  
df_s['GS-Total'] = df_s['K-Total'] + df_s['E-Total'] + df_s['H-Total']


# In[ ]:


df_s.fillna(0, inplace=True)
df_s.dtypes


# In[ ]:


df_s[['Year','N-Total','K-Total','E-Total','H-Total','C-Total']].sort_values('Year').plot.area(x='Year',figsize=(10,5),colormap='tab20c', title="Total US school populations (1955-present)").legend(['Kindergarten','Elementary/Middle','High School','College'], title=None, loc='center left', bbox_to_anchor=(1, 0.5))


# In[ ]:


# create a set for 1990 and later only (to match the shooting dataset)
df_s_1990 = df_s[df_s['Year'] >= 1990]


# In[ ]:


df_s_1990[['Year','K-Total','E-Total','H-Total','C-Total']].sort_values('Year').plot.area(x='Year',figsize=(10,5),colormap='tab20c',title="Total US school populations (1990-present)").legend(['Kindergarten','Elementary/Middle','High School','College'], title=None, loc='center left', bbox_to_anchor=(1, 0.5))


# # School Shootings

# Now lets look at the data around school shootings.  This dataset includes everything from accidental discharges to suicides to gang violence to mass killings.  The commonality is that the victims were wounded or killed with guns; motive is not considered.

# In[ ]:


# import the dataset
df_v = pd.read_csv('../input/pah_wikp_combo.csv', parse_dates=['Date'], encoding="utf-8")


# In[ ]:


df_v.tail()


# In[ ]:


df_v['Year'] = df_v['Date'].dt.year

# drop dupes and unknown schoolsa
df_v['Dupe'].fillna(0, inplace=True)
df_v = df_v[df_v['Dupe'] == 0]
df_v = df_v[df_v['School'] != '-']

# calculate casualties
df_v['Casualties'] = df_v['Wounded'] + df_v['Fatalities']

# mod schools
df_v['SchoolName'] = df_v['School'].map(lambda x: 'College' if x == 'C' else 'Elementary/Middle' if x in ['MS','ES'] else 'High School')
df_gs = df_v[df_v['School'].isin(['ES','MS','HS'])]
df_c = df_v[df_v['School'].isin(['C'])]


# In[ ]:


df_v.head()


# First some raw numbers on casualties at the grade school and college level:

# In[ ]:


print("Total US grade school shooting casualties (1990-present): \n{0}".format(df_gs[['Wounded','Fatalities']].sum()))


# In[ ]:


print("Total US college shooting casualties (1990-present): \n{0}".format(df_c[['Wounded','Fatalities']].sum()))


# Here's how it looks over time.  The first graph is all casualties (wounded and killed) while the second focusses only on fatalities:

# In[ ]:


# look at overall casualties by year
pd.pivot_table(data=df_v, index='Year', columns='SchoolName', values='Casualties', aggfunc="sum").plot.bar(stacked=True, figsize=(15,5),colormap='tab20c', title="US school shooting casualties (1990-present)").legend(title=None)


# In[ ]:


# zoom in on fatalities by year
pd.pivot_table(data=df_v, index='Year', columns='SchoolName', values='Fatalities', aggfunc="sum").plot.bar(stacked=True, figsize=(15,5),colormap='tab20c', title="US school shooting fatalities (1990-present)").legend(title=None)


# In[ ]:


# join with census data
df_p = pd.pivot_table(data=df_v, index='Year', columns='SchoolName', values='Fatalities', aggfunc="sum")
df_vp = df_p.reset_index().fillna(0)
df_m = df_s_1990.merge(df_vp, how="left", on="Year")

# also get count of events vs sum
df_pct = pd.pivot_table(data=df_v, index='Year', columns='SchoolName', values='Fatalities', aggfunc=len)
df_pct = df_pct.reset_index().fillna(0)


# Next let's consider the number of shooting incidents over time and look at the change in lethality of each incident:

# In[ ]:


df_pct[['Year', 'Elementary/Middle', 'High School', 'College']].sort_values('Year').plot.bar(x='Year', stacked=False, figsize=(15,5),colormap='tab20c', title='Count of incidents').legend(title=None)


# In[ ]:


# divide casualties (dataframe) by incidents (series) to get casualities per incident trend
(df_v[['Year','Casualties']].groupby('Year').sum()['Casualties'] / df_v[['Year','Casualties']].groupby('Year').size()).sort_index().plot.bar(stacked=False, figsize=(15,5),colormap='tab20c', title='Mean Casualities per Incident', legend=False)


# In[ ]:


df_m.head()


# In[ ]:


# calculate the victims per million students 
df_m['Elem/Mid (vpm)'] = (df_m['Elementary/Middle'])/((df_m['E-Total']+df_m['K-Total'])/1000)
df_m['High School (vpm)'] = (df_m['High School'])/(df_m['H-Total']/1000)
df_m['College (vpm)'] = (df_m['College'])/(df_m['C-Total']/1000)


# In[ ]:


df_m.head()


# The raw numbers tell one story.  Another perspective that interests me is relative risk - how much danger do students face from guns relative to other risks?

# In[ ]:


df_m[['Year', 'Elem/Mid (vpm)', 'High School (vpm)', 'College (vpm)']].sort_values('Year').plot.bar(x='Year', stacked=False, figsize=(15,5),colormap='tab20c', title='Risk (victims/million) of being killed in a school shooting').legend(title=None)


# In[ ]:


print('Mean deaths/million from school shootings (1990-present): \n College: {0:0.3}\n High School: {1:0.3}\n Elementary and Middle School: {2:0.3}'      .format(
          df_m['College (vpm)'].mean(), 
          df_m['High School (vpm)'].mean(), 
          df_m['Elem/Mid (vpm)'].mean())
     )


# High school is the most dangerous place to be, with a roughly 1 in 2M chance of being killed in a school shooting in any given year; over 4 years, a high school student has a roughly 1 in 500K chance of being killed.  
# 
# College comes in at roughly 1 in 3M (1 in 750K over 4 years), and elementary-middle school at roughly 1 in 10M (or ~1 in 1M from K-8).
# 

# How does this compare with other forms of death?
# 
# According to the NIH (https://www.ncbi.nlm.nih.gov/books/NBK220806/) in 1999:
# * Total children age 5-14: ~39.5M (roughly corresponds to Elementary and Middle School data above)
# * Total young adults age 15-24: ~37.8M (roughly corresponds to High School and College data above)
# 
# Based on the top 10 causes for each group, here's the death rate by category:

# In[ ]:


num_5_14 = (7595/19.2)*100000
num_15_24 = (30656/81.2)*100000

df_nih_5_14 = pd.DataFrame(data=[
    ['Accidents', 'accident', 3091],
    ['Malignant neoplasms', 'illness', 1021],
    ['Homicide', 'crime', 432],
    ['Congenital anomalies', 'illness', 428],
    ['Diseases of the heart', 'illness', 277],
    ['Suicide', 'suicide', 242],
    ['Chronic lower respiratory diseases ', 'illness', 139],
    ['Benign neoplasms', 'illness', 101],
    ['Pneumonia and influenza', 'illness', 93],
    ['Septicemia', 'illness', 77],
], columns=['Cause', 'Category', 'Count'])
df_nih_5_14['VPM'] = df_nih_5_14['Count']/num_5_14*1000000


df_nih_15_24 = pd.DataFrame(data=[
    ['Accidents', 'accident', 13656],
    ['Homicide', 'crime', 4998],
    ['Suicide', 'suicide', 3901],
    ['Malignant neoplasms', 'illness', 1724],
    ['Diseases of the heart', 'illness', 1069],
    ['Congenital anomalies', 'illness', 434],
    ['Chronic lower respiratory diseases ', 'illness', 209],
    ['HIV', 'illness', 198],
    ['Stroke', 'illness', 182],
    ['Pneumonia and influenza', 'illness', 179],
], columns=['Cause', 'Category', 'Count'])
df_nih_15_24['VPM'] = df_nih_15_24['Count']/num_15_24*1000000


# In[ ]:


(df_nih_5_14.groupby('Category')['Count'].sum().sort_values(ascending=False)/num_15_24*1000000).plot.bar(x='Year', stacked=False, figsize=(10,5),colormap='tab20c', title='Risk (victims/million) of dying from other factors (5-14 years)', legend=False)


# In[ ]:


(df_nih_15_24.groupby('Category')['Count'].sum().sort_values(ascending=False)/num_15_24*1000000).plot.bar(x='Year', stacked=False, figsize=(10,5),colormap='tab20c', title='Risk (victims/million) of dying from other factors (15-24 years)', legend=False)

