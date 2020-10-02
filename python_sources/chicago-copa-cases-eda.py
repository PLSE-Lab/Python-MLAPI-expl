#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set()
p = 'magma'


# In[ ]:


df = pd.read_csv("../input/copa-cases-summary.csv")


# In[ ]:


df.head()


# In[ ]:


df.info()


# # Feature Engineering

# In[ ]:


# Original race category is redundant
df.RACE_OF_COMPLAINANTS.value_counts()


# ## Process Race Column

# In[ ]:


def process_race(x):
    if x == "nan":
        return np.nan
        
    xl = x.replace(" | ", "|").split("|")
    if all(r == xl[0] for r in xl):
        return xl[0]
    else:
        return "Multirace"
df.RACE_OF_COMPLAINANTS = df.RACE_OF_COMPLAINANTS.astype('str')
df.RACE_OF_COMPLAINANTS = df.RACE_OF_COMPLAINANTS.apply(process_race)
df.RACE_OF_COMPLAINANTS = df.RACE_OF_COMPLAINANTS.astype('object')


# In[ ]:


# Nice and clean :D
df.RACE_OF_COMPLAINANTS.replace("nan", np.nan)
df.RACE_OF_COMPLAINANTS.value_counts()


# ## Process Date

# In[ ]:


# Convert to date format
df.COMPLAINT_DATE = pd.to_datetime(df.COMPLAINT_DATE)


# In[ ]:


# Add YEAR column for later use
df["YEAR"] = df.COMPLAINT_DATE.dt.year
df["DAY"] = df.COMPLAINT_DATE.dt.day
df["DAY_OF_WEEK"] = df.COMPLAINT_DATE.dt.dayofweek
df.index = pd.DatetimeIndex(df["COMPLAINT_DATE"])


# In[ ]:


df.head()


# ## Process Categories

# In[ ]:


df.CURRENT_CATEGORY = df.CURRENT_CATEGORY.replace("Unlawful Denial of Counsel", "Legal Violation")
df.CURRENT_CATEGORY = df.CURRENT_CATEGORY.replace("Coercion", "Legal Violation")
df.CURRENT_CATEGORY = df.CURRENT_CATEGORY.replace("Operational Violation", "Legal Violation")
df.CURRENT_CATEGORY = df.CURRENT_CATEGORY.replace("Firearm Discharge - No Hits", "Firearm Discharge")
df.CURRENT_CATEGORY = df.CURRENT_CATEGORY.replace("Firearm Discharge - Hits", "Firearm Discharge")
df.CURRENT_CATEGORY = df.CURRENT_CATEGORY.replace("Firearm Discharge at Animal", "Firearm Discharge")


# # Exploratory Data Analysis

# ## Distribution of complaint categories

# In[ ]:


plt.figure(figsize=(10,8))
sns.countplot(y="CURRENT_CATEGORY", data=df, order=df["CURRENT_CATEGORY"].value_counts().index, palette=p)


# ## Distribution of complaint categories

# In[ ]:


sns.countplot(x="COMPLAINT_MONTH", data=df, palette=p)


# ## Distribution of complaints per hour

# In[ ]:


sns.countplot(x="COMPLAINT_HOUR", data=df, palette=p)


# ## Distribution of race of complainants

# In[ ]:


plt.figure(figsize=(10,6))
sns.countplot(y="RACE_OF_COMPLAINANTS", data=df, order=df["RACE_OF_COMPLAINANTS"].value_counts().index, palette=p)


# ## What's the distribution of complaints per day? 

# In[ ]:


sns.distplot(df.resample('D').size(), color="purple")


# In[ ]:


# mean and standard deviation of complaints per day
complaints_per_day = pd.DataFrame(df.resample('D').size())
complaints_per_day["MEAN"] = df.resample('D').size().mean()
complaints_per_day["STD"] = df.resample('D').size().std()
# upper control limit and lower control limit
UCL = complaints_per_day['MEAN'] + 3 * complaints_per_day['STD']
LCL = complaints_per_day['MEAN'] - 3 * complaints_per_day['STD']


# ## Total complaints per day

# In[ ]:


plt.figure(figsize=(15,6))
df.resample('D').size().plot(label='Complaints per day', color='purple')
UCL.plot(color='red', ls='--', linewidth=1.5, label='UCL')
LCL.plot(color='red', ls='--', linewidth=1.5, label='LCL')
complaints_per_day['MEAN'].plot(color='red', linewidth=2, label='Average')
plt.title('Total complaints per day', fontsize=16)
plt.xlabel('Day')
plt.ylabel('Number of complaints')
plt.tick_params(labelsize=14)


# ## Total complaints per month

# In[ ]:


plt.figure(figsize=(15,6))
df.resample('M').size().plot(label='Total complaints per month', color='purple')
df.resample('M').size().rolling(window=12).mean().plot(color='red', linewidth=5, label='12-Months Average')
plt.title('Total complaints per month', fontsize=16)


# ## Is the trend same for all races?

# In[ ]:


df.pivot_table(index='COMPLAINT_DATE', columns='RACE_OF_COMPLAINANTS', aggfunc='size', fill_value=0).resample('M').sum().rolling(window=12).mean().plot(figsize=(15,6), linewidth=4, cmap='magma')
plt.title('Moving Average of Crimes per month by Category', fontsize=16)
plt.xlabel('')
plt.legend(prop={'size':16})
plt.tick_params(labelsize=16);


# ## Is the trend same for all categories of complaints?

# In[ ]:


df.pivot_table(index='COMPLAINT_DATE', columns='CURRENT_CATEGORY', aggfunc='size', fill_value=0).resample('M').sum().rolling(window=12).mean().plot(figsize=(12,25), linewidth=4, cmap='magma', subplots=True, layout=(-1, 3))
plt.xlabel('')
plt.legend(prop={'size':16})
plt.tick_params(labelsize=16);


# ## Which days have the highest and lowest average number of complaints?

# In[ ]:


months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
complaints_pt = df.pivot_table(values='YEAR', index='DAY', columns='COMPLAINT_MONTH', aggfunc=len)
complaints_pt_year_count = df.pivot_table(values='YEAR', index='DAY', columns='COMPLAINT_MONTH', aggfunc=lambda x: len(x.unique()))
complaints_avg = complaints_pt / complaints_pt_year_count
complaints_avg.columns = months
plt.figure(figsize=(10,12))
plt.title('Average Number of Complaints per Day and Month', fontsize=14)
sns.heatmap(complaints_avg.round(), cmap='magma', linecolor='grey',linewidths=0.1, cbar=True, annot=True, fmt=".0f");


# - The worst day is Feb 29th (well, this only happens every four years)
# - So 27 complaints on July 16 is a more reasonable estimate
# - The day with the least complaints is Christmas Day, nobody wants to spoil the celebration!

# ## Average Number of Complaints per Category and Month

# In[ ]:


complaints_pt = df.pivot_table(index='CURRENT_CATEGORY', columns='COMPLAINT_MONTH', aggfunc='size')
complaints_scaled = complaints_pt.apply(lambda x: x / complaints_pt.max(axis=1))
complaints_scaled.columns = months
plt.figure(figsize=(8,10))
plt.title('Average Number of Complaints per Category and Month', fontsize=14)
sns.heatmap(complaints_scaled, cmap='magma', cbar=True, annot=False, fmt=".0f");


# ## Average Number of Complaints per Category and Hour

# In[ ]:


complaints_pt = df.pivot_table(index='CURRENT_CATEGORY', columns='COMPLAINT_HOUR', aggfunc='size')
complaints_scaled = complaints_pt.apply(lambda x: x / complaints_pt.max(axis=1))
plt.figure(figsize=(12,8))
plt.title('Average Number of Complaints per Category and Hour', fontsize=14)
sns.heatmap(complaints_scaled, cmap='magma', cbar=True, annot=False, fmt=".0f");


# ## Average Number of Complaints per Category and Hour

# In[ ]:


complaints_pt = df.pivot_table(index='CURRENT_CATEGORY', columns='DAY_OF_WEEK', aggfunc='size')
complaints_scaled = complaints_pt.apply(lambda x: x / complaints_pt.max(axis=1))
plt.figure(figsize=(5,10))
plt.title('Average Number of Complaints per Category and Hour', fontsize=14)
sns.heatmap(complaints_scaled, cmap='magma', cbar=True, annot=False, fmt=".0f");

