#!/usr/bin/env python
# coding: utf-8

# Airline Fleet Dataset Exploration and Visualization

# Dataset includes the following data:
# 
# Parent Airline: i.e. International Airlines Group (IAG)
# 
# Airline: i.e. Iberia, Aer Lingus, British Airways...etc. which are owned by IAG
# 
# Aircraft Type: Manufacturer & Model
# 
# Current: Quantity of airplanes in Operation
# 
# Future: Quantity of airplanes on order, from planespotter.net
# 
# Order: Quantity airplanes on order, from Wikipedia
# 
# Unit Cost: Average unit cost ($M) of Aircraft Type, as found by Wikipedia and various google searches
# 
# Total Cost: Current quantity * Unit Cost ($M)
# 
# Average Age: Average age of "Current" airplanes by "Aircraft Type"

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


fleet = pd.read_csv('../input/Fleet Data.csv')


# Exploring the dataset:

# In[3]:


fleet.head()


# In[4]:


fleet.describe()


# In[5]:


fleet.info()


# Number of unique parent airlines:

# In[6]:


fleet['Parent Airline'].nunique()


# Number of unique airlines:

# In[7]:


fleet['Airline'].nunique()


# In[8]:


fleet['Aircraft Type'].nunique()


# In[9]:


fleet.columns


# In[10]:


#Separating quantity and cost data from age:
aircraftfleet = fleet[['Airline','Aircraft Type', 'Current', 'Future', 'Historic', 'Total', 'Orders', 'Total Cost (Current)']]
parentfleet = fleet[['Parent Airline','Aircraft Type', 'Current', 'Future', 'Historic', 'Total', 'Orders', 'Total Cost (Current)']]


# In[11]:


aircraftfleet.head(5)


# In[12]:


parentfleet.head(5)


# <h2> Airlines</h2>

# In[13]:


# Grouping data by Airline name to see tha aircraft fleet size including all aircraft type:
aircraftfleet.groupby(axis=0, by='Airline').sum().head(10)


# In[14]:


# Top 20 Parent Airlines with the biggest currently active aircraft fleet:
aircraftfleet.groupby(by='Airline').sum()['Current'].sort_values(ascending=False).head(20)


# <h2> Parent Airlines </h2>

# In[15]:


# Grouping data by Parent Airline name to see tha aircraft fleet size including all aircraft type:
parentfleet.groupby(axis=0, by='Parent Airline').sum().head(10)


# In[16]:


# Top20 Airlines with the biggest currently active aircraft fleet:
parentfleet.groupby(by='Parent Airline').sum()['Current'].sort_values(ascending=False).head(50)


# <h4>Let's explore a bit more in details information about some specific major Parent Airlines operating in Europe, Middle-East and Asia.</h4>

# <h4> I'll choose the following airlines:
# 
# Air France/KLM,
# 
# Aeroflot,
# 
# Lufthansa,
# 
# Emirates,
# 
# American Airlines;</h4>

# In[17]:


# Fleet size for these specific airlines:
fleet[(fleet['Parent Airline'] == 'Air France/KLM') | (fleet['Parent Airline'] == 'Aeroflot') | (fleet['Parent Airline'] == 'Lufthansa') | (fleet['Parent Airline'] == 'Emirates') | (fleet['Parent Airline'] == 'American Airlines')].groupby(by='Parent Airline').sum()['Current'].sort_values(ascending=False).head(50)


# In[18]:


# selected_airlines = fleet[(fleet['Parent Airline'] == 'Air France/KLM') | (fleet['Parent Airline'] == 'Aeroflot') | (fleet['Parent Airline'] == 'Lufthansa') | (fleet['Parent Airline'] == 'Emirates') | (fleet['Parent Airline'] == 'American Airlines')]
selected_airlines = fleet[(fleet['Parent Airline'] == 'Air France/KLM') | (fleet['Parent Airline'] == 'Aeroflot') | (fleet['Parent Airline'] == 'Lufthansa') | (fleet['Parent Airline'] == 'Emirates') | (fleet['Parent Airline'] == 'American Airlines')].copy()


# In[19]:


# Top20 oldest currently active aircraft types based on the average age:
selected_airlines[['Parent Airline', 'Aircraft Type','Average Age']].dropna(axis=0).sort_values(by='Average Age', ascending=False).head(20)


# In[20]:


# Top20 newest currently active planes:
selected_airlines[['Parent Airline', 'Aircraft Type','Average Age']].dropna(axis=0).sort_values(by='Average Age').head(20)


# In[21]:


selected_airlines.columns


# <h4> The following plot demonstrates the distribution of the aircraft types among daughter airlines.
# We can see which airline has the biggest variety of the planes.</h4>

# In[22]:


plt.figure(figsize=(14,10))
sns.countplot(data=selected_airlines, x='Parent Airline', hue='Airline')
plt.legend(bbox_to_anchor=(1, 1.0))


# In[23]:


# Number of unique aircraft types
selected_airlines['Aircraft Type'].nunique()


# In[24]:


# Here we can find the number of unique aircraft types used by Emirates:
selected_airlines[selected_airlines['Parent Airline'] == 'Emirates']['Aircraft Type'].nunique()


# In[25]:


selected_airlines[selected_airlines['Parent Airline'] == 'Emirates'][['Aircraft Type', 'Current', 'Future',
       'Historic', 'Total', 'Orders']]


# <h4> Let's explore the average age of the aircraft accross selected Parent Airlines </h4>

# In[26]:


sns.set_style('darkgrid')
plt.figure(figsize=(14,10))
sns.boxplot(data=selected_airlines, x='Parent Airline', y='Average Age', palette='coolwarm')


# In[27]:


avg = selected_airlines.dropna(axis=0, subset=['Average Age',])[['Parent Airline','Airline','Aircraft Type','Average Age']]


# In[28]:


# List of unique airplanes for these airlines:
avg['Aircraft Type'].unique()


# In[29]:


plt.figure(figsize=(14,10))
sns.boxplot(data=avg, y='Aircraft Type', x='Average Age')


# <h4> Now let's see the distribution of the big planes like Airbus A380, Airbus A330, Airbus A340, Boeing 747, Boeing 777, Boeing 787 among these selected airlines.<h4>

# In[30]:


biggies = selected_airlines[(selected_airlines['Aircraft Type'] == 'Airbus A380') | (selected_airlines['Aircraft Type'] == 'Airbus A330') | (selected_airlines['Aircraft Type'] == 'Airbus A340') | (selected_airlines['Aircraft Type'] == 'Boeing 747') | (selected_airlines['Aircraft Type'] == 'Boeing 777') | (selected_airlines['Aircraft Type'] == 'Boeing 787 Dreamliner')]


# In[31]:


biggies.head(5)


# In[32]:


biggies.sort_values('Aircraft Type')[biggies['Current'] > 0][['Parent Airline', 'Airline', 'Aircraft Type', 'Current']].head(20)


# In[33]:


plt.figure(figsize=(10,6))
sns.countplot(data=biggies, x='Aircraft Type', hue='Parent Airline')
plt.legend(bbox_to_anchor=(1, 1.0))

# The plot demonstrates how many Airlines under major daughter airlines use big aircraft.


# In[34]:


selected_airlines.columns


# In[35]:


bigsorted = biggies.drop(axis=1, columns='Average Age').groupby('Aircraft Type').sum()


# In[36]:


bigsorted.head(10).sort_values('Current', ascending=False)

# As we can see the most used type of Aircraft among the selected Airlines is Boeing 777.
# Boeing 787 Dreamliner is quite a new plane and is not that widely used yet.
# Boeing 747 is quite old already and probably most airlines will soone replace these planes with newer ones, like 787.


# In[37]:


bigsorted = biggies.groupby('Aircraft Type').mean().sort_values('Average Age', ascending=False)


# In[38]:


bigsorted['Average Age']

# As we can see below Boeing 747 is indeed in general older that other planes, while Boeing 787 planes are the youngest ones.


# In[39]:


airplanes = biggies[['Parent Airline', 'Aircraft Type', 'Current']].copy()


# In[40]:


airplanes.dropna(axis=0, subset=['Current',], inplace=True)


# In[41]:


airplanes.sort_values('Aircraft Type')


# In[42]:


airplanes = airplanes.groupby(by=['Parent Airline', 'Aircraft Type']).sum()


# In[43]:


airplanes = airplanes.reset_index()


# In[44]:


sns.lmplot(x='Parent Airline', y='Current', hue='Aircraft Type', data=airplanes, fit_reg=False, size=6)


# In[45]:


g = sns.FacetGrid(airplanes, row='Parent Airline' , col="Aircraft Type", hue='Aircraft Type', margin_titles=True)
g = g.map(plt.bar, "Aircraft Type", "Current")

# Facit Grid plot to display data abour Airline fleet in details. Columns - Aircraft Type, Rows - Airline


# In[46]:


plt.figure(figsize=(10,6))
sns.barplot(data=airplanes, x='Parent Airline',y='Current', hue='Aircraft Type')
plt.legend(bbox_to_anchor=(1, 1.0))

# ALternative barplot graph to make it easier to compare the fleets of different Airlines.


# In[ ]:




