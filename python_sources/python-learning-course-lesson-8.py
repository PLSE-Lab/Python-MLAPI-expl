#!/usr/bin/env python
# coding: utf-8

# # Python Learning Course - Lesson 8
# ## 1. Importing Libraries and Data

# In[1]:


# importing necessary libraries

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')

import os
print(os.listdir("../input"))


# In[2]:


# importing dataset
df_original = pd.read_csv('../input/outbreaks.csv')
df_original.head(1)


# ## 2. Preliminary Data Exploration

# In[3]:


df_original.info() # data exploration - notice Ingredients and Serotype/Genotype columns. many missing values expected


# In[4]:


len(df_original) # number of dataset records


# In[5]:


df_original.isnull() # identifying missing values per column


# In[6]:


df_original.isnull().sum() # counting missing values per column


# In[7]:


df_original.isnull().sum() * 100 / len(df_original) # creating percentages out of the number of missing values


# In[8]:


# copy dataset excluding columns with high missing value percentage - Ingredient, Serotype/Genotype
df = df_original[['Year', 'Month', 'State', 'Location', 'Food', 'Species', 'Status', 'Illnesses', 'Hospitalizations', 'Fatalities']].copy()
df.head(1)


# In[9]:


df_state = df.filter(items=['State', 'Illnesses', 'Fatalities']) # using .filter to alter the data perspective
df_state


# In[10]:


df_state.groupby(['State']) # using .groupby


# In[11]:


type(df_state) # understanding .groupby


# In[12]:


type(df_state.groupby(['State']))


# In[13]:


df_state.groupby(['State']).sum() # getting results from .groupby


# In[14]:


df.filter(items=['State', 'Illnesses']).groupby(['State']).describe() # illnesses statistics for all years by state


# In[15]:


df.filter(items=['State', 'Fatalities']).groupby(['State']).describe() # fatalities statistics for all years by state


# In[16]:


df.loc[df['Year'] == 2015].filter(items=['State', 'Illnesses']).groupby(['State']).describe() # descriptive statistics for 2015 per state


# In[17]:


df.Year


# In[18]:


set(df.Year)


# In[19]:


unique_years = set(df.Year) # taking unique years
unique_years


# In[20]:


for year in unique_years: # testing a loop of each year in set
    print(str(year))


# In[21]:


def describe_by_year(my_set, my_df, use_data):
    for year in my_set:
        print('Descriptive Statistics per State for the Year: ' + str(year))
        print('')
        print(my_df.loc[my_df['Year'] == year].filter(items=['State', use_data]).groupby(['State']).describe())
        print('')
        print('')
        print('')
    return


# In[22]:


pd.set_option('display.expand_frame_repr', False) # changes the display options to fit the output
describe_by_year(unique_years, df, "Illnesses")


# In[23]:


output = describe_by_year(unique_years, df, "Fatalities")
output.to_csv('descriptive_tables', sep='\t', encoding='utf-8', index=False)


# In[24]:


# getting information per year for the state of California
df.loc[df['State'] == 'California'].filter(items=['Year', 'Illnesses', 'Fatalities']).groupby(['Year']).sum()


# In[25]:


def describe_by_state(my_set, my_df, use_data):
    for my_state in my_set:
        print('Number of Illness and Fatal incidents per Year in State: ' + str(my_state))
        print('')
        print(my_df.loc[my_df['State'] == my_state].filter(items=['Year', use_data]).groupby(['Year']).sum())
        print('')
        print('')
        print('')
    print("Total States: " + str(len(my_set)))
    return


# In[26]:


describe_by_state(set(df.State), df, "Illnesses")


# In[27]:


# more .filter and .groupby combinations
temp = df.filter(items=['State', 'Year','Illnesses']).groupby(['State', 'Year']).sum()
temp


# In[28]:


# database format of the above result
df_state_year = df.filter(['State', 'Year', 'Illnesses'])
df_state_year


# # 3. Introduction to Data Visualization

# In[29]:


plt.rcParams['figure.figsize'] = [16.0, 12.0]
temp = df.loc[df['State'] == 'California'].filter(items=['Year', 'Illnesses']).groupby(['Year']).sum()
temp.plot(kind='bar')
plt.show()


# In[30]:


temp = df.loc[df['State'] == 'California'].filter(items=['Year', 'Illnesses', 'Fatalities']).groupby(['Year']).sum()
temp.plot.bar(stacked=True)
plt.show()


# In[32]:


temp = df.filter(items=['State', 'Illnesses', 'Fatalities']).groupby(['State']).sum()
temp.plot.bar(stacked=True)
plt.show()

