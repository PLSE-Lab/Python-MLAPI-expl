#!/usr/bin/env python
# coding: utf-8

# ![cdc-k0KRNtqcjfw-unsplash.jpg](attachment:cdc-k0KRNtqcjfw-unsplash.jpg)

# In[ ]:


import pandas as pd
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# # Importing data

# In[ ]:


# Path of the file to read
indiaCovid19_filepath = "../input/coronavirus-cases-in-india/Covid cases in India.csv"

# Reading the file
indiaCovid19_data = pd.read_csv(indiaCovid19_filepath, index_col="S. No.")


# ## A first look at the data

# In[ ]:


indiaCovid19_data.columns


# #### What are the types of data in each column?

# In[ ]:


indiaCovid19_data.dtypes


# In[ ]:


indiaCovid19_data.describe()


# In[ ]:


indiaCovid19_data.head()


# # Visualization

# ### Number of confirmed cases by state

# In[ ]:


plt.figure(figsize=(10,6))
plt.title("Number of confirmed cases by state in India")

sns.barplot(x=indiaCovid19_data['Name of State / UT'], y=indiaCovid19_data['Total Confirmed cases'])

# Add label for vertical axis
plt.ylabel("Number of cases")

# Rotating xlabels so that we can read the text displayed
plt.xticks(
    rotation=45, 
    horizontalalignment='right',
    fontweight='light' 
)


# There are the most cases in the city of Maharashtra (almost 700 cases). Several states are found to be very little affected. 

# In[ ]:


# What is the exact number of cases in Maharashtra ?
indiaCovid19_data[indiaCovid19_data['Name of State / UT'] == 'Maharashtra']


# ### Number of Cured/Discharged/Migrated cases by state

# In[ ]:


plt.figure(figsize=(10,6))
plt.title("Number of Cured/Discharged/Migrated cases by state in India")

sns.barplot(x=indiaCovid19_data['Name of State / UT'], y=indiaCovid19_data['Cured/Discharged/Migrated'])

# Add label for vertical axis
plt.ylabel("Number of cases")

# Rotating xlabels so that we can read the text displayed
plt.xticks(
    rotation=45, 
    horizontalalignment='right',
    fontweight='light' 
)


# In[ ]:


indiaCovid19_data[indiaCovid19_data['Name of State / UT'] == 'Kerala']


# 
# The values in Kerala (55) and Maharashtra (42) are quite similar. Maharashtra has the highest number of cases and does quite well in healing (compared to the rest of the country). However, since these figures are cumulative with cases migrated and discharged, it is not possible to really judge treatment performance on this indicator alone in these states. 
# In this situation, it would be better to look at new cases and recoveries across the country. 

# What about death records?

# ### Number of Deaths by state

# In[ ]:


plt.figure(figsize=(10,6))
plt.title("Number of deaths cases by state in India")

sns.barplot(x=indiaCovid19_data['Name of State / UT'], y=indiaCovid19_data['Deaths'])

# Add label for vertical axis
plt.ylabel("Number of cases")

# Rotating xlabels so that we can read the text displayed
plt.xticks(
    rotation=45, 
    horizontalalignment='right',
    fontweight='light' 
)


# With these three diagrams, we can see that Maharashtra is at the heart of the fight against the coronavirus. The other states have fewer cases of illness, more cases of cure/discharge/migration, and far fewer deaths (or no deaths at all for some states!). 
# 
# In Kerala there are fewer sick people than in Delhi and Tamil Nadu for example, but many more Cured/Discharged/Migrated in Kerala and fewer deaths.
# 
# Let's look at those specific values :

# In[ ]:


indiaCovid19_data[indiaCovid19_data['Name of State / UT'].isin(['Kerala', 'Delhi', 'Maharashtra', 'Tamil Nadu'])]


# ### Let's look at the relationships between these data via some scatter plots

# In[ ]:


sns.scatterplot(x=indiaCovid19_data['Total Confirmed cases'], y=indiaCovid19_data['Cured/Discharged/Migrated'])


# Drawing a regression line.

# In[ ]:


sns.regplot(x=indiaCovid19_data['Total Confirmed cases'], y=indiaCovid19_data['Cured/Discharged/Migrated'])


# In[ ]:


sns.scatterplot(x=indiaCovid19_data['Total Confirmed cases'], y=indiaCovid19_data['Deaths'])


# In[ ]:


sns.regplot(x=indiaCovid19_data['Total Confirmed cases'], y=indiaCovid19_data['Deaths'])


# Number of cases, Cured/Discharged/Migrated and Deaths

# In[ ]:


sns.scatterplot(x=indiaCovid19_data['Total Confirmed cases'], 
                y=indiaCovid19_data['Cured/Discharged/Migrated'], 
                hue=indiaCovid19_data['Deaths'])


# In[ ]:


sns.jointplot(x=indiaCovid19_data['Total Confirmed cases'], y=indiaCovid19_data['Cured/Discharged/Migrated'], kind="kde")


# In[ ]:


sns.jointplot(x=indiaCovid19_data['Total Confirmed cases'], y=indiaCovid19_data['Deaths'], kind="kde")


# In[ ]:


sns.jointplot(x=indiaCovid19_data['Deaths'], y=indiaCovid19_data['Cured/Discharged/Migrated'], kind="kde")


# Values close to zero - zero are more frequent. That's reassuring enough. Overall the country is doing well (considering the data we have here of course).
