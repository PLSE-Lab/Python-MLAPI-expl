#!/usr/bin/env python
# coding: utf-8

# ## Which National Parks to prioritize in light of species conservation##
# The intent is to use the US National Park Biodiversity 2016 database  to specifically see which National Parks contain the most amount of imperiled species. And thus, which ones could in theory have higher priority.
# 
# 
# DISCLAIMER: I am of the opinion that any national park, anywhere, is a worth preserving. But the purpose of this exercise is that given the scenario of limited resources, which parks are 'the most important for wildlife'. 
# 
# It is also my first attempt to do something like this with python and on kaggle, so any feedback is most welcomed.

# **Import libraries and set up**

# In[ ]:


from __future__ import division
import pandas as pd
import matplotlib.pyplot  as plt

# for prettier plots
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (10, 9)


# **Data import**

# In[ ]:


species = pd.read_csv('../input/species.csv', usecols=range(1,13))
parks = pd.read_csv('../input/parks.csv')

#change NaN to 'Not Threatened' in the Conservation Status column
species['Conservation Status'].fillna('Not Threatened', inplace=True)

# On inspection, the 'Conservation Category' contains Seasonality information - get rid of these rows
species = species[species['Conservation Status'].str.contains("Breeder|Resident|Migratory") == False]


# **Analysis**

# In[ ]:


# Calculate the number of each conservation category for each park
props = species[['Park Name', 'Conservation Status']].groupby(['Park Name', 'Conservation Status']).size()
# Convert to dataframe
props_df = props.to_frame().reset_index()
props_df.columns = ['Park Name', 'Conservation Status', 'Count']


# *1. Absolute count*
# 
# One way of judging the importance of individual NP is to see which one has the maximum absolute number of species for some key conservation categories:

# In[ ]:


cat_to_plot = ['Endangered', 'In Recovery', 'Species of Concern', 'Threatened']
for i in cat_to_plot:
	subset = props_df[props_df['Conservation Status'] == i]
	subset.sort_values(by='Count', ascending=False).plot(x='Park Name', y='Count', kind='bar', legend=None)
	plt.ylabel('Absolute number of species')
	plt.title('Conservation category: %s' % (i))
	plt.tight_layout()
plt.show()


# NP with:
# 
# - Maximum number of endangered species: Hawaii Volcanoes National Park (44)
# - Maximum number of in recovery species: Redwood National Park (7)
# - Maximum number of species of concern : Death Valley National Park (177)
# - Maximum number of threatened species: Death Valley National Park (16)

# *2. Taking park size into account*
# 
# Absolute number of species may however skew the results. It could simply be that bigger parks can sustain larger number of species. We can therefore look how many valuable species per acre of land each park holds:

# In[ ]:


# Count of each conservation category per acre for each NP - i.e. per acre of land, which park harbors the most e.g. endangered species
#1. Create a dictionary with 'Park' : 'Acres'
park_dict = dict(zip(parks['Park Name'], parks['Acres']))
#2. Create a function that divides each row's conservation category count by the park's area
def divide_count(row):
	return row['Count']/(park_dict[row['Park Name']])
#3. Create a new column with the count per acre measure
props_df['CountPerAcre'] = props_df.apply(divide_count, axis=1)
#4. Plot the results
for i in cat_to_plot:
	subset = props_df[props_df['Conservation Status'] == i]
	subset.sort_values(by='CountPerAcre', ascending=False).plot(x='Park Name', y='CountPerAcre', kind='bar', legend=None)
	plt.ylabel('Number of species per acre of park land')
	plt.title('Conservation category: %s' % (i))
	plt.tight_layout()
plt.show()


# NP with:
# 
# - The per acre maximum number of endangered species: Haleakala National Park (0.001375)
# - The per acre maximum number of in recovery species: Hot Springs National Park (0.000180)
# - The per acre maximum number of species of concern: Hot Springs National Park (0.010991)
# - The per acre maximum number of threatened species: Hot Springs National Park (0.000360

# *3. Proportion of all species in a park*
# 
# Ultimately, we can also look at the number of species in each conservation category as a proportion of all species observed in the park:

# In[ ]:


# Count the proportion of each conservation category for each park - i.e. find the park with the highest proportion of endangered species
#1. Sum up all conservation classes per park
park_sums = props_df.groupby(['Park Name']).agg('sum').reset_index()
# 2. Create a dictionary with 'Park' : 'Total count'
park_sums_dict = dict(zip(park_sums['Park Name'], park_sums['Count']))
# Create function that divides each conservation category count by the total count
def divide_total(row):
	return row['Count']/(park_sums_dict[row['Park Name']])
# 3. Create new column with proportional count of conservation category
props_df['ProportionalCount'] = props_df.apply(divide_total, axis=1)
# 4. Plot the results
for i in cat_to_plot:
	subset = props_df[props_df['Conservation Status'] == i]
	subset.sort_values(by='ProportionalCount', ascending=False).plot(x='Park Name', y='ProportionalCount', kind='bar', legend=None)
	plt.ylabel('Proportion of all species in the park')
	plt.title('Conservation category: %s' % (i))
	plt.tight_layout()
plt.show()


# NP with:
# 
# - Maximum proportion of endangered species: Haleakala National Park (0.015504)
# - Maximum proportion of in recovery species: Channel Islands National Park (0.002653)
# - Maximum proportion of species of concern: Petrified Forest National Park (0.086753)
# - Maximum proportion of threatened species: Dry Tortugas National Park (0.007075)

# **Implications**
# 
# If the allocation of resources is tied to the size of the park, then the Haleakala NP and Hot Springs NP should be prioritized, as these have the highest amount of vulnerable species per acre of land. 
# 
# The Haleakala NP also contains the highest proportion of endangered species (1.5%) so even if resources are not allocated per acre of park land, this NP would seems like the top candidate to prioritize.  
