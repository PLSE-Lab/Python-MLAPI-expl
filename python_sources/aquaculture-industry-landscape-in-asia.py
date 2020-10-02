#!/usr/bin/env python
# coding: utf-8

# In[85]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# In[86]:


#read file
data_global_value = pd.read_csv('../input/Global_Value.csv', index_col=['Land Area'])

#clean data
for col in data_global_value.columns:
    if 'S_' in col:
        del data_global_value[col]

print(data_global_value)

#read file
data_global_quantity = pd.read_csv('../input/Global_Quantity.csv',index_col=['Land Area'])

#clean the data
for col in data_global_quantity.columns:
    if 'S_' in col:
        del data_global_quantity[col]

print(data_global_quantity)


# In[87]:


global_fish_export_quantity = data_global_quantity[(data_global_quantity['Trade flow'] == 'Export') & (data_global_quantity['Commodity'] == 'Fish')]
global_fish_export_quantity = global_fish_export_quantity.drop(['Trade flow', 'Commodity'], axis=1)
print(global_fish_export_quantity)

global_fish_export_value = data_global_value[(data_global_value['Trade flow'] == 'Export') & (data_global_value['Commodity'] == 'Fish')]
global_fish_export_value = global_fish_export_value.drop(['Trade flow', 'Commodity'], axis=1)
print(global_fish_export_value)


# In[88]:


africa = global_fish_export_quantity.loc['Africa']
americas = global_fish_export_quantity.loc['Americas']
asia = global_fish_export_quantity.loc['Asia']
europe = global_fish_export_quantity.loc['Europe']
oceania = global_fish_export_quantity.loc['Oceania']

#fig size
plt.figure(figsize=(20,10))

#plot each region
plt.subplot(2,1,1)
plt.plot(africa, label='Africa')
plt.plot(americas, label='Americas')
plt.plot(asia, label='Asia')
plt.plot(europe, label='Europe')
plt.plot(oceania, label='Oceania')

#y scale
plt.yscale('linear')

#plot label
plt.ylabel('tonnes (t)')

# Add a legend in the top left corner of the plot
plt.legend(loc='upper left')

# Specify the orientation of the xticks
plt.xticks(rotation=60)

#customize grid
plt.grid(color='b', linestyle='-', linewidth=1, alpha=0.1)

#plot title
plt.title('Fish Export Quantity (t)')

#####
africa = global_fish_export_value.loc['Africa']
americas = global_fish_export_value.loc['Americas']
asia = global_fish_export_value.loc['Asia']
europe = global_fish_export_value.loc['Europe']
oceania = global_fish_export_value.loc['Oceania']

#fig size
plt.figure(figsize=(20,10))

#plot each region
plt.subplot(2,1,2)
plt.plot(africa, label='Africa')
plt.plot(americas, label='Americas')
plt.plot(asia, label='Asia')
plt.plot(europe, label='Europe')
plt.plot(oceania, label='Oceania')

#y scale
plt.yscale('linear')

#plot label
plt.xlabel('Year')
plt.ylabel("USD '000")

# Add a legend in the top left corner of the plot
plt.legend(loc='upper left')

# Specify the orientation of the xticks
plt.xticks(rotation=60)

#customize grid
plt.grid(color='b', linestyle='-', linewidth=1, alpha=0.1)

#plot title
plt.title('Fish Export Value (USD 000)')

# Display plot
plt.show()


# In[89]:


global_shrimp_export_quantity = data_global_quantity[(data_global_quantity['Trade flow'] == 'Export') & (data_global_quantity['Commodity'] == 'Crustaceans')]
global_shrimp_export_quantity = global_shrimp_export_quantity.drop(['Trade flow', 'Commodity'], axis=1)
print(global_shrimp_export_quantity)

global_shrimp_export_value = data_global_value[(data_global_value['Trade flow'] == 'Export') & (data_global_value['Commodity'] == 'Crustaceans')]
global_shrimp_export_value = global_shrimp_export_value.drop(['Trade flow', 'Commodity'], axis=1)
print(global_shrimp_export_value)


# In[90]:


africa = global_shrimp_export_quantity.loc['Africa']
americas = global_shrimp_export_quantity.loc['Americas']
asia = global_shrimp_export_quantity.loc['Asia']
europe = global_shrimp_export_quantity.loc['Europe']
oceania = global_shrimp_export_quantity.loc['Oceania']

#fig size
plt.figure(figsize=(20,10))

#plot each region
plt.subplot(2,1,1)
plt.plot(africa, label='Africa')
plt.plot(americas, label='Americas')
plt.plot(asia, label='Asia')
plt.plot(europe, label='Europe')
plt.plot(oceania, label='Oceania')

#y scale
plt.yscale('linear')

#plot label
plt.ylabel('tonnes (t)')

# Add a legend in the top left corner of the plot
plt.legend(loc='upper left')

# Specify the orientation of the xticks
plt.xticks(rotation=60)

#customize grid
plt.grid(color='b', linestyle='-', linewidth=1, alpha=0.1)

#plot title
plt.title('Shrimp Export Quantity (t)')

# Display plot
plt.show()

#####
africa = global_shrimp_export_value.loc['Africa']
americas = global_shrimp_export_value.loc['Americas']
asia = global_shrimp_export_value.loc['Asia']
europe = global_shrimp_export_value.loc['Europe']
oceania = global_shrimp_export_value.loc['Oceania']

#fig size
plt.figure(figsize=(20,10))

#plot each region
plt.subplot(2,1,2)
plt.plot(africa, label='Africa')
plt.plot(americas, label='Americas')
plt.plot(asia, label='Asia')
plt.plot(europe, label='Europe')
plt.plot(oceania, label='Oceania')

#y scale
plt.yscale('linear')

#plot label
plt.xlabel('Year')
plt.ylabel("USD '000")

# Add a legend in the top left corner of the plot
plt.legend(loc='upper left')

# Specify the orientation of the xticks
plt.xticks(rotation=60)

#customize grid
plt.grid(color='b', linestyle='-', linewidth=1, alpha=0.1)

#plot title
plt.title('Shrimp Export Value (USD 000)')

# Display plot
plt.show()


# In[91]:


#read file
asia_quantity = pd.read_csv('../input/Asia_Quantity.csv', index_col=0)

for col in asia_quantity.columns:
    if 'S_' in col:
        del asia_quantity[col]

asia_export_shrimp_quantity = asia_quantity[(asia_quantity['Trade flow'] == 'Export') & (asia_quantity['Commodity'] == 'Crustaceans')]
asia_export_shrimp_quantity = asia_export_shrimp_quantity['2015'].sort_values(ascending=False)

print(asia_export_shrimp_quantity)

