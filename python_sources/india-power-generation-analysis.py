#!/usr/bin/env python
# coding: utf-8

# ## Introduction
# 
# The main goal of thi project is to get useful and meaningful information from India Power Generation between 2017 and 2020. 
# 
# For this project it's going to be consider only Thermo and Hydro generations.
# 
# There are some visualizations exploring the relationshipd between actual generation and the regions, and I also created a indicator of MU/km2 for the actual generation at each region.

# ## Libraries

# In[ ]:


import pandas as pd
import datetime as dt
import seaborn as sns
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# ## Accessing the data

# In[ ]:


files = r'../input/daily-power-generation-in-india-20172020/file.csv'
state_region =r'../input/daily-power-generation-in-india-20172020/State_Region_corrected.csv'

plant_types = pd.read_csv(files, thousands = ',', decimal = '.')
region_data = pd.read_csv(state_region)


# ### Plant Types
# 
# In this DataFrame we have information daily generation in India in MU.
# 
# 1MU = 1000 MWh

# In[ ]:


plant_types.sample(4)


# ### Cleaning

# In[ ]:


## There's a lot of missing data for Nuclear Generation, so I deceided to drop it.
plant_types.drop('Nuclear Generation Actual (in MU)', axis= 1, inplace= True)
plant_types.drop('Nuclear Generation Estimated (in MU)', axis= 1, inplace= True)


# In[ ]:


## The thermal columns should be floats, not strings
plant_types['Thermal Generation Actual (in MU)'] = plant_types['Thermal Generation Actual (in MU)'].astype(float)
plant_types['Thermal Generation Actual (in MU)'] = plant_types['Thermal Generation Estimated (in MU)'].astype(float)


# In[ ]:


## Converting Data into datetime type
plant_types['Date'] = pd.to_datetime(plant_types['Date'])


# In[ ]:


## Rename NorthEastern at region_data to make it the same as in plant_Types
region_data['Region'] = region_data['Region'].replace({'Northeastern':'NorthEastern'})


# ### Creating new columns to plant_types dataframe

# In[ ]:


## Total columns
plant_types['Total Generation Actual (in MU)'] = plant_types['Thermal Generation Actual (in MU)'] + plant_types['Hydro Generation Actual (in MU)']
plant_types['Total Generation Estimated (in MU)'] = plant_types['Thermal Generation Estimated (in MU)'] + plant_types['Hydro Generation Estimated (in MU)']


# In[ ]:


plant_types.sample(4)


# ## Visualizations

# In[ ]:


plt.figure(figsize = [12, 5])
base_color = sns.color_palette()
regions_order = ['Northern', 'NorthEastern', 'Western', 'Eastern', 'Southern']


ax1 = sns.boxplot(data = plant_types, x = 'Region', y = 'Total Generation Actual (in MU)', 
                  color = base_color[0], order=regions_order);
plt.title('Total generation by Region');


# In[ ]:


df = plant_types.groupby('Region')[['Thermal Generation Actual (in MU)', 'Hydro Generation Actual (in MU)']].mean()
df


# In[ ]:


plt.figure(figsize=[15, 5])

plt.subplot(1, 2, 1)
sns.barplot(data = df, x=df.index, y = 'Thermal Generation Actual (in MU)', color=base_color[0], order=regions_order);
plt.title('Total Thermal Generation by Region');

plt.subplot(1, 2, 2)
sns.barplot(data = df, x=df.index, y = 'Hydro Generation Actual (in MU)', color=base_color[0], order=regions_order);
plt.title('Total Hydro Generation by Region');


# In[ ]:


df2 = region_data.groupby('Region').sum()
df3 = pd.merge(df, df2, left_index=True, right_index=True)


# In[ ]:


df3['%Thermo Generation (MU/km2)'] = (df3['Thermal Generation Actual (in MU)'] / df3['Area (km2)'])*100
df3['%Hydro Generation (MU/km2)'] = (df3['Hydro Generation Actual (in MU)'] / df3['Area (km2)'])*100
df3['%Total Generation (MU/km2)'] = df3['%Thermo Generation (MU/km2)'] + df3['%Hydro Generation (MU/km2)']


# In[ ]:


df3


# In[ ]:


df3.plot(y=['%Total Generation (MU/km2)', '%Thermo Generation (MU/km2)', '%Hydro Generation (MU/km2)'], kind='line');
plt.title('Line plot for generation per area (MU/km2)');
plt.ylabel('MU/km2 * 100');

