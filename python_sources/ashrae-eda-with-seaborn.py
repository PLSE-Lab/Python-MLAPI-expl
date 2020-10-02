#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)


# Using .pkl files from data minification.

# In[ ]:


dftrain = pd.read_pickle('/kaggle/input/ashraepkl/train.pkl')
dftest = pd.read_pickle('/kaggle/input/ashraepkl/test.pkl')
print ('Train: %s , Test %s' % (dftrain.shape, dftest.shape) )


# There is 16 columns with informations like the moment of meter reading, information about the buildings, and the weather. There is also our target variable, meter reading.

# In[ ]:


dftrain.head()


# Most of the variables are in numeric format.

# In[ ]:


dftrain.info()


# Lets see at first how much data is missing in both datasets.

# In[ ]:


ind = np.arange(len(dftrain.columns))
width = 0.35

fig, axes = plt.subplots(figsize=(14, 6))
plt.title('Missing values by column', fontsize=14)

axes.bar(dftrain.columns, dftrain.isnull().sum(), width, color='orangered')
axes.bar(ind+width, dftest.isnull().sum(), width, color='lightseagreen')

plt.ylabel('Missing values')
plt.xticks(rotation=40)
axes.set_xticks(ind + width / 2)
plt.legend(title='Data',loc='upper right', labels=['Train', 'Test'])
plt.show()


# Looks like both train and test datasets have an equivalent quantity of missing values. The floor count column is the one with most missing values followed by year built and cloud coverage

# In[ ]:


dftrain.describe()


# The meter reading have a big quantity of near 0 values. The type of consumed energy have more values on electricity (0).

# In[ ]:


fig, axes = plt.subplots(figsize=(12, 6), ncols=2)
fig.suptitle('Meter reading and meter type distribution', fontsize=14)
dftrain.meter_reading.plot.hist(color='darkcyan', bins=50, ax=axes[0],log=True)
dftrain.meter.plot.hist(color='darkcyan', bins=4, ax=axes[1])
axes[0].set(xlabel='Meter reading')
axes[1].set(xlabel='Meter type')
axes[1].set_xticks(np.arange(0, 4, step=1))
plt.show()


# In[ ]:


fig, axes = plt.subplots(figsize=(15, 10), ncols=3, nrows=3)
fig.suptitle('Features values distribution', fontsize=16)

dftrain.air_temperature.plot.hist(edgecolor='black',color='c', bins=50, ax=axes[0,0])
axes[0,0].set(xlabel='Air temperature')

dftrain.dew_temperature.plot.hist(edgecolor='black',color='royalblue', bins=50, ax=axes[0,1])
axes[0,1].set(xlabel='Dew temperature')

dftrain.wind_speed.plot.hist(edgecolor='black',color='c', bins=50, ax=axes[0,2])
axes[0,2].set(xlabel='Wind speed')

dftrain.sea_level_pressure.plot.hist(edgecolor='black',color='royalblue', bins=50, ax=axes[1,0])
axes[1,0].set(xlabel='Sea level pressure')

dftrain.wind_direction.plot.hist(edgecolor='black',color='c', bins=50, ax=axes[1,1])
axes[1,1].set(xlabel='Wind direction')

dftrain.precip_depth_1_hr.plot.hist(edgecolor='black',color='royalblue', bins=50, ax=axes[1,2],log=True)
axes[1,2].set(xlabel='Precip depth 1hr')

dftrain.cloud_coverage.plot.hist(edgecolor='black',color='c', bins=50, ax=axes[2,0])
axes[2,0].set(xlabel='Cloud coverage')

dftrain.year_built.plot.hist(edgecolor='black',color='royalblue', bins=50, ax=axes[2,1])
axes[2,1].set(xlabel='Year built')

dftrain.floor_count.plot.hist(edgecolor='black',color='c', bins=50, ax=axes[2,2])
axes[2,2].set(xlabel='Floor count')

plt.subplots_adjust(wspace=0.3,hspace=0.3)
plt.show()


# Air temperature, dew temperature and sea level pressure are left-tailed distributions. The amount of precipitation depth values is much higher near 0. There are more analyzed buildings that were built in the 60-70s and 2000s. Most buildings have 1 floor.

# In[ ]:


temp = dftrain
corrmat = round(temp.corr(method='pearson'),2)
plt.subplots(figsize=(10, 8))
sns.heatmap(corrmat, vmax=1.0, vmin=-1.0, square=True, annot=True, cmap='RdYlBu')
plt.title('Correlation', fontsize=15)
plt.show()


# Looking at the correlation matrix, there is not a big problem of multicollinearity. Some columns like air temperature and dew temperature have a correlation that seems large. Also the floor count column and area size seem to have a correlation. Let's investigate this further.

# In[ ]:


fig, axes = plt.subplots(figsize=(15, 6), ncols=2, nrows=1)

sns.scatterplot(x='dew_temperature', y='air_temperature', data=dftrain.sample(10000), alpha=0.4, color="purple", ax=axes[0])
axes[0].set_title('Air temperature by dew temperature', fontsize=14)
plt.xlabel('dew_temperature', fontsize=12)
plt.ylabel('air_temperature', fontsize=12)
sns.scatterplot(x='floor_count', y='square_feet', data=dftrain.sample(10000), alpha=0.4, color="purple", ax=axes[1])
axes[1].set_title('Floor count by square feet', fontsize=14)
plt.xlabel('floor_count', fontsize=12)
plt.ylabel('square_feet', fontsize=12)
plt.show()


# The correlation between air temperature and dew point seems quite evident, as both features say much the same thing to the model, so it may be interesting to evaluate the exclusion of either later. Since the floor count per area also has a correlation, but not so strong, they can possibly bring different information to the model, so it is better to leave them as features for the model.

# goupying by site Id.

# In[ ]:


site_id_means = dftrain.groupby(by='site_id').mean().reset_index()
site_id_means.head()


# In[ ]:


plt.figure(figsize=(6,5))
plt.title('Air temperature mean in each site_id', fontsize=14)
sns.barplot(x='site_id', y='air_temperature', 
            data = site_id_means,
            order = site_id_means.sort_values('air_temperature').site_id,
            palette = 'YlOrRd')
plt.xlabel('site_id', fontsize=12)
plt.ylabel('Air temperature', fontsize=12)
plt.show()


# Looking at the average temperature, we can see that it is quite different for different locations. Location 11 being the coldest and 2 being the hottest. The temperature difference is considerable, indicating that perhaps the buildings may even be in different countries.

# In[ ]:


plt.figure(figsize=(6,5))
plt.title('Meter reading mean in each site_id', fontsize=14)
sns.barplot(x='site_id', y='meter_reading', 
            data = site_id_means,
            order = site_id_means.sort_values('meter_reading').site_id,
            palette = 'RdYlBu')
plt.xlabel('site_id', fontsize=12)
plt.ylabel('meter_reading', fontsize=12)
plt.show()


# It seems that site 13 consumes a much larger amount of energy than other sites, let's take a closer look at this. I select only the constructions from site 13 and group by construction id, averaging features.

# In[ ]:


dftrain[dftrain.site_id==13].groupby('building_id').mean().sort_values(by='meter_reading',ascending=False).head()


# The energy consumption of a specific id 1099 building is much higher than all the others. This building probably has some data collection error or is an outlier. Further analysis is required later. For now let's remove this building from our dataframe.

# In[ ]:


dftrain = dftrain[dftrain.building_id!=1099]


# In[ ]:


plt.figure(figsize=(6,5))
plt.title('Meter reading mean in each site_id', fontsize=14)
sns.barplot(x='site_id', y='meter_reading', 
            data = dftrain.groupby(by='site_id').mean().reset_index(),
            order = site_id_means.sort_values('meter_reading').site_id,
            palette = 'RdYlBu')
plt.xlabel('site_id', fontsize=12)
plt.ylabel('meter_reading', fontsize=12)
plt.show()


# Now it seems that everything is more natural. Let's now check the consumption by type of building use.

# In[ ]:


plt.figure(figsize=(10,6))
sns.boxplot(x='primary_use', y='meter_reading', data=dftrain, showfliers=False)
plt.xticks(rotation='vertical')
plt.show()


# The most energy-consuming buildings are health and utilities buildings. The least consuming are religious. For now we finish the exploratory analysis, I will maintain this kernel updated.
