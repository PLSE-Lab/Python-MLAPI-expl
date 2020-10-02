#!/usr/bin/env python
# coding: utf-8

# The task is to build a model which will predict the consumtion of the energy for the building based on the weather condition. These will allow to assess how the improvment of building energy efficiency will influence the energy consumption. I will start with looking at the data about building. 
# ![.](https://images-wixmp-ed30a86b8c4ca887773594c2.wixmp.com/f/46e8636d-7443-4725-b69f-1a7cd3eee693/d620ip0-c6be5721-ed05-42cf-88db-1e8ea25f1d6e.jpg/v1/fill/w_1024,h_768,q_75,strp/window_in_window_by_yenneferx_d620ip0-fullview.jpg?token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1cm46YXBwOjdlMGQxODg5ODIyNjQzNzNhNWYwZDQxNWVhMGQyNmUwIiwiaXNzIjoidXJuOmFwcDo3ZTBkMTg4OTgyMjY0MzczYTVmMGQ0MTVlYTBkMjZlMCIsIm9iaiI6W1t7ImhlaWdodCI6Ijw9NzY4IiwicGF0aCI6IlwvZlwvNDZlODYzNmQtNzQ0My00NzI1LWI2OWYtMWE3Y2QzZWVlNjkzXC9kNjIwaXAwLWM2YmU1NzIxLWVkMDUtNDJjZi04OGRiLTFlOGVhMjVmMWQ2ZS5qcGciLCJ3aWR0aCI6Ijw9MTAyNCJ9XV0sImF1ZCI6WyJ1cm46c2VydmljZTppbWFnZS5vcGVyYXRpb25zIl19.W5iJUg6YB0_uKaoT0ytLbHAFV8JDoAc1rbJkopQm1kA)

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from bokeh import palettes as bh
from datetime import datetime


# In[ ]:


folder = '../input/ashrae-energy-prediction/'
weather_train_df = pd.read_csv(folder + 'weather_train.csv')
building_meta_df = pd.read_csv(folder + 'building_metadata.csv')
train_df = pd.read_csv(folder + 'train.csv')


# REducing memory function is taken from [ASHRAE -Start Here: A Gentle Introduction](https://www.kaggle.com/caesarlupum/ashrae-start-here-a-gentle-introduction )

# In[ ]:


## Function to reduce the DF size
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df
## REducing memory

weather_train_df = reduce_mem_usage(weather_train_df)
building_meta_df = reduce_mem_usage(building_meta_df)
train_df = reduce_mem_usage(train_df)
merged_train = pd.merge(train_df, building_meta_df, how="left", on=["building_id"])


# # Buildings
# Let's start with `building_metadata.csv` dataset. 
# We have 5 columns:
# * `site_id`- we will use it to join with weather files
# * `building_id` - id of the building, we have 1449 buildings (one row per building)	
# * `primary_use` - we have 16 different primary use of the buildings, for Education and Office the number of buildings is quite reasonable, but for example for Religious worship we have only 3 buildings. 	
# * `square_feet` - Size of building, we have building as small as 283 ft^2 	(26 m^2) and as big as 875000 ft^2 81290 m^2
# * `year_built` - 	Newest building come from 2017, the oldest one from 1900. It may be important information as the technology changed significantly in between. We miss this information for all of the building. 
# * `floor_count` - Higest building is 26 floors, lowest 1. Again we don't have this information for some of the buildings. 

# In[ ]:


print('The shape of our data is:', building_meta_df.shape)
print(f"There are {len(building_meta_df['building_id'].unique())} distinct buildings." )


# In[ ]:


building_meta_df.describe()


# In[ ]:


building_meta_pu = building_meta_df.groupby('primary_use').agg({'building_id':'count',
                                             'year_built':['min','max','mean'],
                                            'square_feet':['min','max','mean']})
building_meta_pu_mi = building_meta_pu.columns
building_meta_pu_mi = pd.Index([e[0] + ' ' + e[1] for e in building_meta_pu_mi.tolist()])
building_meta_pu.columns = building_meta_pu_mi
building_meta_pu = building_meta_pu.sort_values('building_id count', ascending = False)
building_meta_pu


# In[ ]:





# In[ ]:


fig, ax = plt.subplots()

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.hist(building_meta_df['square_feet'], 20, facecolor=bh.magma(6)[1], alpha=0.75)
plt.xlabel('Size in square feet')
plt.ylabel('Number of buildings')
plt.title('Histogram of size in square feet')
plt.show()


# Let's see if there is any difference in the size dependent of the primary use. 

# In[ ]:


fig, (ax1,ax2,ax3,ax4)  = plt.subplots(4, sharex=True, figsize=(12,10))

ax1.hist(building_meta_df[building_meta_df['primary_use']==building_meta_pu.index[0]]['square_feet'],
                     20, facecolor=bh.viridis(6)[1], alpha=0.75, label = building_meta_pu.index[0])
ax1.legend(prop={'size': 10})
a2 = ax2.hist(building_meta_df[building_meta_df['primary_use']==building_meta_pu.index[1]]['square_feet'],
                     20, facecolor=bh.viridis(6)[2], alpha=0.75, label = building_meta_pu.index[1])
ax2.legend(prop={'size': 10})
a3 = ax3.hist(building_meta_df[building_meta_df['primary_use']==building_meta_pu.index[2]]['square_feet'],
                     20, facecolor=bh.viridis(6)[3], alpha=0.75, label = building_meta_pu.index[2])
ax3.legend(prop={'size': 10})
a4 = ax4.hist(building_meta_df[building_meta_df['primary_use']==building_meta_pu.index[3]]['square_feet'],
                     20, facecolor=bh.viridis(6)[4], alpha=0.75, label = building_meta_pu.index[3])
ax4.legend(prop={'size': 10})
ax1.set_xlim([0, 500000])
plt.show()


# Now we will look on the histogram of Year of built. 

# In[ ]:


fig, ax = plt.subplots()

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.hist(building_meta_df['year_built'], 20, facecolor=bh.magma(6)[2], alpha=0.75)
plt.xlabel('Year of building')
plt.ylabel('Number of buildings')
plt.title('Histogram of year of building')
plt.show()


# In[ ]:


fig, (ax1,ax2,ax3,ax4)  = plt.subplots(4, sharex=True, figsize=(12,10))

ax1.hist(building_meta_df[building_meta_df['primary_use']==building_meta_pu.index[0]]['year_built'],
                     20, facecolor=bh.viridis(6)[1], alpha=0.75, label = building_meta_pu.index[0])
ax1.legend(prop={'size': 10})
a2 = ax2.hist(building_meta_df[building_meta_df['primary_use']==building_meta_pu.index[1]]['year_built'],
                     20, facecolor=bh.viridis(6)[2], alpha=0.75, label = building_meta_pu.index[1])
ax2.legend(prop={'size': 10})
a3 = ax3.hist(building_meta_df[building_meta_df['primary_use']==building_meta_pu.index[2]]['year_built'],
                     20, facecolor=bh.viridis(6)[3], alpha=0.75, label = building_meta_pu.index[2])
ax3.legend(prop={'size': 10})
a4 = ax4.hist(building_meta_df[building_meta_df['primary_use']==building_meta_pu.index[3]]['year_built'],
                     20, facecolor=bh.viridis(6)[4], alpha=0.75, label = building_meta_pu.index[3])
ax4.legend(prop={'size': 10})
ax1.set_xlim([1900, 2020])
plt.show()


# And the last will be the number of floor. 

# In[ ]:


fig, ax = plt.subplots()

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.hist(building_meta_df['floor_count'], 20, facecolor=bh.magma(6)[4], alpha=0.75)
plt.xlabel('Number of floor')
plt.ylabel('Number of buildings')
plt.title('Histogram of floor count')
plt.show()


# In[ ]:


fig, (ax1,ax2,ax3,ax4)  = plt.subplots(4, sharex=True, figsize=(12,10))

ax1.hist(building_meta_df[building_meta_df['primary_use']==building_meta_pu.index[0]]['floor_count'],
                     10, facecolor=bh.viridis(6)[1], alpha=0.75, label = building_meta_pu.index[0])
ax1.legend(prop={'size': 10})
a2 = ax2.hist(building_meta_df[building_meta_df['primary_use']==building_meta_pu.index[1]]['floor_count'],
                     10, facecolor=bh.viridis(6)[2], alpha=0.75, label = building_meta_pu.index[1])
ax2.legend(prop={'size': 10})
a3 = ax3.hist(building_meta_df[building_meta_df['primary_use']==building_meta_pu.index[2]]['floor_count'],
                     10, facecolor=bh.viridis(6)[3], alpha=0.75, label = building_meta_pu.index[2])
ax3.legend(prop={'size': 10})
a4 = ax4.hist(building_meta_df[building_meta_df['primary_use']==building_meta_pu.index[3]]['floor_count'],
                     10, facecolor=bh.viridis(6)[4], alpha=0.75, label = building_meta_pu.index[3])
ax4.legend(prop={'size': 10})
ax1.set_xlim([0, 25])
plt.show()


# # Weather 
# Let's now look at `weather_train_df`. 
# * site_id - We have 16 unique site_id. 
# * air_temperature - First of all we have some rows with null values for air_temputere. It's not null for 139 718 rows from 139 773. The lowest air_temputere is -28.9, the higest is 47.2. We will soon see if the air temputere is significantly different dependent from the site_id. 
# * cloud_coverage - We have data only for 70 600 rows. 
# * dew_temperature[1] - We have data for 139 660 rows. 
# * precip_depth_1_hr - We have data for 89 484 rows. 
# * sea_level_pressure - We have data for 129155 rows. 
# * wind_direction - We have data for 133 505 rows. Interesting is that it is numeric data from range 0-360, where 0&deg;/360&deg; can be interpreted as directly North. 
# * wind_speed - We have data for 139 469 rows. The minimum value is 0, maximum value is 19m/s (68.4 km/h).
# 
# [1] Dew temputere is the temperature to which air must be cooled to become saturated with water vapor. When further cooled, the airborne water vapor will condense to form liquid water (dew).

# In[ ]:


print('The shape of our data is:', weather_train_df.shape)
print(f"There are {len(weather_train_df.site_id.unique())} unique site_id." )


# In[ ]:


weather_train_df.describe()


# Let's look at the histogram of temputere from different sites.
# We can see that in some site_id we observe extreme cold weather (for example 11 or 13), in some other extreme hot weather (2), but we also have some with moderate weather (12, 13). 

# In[ ]:


site_ids = weather_train_df['site_id'].unique()
fig, axs   = plt.subplots(4,4, sharex=True,sharey=True, figsize=(12,12))

for i in range(0,16):
    axs[i//4,i%4 ].hist(weather_train_df[weather_train_df['site_id']==site_ids[i]]['air_temperature'],
                     20, facecolor=bh.magma(19)[i], alpha=0.75, label = site_ids[i])
    axs[i//4,i%4].set_title(''.join(['Site id:',str(site_ids[i])]))
plt.show()


# 

# And the wind.

# In[ ]:


site_ids = weather_train_df['site_id'].unique()
fig, axs   = plt.subplots(4,4, sharex=True,sharey=True, figsize=(12,10))

for i in range(0,16):
    axs[i//4,i%4 ].hist(weather_train_df[weather_train_df['site_id']==site_ids[i]]['wind_speed'],
                     20, facecolor=bh.magma(19)[i], alpha=0.75, label = site_ids[i])
    axs[i//4,i%4].set_title(''.join(['Site id:',str(site_ids[i])]))
plt.show()


# And the pressure. We have data only from the first 5 site_id. 

# In[ ]:


site_ids = weather_train_df['site_id'].unique()
fig, axs   = plt.subplots(5, sharex=True,sharey=True, figsize=(12,10))

for i in range(0,5):
    axs[i ].hist(weather_train_df[weather_train_df['site_id']==site_ids[i]]['sea_level_pressure'],
                     20, facecolor=bh.magma(19)[i], alpha=0.75, label = site_ids[i])
    axs[i].set_title(''.join(['Site id:',str(site_ids[i])]))
plt.show()


# # Train
# Train dataset has only 4 columns:
# * building_id - described in part about buildings, we will use it to join these datasets.
# * meter - we have 4 posible option: 0 is electricity, 1 - chilledwater, 2 - steam, hotwater -3
# * timestamp - all data come from one year - 2016. 
# * meter_reading - target value

# In[ ]:


print('The shape of our data is:', train_df.shape)
print(f"There are {len(train_df.building_id.unique())} unique building_id." )


# In[ ]:


train_df.head()


# In[ ]:


train_df['meter'].unique()


# In[ ]:


train_df.groupby('meter').agg({'building_id':'nunique', 'meter_reading':sum, 'timestamp': ['min','max']})


# Let's look more detailed on the meter reading. 

# In[ ]:


train_df['log_meter_reading']=np.log1p(train_df['meter_reading'])


# In[ ]:


meter_ids = train_df['meter'].unique()
meter_ids_map = {0: 'electricity', 1: 'chilledwater', 2: 'steam', 3:'hotwater'}

fig, axs   = plt.subplots(2,2, sharex=True,sharey=True, figsize=(12,12))

for i in range(0,4):
    axs[i//2,i%2 ].hist(train_df[train_df['meter']==meter_ids[i]]['log_meter_reading'],
                     20, facecolor=bh.cividis(4)[i], alpha=0.75, label = meter_ids_map[meter_ids[i]])
    axs[i//2,i%2].set_title(''.join(['Logaritm of meter id:',meter_ids_map[meter_ids[i]]]))
plt.show()


# I put 0 in place of -Inf. But we generally should look more carefully at those 0.0 values. It seems that we have quite a lot of them. 

# In[ ]:


len(train_df[train_df['meter_reading']==0]['meter_reading'])


# # Important concepts
# I've decided to check if some of the concepts from [discussion topic](https://www.kaggle.com/c/ashrae-energy-prediction/discussion/112871) are true. 
# 1. Weekly cycle of using energy. We expect Education buildings to use energy mostly during week days, while Religious only during the special weekday. 
# 2. For some building we can see trend independent from the weather, connected with increasing/decreasing of amount of people using the building. 
# 3. There is singnificant difference between old and new building because of technology. These difference is not linear, but caused by the change of technology or practise (for example some old building have high ceilings, what is not so popular nowadays). 
# 4. Wind is more significant for building where you are able to open the window, so it's not so significant for very high, new builidings, where windows often are fix. 
# 5. Buildings built in the places where weather often is extreme are better prepared for different temperature. 

# In[ ]:



merged_train.head()


# In[ ]:


merged_train['Weekday'] = merged_train['timestamp'].apply(lambda row: datetime.strptime(row, '%Y-%m-%d %H:%M:%S'))


# In[ ]:


merged_train['Weekday_num'] = merged_train['Weekday'].apply(lambda row: row.strftime('%w'))
merged_train['Weekday'] = merged_train['Weekday'].apply(lambda row: row.strftime('%A'))
merged_train['log_meter_reading']=np.log1p(merged_train['meter_reading'])


# In[ ]:


merged_train.head()


# In[ ]:


groupby_day = merged_train.groupby(['primary_use','Weekday','Weekday_num','site_id']).agg({'log_meter_reading':['sum','mean']})
merged_train


# In[ ]:


groupby_day_mi = groupby_day.columns
groupby_day_mi = pd.Index([e[0] + ' ' + e[1] for e in groupby_day_mi.tolist()])
groupby_day.columns = groupby_day_mi
groupby_day = groupby_day.reset_index()


# In[ ]:





# In[ ]:



site_ids = list(groupby_day['site_id'].unique())
primary_uses = list(building_meta_pu[building_meta_pu['building_id count']>20].index)
fig, axs   = plt.subplots(4,4, sharex=True,sharey=True, figsize=(12,12))
width = 1/9
for i in range(0,16):
    df_site = groupby_day[groupby_day['site_id']==site_ids[i]]
    for j in range(0,8):
        axs[i//4,i%4 ].bar(df_site[df_site['primary_use']==primary_uses[j]]['Weekday_num'].astype(float)+width*j,df_site[df_site['primary_use']==primary_uses[j]]['log_meter_reading mean']
                     , facecolor=bh.viridis(8)[j], alpha=0.75, label = primary_uses[j],width=width )
        axs[i//4,i%4].set_title(''.join(['Site id:',str(site_ids[i])]))
plt.show()


# In[ ]:





# Work in progress....
