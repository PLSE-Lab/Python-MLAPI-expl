#!/usr/bin/env python
# coding: utf-8

# Enexis, Liander, and Stedin are the three major network administrators of the Netherlands and, together, they provide energy to nearly the entire country. Every year, they release on their websites a table with the energy consumption of the areas under their administration.
# 
# The data are anonymized by aggregating the Zipcodes so that every entry describes at least 10 connections.
# 
# This market is not competitive, meaning that the zones are assigned. This means that every year they roughly provide energy to the same zipcodes. Small changes can happen from year to year either for a change of management or for a different aggregation of zipcodes.
# 
# This kernel aims to explore and spark some ideas on how to use this new dataset. 
# 
# In drafting it, I was able to spot a few issues in the dataset and correct them. However, the data are coming from different companies and cover several years: assume that there will be inconsistencies for a while.
# 
# ***v14 notes***: this is a run to test that the new version of the dataset is healthy. The 2018 data are added for 2 companies out of 3, they will not be displayed by the current analysis. As soon as Stedin put their data out, a new version will be released
# 
# ***v16 notes***: This run includes the data for Stedin in 2018 as well, the purpose is again to test that the dataset preparation went reasonably well
# 
# ***v17 notes***: A run to test 2019 data.

# In[ ]:


import numpy as np
import pandas as pd

#visualization
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from os.path import join, isfile
from os import path, scandir, listdir

import gc


# In[ ]:


def list_all_files(location='kaggle/input/', pattern=None, recursive=True):
    """
    This function returns a list of files at a given location (including subfolders)
    
    - location: path to the directory to be searched
    - pattern: part of the file name to be searched (ex. pattern='.csv' would return all the csv files)
    - recursive: boolean, if True the function calls itself for every subdirectory it finds
    """
    subdirectories= [f.path for f in scandir(location) if f.is_dir()]
    files = [join(location, f) for f in listdir(location) if isfile(join(location, f))]
    if recursive:
        for directory in subdirectories:
            files.extend(list_all_files(directory))
    if pattern:
        files = [f for f in files if pattern in f]
    return files


# The data are structured as follows

# In[ ]:


list_all_files('/kaggle/input/Electricity/', pattern='stedin')


# *Note*: Every file actually refers to the energy consumption of the year before. Thus `stedin_electricity_2011.csv` contains the data about 2010, for Stedin administrated connections.
# 
# Let's proceed in importing everything in a convenient structure.

# In[ ]:


def importer(file_list):
    imported = {}
    for file in file_list:
        yr = file.split('_')[-1].split('.')[0]
        if '0101' in yr:
            yr = yr.replace('0101', '')
        name = file.split('/')[-1].split('_')[0]
        # print(name, yr)
        df = pd.read_csv(file)
        # print(df.shape)
        imported[name + '_' + yr] = df
        del df
    return imported


# In[ ]:


elec_list = list_all_files('/kaggle/input/Electricity/')
gas_list = list_all_files('/kaggle/input/Gas/')
imp_elec = importer(elec_list)
imp_gas = importer(gas_list)
print('Done!')


# # A first look at the data
# 
# In this section, we will try to merge all these files together so that we can have an overview of the energy consumption at a given zip code (or group of zip codes) every year. We will first merge by company and then concatenate the results
# 
# There are a few obstacles:
# * the zip codes can be grouped differently every year
# * the zip codes can change from year to year (some of them got redefined during this period)
# * not only the annual consumption changes but also the number of connections and other indicators change every year, we have to account for that.
# 
# As we will see, this approach is useful to keep track of what happens to a specific group of zip codes year by year, but it is not very good if we try some kind of aggregation.

# In[ ]:


def merge_manager(data_dict):
    all_man = pd.DataFrame()
    n_rows = 0
    for key in data_dict.keys():
        df = data_dict[key].copy()
        yr = key.split('_')[1]
        yr = str(int(yr) - 1) # account for the "delayed data issue"
        df = df.rename(columns={'annual_consume' : 'annual_consume_' + yr,
                               'delivery_perc': 'delivery_perc_' + yr,
                               'num_connections': 'num_connections_' + yr,
                               'perc_of_active_connections': 'perc_of_active_connections_' + yr,
                               'annual_consume_lowtarif_perc': 'annual_consume_lowtarif_perc_' + yr,
                               'smartmeter_perc': 'smartmeter_perc_' + yr})
        del df['type_conn_perc']
        del df['type_of_connection']
        del df['net_manager']
        del df['purchase_area']
        n_rows += df.shape[0]
        if len(all_man) == 0:
            all_man = df.copy()
        else:
            del df['street']
            del df['city']
            all_man = pd.merge(all_man, df, on=['zipcode_from', 'zipcode_to'], how='inner') # 'city', 'street',  
        del df
        gc.collect()
    print(f"Total rows before merge: {n_rows}")
    print(f"Total rows after merge: {all_man.shape[0]}")
    return all_man


def merge_yr(data_dict):
    all_yr = pd.DataFrame()
    for manager in ['enexis', 'liander', 'stedin']:
        print(manager)
        tmp = { key: data_dict[key] for key in data_dict.keys() if manager in key}
        all_man = merge_manager(tmp)
        if len(all_yr) == 0:
            all_yr = all_man.copy()
        else:
            all_yr = pd.concat([all_yr, all_man], ignore_index=True, join='inner')
        del all_man
        gc.collect()
        print("_"*40)
    print(f"Final shape: {all_yr.shape}")
    return all_yr


# In[ ]:


print("Electricity merging...")
elec_full = merge_yr(imp_elec)
print('_'*40)
print('_'*40)
print("Gas merging...")
gas_full = merge_yr(imp_gas)


# As we see, we lose a considerable amount of entries due to the merge, we will think about a solution for this later and move on for now.
# 
# Moreover, one company does not have a `2009` table and I am silently dropping the other `2009` tables. 
# 
# During the merge, I have also corrected for the year ambiguity. Thus now, for example, `num_connections_2014` really refers to the value of 2014.

# In[ ]:


elec_full.head()


# One thing we can do is to calculate the consumption per connection. Thus we divide the annual consumption by the number of active connections.

# In[ ]:


def consume_per_connection(data, consume_list):
    for col in consume_list:
        yr = col.split('_')[-1]
        data['consume_per_conn_'+yr] = data[col] / (data['num_connections_' + yr] * 
                                                   data['perc_of_active_connections_' + yr] / 100)
        data.loc[data['consume_per_conn_' + yr] == np.inf, 'consume_per_conn_' + yr] = 0
    return data


# In[ ]:


consume = [col for col in elec_full.columns if 'annual_consume_2' in col]
consume.sort()


# In[ ]:


elec_full = consume_per_connection(elec_full, consume)
elec_full[consume + [col for col in elec_full.columns if 'consume_per_conn_' in col]].describe()


# In[ ]:


gas_full = consume_per_connection(gas_full, consume)
gas_full[consume + [col for col in gas_full.columns if 'consume_per_conn_' in col]].describe()


# Nothin looks particulary weird but it was at this point that I have noticed a problem with a few files that were using the `.` for the thousands in a non obvious way. The solution proposed is temporary but keep that in mind.
# 
# We can make some  plots with the distribution of the consumption of electricity and gas every year. (We exclude the ectremely large values)
# 
# (The warning happens only on the kaggle kernel, probably due to a different scipy/seaborn version)

# In[ ]:


fig, ax = plt.subplots(1,2, figsize=(17,8))

for col in consume:
    sns.distplot(elec_full.loc[elec_full[col] < 20000, col], 
                 hist=False, label=col.split('_')[-1], ax=ax[0], axlabel='Annual Consumption')
    sns.distplot(gas_full.loc[gas_full[col] < 6000, col], 
                 hist=False, label=col.split('_')[-1], ax=ax[1], axlabel='Annual Consumption') 

ax[0].set_title('Electricity', fontsize=15)
ax[1].set_title('Gas', fontsize=15)
fig.suptitle('Annual consumption', fontsize=22)
plt.show()


# And the same can be done for the consumption per connection we created above

# In[ ]:


cons_per_conn = [col for col in gas_full.columns if 'consume_per_conn_' in col]
cons_per_conn.sort()


# In[ ]:


fig, ax = plt.subplots(1,2, figsize=(17,8))

for col in cons_per_conn:
    sns.distplot(elec_full.loc[elec_full[col] < 1000, col], 
                 hist=False, label=col.split('_')[-1], ax=ax[0], axlabel='Annual Consumption')
    sns.distplot(gas_full.loc[gas_full[col] < 400, col], 
                 hist=False, label=col.split('_')[-1], ax=ax[1],  axlabel='Annual Consumption')

ax[0].set_title('Electricity', fontsize=15)
ax[1].set_title('Gas', fontsize=15)
fig.suptitle('Annual consume per connection', fontsize=22)
plt.show()


# It appears to be very consistent every year, I wonder if we can observe that better (yes, it is pairplot time)

# In[ ]:


plt.figure(figsize=(10,10))
sns.pairplot(elec_full[consume].sample(10000), kind="reg")

plt.suptitle('Correlations between electricity consumptions at different years', fontsize=22, y=1.01)
plt.show()


# Yep, it is pretty consistent.
# 
# Next, we could focus on aggregating by city

# In[ ]:


elec_city = elec_full[['city', 'annual_consume_2009']].groupby('city', as_index=False).sum()

for col in consume:
    if col == 'annual_consume_2009':
        continue
    tmp = elec_full[['city', col]].groupby('city', as_index=False).sum()
    elec_city = pd.merge(elec_city, tmp, on='city')

elec_city = elec_city.set_index('city')
elec_city['mean_consume'] = elec_city.mean(axis=1)
elec_city.sample(5)


# Let's see the top 10 cities by electricity consumption

# In[ ]:


tmp = elec_city.nlargest(10, 'mean_consume')
del tmp['mean_consume'] # so it doesn't show up in the plot
tmp.columns = tmp.columns.str.replace('annual_consume_', '')
ax = tmp.T.plot(figsize=(10,8))
ax.set_xticklabels(['','2009', '2011','2013','2015', '2017'])
ax.set_ylabel("kWh")
ax.set_xlabel("Year")
ax.set_title("Electricity consumption by year (top 10 cities)", fontsize=18)
del tmp


# Which is nice but it is just confirming that some cities are more populated than others. Let's do it again with consumption per connection.

# In[ ]:


elec_city = elec_full[['city', 'annual_consume_2009', 'num_connections_2009']].groupby('city', as_index=False).sum()
elec_city['cons_per_con_2009'] = elec_city['annual_consume_2009'] / elec_city['num_connections_2009']
del elec_city['num_connections_2009']
del elec_city['annual_consume_2009']

for col in consume:
    if col == 'annual_consume_2009':
        continue
    yr = col.split('_')[-1]
    tmp = elec_full[['city', col, 'num_connections_'+yr]].groupby('city', as_index=False).sum()
    tmp['cons_per_con_'+yr] = tmp[col] / tmp['num_connections_'+yr]
    del tmp[col]
    del tmp['num_connections_'+yr]
    elec_city = pd.merge(elec_city, tmp, on='city')

elec_city = elec_city.set_index('city')
elec_city['mean_consume'] = elec_city.mean(axis=1)
tmp = elec_city.nlargest(10, 'mean_consume')
del tmp['mean_consume']
tmp.columns = tmp.columns.str.replace('cons_per_con_', '')
ax = tmp.T.plot(figsize=(10,8), title='Electricity consumption per connection by year')
ax.set_xticklabels(['','2009', '2011', '2013','2015','2017', '2019'])
ax.set_ylabel("kWh")
ax.set_xlabel("Year")
ax.set_title("Electricity consumption per connection by year (top 10 cities)", fontsize=18)
del tmp
plt.show()


# In[ ]:


gas_city = gas_full[['city', 'annual_consume_2009', 'num_connections_2009']].groupby('city', as_index=False).sum()
gas_city['cons_per_con_2009'] = gas_city['annual_consume_2009'] / gas_city['num_connections_2009']
del gas_city['num_connections_2009']
del gas_city['annual_consume_2009']

for col in consume:
    if col == 'annual_consume_2009':
        continue
    yr = col.split('_')[-1]
    tmp = gas_full[['city', col, 'num_connections_'+yr]].groupby('city', as_index=False).sum()
    tmp['cons_per_con_'+yr] = tmp[col] / tmp['num_connections_'+yr]
    del tmp[col]
    del tmp['num_connections_'+yr]
    gas_city = pd.merge(gas_city, tmp, on='city')

gas_city = gas_city.set_index('city')
gas_city['mean_consume'] = gas_city.mean(axis=1)
tmp = gas_city.nlargest(10, 'mean_consume')
del tmp['mean_consume']
tmp.columns = tmp.columns.str.replace('cons_per_con_', '')
ax = tmp.T.plot(figsize=(10,8), title='Gas consumption per connection by year')
ax.set_xticklabels(['','2009', '2011', '2013','2015','2017', '2019'])
ax.set_ylabel("m3")
ax.set_xlabel("Year")
ax.set_title("Gas consumption per connection by year (top 10 cities)", fontsize=18)
del tmp


# In[ ]:


print(elec_city.nlargest(10, 'mean_consume')['mean_consume'])
print('_'*40)
print(gas_city.nlargest(10, 'mean_consume')['mean_consume'])


# Not only this plot is very bad looking, but also somewhat misleading since we are aggregating by city a dataset that does not have all the data of every city (due to the merge). Let's free up some memory and do it in a better way and see if we still get what it appears to be a slightly negative trend.
# 
# # Aggregating data by city, the better way

# In[ ]:


del elec_full
del gas_full
del elec_city
del gas_city

gc.collect()


# This time, we aggregate the data *before* the merge.

# In[ ]:


def aggr_yr(data, yr):
    # useful features
    data['net_annual_cons_'+yr] = data['annual_consume'] * data['delivery_perc'] / 100
    data['self_production_'+yr] = data['annual_consume'] - data['net_annual_cons_'+yr]
    data['low_tarif_cons_'+yr] = data['annual_consume'] * data['annual_consume_lowtarif_perc'] / 100
    data['active_conn_'+yr] = data['num_connections'] * data['perc_of_active_connections'] / 100
    data['num_smartmeters_'+yr] = data['num_connections'] * data['smartmeter_perc'] / 100
    data = data.rename(columns={'annual_consume': 'annual_consume_'+yr})
    # aggregations
    aggregation = data[['city', 'annual_consume_'+yr, 'net_annual_cons_'+yr,
                        'self_production_'+yr, 'low_tarif_cons_'+yr,
                        'active_conn_'+yr, 'num_smartmeters_'+yr]].groupby('city', as_index=False).sum()
    return aggregation

def aggr_mng(data_dict):
    all_man = pd.DataFrame()
    for key in data_dict.keys():
        df = data_dict[key].copy()
        yr = key.split('_')[-1]
        yr = str(int(yr) - 1) # account for the "delayed data issue"
        if len(all_man) == 0:
            all_man = aggr_yr(df, yr)
        else:
            df = aggr_yr(df,yr)
            all_man = pd.merge(all_man, df, on='city')
        del df
        gc.collect()
    all_man = all_man.set_index('city')
    return all_man

def aggregations(data_dict):
    result = pd.DataFrame()
    for manager in ['enexis', 'liander', 'stedin']:
        print(manager)
        tmp = { key: data_dict[key] for key in data_dict.keys() if manager in key}
        all_man = aggr_mng(tmp)
        if len(result) == 0:
            result = all_man.copy()
        else:
            result = pd.concat([result, all_man], join='inner')
        del all_man
        gc.collect()
        print("_"*40)
    print(f"Final shape: {result.shape}")
    return result


# In[ ]:


cities_el = aggregations(imp_elec)
cities_el.sample(10)


# In[ ]:


cities_el.describe()


# In[ ]:


cities_gas = aggregations(imp_gas)
cities_gas.sample(10)


# In[ ]:


cities_gas.describe()


# ## Energy consumption
# 
# Let's have a look at the total energy consumption of the country.

# In[ ]:


consume = [col for col in cities_el.columns if 'annual_consume_' in col]
consume.sort()


# In[ ]:


fig, ax = plt.subplots(1,2, figsize=(20, 8))

cities_el[consume].sum().plot(title='Total Electricity consumption per year', ax=ax[0])
ax[0].set_xticklabels(['','2009', '2011', '2013','2015','2017', '2019'])
ax[0].set_ylabel("kWh")
ax[0].set_xlabel("Year")
cities_gas[consume].sum().plot(title='Total Gas consumption per year', ax=ax[1])
ax[1].set_xticklabels(['','2009', '2011', '2013','2015','2017', '2019'])
ax[1].set_ylabel("m3")
ax[1].set_xlabel("Year")
plt.show()


# We indeed observe a descending trend, with something odd happening in 2017. **This can very well be a mistake during the data cleaning or data preparation.**
# 
# Again, focusing on cities with high consumption will just lead to the most populated cities.

# In[ ]:


tmp = cities_el[consume].copy()
tmp['mean_consume'] = tmp.mean(axis=1)
tmp = tmp.nlargest(10, 'mean_consume')
del tmp['mean_consume']
tmp.columns = tmp.columns.str.replace('annual_consume_', '')

fig, ax = plt.subplots(1,2, figsize=(20, 8))
tmp.T.sum().plot(kind='bar', title='Total Electricity top 10 cities', ax=ax[0])
tmp.T.plot(title='Electricity consumption per year top 10 cities',ax=ax[1])
ax[0].set_ylabel("kWh")
ax[1].set_xticklabels(['','2009', '2011', '2013','2015','2017', '2019'])
ax[1].set_xlabel("Year")
plt.show()


# In[ ]:


tmp = cities_gas[consume].copy()
tmp['mean_consume'] = tmp.mean(axis=1)
tmp = tmp.nlargest(10, 'mean_consume')
del tmp['mean_consume']
tmp.columns = tmp.columns.str.replace('annual_consume_', '')

fig, ax = plt.subplots(1,2, figsize=(20, 8))
tmp.T.sum().plot(kind='bar', title='Total Gas top10 cities', ax=ax[0])
tmp.T.plot(title='Gas consumption per year top10 cities',ax=ax[1])
ax[0].set_ylabel("m3")
ax[1].set_xticklabels(['','2009', '2011', '2013','2015','2017', '2019'])
ax[1].set_xlabel("year")
plt.show()


# ## Self-production
# 
# Let's see if we can identify some trend in the energy produced by the population (most likely solar panels, for which there have been some incentives given by the government)

# In[ ]:


self_prod = [col for col in cities_el.columns if 'self_production_' in col]
self_prod.sort()


# In[ ]:


ax = cities_el[self_prod].sum().plot(figsize=(12, 8), fontsize=12)
ax.set_xticklabels(['','2009', '2011', '2013','2015','2017', '2019'])
ax.set_title('Total Electricity self-produced per year', fontsize=22)
ax.set_ylabel("kWh", fontsize=12)
ax.set_xlabel("year", fontsize=12)
plt.show()


# If we define the top cities as the one that had the maximum production at some point in the period under analysis, we get some **Almere pride** in the air

# In[ ]:


tmp = cities_el[self_prod].copy()
tmp['max_prod'] = tmp.max(axis=1)
tmp = tmp.nlargest(10, 'max_prod')
del tmp['max_prod']
tmp.columns = tmp.columns.str.replace('self_production_', '')

fig, ax = plt.subplots(1,2, figsize=(20, 8))
tmp['2018'].T.plot(kind='bar', title='Electricity self-produced in 2018, top10 cities', ax=ax[0])
tmp.T.plot(title='Electricity self-produced per year top10 cities',ax=ax[1])
ax[0].set_ylabel("kWh")
ax[1].set_xticklabels(['','2009', '2011', '2013','2015','2017', '2019'])
ax[1].set_xlabel("Year")
plt.show()


# ## Smart meters

# In[ ]:


smrt = [col for col in cities_el.columns if 'num_smartmeters_' in col]
smrt.sort()


# In[ ]:


ax = cities_el[smrt].sum().plot(figsize=(12, 8),fontsize=12)
ax.set_xticklabels(['','2009', '2011', '2013','2015','2017', '2019'])
ax.set_ylabel("Number of smart meters", fontsize=12)
ax.set_xlabel("year", fontsize=12)
ax.set_title('Total smart meters per year', fontsize=22)
plt.show()


# In[ ]:


tmp = cities_el[smrt].copy()
tmp['max_num'] = tmp.max(axis=1)
tmp = tmp.nlargest(10, 'max_num')
del tmp['max_num']
tmp.columns = tmp.columns.str.replace('num_smartmeters_', '')

fig, ax = plt.subplots(1,2, figsize=(20, 8))
tmp['2018'].T.plot(kind='bar', title='Total number of smart meters in 2018, top10 cities', ax=ax[0])
tmp.T.plot(title='Total number of smart meters per year top10 cities',ax=ax[1])
ax[0].set_ylabel("Number of smart meters")
ax[1].set_xticklabels(['','2009', '2011', '2013','2015','2017', '2019'])
ax[1].set_xlabel("Year")
plt.show()


# While it was predictable to find Amsterdam and Rotterdam leading this chart (being the biggest cities), it is interesting to notice that they grew similarly even though their networks are managed by different companies (Liander and Stedin, respectively)
# 
# # Conclusions
# 
# I hope this kernel gave you some inspiration on how to use this dataset, please share your thoughts about it, use it, make something beautiful, find problems in the data and help me fix them :)
# 
# On a more personal note, I am not used (as you see) to making visualizations so every feedback will be very much appreciated.
# 
# Cheers.

# In[ ]:




