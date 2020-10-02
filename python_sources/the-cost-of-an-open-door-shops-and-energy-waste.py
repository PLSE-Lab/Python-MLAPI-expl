#!/usr/bin/env python
# coding: utf-8

# In the past months, the longtime-coming debate about environmental sustainability became finally of primary interest. While the importance of having green policies coming from the Governments and from other international institutions is fundamental for a structured and effective revolution, we should not forget the impact that the individuals can have on the path towards a more environmentally sustainable lifestyle.
# 
# Almost everywhere I have been in Europe (either for living or just visiting) I noticed one thing: **every shop has the door open all day, every day, no matter the season**.
# 
# I thus wonder: **how much energy do we waste for that open door?**
# 
# To investigate this, I want to tell you a story.
# 
# 
# # A tale of three streets
# 
# In my 5 years in the Netherlands, I can safely say that most of the cities have that one street in the city center entirely devoted to shops. Imagine a 200-300 meters long line of shopping windows and open doors on both sides, giving you cold air in the summer and warm in the winter when you walk by them. Such streets are very common in Europe (and I imagine not only there). Three of these streets are the one that caught my attention the most:
# 
# * Herestraat in Groningen
# * Haarlemmerstraat in Leiden
# * Grote Houtstraat in Haarlem
# 
# None of these cities is what we could call a big city (even for Dutch standards) and, for how beautiful they are (and they are, you should visit them), they are not a major touristic destination (i.e. there are streets in Amsterdam that even more extreme but they are flooded by tourists almost 24/7 and this make their energy consumption less surprising).
# 
# Since the data are anonymized by grouping at least 10 connections together, it is difficult to isolate the consumption of a single shop but we can observe the annual consumption of a street. Due to the fact that every street has a different number of connections, in order to compare one street with another one, we will focus on energy consumption per connection.

# In[ ]:


import pandas as pd
import numpy as np

from os.path import join, isfile
from os import path, scandir, listdir

import warnings

pd.set_option('max_columns', 100)

#visualization
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


def list_all_files(location='/kaggle/input/', pattern=None, recursive=True):
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


def find_street(energy_list, city_name, street_name):
    res_street = pd.DataFrame({'city': [], 'street': [], 'year': [], #'provider': [], 
                           'num_connections': [], 'active_connections': [], 
                           'annual_consume': []})
    for key in energy_list.keys():
        yr = str(int(key.split('_')[1]) - 1)
        provider = key.split('_')[0]
        tmp = energy_list[key].copy()
        if (city_name in set(tmp.city)) and (street_name in set(tmp.street)):
            tmp = tmp[(tmp.city==city_name) & (tmp.street==street_name)].copy()
            tmp['active_connections'] = (tmp['num_connections'] * tmp['perc_of_active_connections'] / 100).astype(int)
            tmp = tmp.groupby(['city', 'street'], as_index=False).sum()
            tmp['year'] = yr
            #tmp['provider'] = provider
            tmp = tmp[['city', 'street', 'year', 'num_connections', 
                       'active_connections', 'annual_consume']].copy()
            res_street = pd.concat([res_street, tmp])
    res_street = res_street.groupby(['city', 'street', 'year'], as_index=False).sum()
    res_street['consume_per_connection'] = res_street['annual_consume'] / res_street['active_connections']
    res_street = res_street.sort_values('year')
    return res_street


def avg_consume_city(energy_list, city_name):
    res_city = pd.DataFrame({'city': [], 'year': [], #'provider': [], 
                           'num_connections': [], 'active_connections': [], 
                           'annual_consume': []})
    for key in energy_list.keys():
        yr = str(int(key.split('_')[1]) - 1)
        provider = key.split('_')[0]
        tmp = energy_list[key].copy()
        if city_name in set(tmp.city):
            tmp = tmp[tmp.city==city_name].copy()
            tmp['active_connections'] = (tmp['num_connections'] * tmp['perc_of_active_connections'] / 100).astype(int)
            tmp = tmp.groupby(['city'], as_index=False).sum()
            tmp['year'] = yr
            #tmp['provider'] = provider
            tmp = tmp[['city', 'year', 'num_connections', 
                       'active_connections', 'annual_consume']].copy()
            res_city = pd.concat([res_city, tmp])
    res_city = res_city.groupby(['city', 'year'], as_index=False).sum()
    res_city['consume_per_connection'] = res_city['annual_consume'] / res_city['active_connections']
    res_city = res_city.sort_values('year')
    return res_city


def elec_gas_comparison(el_list, ga_list, city_name, street_name):
    el_city = avg_consume_city(el_list, city_name)
    ga_city = avg_consume_city(ga_list, city_name)
    el_street = find_street(el_list, city_name, street_name)
    ga_street = find_street(ga_list, city_name, street_name)
    
    ren_city = {'active_connections': 'city_active_connections', 
            'num_connections': 'city_num_connections',
            'annual_consume': 'city_annual_consume', 
            'consume_per_connection': 'city_consume_per_connection'}
    
    el_city = el_city.rename(columns=ren_city)
    ga_city = ga_city.rename(columns=ren_city)
    el_city.columns = ['city', 'year'] + [col+'_e' for col in el_city.columns if col not in ['city', 'year']]
    ga_city.columns = ['city', 'year'] + [col+'_g' for col in ga_city.columns if col not in ['city', 'year']]
    el_street.columns = ['city', 'street', 'year'] +                         [col+'_e' for col in el_street.columns if col not in ['city', 'street', 'year']]
    ga_street.columns = ['city', 'street', 'year'] +                         [col+'_g' for col in ga_street.columns if col not in ['city', 'street', 'year']]
    
    el_street = pd.merge(el_street, el_city[['year', 'city_num_connections_e', 'city_active_connections_e', 
                                             'city_annual_consume_e', 'city_consume_per_connection_e']], on='year')
    el_street['excess_perc_e'] = el_street['consume_per_connection_e'] / el_street['city_consume_per_connection_e']*100
    ga_street = pd.merge(ga_street, ga_city[['year', 'city_num_connections_g', 'city_active_connections_g', 
                                             'city_annual_consume_g', 'city_consume_per_connection_g']], on='year')
    ga_street['excess_perc_g'] = ga_street['consume_per_connection_g'] / ga_street['city_consume_per_connection_g']*100
    result = pd.merge(el_street, ga_street, on=['city', 'street', 'year'])
    return result


def consume_city_distribution(energy_list, city_name):
    res_city = pd.DataFrame({'city': [], 'street': [], 'year': [],
                           'num_connections': [], 'active_connections': [], 
                           'annual_consume': []})
    for key in energy_list.keys():
        yr = str(int(key.split('_')[1]) - 1)
        tmp = energy_list[key].copy()
        if city_name in set(tmp.city):
            tmp = tmp[tmp.city==city_name].copy()
            tmp['active_connections'] = (tmp['num_connections'] * tmp['perc_of_active_connections'] / 100).astype(int)
            tmp = tmp.groupby(['city', 'street'], as_index=False).sum()
            tmp['year'] = yr
            tmp = tmp[['city', 'street', 'year', 'num_connections', 
                       'active_connections', 'annual_consume']].copy()
            res_city = pd.concat([res_city, tmp])
    res_city = res_city.groupby(['city', 'street', 'year'], as_index=False).sum()
    res_city = res_city.groupby(['city', 'street'], as_index=False).sum()
    res_city['consume_per_connection'] = res_city['annual_consume'] / res_city['active_connections']
    res_city = res_city.sort_values('consume_per_connection', ascending=False)
    return res_city


def consume_by_street(energy_list):
    res_en = pd.DataFrame({'city': [], 'street': [], 'year': [],
                           'num_connections': [], 'active_connections': [], 
                           'annual_consume': []})
    for key in energy_list.keys():
        yr = str(int(key.split('_')[1]) - 1)
        tmp = energy_list[key].copy()
        tmp['active_connections'] = (tmp['num_connections'] * tmp['perc_of_active_connections'] / 100).astype(int)
        tmp = tmp.groupby(['city', 'street'], as_index=False).sum()
        tmp['year'] = yr
        tmp = tmp[['city', 'street', 'year', 'num_connections', 
                   'active_connections', 'annual_consume']].copy()
        res_en = pd.concat([res_en, tmp])
    res_en = res_en.groupby(['city', 'street', 'year'], as_index=False).sum()
    res_en['consume_per_connection'] = res_en['annual_consume'] / res_en['active_connections']
    return res_en


# In[ ]:


elec_list = list_all_files('/kaggle/input/Electricity/')
gas_list = list_all_files('/kaggle/input/Gas/')
imp_elec = importer(elec_list)
imp_gas = importer(gas_list)

elec = consume_by_street(imp_elec)
gas = consume_by_street(imp_gas)


# In[ ]:


gro_her = elec_gas_comparison(imp_elec, imp_gas, 'GRONINGEN', 'Herestraat')

gro_her


# We start with Heerestraat in Groningen, a very cold city in the far north of the country. The electricity connections in this shopping street have consistently consumed **between 600% and 700% of the city average**, while the gas ones have consumed about **350-400%** of the city average. Even in terms of absolute values, the electricity consumption in Hererstraat (for 2018) was about 1% of the city consumption with only 0.1% of the connections.
# 
# Interesting to notice, and it is going to be a trend for all three streets, growth in the number of connections and a negative trend in the consumption per connection, possibly indicating a more energy-efficient consume.

# In[ ]:


fig, ax = plt.subplots(1,2, figsize=(18, 6))

gro_her[['year', 'consume_per_connection_e']].plot(kind='bar', x= 'year', ax=ax[0], label='Consume per connection')
gro_her[['year', 'city_consume_per_connection_e']].plot(color='r', x= 'year', ax=ax[0], label='City average', linestyle='--')
ax[0].legend(["City average", "Herestraat"])
ax[0].set_title('Electricity', fontsize=14)
ax[0].set_xlabel('')
ax[0].set_ylabel('kWh')

gro_her[['year', 'consume_per_connection_g']].plot(kind='bar', x= 'year', ax=ax[1], label='Consume per connection')
gro_her[['year', 'city_consume_per_connection_g']].plot(color='r', x= 'year', ax=ax[1], label='City average', linestyle='--')
ax[1].legend(["City average", "Herestraat"])
ax[1].set_title('Gas', fontsize=14)
ax[1].set_xlabel('')
ax[1].set_ylabel('m3')

fig.suptitle('Energy consumption per connection, Groningen', fontsize=18)
plt.show()


# This is how the consumption per connection is distributed in Groningen (averaged across the years)
# 
# *Note: The connections with 0 consumption are not taken into consideration when the mean is computed*

# In[ ]:


fig, ax = plt.subplots(1,2, figsize=(18, 6))

test = consume_city_distribution(imp_elec, 'GRONINGEN')
test.consume_per_connection.replace(np.inf, 0).hist(bins=100, ax=ax[0])
ax[0].axvline(x=test[test.street == 'Herestraat'].consume_per_connection.values[0], color='k', linestyle='--')
ax[0].axvline(x=test.consume_per_connection.replace(np.inf, 0).mean(), color='r', linestyle='--')
ax[0].annotate('Herestraat',
            xy=(test[test.street == 'Herestraat'].consume_per_connection.values[0], 100), 
            xycoords='data', xytext=(15, 25), textcoords='offset points',
            arrowprops=dict(facecolor='black', shrink=0.05))
ax[0].annotate('University and crematory',
            xy=(test.consume_per_connection.max(), 30), 
            xycoords='data', xytext=(-125, 25), textcoords='offset points',
            arrowprops=dict(facecolor='black', shrink=0.05))
ax[0].annotate('Mean',
            xy=(test.consume_per_connection[test.consume_per_connection>0].mean(), 300), xycoords='data',
            xytext=(55, 25), textcoords='offset points',
            arrowprops=dict(facecolor='black', shrink=0.05),
            horizontalalignment='right', verticalalignment='bottom')
ax[0].grid(False)
ax[0].set_title('Electricity', fontsize=14)
ax[0].set_xlabel('kWh')
ax[0].set_ylabel('Count')

test = consume_city_distribution(imp_gas, 'GRONINGEN')
test.consume_per_connection.replace(np.inf, 0).hist(bins=100, ax=ax[1])
ax[1].axvline(x=test[test.street == 'Herestraat'].consume_per_connection.values[0], color='k', linestyle='--')
ax[1].axvline(x=test.consume_per_connection.replace(np.inf, 0).mean(), color='r', linestyle='--')
ax[1].annotate('Herestraat',
            xy=(test[test.street == 'Herestraat'].consume_per_connection.values[0], 100), 
            xycoords='data', xytext=(15, 25), textcoords='offset points',
            arrowprops=dict(facecolor='black', shrink=0.05))
ax[1].annotate('Hospital',
            xy=(1048.89, 30), 
            xycoords='data', xytext=(-15, 25), textcoords='offset points',
            arrowprops=dict(facecolor='black', shrink=0.05))
ax[1].annotate('Mean',
            xy=(test.consume_per_connection[test.consume_per_connection>0].mean(), 200), xycoords='data',
            xytext=(45, 25), textcoords='offset points',
            arrowprops=dict(facecolor='black', shrink=0.05),
            horizontalalignment='right', verticalalignment='bottom')
ax[1].grid(False)
ax[1].set_title('Gas', fontsize=14)
ax[1].set_xlabel('m3')
ax[1].set_ylabel('Count')

fig.suptitle('Mean energy consumption per connection, Groningen', fontsize=18)
plt.show()


# We see how the shopping street consumption is significantly above the city average. Other outliers in this distribution are generally associated with the presence of several university buildings, the crematory, a hospital, and an industrial area.
# 
# Let's then move on to the second street of our story.

# In[ ]:


lei_haa = elec_gas_comparison(imp_elec, imp_gas, 'LEIDEN', 'Haarlemmerstraat')
lei_haa


# Haarlemmerstraat in Leiden, which is in South Holland, exhibits a very similar behavior: **about 400-450% of the average consumption of electricity** and 220% for the gas consumption. 

# In[ ]:


fig, ax = plt.subplots(1,2, figsize=(18, 6))

lei_haa[['year', 'consume_per_connection_e']].plot(kind='bar', x= 'year', ax=ax[0], label='Consume per connection')
lei_haa[['year', 'city_consume_per_connection_e']].plot(color='r', x= 'year', ax=ax[0], label='City average', linestyle='--')
ax[0].legend(["City average", "Haarlemmerstraat"])
ax[0].set_title('Electricity', fontsize=14)
ax[0].set_xlabel('')
ax[0].set_ylabel('kWh')

lei_haa[['year', 'consume_per_connection_g']].plot(kind='bar', x= 'year', ax=ax[1], label='Consume per connection')
lei_haa[['year', 'city_consume_per_connection_g']].plot(color='r', x= 'year', ax=ax[1], label='City average', linestyle='--')
ax[1].legend(["City average", "Haarlemmerstraat"])
ax[1].set_title('Gas', fontsize=14)
ax[1].set_xlabel('')
ax[1].set_ylabel('m3')

fig.suptitle('Energy consumption per connection, Leiden', fontsize=18)
plt.show()


# To put things into perspective, here is how the consumption per connection is distributed in Leiden
# 
# *Note: The connections with 0 consumption are not taken into consideration when the mean is computed*

# In[ ]:


fig, ax = plt.subplots(1,2, figsize=(18, 6))

test = consume_city_distribution(imp_elec, 'LEIDEN')
test.consume_per_connection.replace(np.inf, 0).hist(bins=100, ax=ax[0])
ax[0].axvline(x=test[test.street == 'Haarlemmerstraat'].consume_per_connection.values[0], color='darkviolet', linestyle='--')
ax[0].axvline(x=test.consume_per_connection.replace(np.inf, 0).mean(), color='r', linestyle='--')
ax[0].annotate('Haarlemmerstraat',
            xy=(test[test.street == 'Haarlemmerstraat'].consume_per_connection.values[0], 100), 
            xycoords='data', xytext=(15, 25), textcoords='offset points',
            arrowprops=dict(facecolor='black', shrink=0.05))
ax[0].annotate('Cryogenic laboratories',
            xy=(test.consume_per_connection.replace(np.inf, 0).max(), 10), 
            xycoords='data', xytext=(-105, 25), textcoords='offset points',
            arrowprops=dict(facecolor='black', shrink=0.05))
ax[0].annotate('Mean',
            xy=(test.replace(np.inf, 0)[test.consume_per_connection>0].consume_per_connection.mean(), 200), xycoords='data',
            xytext=(45, 25), textcoords='offset points',
            arrowprops=dict(facecolor='black', shrink=0.05),
            horizontalalignment='right', verticalalignment='bottom')
ax[0].grid(False)
ax[0].set_title('Electricity', fontsize=14)
ax[0].set_xlabel('kWh')
ax[0].set_ylabel('Count')

test = consume_city_distribution(imp_gas, 'LEIDEN')
test.consume_per_connection.replace(np.inf, 0).hist(bins=100, ax=ax[1])
ax[1].axvline(x=test[test.street == 'Haarlemmerstraat'].consume_per_connection.values[0], color='darkviolet', linestyle='--')
ax[1].axvline(x=test.consume_per_connection.replace(np.inf, 0).mean(), color='r', linestyle='--')
ax[1].annotate('Haarlemmerstraat',
            xy=(test[test.street == 'Haarlemmerstraat'].consume_per_connection.values[0], 60), 
            xycoords='data', xytext=(15, 25), textcoords='offset points',
            arrowprops=dict(facecolor='black', shrink=0.05))
ax[1].annotate('Sport center',
            xy=(test.consume_per_connection.replace(np.inf, 0).max(), 10), 
            xycoords='data', xytext=(-75, 25), textcoords='offset points',
            arrowprops=dict(facecolor='black', shrink=0.05))
ax[1].annotate('Mean',
            xy=(test.replace(np.inf, 0)[test.consume_per_connection>0].consume_per_connection.mean(), 100), xycoords='data',
            xytext=(35, 25), textcoords='offset points',
            arrowprops=dict(facecolor='black', shrink=0.05),
            horizontalalignment='right', verticalalignment='bottom')
ax[1].grid(False)
ax[1].set_title('Gas', fontsize=14)
ax[1].set_xlabel('m3')
ax[1].set_ylabel('Count')

fig.suptitle('Mean energy consumption per connection, Leiden', fontsize=18)
plt.show()


# Again, while the shopping street is not the only anomaly of the city, it does not have a good reason for it as much as, for example, a street with cryogenic laboratories in it.
# 
# The third and last protagonist of our story is in Haarlem, a small city next to Amsterdam.

# In[ ]:


haa_gro = elec_gas_comparison(imp_elec, imp_gas, 'HAARLEM', 'Grote Houtstraat')
haa_gro


# Grote Houtstraat in the close-by Haarlem has even more extreme excess, maching what we were able to observe in the much colder and darker Groningen.

# In[ ]:


fig, ax = plt.subplots(1,2, figsize=(18, 6))

haa_gro[['year', 'consume_per_connection_e']].plot(kind='bar', x= 'year', ax=ax[0], label='Consume per connection')
haa_gro[['year', 'city_consume_per_connection_e']].plot(color='r', x= 'year', ax=ax[0], label='City average', linestyle='--')
ax[0].legend(["City average", "Haarlemmerstraat"])
ax[0].set_title('Electricity', fontsize=14)
ax[0].set_xlabel('')
ax[0].set_ylabel('kWh')

haa_gro[['year', 'consume_per_connection_g']].plot(kind='bar', x= 'year', ax=ax[1], label='Consume per connection')
haa_gro[['year', 'city_consume_per_connection_g']].plot(color='r', x= 'year', ax=ax[1], label='City average', linestyle='--')
ax[1].legend(["City average", "Haarlemmerstraat"])
ax[1].set_title('Gas', fontsize=14)
ax[1].set_xlabel('')
ax[1].set_ylabel('m3')

fig.suptitle('Energy consumption per connection, Haarlem', fontsize=18)
plt.show()


# To observe the distribution of the consumption as above, we have to cut out some strong outliers coming from a very inconsistent count of the number of connections (thus blowing the consumption per connection out of proportion)
# 
# *Note: The connections with 0 consumption are not taken into consideration when the mean is computed*

# In[ ]:


fig, ax = plt.subplots(1,2, figsize=(18, 6))

test = consume_city_distribution(imp_elec, 'HAARLEM')
test[test.consume_per_connection.replace(np.inf, 0) < 5000].consume_per_connection.replace(np.inf, 0).hist(bins=100, ax=ax[0])
ax[0].axvline(x=test[test.street == 'Grote Houtstraat'].consume_per_connection.values[0], color='lime', linestyle='--')
ax[0].axvline(x=test.consume_per_connection.replace(np.inf, 0).mean(), color='r', linestyle='--')
ax[0].annotate('Grote Houtstraat',
            xy=(test[test.street == 'Grote Houtstraat'].consume_per_connection.values[0], 100), 
            xycoords='data', xytext=(15, 25), textcoords='offset points',
            arrowprops=dict(facecolor='black', shrink=0.05))
ax[0].annotate('Huge commercial area',
            xy=(2435, 10), 
            xycoords='data', xytext=(-125, 25), textcoords='offset points',
            arrowprops=dict(facecolor='black', shrink=0.05))
ax[0].annotate('Mean',
            xy=(test.replace(np.inf, 0)[test.consume_per_connection>0].consume_per_connection.mean(), 200), xycoords='data',
            xytext=(45, 25), textcoords='offset points',
            arrowprops=dict(facecolor='black', shrink=0.05),
            horizontalalignment='right', verticalalignment='bottom')
ax[0].grid(False)
ax[0].set_title('Electricity', fontsize=14)
ax[0].set_xlabel('kWh')
ax[0].set_ylabel('Count')

test = consume_city_distribution(imp_gas, 'HAARLEM')
test.consume_per_connection.replace(np.inf, 0).hist(bins=100, ax=ax[1])
ax[1].axvline(x=test[test.street == 'Grote Houtstraat'].consume_per_connection.values[0], color='lime', linestyle='--')
ax[1].axvline(x=test.consume_per_connection.replace(np.inf, 0).mean(), color='r', linestyle='--')
ax[1].annotate('Grote Houtstraat',
            xy=(test[test.street == 'Grote Houtstraat'].consume_per_connection.values[0], 100), 
            xycoords='data', xytext=(15, 25), textcoords='offset points',
            arrowprops=dict(facecolor='black', shrink=0.05))
ax[1].annotate('Schools and parks',
            xy=(test.consume_per_connection.replace(np.inf, 0).max(), 10), 
            xycoords='data', xytext=(-125, 25), textcoords='offset points',
            arrowprops=dict(facecolor='black', shrink=0.05))
ax[1].annotate('Mean',
            xy=(test.replace(np.inf, 0)[test.consume_per_connection>0].consume_per_connection.mean(), 200), xycoords='data',
            xytext=(45, 25), textcoords='offset points',
            arrowprops=dict(facecolor='black', shrink=0.05),
            horizontalalignment='right', verticalalignment='bottom')
ax[1].grid(False)
ax[1].set_title('Gas', fontsize=14)
ax[1].set_xlabel('m3')
ax[1].set_ylabel('Count')

fig.suptitle('Mean energy consumption per connection, Haarlem', fontsize=18)
plt.show()


# At last, this is an overview of the energy consumption of the entire country (we removed some outliers to make the graphs more readable)

# In[ ]:


fig, ax = plt.subplots(2,1, figsize=(18, 12))

elec = pd.concat([data[['city', 'street', 'num_connections', 
                        'annual_consume', 'perc_of_active_connections']] for data in imp_elec.values()])
elec['active_connections'] = (elec['num_connections'] * elec['perc_of_active_connections'] / 100).astype(int)
elec = elec.groupby(['city', 'street'], as_index=False).sum()
elec['consume_per_connection'] = elec['annual_consume'] / elec['active_connections']
elec[elec.consume_per_connection < 3000].consume_per_connection.hist(bins=100, ax=ax[0])
ax[0].axvline(x=elec[(elec.street == 'Grote Houtstraat') & 
                     (elec.city=='HAARLEM')].consume_per_connection.values[0], color='lime', linestyle='--')
ax[0].axvline(x=elec[(elec.street == 'Haarlemmerstraat') & 
                     (elec.city=='LEIDEN')].consume_per_connection.values[0], color='darkviolet', linestyle='--')
ax[0].axvline(x=elec[(elec.street == 'Herestraat') & 
                     (elec.city=='GRONINGEN')].consume_per_connection.values[0], color='k', linestyle='--')
ax[0].axvline(x=elec.consume_per_connection.replace(np.inf, 0).mean(), color='r', linestyle='--')
ax[0].annotate('Mean',
            xy=(elec.replace(np.inf, 0)[elec.consume_per_connection>0].consume_per_connection.mean(), 20000), xycoords='data',
            xytext=(45, 25), textcoords='offset points',
            arrowprops=dict(facecolor='black', shrink=0.05),
            horizontalalignment='right', verticalalignment='bottom')
ax[0].annotate('Herestraat',
            xy=(elec[(elec.street == 'Herestraat') & 
                     (elec.city=='GRONINGEN')].consume_per_connection.values[0], 8000), xycoords='data',
            xytext=(70, 25), textcoords='offset points',
            arrowprops=dict(facecolor='black', shrink=0.05),
            horizontalalignment='right', verticalalignment='bottom')
ax[0].annotate('Haarlemmerstraat',
            xy=(elec[(elec.street == 'Haarlemmerstraat') & 
                     (elec.city=='LEIDEN')].consume_per_connection.values[0], 15000), xycoords='data',
            xytext=(-10, 25), textcoords='offset points',
            arrowprops=dict(facecolor='black', shrink=0.05),
            horizontalalignment='right', verticalalignment='bottom')
ax[0].annotate('Grote Houtstraat',
            xy=(elec[(elec.street == 'Grote Houtstraat') & 
                     (elec.city=='HAARLEM')].consume_per_connection.values[0], 12000), xycoords='data',
            xytext=(120, 25), textcoords='offset points',
            arrowprops=dict(facecolor='black', shrink=0.05),
            horizontalalignment='right', verticalalignment='bottom')
ax[0].grid(False)
ax[0].set_title('Electricity', fontsize=16)
ax[0].set_xlabel('kWh')
ax[0].set_ylabel('Count')

gas = pd.concat([data[['city', 'street', 'num_connections', 
                       'annual_consume', 'perc_of_active_connections']] for data in imp_gas.values()])
gas['active_connections'] = (gas['num_connections'] * gas['perc_of_active_connections'] / 100).astype(int)
gas = gas.groupby(['city', 'street'], as_index=False).sum()
gas['consume_per_connection'] = gas['annual_consume'] / gas['active_connections']
gas[gas.consume_per_connection < 1000].consume_per_connection.hist(bins=100, ax=ax[1])
ax[1].axvline(x=gas[(gas.street == 'Grote Houtstraat') & 
                     (gas.city=='HAARLEM')].consume_per_connection.values[0], color='lime', linestyle='--')
ax[1].axvline(x=gas[(gas.street == 'Haarlemmerstraat') & 
                     (gas.city=='LEIDEN')].consume_per_connection.values[0], color='darkviolet', linestyle='--')
ax[1].axvline(x=gas[(gas.street == 'Herestraat') & 
                     (gas.city=='GRONINGEN')].consume_per_connection.values[0], color='k', linestyle='--')
ax[1].axvline(x=gas.consume_per_connection.replace(np.inf, 0).mean(), color='r', linestyle='--')
ax[1].annotate('Mean',
            xy=(gas.replace(np.inf, 0)[gas.consume_per_connection>0].consume_per_connection.mean(), 12000), xycoords='data',
            xytext=(-45, 25), textcoords='offset points',
            arrowprops=dict(facecolor='black', shrink=0.05),
            horizontalalignment='right', verticalalignment='bottom')
ax[1].annotate('Herestraat',
            xy=(gas[(gas.street == 'Herestraat') & 
                     (gas.city=='GRONINGEN')].consume_per_connection.values[0], 2000), xycoords='data',
            xytext=(75, 25), textcoords='offset points',
            arrowprops=dict(facecolor='black', shrink=0.05),
            horizontalalignment='right', verticalalignment='bottom')
ax[1].annotate('Haarlemmerstraat',
            xy=(gas[(gas.street == 'Haarlemmerstraat') & 
                     (gas.city=='LEIDEN')].consume_per_connection.values[0], 8000), xycoords='data',
            xytext=(-65, 25), textcoords='offset points',
            arrowprops=dict(facecolor='black', shrink=0.05),
            horizontalalignment='right', verticalalignment='bottom')
ax[1].annotate('Grote Houtstraat',
            xy=(gas[(gas.street == 'Grote Houtstraat') & 
                     (gas.city=='HAARLEM')].consume_per_connection.values[0], 6000), xycoords='data',
            xytext=(55, 25), textcoords='offset points',
            arrowprops=dict(facecolor='black', shrink=0.05),
            horizontalalignment='right', verticalalignment='bottom')
ax[1].grid(False)
ax[1].set_title('Gas', fontsize=16)
ax[1].set_xlabel('m3')
ax[1].set_ylabel('Count')

fig.suptitle('Mean energy consumption per connection', fontsize=20)
plt.show()


# At this point, one may wonder how the previous graphs would look like if we stop studying the *consumption per connection* and focus on the total consumption. The reason why we are taking into account the number of active connections is that these 3 streets are fairly large streets and it would be unfair to compare them with the many tiny streets of the Dutch cities.
# 
# However, at this stage, it doesn't hurt to have a look at how unfair the comparison would be and see how the **total energy consumption of the Netherlands in the past 10 years** is distributed and where can we find the 3 streets.
# 
# *Note: in the next graphs the mean is calculated without consumptions below 1000 kWh or below 100 m3 of gas*

# In[ ]:


fig, ax = plt.subplots(2,1, figsize=(18, 12))

elec = pd.concat([data[['city', 'street', 'num_connections', 
                        'annual_consume', 'perc_of_active_connections']] for data in imp_elec.values()])
elec = elec.groupby(['city', 'street'], as_index=False).sum()
elec.annual_consume.hist(bins=1000, ax=ax[0])
ax[0].axvline(x=elec[(elec.street == 'Grote Houtstraat') & 
                     (elec.city=='HAARLEM')].annual_consume.values[0], color='lime', linestyle='--')
ax[0].axvline(x=elec[(elec.street == 'Haarlemmerstraat') & 
                     (elec.city=='LEIDEN')].annual_consume.values[0], color='darkviolet', linestyle='--')
ax[0].axvline(x=elec[(elec.street == 'Herestraat') & 
                     (elec.city=='GRONINGEN')].annual_consume.values[0], color='k', linestyle='--')
ax[0].axvline(x=elec[elec.annual_consume > 1000].annual_consume.mean(), color='r', linestyle='--')
ax[0].annotate('Mean',
            xy=(elec[elec.annual_consume > 1000].annual_consume.mean(), 15000), xycoords='data',
            xytext=(45, 25), textcoords='offset points',
            arrowprops=dict(facecolor='black', shrink=0.05),
            horizontalalignment='right', verticalalignment='bottom')
ax[0].annotate('Herestraat',
            xy=(elec[(elec.street == 'Herestraat') & 
                     (elec.city=='GRONINGEN')].annual_consume.values[0], 14000), xycoords='data',
            xytext=(-30, 25), textcoords='offset points',
            arrowprops=dict(facecolor='black', shrink=0.05),
            horizontalalignment='right', verticalalignment='bottom')
ax[0].annotate('Haarlemmerstraat',
            xy=(elec[(elec.street == 'Haarlemmerstraat') & 
                     (elec.city=='LEIDEN')].annual_consume.values[0], 12500), xycoords='data',
            xytext=(-10, 25), textcoords='offset points',
            arrowprops=dict(facecolor='black', shrink=0.05),
            horizontalalignment='right', verticalalignment='bottom')
ax[0].annotate('Grote Houtstraat',
            xy=(elec[(elec.street == 'Grote Houtstraat') & 
                     (elec.city=='HAARLEM')].annual_consume.values[0], 10000), xycoords='data',
            xytext=(120, 25), textcoords='offset points',
            arrowprops=dict(facecolor='black', shrink=0.05),
            horizontalalignment='right', verticalalignment='bottom')
ax[0].annotate('Amsterdam\'s shopping street',
            xy=(elec[(elec.street == 'Keizersgracht') & 
                     (elec.city=='AMSTERDAM')].annual_consume.values[0], 1000), xycoords='data',
            xytext=(30, 25), textcoords='offset points',
            arrowprops=dict(facecolor='black', shrink=0.05),
            horizontalalignment='right', verticalalignment='bottom')
ax[0].grid(False)
ax[0].set_title('Electricity', fontsize=16)
ax[0].set_xlabel('kWh')
ax[0].set_ylabel('Count')


gas = pd.concat([data[['city', 'street', 'num_connections', 
                       'annual_consume', 'perc_of_active_connections']] for data in imp_gas.values()])
gas = gas.groupby(['city', 'street'], as_index=False).sum()
gas.annual_consume.hist(bins=1000, ax=ax[1])
ax[1].axvline(x=gas[(gas.street == 'Grote Houtstraat') & 
                     (gas.city=='HAARLEM')].annual_consume.values[0], color='lime', linestyle='--')
ax[1].axvline(x=gas[(gas.street == 'Haarlemmerstraat') & 
                     (gas.city=='LEIDEN')].annual_consume.values[0], color='darkviolet', linestyle='--')
ax[1].axvline(x=gas[(gas.street == 'Herestraat') & 
                     (gas.city=='GRONINGEN')].annual_consume.values[0], color='k', linestyle='--')
ax[1].axvline(x=gas[gas.annual_consume > 100].annual_consume.mean(), color='r', linestyle='--')
ax[1].annotate('Mean',
            xy=(gas[gas.annual_consume > 100].annual_consume.mean(), 12000), xycoords='data',
            xytext=(35, 25), textcoords='offset points',
            arrowprops=dict(facecolor='black', shrink=0.05),
            horizontalalignment='right', verticalalignment='bottom')
ax[1].annotate('Herestraat',
            xy=(gas[(gas.street == 'Herestraat') & 
                     (gas.city=='GRONINGEN')].annual_consume.values[0], 10000), xycoords='data',
            xytext=(-15, 25), textcoords='offset points',
            arrowprops=dict(facecolor='black', shrink=0.05),
            horizontalalignment='right', verticalalignment='bottom')
ax[1].annotate('Haarlemmerstraat',
            xy=(gas[(gas.street == 'Haarlemmerstraat') & 
                     (gas.city=='LEIDEN')].annual_consume.values[0], 8000), xycoords='data',
            xytext=(105, 25), textcoords='offset points',
            arrowprops=dict(facecolor='black', shrink=0.05),
            horizontalalignment='right', verticalalignment='bottom')
ax[1].annotate('Grote Houtstraat',
            xy=(gas[(gas.street == 'Grote Houtstraat') & 
                     (gas.city=='HAARLEM')].annual_consume.values[0], 5000), xycoords='data',
            xytext=(105, 25), textcoords='offset points',
            arrowprops=dict(facecolor='black', shrink=0.05),
            horizontalalignment='right', verticalalignment='bottom')
ax[1].annotate('Amsterdam\'s shopping street',
            xy=(gas[(gas.street == 'Keizersgracht') & 
                     (gas.city=='AMSTERDAM')].annual_consume.values[0], 1000), xycoords='data',
            xytext=(30, 25), textcoords='offset points',
            arrowprops=dict(facecolor='black', shrink=0.05),
            horizontalalignment='right', verticalalignment='bottom')
ax[1].grid(False)
ax[1].set_title('Gas', fontsize=16)
ax[1].set_xlabel('m3')
ax[1].set_ylabel('Count')

fig.suptitle('Total energy consumption in the Netherlands (2008-2018)', fontsize=20)
plt.show()


# This graph shows also why I didn't consider the shopping street of Amsterdam (whose shops keep their door open too): **it is the street with the highest energy consumption of the country**. Please keep in mind that in these data are present industrial facilities too, including, for example, the production facilities of Heineken. 
# 
# Even by breaking down the consumption year by year we can observe how atypical these 3 streets are. (Thanks @bberghuis and [his kernel](https://www.kaggle.com/bberghuis/dutch-electricity-eda-fs-clustering-maps) for the idea of using a gif)
# ![](https://i.imgur.com/qkn5RSd.gif)

# # Is it about the shopping or is it about the door?
# 
# One doubt easily arises here: the city average is based mostly on houses and a street with a large number of shops is naturally consuming more energy. It is fair to admit that a shop can consume more than a house (larger spaces to warm up or cool down, more lights), does this justify an average consumption 500% higher?
# 
# Let's take Leiden. Not far from Haarlemmerstraat there are 2 streets with 2 fairly big (for Dutch standards) supermarkets, a few other shops, and a few houses. I do expect to see a high energy consumption because the supermarkets come with a 2 story parking lot, they have refrigerators (something not much present in Harlemmerstraat since the shops are mostly of clothes and mobile phones), they are open more than anything else in the city (12/7 at least). There is one big difference they (as well as the other shops in their streets) have multiple doors that are not constantly open.
# 
# Here is what their energy consumption looks like

# In[ ]:


elec_gas_comparison(imp_elec, imp_gas, "LEIDEN", 'Korevaarstraat')


# In[ ]:


elec_gas_comparison(imp_elec, imp_gas, "LEIDEN", 'Hooigracht')


# The excess of energy consumption for these 2 streets is, as expected, well above the city average but still below 200% of the city average. In other words, **they consume less than half of Haarlemmerstraat just by having their doors closed** and despite having machines that consume way more energy.

# In[ ]:


fig, ax = plt.subplots(1,2, figsize=(18, 6))

test = consume_city_distribution(imp_elec, 'LEIDEN')
test[test.consume_per_connection.replace(np.inf, 0) < 2000].consume_per_connection.replace(np.inf, 0).hist(bins=100, ax=ax[0])
ax[0].axvline(x=test[test.street == 'Haarlemmerstraat'].consume_per_connection.values[0], color='darkviolet', linestyle='--')
ax[0].axvline(x=test[test.street == 'Hooigracht'].consume_per_connection.values[0], color='g', linestyle='--')
ax[0].axvline(x=test[test.street == 'Korevaarstraat'].consume_per_connection.values[0], color='g', linestyle='--')
ax[0].axvline(x=test.consume_per_connection.replace(np.inf, 0).mean(), color='r', linestyle='--')
ax[0].annotate('Haarlemmerstraat',
            xy=(test[test.street == 'Haarlemmerstraat'].consume_per_connection.values[0], 60), 
            xycoords='data', xytext=(15, 25), textcoords='offset points',
            arrowprops=dict(facecolor='black', shrink=0.05))
ax[0].annotate('Hooigracht',
            xy=(test[test.street == 'Hooigracht'].consume_per_connection.values[0], 100), 
            xycoords='data', xytext=(15, 25), textcoords='offset points',
            arrowprops=dict(facecolor='black', shrink=0.05))
ax[0].annotate('Mean',
            xy=(test.replace(np.inf, 0)[test.consume_per_connection>0].consume_per_connection.mean(), 120), xycoords='data',
            xytext=(-15, 25), textcoords='offset points',
            arrowprops=dict(facecolor='black', shrink=0.05),
            horizontalalignment='right', verticalalignment='bottom')
ax[0].grid(False)
ax[0].set_title('Electricity', fontsize=14)
ax[0].set_xlabel('kWh')
ax[0].set_ylabel('Count')

test = consume_city_distribution(imp_gas, 'LEIDEN')
test[test.consume_per_connection.replace(np.inf, 0) < 400].consume_per_connection.replace(np.inf, 0).hist(bins=100, ax=ax[1])
ax[1].axvline(x=test[test.street == 'Haarlemmerstraat'].consume_per_connection.values[0], color='darkviolet', linestyle='--')
ax[1].axvline(x=test[test.street == 'Hooigracht'].consume_per_connection.values[0], color='g', linestyle='--')
ax[1].axvline(x=test[test.street == 'Korevaarstraat'].consume_per_connection.values[0], color='g', linestyle='--')
ax[1].axvline(x=test.consume_per_connection.replace(np.inf, 0).mean(), color='r', linestyle='--')
ax[1].annotate('Haarlemmerstraat',
            xy=(test[test.street == 'Haarlemmerstraat'].consume_per_connection.values[0], 50), 
            xycoords='data', xytext=(15, 25), textcoords='offset points',
            arrowprops=dict(facecolor='black', shrink=0.05))
ax[1].annotate('Hooigracht',
            xy=(test[test.street == 'Hooigracht'].consume_per_connection.values[0], 80), 
            xycoords='data', xytext=(-65, 25), textcoords='offset points',
            arrowprops=dict(facecolor='black', shrink=0.05))
ax[1].annotate('Mean',
            xy=(test.replace(np.inf, 0)[test.consume_per_connection>0].consume_per_connection.mean(), 100), xycoords='data',
            xytext=(-15, 25), textcoords='offset points',
            arrowprops=dict(facecolor='black', shrink=0.05),
            horizontalalignment='right', verticalalignment='bottom')
ax[1].grid(False)
ax[1].set_title('Gas', fontsize=14)
ax[1].set_xlabel('m3')
ax[1].set_ylabel('Count')

fig.suptitle('Comparison with a "closed-door" commercial area', fontsize=18)
plt.show()


# It is interesting to notice how Haarlemmerstraat burns so much more electricity than the other two streets (with Korevaarstraat being almost perfectly aligned with the city mean) and how, looking at gas consumption, Hooigracht and Haarlemmerstraat are essentially the same. One reason could be that the air conditioning of these commercial places is fueled by electricity and a constantly open door can make a lot of difference while gas consumption (the main source of heat for houses, especially for old houses as the one in that area) is not influenced by the shops' open doors. In the absence of further information, everything else would be pure speculation.

# # What else consume as much energy?
# 
# At last, we can look for similar streets either by energy consumption or number of connections. First, let's have an overview of the yearly consumption of our three streets collectively

# In[ ]:


elec = consume_by_street(imp_elec)
gas = consume_by_street(imp_gas)


# In[ ]:


test = elec[((elec.city == 'LEIDEN') & (elec.street == 'Haarlemmerstraat')) | 
     ((elec.city == 'HAARLEM') & (elec.street == 'Grote Houtstraat')) |  
     ((elec.city == 'GRONINGEN') & (elec.street == 'Herestraat'))].copy()
print('----- Electricity ------')
print('Active connections: ')
print(f'\t Min: {test.active_connections.min()}')
print(f'\t Max: {test.active_connections.max()}')
print('Annual consume: ')
print(f'\t Min: {test.annual_consume.min()}')
print(f'\t Max: {test.annual_consume.max()}')
print('Consume per connections: ')
print(f'\t Min: {test.consume_per_connection.min()}')
print(f'\t Max: {test.consume_per_connection.max()}')
test = gas[((gas.city == 'LEIDEN') & (gas.street == 'Haarlemmerstraat')) | 
     ((gas.city == 'HAARLEM') & (gas.street == 'Grote Houtstraat')) |  
     ((gas.city == 'GRONINGEN') & (gas.street == 'Herestraat'))].copy()
print('----- Gas ------')
print('Active connections: ')
print(f'\t Min: {test.active_connections.min()}')
print(f'\t Max: {test.active_connections.max()}')
print('Annual consume: ')
print(f'\t Min: {test.annual_consume.min()}')
print(f'\t Max: {test.annual_consume.max()}')
print('Consume per connections: ')
print(f'\t Min: {test.consume_per_connection.min()}')
print(f'\t Max: {test.consume_per_connection.max()}')


# We now can look for streets whose yearly consumption was similar to these 3 streets.
# 
# Interestingly, the only street **consuming more energy with fewer connections** than our 3 streets is a huge commercial area in the north of Eindhoven

# In[ ]:


elec[(elec.annual_consume > 337248) & (elec.active_connections < 283) ] 


# If we instead look for streets that (in at least one year) exceeded the lowest yearly consumption we find 112 streets that consumed more than the lowest consumption of the main characters of our tale (with fewer connections than the maximum number of connections in our streets). I am still looking for one of them that is not a shopping street.

# In[ ]:


test = elec[(elec.annual_consume > 121425) & (elec.active_connections < 283) ][['city', 'street']].drop_duplicates()
print(test.shape[0])
test.sample(10)


# A similar search for gas consumption identifies only 4 streets that managed to consume more gas with fewer connections and 241 that exceeded the lowest consumption across the years of our three streets (with fewer connections than the maximum number)

# In[ ]:


gas[(gas.annual_consume > 59178) & (gas.active_connections < 278) ] 


# In[ ]:


test = gas[(gas.annual_consume > 32346) & (gas.active_connections < 278) ][['city', 'street']].drop_duplicates()
print(test.shape[0])
test.sample(10)


# In this case, we find a few streets that do not look like they are overcrowded with shops. As we have mentioned before, the effect of an open door is mainly to make more difficult to maintain the temperature set with a thermostat and, since the shops do not warm up or cool down their spaces by using gas, this can be more clearly observed by analyzing their electricity consumption. One may thus wonder if the relationship between the electricity and the gas consumption is somewhat different when we take into account the 112 streets found above. To do so, we need to merge the electricity and gas consumption datasets and, unfortunately, we lose some street in the process.

# In[ ]:


el_con = elec.copy()
ga_con = gas.copy()

el_con.columns = ['city', 'street', 'year'] +                 ['el_'+col for col in el_con.columns if col not in ['city', 'street', 'year']]
ga_con.columns = ['city', 'street', 'year'] +                 ['ga_'+col for col in ga_con.columns if col not in ['city', 'street', 'year']]

tot_con = pd.merge(el_con, ga_con, on=['city', 'street', 'year'])

test = elec[(elec.annual_consume > 121425) & (elec.active_connections <= 283) ][['city', 'street']].drop_duplicates()

test['Is shopping street?'] = 'Yes'

tot_con = pd.merge(tot_con, test, on=['city', 'street'], how='left').fillna('No')

tot_con['tot_connections'] = tot_con['el_active_connections'] + tot_con['ga_active_connections']

plt.figure(figsize=(12,8))
ax = sns.scatterplot(x='el_annual_consume', y='ga_annual_consume', size='tot_connections',
                     hue='Is shopping street?', data=tot_con)
ax.set_title('Electricity vs Gas consumtion', fontsize=18)
ax.set_xlabel('Electricity consumption', fontsize=14)
ax.set_ylabel('Gas consumption', fontsize=14)
handles, labels = ax.get_legend_handles_labels()
plt.legend(handles[0:3], labels[0:3])
plt.show()


# *Note: the streets flagged with 'Is Shopping Street?' are only the ones we found above, thus consuming a comparable amount of electricity with a comparable amount of connections. Bigger or smaller streets are not included. The size of the bubbles indicates the total connections (electricity and gas).*
# 
# We can indeed see that these streets stand out by consuming more electricity than it is expected from their gas consumption. For those of you curious about the group of points on the bottom right, they represent Laan van Vollenhove in Zeist (next to Utrecht) from 2009 to 2018. This street didn't appear in our selection because, being a very long street, it has more than 2000 active connections (and we are highlighting those with less than 283) and a great variety of buildings in it, from schools to houses and shops.
# 
# We can also see how this graph changes year by year 
# ![](https://i.imgur.com/mGTGnwl.gif)

# We thus see that the 112 *comparable streets* listed above are in reality much less and that their number appear to decrease with the years. On the other hand, both Haarlemmerstraat and Grote Houdstraat appear to keep decreasing their electricity consumption (keeping the gas consumption fairly stationary), while Herestraat is very consistent every year.
# 
# The two points with high electricity consumption appearing out of nowhere since 2017, are the huge commercial area outside of Eindhoven mentioned before (and this dataset does not have data about that area prior 2017, unfortunately)

# # Conclusions and limits
# 
# Every analysis can be only as good as the data upon which it is founded and these data are far from perfect. For example, it is not extremely clear what it is meant with *connection* since the mean energy consumption *per connection* is well below the one of a normal house. However, since the data were used mainly to compare different streets, we are confident that this ambiguity would not influence the final conclusions of this analysis (to confirm that, an analysis of the total consumption was included).
# 
# The conclusion is that, in proportion with the number of connections in that area, streets with several shops that keep their doors open are consuming way too much energy, considering they are not industrial areas. The reader should, however, be aware that, despite the attempts of putting this high consumption in context, the perfect comparison would be between one of these 3 streets and an analogous street with several shops but without the open doors. Unfortunately, I am not aware of the existence of such street (which is not a scientific statement, but it is the truth) and I could only compare them with places with buildings that are expected to be energy-demanding (supermarkets and parking lots). 
# 
# While one can be glad to observe the negative trend in the electricity consumption, an encouraging result for a more energy-efficient future, we must remember that **having a more efficient refrigerator is not enough if we keep its door constantly open**.
# 
# At last, the goal of this kernel is not to point fingers against the shops in these 3 streets. It is well known that every shop around the world keep the door open because customers are more likely to come in if the door is open. What this kernel wants to point out is the cost, for the shop-owners but also for the environment, that an open door can have. 
# 
# At this point, I can see 3 solutions (not mutually exclusive):
# 
# * the shop-owners close their doors (even better if they invest in a double door entrance), save a considerable amount of money in their electric bill, and send a good message. This holds for every individual that, for example, open a window with the air conditioning on. In other words, let's cut the most obvious waste first.
# * the municipalities make a law to have those doors closed. Fairly invasive but this would guarantee that the commercial benefit of keeping a door open is going to disappear for everyone equally.
# * we, as customers, start to be mindful of that, learn to not be discouraged by a closed door (no encouragement to breaking and entering here), and close the doors behind us.
# 
# Small things that have big impacts in the long run.
# 
# ### Thank you for reading this far, please leave some feedback and/or share this notebook if you feel like it
