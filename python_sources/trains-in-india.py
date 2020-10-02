#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import folium
from folium.map import Icon

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# ## Datasets exploration and cleaning

# In[ ]:


df_trains = pd.read_csv('../input/indiantrains/All_Indian_Trains.csv')
df_cities = pd.read_csv("../input/top-500-indian-cities/cities_r2.csv")
df_wcities = pd.read_csv("../input/world-cities-database/worldcitiespop.csv")


# About cities, as the World Cities dataset is very big (more than 2 millions entries), we will focus on the cities from India and some neighbour countries, which should be enough.

# In[ ]:


sub_wcities = pd.concat([df_wcities[df_wcities['Country'] == 'in'], df_wcities[df_wcities['Country'] == 'bd'] ,df_wcities[df_wcities['Country'] == 'bt'],df_wcities[df_wcities['Country'] == 'np'],df_wcities[df_wcities['Country'] == 'pk']])


# In[ ]:


df_trains.head(10)


# In[ ]:


len(df_trains)


# In[ ]:


sub_wcities.head(10)


# In[ ]:


len(sub_wcities[sub_wcities['Country'] == 'in'])


# In[ ]:


df_cities.head(10)


# The World Cities database contains much more entries than the Top 500 Indian Cities one. For India alone, it provides data on almost 40 000 cities, much than the top 500. Nonetheless, the Top 500 Indian Cities dataset provides much more information on each entry, so we will try to see later what can be found out of it.

# In[ ]:


''' Here we will correct some mispellings that had been discovered during the analysis, but deleted from the notebook,
    for the sake of readibility. '''

def corr(name):
    if name == 'Velankanni' or name == 'Vellankanni':
        return 'Velanganni'
    elif name == 'Raxual Junction':
        return 'Raxaul Junction'
    elif name == 'Alipur Duar Junction':
        return 'Alipurduar Junction'
    elif name == 'Chamarajanagar':
        return 'Chamarajnagar'
    elif name == 'Dehradun':
        return 'Dehra Dun'
    elif name == 'Eranakulam Junction':
        return 'Ernakulam Junction'
    elif name == 'Machelipatnam':
        return 'Machilipatnam'
    elif name == 'Metupalaiyam':
        return 'Mettupalaiyam'
    elif name == 'Mathura Junction':
        return 'Vrindavan'               # This one is because the World Cities dataset provided Mathura in Andaman Islands first.
    elif name == 'Murkeongselek':
        return 'Murkong Selek'
    elif name == 'Nagarsol':
        return 'Nagarsul'
    elif name == 'New Delhi':
        return 'Newdelhi'
    elif name == 'Tiruchchirapali':
        return 'Tiruchchirappalli'
    elif name == 'Villuparam Junction':
        return 'Villupuram Junction'
    elif name == 'Vishakapatnam':
        return 'Vishakhapatnam'
    else:
        return name


# ## Map of the stations 

# In[ ]:


ds = df_trains['Starts'].apply(corr)
de = df_trains['Ends'].apply(corr)

df_trains_aug = pd.DataFrame()      # Will be used to draw the map
df_trains_aug['Train no.'] = df_trains['Train no.']
df_trains_aug['Train name'] = df_trains['Train name']
df_trains_aug['Starts'] = ds
df_trains_aug['Ends'] = de

df_trains_aug.head(10)


# Let's build a DataFrame focused on the train stations.

# In[ ]:


df_stations = pd.DataFrame()       # Will group info about all the stations
sta_name = []
sta_city = []
sta_lat = []
sta_long = []
sta_starts = []
sta_ends = []
sta_trains = []
sta_state = []
sta_country = []
unfound = []
stations_set = set(df_trains_aug['Starts']).union(set(df_trains_aug['Ends']))


# In[ ]:


#sub_wcities[sub_wcities['City'] == 'adirampatnam']['Latitude'].to_numpy()[0]


# In[ ]:


for s in stations_set:
    found = False
    for w in sub_wcities['City']:
        if not found:
            if s.lower() in str(w).split(' ') or str(w) in s.lower().split(' ') or str(w) == s.lower():
                sta_name.append(s)
                sta_city.append(str(w))
                sta_lat.append(sub_wcities[sub_wcities['City'] == str(w)]['Latitude'].to_numpy()[0])
                sta_long.append(sub_wcities[sub_wcities['City'] == str(w)]['Longitude'].to_numpy()[0])
                sta_starts.append(len(df_trains_aug[df_trains_aug['Starts'] == s]))
                sta_ends.append(len(df_trains_aug[df_trains_aug['Ends'] == s]))
                sta_trains.append(len(df_trains_aug[df_trains_aug['Starts'] == s]) + len(df_trains_aug[df_trains_aug['Ends'] == s]))
                sta_state.append(sub_wcities[sub_wcities['City'] == str(w)]['Region'].to_numpy()[0])
                sta_country.append(sub_wcities[sub_wcities['City'] == str(w)]['Country'].to_numpy()[0])
                found = True
    if not found:
        unfound.append(s)


# In[ ]:


sta_starts[:10]
sta_ends[:10]


# In[ ]:


len(unfound)


# In[ ]:


unfound


# There are 30 unfound stations. This is something we can handle by hand, by looking for information on the internet, althought it will not be very fun... The idea is to attribute them the closest city in the cities dataset.

# In[ ]:


manual_handle = {'Chirmiri':'korea', 'Manduadih':'varanasi', 'Sadulpur Junction':'churu', 'Manuguru':'kothagudem', 'Mayiladuturai J':'mayuram', 'Sengottai':'tenkasi',
                'Kochuveli':'thiruvananthapuram', 'Patliputra':'danapur', 'Chamarajnagar':'mysore', 'C Shahumharaj T':'kolhapur', 'Lokmanyatilak T':'kurla', 'Gevra Road':'korba',
                'Singrauli':'churki', 'Shmata V D Ktra':'dudura', 'New Alipurdaur':'alipur duar', 'Alipurduar Junction':'alipur duar', 'Habibganj':'bhopal', 'Banaswadi':'bangalore', 'Jhajha':'jamui',
                'Sawantwadi Road':'talavada', 'H Nizamuddin':'delhi', 'Naharlagun':'itanagar', 'Nilaje':'mumbai', 'Khairthal':'alwar', 'Udhna Junction':'surat', 'Kirandul':'dantewara',
                'Kacheguda':'hyderabad', 'Belampalli':'mancherial', 'Radhikapur':'raiganj', 'Borivali':'mumbai', 'Dekargaon':'tezpur', 'Newdelhi': 'new delhi'}

for s in manual_handle.keys():
    sta_name.append(s)
    sta_city.append(manual_handle[s])
    sta_lat.append(sub_wcities[sub_wcities['City'] == manual_handle[s]]['Latitude'].to_numpy()[0])
    sta_long.append(sub_wcities[sub_wcities['City'] == manual_handle[s]]['Longitude'].to_numpy()[0])
    sta_starts.append(len(df_trains_aug[df_trains_aug['Starts'] == s]))
    sta_ends.append(len(df_trains_aug[df_trains_aug['Ends'] == s]))
    sta_trains.append(len(df_trains_aug[df_trains_aug['Starts'] == s]) + len(df_trains_aug[df_trains_aug['Ends'] == s]))
    sta_state.append(sub_wcities[sub_wcities['City'] == manual_handle[s]]['Region'].to_numpy()[0])
    sta_country.append(sub_wcities[sub_wcities['City'] == manual_handle[s]]['Country'].to_numpy()[0])


# In[ ]:


df_stations['name'] = sta_name
df_stations['city'] = sta_city
df_stations['latitude'] = sta_lat
df_stations['longitude'] = sta_long
df_stations['nb_starts'] = sta_starts
df_stations['nb_ends'] = sta_ends
df_stations['nb_trains'] = sta_trains
df_stations['state'] = sta_state
df_stations['country'] = sta_country


# In[ ]:


df_stations.describe()


# In[ ]:


stations_map = folium.Map(location=[22.05, 78.94], zoom_start=4.5)
for idx, row in df_stations.iterrows():
    c = 'mediumpurple'
    if row['nb_ends'] == 0:
        c = 'royalblue'
    if row['nb_starts'] == 0:
        c = 'deeppink'
    folium.Circle(location=[row['latitude'], row['longitude']], radius=1 + 400 * row['nb_trains'], color = c, fill = True, popup = row['name']).add_to(stations_map)
stations_map


# This map shows all the stations of the dataset. Circles radius depend on the number of train of the station. Stations in blue are start stations, stations in red are end stations, and purple stations can be start or end. We can see that blue and red stations are only small stations. Stations in foreing countries are small, because we show only their trains for India.  We can guess that are many internal trains as well. We also see that biggest stations are often to other big stations, in the big cities, like New Delhi, Kolkatta, Chennai or Bangalore.

# Another interesting thing to do with maps would be to represent the trains themselves. It can be approximated by a line from the start station to the end station of the trains. We saw at the beginning of this notebook, that there are 4 024 trains on the dataset. Of course, such a high number of lines on a single map would be hideous, whatever the size of the map. But it can still be interesting to draw map that show a specific subset of trains. For example, the ones from the most frequented station. Let's look at the most frequented stations.

# In[ ]:


df_stations.sort_values('nb_trains',ascending=False).head(10)


# We see that there are four stations with more than 200 trains. Howrah Junction is the first one in number of trains, as well as in number of starting and ending trains. So, let's draw the map of Howrah Junction's trains !

# In[ ]:


howrah_lines = folium.Map(location=[22.59,88.31], zoom_start=4.5)
x0 = df_stations[df_stations['name'] == 'Howrah Junction']['latitude'].to_numpy()[0]
x1 = df_stations[df_stations['name'] == 'Howrah Junction']['longitude'].to_numpy()[0]
folium.Marker(location=(x0, x1), icon=Icon(color='purple', icon='train')).add_to(howrah_lines)
for idx, row in df_trains_aug.iterrows():
    if row['Starts'] == 'Howrah Junction':
        y0 = df_stations[df_stations['name'] == [row['Ends']][0]]['latitude'].to_numpy()[0]
        y1 = df_stations[df_stations['name'] == [row['Ends']][0]]['longitude'].to_numpy()[0]
        folium.Marker(location=(y0, y1), icon=Icon(color='green', icon='train')).add_to(howrah_lines)
    elif row['Ends'] == 'Howrah Junction':
        y0 = df_stations[df_stations['name'] == [row['Starts']][0]]['latitude'].to_numpy()[0]
        y1 = df_stations[df_stations['name'] == [row['Starts']][0]]['longitude'].to_numpy()[0]
        folium.Marker(location=(y0, y1), icon=Icon(color='orange', icon='train')).add_to(howrah_lines)
howrah_lines


# Green markers show the trains starting from Howrah Junction. Orange markers for the trains ending at Howrah Junction. That station seems to cover quite all the territory.
# Now, let's look at the international lines.

# In[ ]:


foreign_lines = folium.Map(location=[22.05, 78.94], zoom_start=4.5)
for idx, row in df_trains_aug.iterrows():
    if df_stations[df_stations['name'] == row['Starts']]['country'].to_numpy()[0] != 'in':
        x0 = df_stations[df_stations['name'] == [row['Starts']][0]]['latitude'].to_numpy()[0]
        x1 = df_stations[df_stations['name'] == [row['Starts']][0]]['longitude'].to_numpy()[0]
        y0 = df_stations[df_stations['name'] == [row['Ends']][0]]['latitude'].to_numpy()[0]
        y1 = df_stations[df_stations['name'] == [row['Ends']][0]]['longitude'].to_numpy()[0]
        folium.PolyLine(locations=[(x0, x1),(y0, y1)], color='limegreen').add_to(foreign_lines)
    elif df_stations[df_stations['name'] == row['Ends']]['country'].to_numpy()[0] != 'in':
        x0 = df_stations[df_stations['name'] == [row['Starts']][0]]['latitude'].to_numpy()[0]
        x1 = df_stations[df_stations['name'] == [row['Starts']][0]]['longitude'].to_numpy()[0]
        y0 = df_stations[df_stations['name'] == [row['Ends']][0]]['latitude'].to_numpy()[0]
        y1 = df_stations[df_stations['name'] == [row['Ends']][0]]['longitude'].to_numpy()[0]
        folium.PolyLine(locations=[(x0, x1),(y0, y1)], color='darkorange').add_to(foreign_lines)
foreign_lines


# In green: trains starting from outside, and coming to India.
# In orange: trains starting from India, and going outside.
# Most of foreign stations are located in Pakistan, plus one in Bhutan.

# ## Demographics analysis
# 
# ### Cities analysis

# In[ ]:


df_cities_stations = pd.DataFrame()
cs_name = []
cs_nb_stations = []
cs_nb_start_trains = []
cs_nb_end_trains = []
cs_nb_trains = []
cs_population = []
cs_literacy = []
cs_literacy_gap = []
cs_graduate = []
cs_state = []
cs_latitude = []
cs_longitude = []


# In[ ]:


stat_cities = set(df_stations['city'])
len(stat_cities)


# In[ ]:


for sc in stat_cities:
    for C in df_cities['name_of_city']:
        if sc in C.lower().split(' ') or C.lower() in sc.split(' '):
            subset = df_stations[df_stations['city'] == sc]
            cs_name.append(C)
            cs_nb_stations.append(len(subset))
            cs_nb_start_trains.append(sum(subset['nb_starts']))
            cs_nb_end_trains.append(sum(subset['nb_ends']))
            cs_nb_trains.append(sum(subset['nb_trains']))
            cs_population.append(df_cities[df_cities['name_of_city'] == C]['population_total'].to_numpy()[0])
            cs_literacy.append(df_cities[df_cities['name_of_city'] == C]['effective_literacy_rate_total'].to_numpy()[0])
            cs_literacy_gap.append(df_cities[df_cities['name_of_city'] == C]['effective_literacy_rate_male'].to_numpy()[0] - df_cities[df_cities['name_of_city'] == C]['effective_literacy_rate_female'].to_numpy()[0])
            cs_graduate.append(df_cities[df_cities['name_of_city'] == C]['total_graduates'].to_numpy()[0])
            cs_state.append(df_cities[df_cities['name_of_city'] == C]['state_name'].to_numpy()[0])
            cs_latitude.append(df_cities[df_cities['name_of_city'] == C]['location'].to_numpy()[0].split(',')[0])
            cs_longitude.append(df_cities[df_cities['name_of_city'] == C]['location'].to_numpy()[0].split(',')[1])

df_cities_stations['name'] = cs_name
df_cities_stations['nb_stations'] = cs_nb_stations
df_cities_stations['nb_start_trains'] = cs_nb_start_trains
df_cities_stations['nb_end_trains'] = cs_nb_end_trains
df_cities_stations['nb_trains'] = cs_nb_trains
df_cities_stations['population'] = cs_population
df_cities_stations['literacy'] = cs_literacy
df_cities_stations['literacy_gap'] = cs_literacy_gap
df_cities_stations['graduate'] = cs_graduate
df_cities_stations['state'] = cs_state
df_cities_stations['latitude'] = cs_latitude
df_cities_stations['longitude'] = cs_longitude


# In[ ]:


df_cities_stations.head(10)


# In[ ]:


df_cities_stations.sort_values('nb_stations', ascending=False).head(20)


# Here, we can see that our algorithm did not treat efficiently the city sharing a same name (Mumbai and Delhi/New Delhi). Let's correct them manually.
# For Mumbai, we will put everything into the 'Greater Mumbai' ensemble, as there is few doubt every Mumbai station is inside of it. We will as well drop the Delhi Cantonment, and distinguish Delhi from New Delhi.

# In[ ]:


df_cities_stations = df_cities_stations.drop([82,83,148,149])
df_cities_stations.sort_values('nb_stations', ascending=False)


# In[ ]:


df_stations[df_stations['city'] == 'new delhi']


# In[ ]:


for idx, row in df_cities.iterrows():
    if 'new delhi' in row['name_of_city'].lower():
        df_cities_stations = df_cities_stations.append(
                                {'name':row['name_of_city'],
                                'nb_stations':1,
                                'nb_start_trains':120,
                                'nb_end_trains':123,
                                'nb_trains':243,
                                'population':df_cities[df_cities['name_of_city'] == row['name_of_city']]['population_total'].to_numpy()[0],
                                'literacy':df_cities[df_cities['name_of_city'] == row['name_of_city']]['effective_literacy_rate_total'].to_numpy()[0],
                                'literacy_gap':df_cities[df_cities['name_of_city'] == row['name_of_city']]['effective_literacy_rate_male'].to_numpy()[0] - df_cities[df_cities['name_of_city'] == row['name_of_city']]['effective_literacy_rate_female'].to_numpy()[0],
                                'graduate':df_cities[df_cities['name_of_city'] == row['name_of_city']]['total_graduates'].to_numpy()[0],
                                'state':df_cities[df_cities['name_of_city'] == row['name_of_city']]['state_name'].to_numpy()[0],
                                'latitude':df_cities[df_cities['name_of_city'] == row['name_of_city']]['location'].to_numpy()[0].split(',')[0],
                                'longitude':df_cities[df_cities['name_of_city'] == row['name_of_city']]['location'].to_numpy()[0].split(',')[1]
                                },
                                ignore_index=True)


# Our new dataset is now complete. So we will first look at a description of it.

# In[ ]:


df_cities_stations.describe()


# In[ ]:


df_cities_stations.hist(bins = 10 , figsize= (12,16))


# The first information we extract from these quick views, is that there are very few cities with more than one station. The number of trains as well is quite low for most cities. It is also interesting to note that the shape of population hist looks like the number of train. We should explore if there is a correlation. The 'graduate' column is not very readible, as it is expressed in population rather than percentage.

# In[ ]:


fig, axs = plt.subplots(1,2)
axs[0].scatter(df_cities_stations['nb_stations'],df_cities_stations['population'])
axs[0].set_xlabel('Number of stations')
axs[0].set_ylabel('Population')
axs[1].scatter(df_cities_stations['nb_trains'],df_cities_stations['population'])
axs[1].set_xlabel('Number of trains')
plt.show()


# The two most populated cities have the most stations (3 and 4). It is less clear for less populated cities, as they can have 1 or 2 stations. The cloud about the number of trains is still more surprising. The city with most trains is far from being the most populated, and so has only 1 or 2 stations.

# In[ ]:


many_stations = df_cities_stations[df_cities_stations['nb_stations'] >= 3]
many_stations


# Delhi and Mumbai are the cities with most stations. 4 for Mumbai and 3 for Delhi.

# In[ ]:


df_cities_stations.sort_values('nb_trains', ascending=False).head(10)


# Nonetheless, Chennai, with 2 stations, and New Delhi, with only one station, have more trains than Mumbai or Delhi. We will focus on the number of trains, which seems to be more interesting than the bumber of stations.

# In[ ]:


fig, ax = plt.subplots()
ax.scatter(df_cities_stations['nb_start_trains'],df_cities_stations['nb_end_trains'])
ax.set_xlabel('Number of starting trains')
ax.set_ylabel('Number of ending trains')
plt.show()


# The previous scatter is pretty linear, which indicates that the number of trains starting from a city is similar to number of trains ending in that same city. Thus, it will not be very useful to conduct separate studies on starting and ending trains. Trains must be enough.

# In[ ]:


fig, axs = plt.subplots(3,1,figsize=(12,12))
axs[0].scatter(df_cities_stations['literacy'],df_cities_stations['nb_trains'])
axs[0].set_ylabel('Number of trains')
axs[0].set_xlabel('Literacy')
axs[1].scatter(df_cities_stations['literacy_gap'],df_cities_stations['nb_trains'])
axs[1].set_ylabel('Number of trains')
axs[1].set_xlabel('Gender inequality against literacy')
axs[2].scatter(df_cities_stations['graduate']/df_cities_stations['population'],df_cities_stations['nb_trains'])
axs[2].set_ylabel('Number of trains')
axs[2].set_xlabel('Rate of graduated inhabitants')
plt.show()


# There is nothing absolutely obvious that comes out these graphs. It means that the number of trains is not a direct way to study demographics of a city. We can, nonetheless, find some interesting features:
# * The cities with the highest number of trains have a good literacy rate.
# * The shape of graduated inhabitants graph does not match perfectly the one of literacy. We can suppose that literacy is required in very frequented cities, but there are jobs for many kinds of people, so graduation is not as much necessary. Cities with highest rates of graduated people have few trains, and are probably cities where elites are living.
# * Cities with highest gender inequality toward literacy have few trains, whereas cities with most trains have a quite good score. So it seems that being connected is a good way to fight geneder inequality.
# 

# ### Stations and trains by state

# In this section, we want to make a study at the level of the state. In the World Cities dataset, there is a column called region, with digital values. In the Top 500 Indian Cities, there is a column call state_name, that contains the names of the states. We want to find a way to match those columns from both dataset, and ideally keep the complete name as reference. So first, let's look if it is perfectly matching.

# In[ ]:


states = set(df_stations['state'])
stations_by_state = {}
for s in states:
    stations_by_state[s] = []
    for idx,row in df_stations.iterrows():
        if row['state'] == s:
            #stations_by_state[s].append(row['city'])
            c = row['city']
            for C in df_cities['name_of_city']:
                if c in C.lower() or C.lower() in c:
                    S = df_cities[df_cities['name_of_city'] == C]['state_name'].to_numpy()[0]
                    stations_by_state[s].append(S)
            
stations_by_state


# As we can see, some states seem to perfectly match, like (7.0 : NCT OF DELHI) or (25.0 : TAMIL NADU), whereas others are still very hard to determine, like the 8 or the 11. We can also note that there are three different formats in the regions "names" (keys): float, int and string. Float is the main type, and is probably for Indian regions. The int is unique, and there are 4 str. I guess str is for Pakistanese regions, as it is almost empty (except '04', which must be a homonymous), and there are 3 totally different values in the integer one (8), so maybe it is also a foreign country. In any case, we are not able to work with it, so it is going to be torn apart as well. Let's admit that when at least 2/3 of the corresponding state for a region is the same, they designate the same thing.

# In[ ]:


stations_by_state.pop(8)
stations_by_state.pop('02')
stations_by_state.pop('04')
stations_by_state.pop('06')
stations_by_state.pop('07')
trad_table = {}
for sbs in stations_by_state:
    threshold = 2 * len(stations_by_state[sbs]) / 3
    sts = set(stations_by_state[sbs])
    aux = {}
    for s in sts:
        aux[s] = stations_by_state[sbs].count(s)
    trad_table[sbs] = ''
    for t in sts:
        if aux[t] >= threshold:
            trad_table[sbs] = t
            
trad_table


# We will now work with the subset of states we identified.

# To be continued...
