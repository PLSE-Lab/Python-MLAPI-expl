#!/usr/bin/env python
# coding: utf-8

# # 0 Imports

# In[ ]:


import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 100)


# ***

# # 1 Plays dataset

# In[ ]:


df = pd.read_csv('../input/atpdata/ATP.csv')


# In[ ]:


df.head()


# In[ ]:


df.info()


# ## 1.1 Winners

# In[ ]:


winners = df[['winner_name', 'winner_hand', 'winner_ioc','winner_rank']]


# In[ ]:


winners.columns = ['name', 'hand', 'country', 'rank']


# ## 1.2 Losers

# In[ ]:


losers = df[['loser_name', 'loser_hand', 'loser_ioc', 'loser_rank']]


# In[ ]:


losers.columns = ['name', 'hand', 'country', 'rank']


# ## 1.3 Union winners w/ losers

# In[ ]:


frames = [winners, losers]
plays = pd.concat(frames)


# In[ ]:


#get the best ranking of each player
plays = plays.groupby(['name', 'hand', 'country']).agg('min').reset_index().sort_values(['rank','name'])


# In[ ]:


#create flag for the players who got into top 10
plays['top10'] = plays['rank'] <= 10


# In[ ]:


plays


# ***

# # 2 Player information dataset

# In[ ]:


inf = pd.read_csv('../input/atp-players-overviews/player_overviews_UNINDEXED.csv', 
                  names=['player_id','player_slug','first_name','last_name','player_url','flag_code','residence', 'birthplace','birthdate','birth_year','birth_month',
                         'birth_day','turned_pro','weight_lbs', 'weight_kg','height_ft','height_inches','height_cm','hand','backhand'])


# In[ ]:


inf.head()


# In[ ]:


#create full name so we can join with plays dataset later
inf['name'] = inf['first_name'] + ' ' + inf['last_name']


# In[ ]:


#replace zero values with nan
inf['turned_pro'] = inf['turned_pro'].replace(0, np.nan)
inf['height_cm'] = inf['height_cm'].replace(0, np.nan)
inf['weight_kg'] = inf['weight_kg'].replace(0, np.nan)


# In[ ]:


#calculate age of the player he turned pro
inf['turned_pro_age'] = inf['turned_pro'] - inf['birth_year']


# In[ ]:


#change birthdate data type to date
inf['birthdate'] = pd.to_datetime(inf['birthdate'], format='%Y.%m.%d')


# In[ ]:


#select only relevant columns
inf = inf[['name', 'residence', 'birthplace', 'birthdate', 'birth_year','birth_month', 'birth_day', 'turned_pro_age', 'weight_kg', 'height_cm', 'hand', 
           'backhand']]


# In[ ]:


inf.info()


# In[ ]:


inf.head()


# ***

# # 3 Zodiacs

# In[ ]:


#create dataset with zodiacs and its dates
data = {'d_start':  ['19', '21', '21', '21', '23', '23', '23', '23', '22', '22', '20', '19'],
        'm_start':  ['3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '1', '2'],
        'd_end':  ['20', '20', '20', '22', '22', '22', '22', '21','21', '19', '18', '20'],        
        'm_end':  ['4', '5', '6', '7', '8', '9', '10', '11','12', '1', '2', '3'],
        'zodiac': ['Aries', 'Taurus', 'Gemini', 'Cancer', 'Leo', 'Virgo', 'Libra', 'Scorpio', 'Sagittarius', 'Capricorn', 'Aquarius', 'Pisces']
        }
zdt = pd.DataFrame(data, columns = ['d_start', 'm_start', 'd_end', 'm_end', 'zodiac'])


# In[ ]:


#ensure correct data types
zdt[['d_start', 'm_start', 'd_end', 'm_end']] = zdt[['d_start', 'm_start', 'd_end', 'm_end']].apply(pd.to_numeric)


# In[ ]:


zdt.head()


# ## 3.1 Date-zodiac

# create table for every day from 1900 to 2020 and corresponding zodiac

# In[ ]:


#generate years
yDF = pd.DataFrame({"Year": pd.Series(range(1900,2020))})


# In[ ]:


#cross join dataset with years only and with month, day and zodiac dataset (created in previous step)
yDF = pd.merge(yDF.assign(key=0), zdt.assign(key=0), on='key').drop('key', axis=1)


# In[ ]:


#create full start and end date for each zodiac
yDF['Date_start'] = pd.to_datetime(yDF.Year*10000+yDF.m_start*100+yDF.d_start,format='%Y%m%d')
yDF['Date_end'] = pd.to_datetime(yDF.Year*10000+yDF.m_end*100+yDF.d_end,format='%Y%m%d')


# In[ ]:


#select only relevant columns
zodiacs = yDF[['Date_start', 'Date_end', 'zodiac']]


# In[ ]:


zodiacs.head()


# ***

# # 4 Join player info w/ zodiacs

# add zodiac info to the player information

# In[ ]:


#create artifical key to cross join every player information with every zodiac
inf['key'] = 0
zodiacs['key'] = 0
infAll = inf.merge(zodiacs)


# In[ ]:


#filter each player with corresponding zodiac
infAll = infAll[(infAll.birthdate >= infAll.Date_start) & (infAll.birthdate <= infAll.Date_end)]


# In[ ]:


#drop useless columns
infAll.drop(['key', 'Date_start', 'Date_end'], axis=1, inplace=True)


# In[ ]:


infAll.info()


# In[ ]:


infAll.head()


# ***

# # 5 Join plays w/ player info

# add complete player information to the original plays dataset

# In[ ]:


#perform join with plays and player information
out = plays.merge(infAll, on='name')


# In[ ]:


out


# ***

# # 6 Exploration

# In[ ]:


import plotly.express as px


# ## 6.1 How many players actually got to top 10?

# In[ ]:


fig = px.histogram(out, x="top10")
fig.show()


# ### 6.1.1 Who they were?

# In[ ]:


print(out[out['top10']].sort_values('name').name.values)


# ## 6.2 Is there any importance on player zodiac?

# ### 6.2.1 All players

# In[ ]:


fig = px.histogram(out, x="zodiac", color='zodiac').update_xaxes(categoryorder = 'total descending')
fig.show()


# ### 6.2.2 Top 10 players

# In[ ]:


fig = px.histogram(out[out['top10'] == True], x="zodiac", color='zodiac').update_xaxes(categoryorder = 'total descending')
fig.show()


# ## 6.3 Playing hand(s)

# In[ ]:


fig = px.histogram(out, x="hand_y", color='top10')
fig.show()


# In[ ]:


fig = px.histogram(out[out['top10'] == True], x="top10", color='hand_x')
fig.show()


# ## 6.4 Backhand

# In[ ]:


fig = px.histogram(out, x="backhand", color='top10')
fig.show()


# ## 6.5 Country

# In[ ]:


fig = px.histogram(out, x="country", color='top10').update_xaxes(categoryorder = 'total descending')
fig.show()


# ### 6.5.1 Top 10 players only

# In[ ]:


fig = px.histogram(out[out['top10'] == True], x="country").update_xaxes(categoryorder = 'total descending')
fig.show()


# ## 6.6 Player height

# In[ ]:


fig = px.histogram(out, x="height_cm", color='top10', nbins = 20)
fig.show()


# ## 6.7 Player weight

# In[ ]:


fig = px.histogram(out, x="weight_kg", color='top10', nbins = 20)
fig.show()


# ## 6.8 Age turned pro

# In[ ]:


fig = px.histogram(out, x="turned_pro_age", color='top10', nbins = 20)
fig.show()


# ## 6.9 Birthplace

# In[ ]:


get_ipython().system('pip install googlemaps')
import googlemaps


# In[ ]:


gmaps_key = googlemaps.Client(key="AIzaSyDREugpyCcRUjp_3KvPl6oOUzip6mm7NRY")


# In[ ]:


#create columns for latitude and longtitude
out['lon'] = None
out['lat'] = None


# In[ ]:


#get coordinates for birthplace using google maps api
for i in range(len(out)):
    geo_result = gmaps_key.geocode(out.loc[i, 'birthplace'])
    try:
        lon = geo_result[0]["geometry"]["location"]["lng"]
        lat = geo_result[0]["geometry"]["location"]["lat"]
        out.loc[i, 'lon'] = lon
        out.loc[i, 'lat'] = lat
    except:
        lat = None
        lon = None


# ### 6.9.1 All players

# In[ ]:


fig = px.scatter_geo(out[out['top10'] == False], lon="lon", lat = 'lat', color_discrete_sequence=["black"],hover_name='birthplace', opacity= 0.2)
fig.show()


# ### 6.9.2 Top 10 players

# In[ ]:


fig = px.scatter_geo(out[out['top10'] == True], lon="lon", lat = 'lat', color='top10', color_discrete_sequence=["green"],hover_name='birthplace', opacity=0.5)
fig.show()


# In[ ]:


#output the complete dataset
out.to_csv('ATP_players_info_full.csv', index = False)


# # 7 Conclusion

# 1. Vast majority of the players in dataset did not get to the top 10 (135 got there vs 2569 did not)
# 2. Aries and Pisces are the most frequent and also most successfull zodiacs in ATP. Leo and Cancer, despite not being the most frequent, are still pretty successfull.
# 3. Most of the players are right handed therefore most of the players who got into top 10 are right handers as well. The same aplies for the two-handed backhand.
# 4. As of a country the most dominant are undoubtedly USA. However only 5,4% (29/528) of all USA players got into top 10. In comparison, Sweden which has 99 players overall has 14 players who broke into top ten, which represent 14% of all Sweden players. Similar to Sweden is situation of Spain players (16:109 = ~15%).
# 5. Looking at body dimensions the most prominent players height is between 175 - 195cm and weight 70-90kg.
# 6. The best age to turn pro is between 16 - 19 years.
# 7. While ATP players borned around whole globe the ones who got into top 10 borned mainly in Europe and USA.

# ## How to break into ATP top 10?
# * Get born as a Aries, Pisces or Cancer zodiac,
# * Play with right hand and both-handed backhand,
# * Play for Spain or Sweden,
# * Be around 185cm tall and weight around 80kg,
# * Turn pro at the age of 18,
# * Get born in Europe or in the USA.
# 
# <div style="text-align: right"><i>This is for fun purpose only.</div>
# <div style="text-align: right"><i>This is by no means any country emphasizing/promotion and author strictly refuses any country differences "conflicts".</div>
