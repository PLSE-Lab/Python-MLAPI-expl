#!/usr/bin/env python
# coding: utf-8

# #                                  ===     Tamir's  Ganor  Foot  ball  Project  ===
# 
# European Soccer Database
# 25k+ matches, players & teams attributes for European Professional Football
# ![z.jpg](attachment:z.jpg)

# [](http://)# The ultimate Soccer database for data analysis and 
# 
# 
# +25,000 matches
# 
# +10,000 players
# 
# 11 European Countries with their lead championship
# 
# Seasons 2008 to 2016
# 
# Players and Teams' attributes* sourced from EA Sports' FIFA video game series,
# 
# Detailed match events (goal types, possession, corner, cross, fouls, cards etc...) for +10,000 matches
# 
# ## insperation from Kaggale Kernel
# https://www.kaggle.com/hugomathien/soccer 
# 
# https://www.kaggle.com/pavanraj159/european-football-data-analysis

# ### EDA + Prediction tool for UK Priemer league
# *** setting up the enviroments

# In[ ]:


#Import libraries
import numpy as np
import pandas as pd
import sqlite3
from datetime import timedelta
import datetime as dt
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.cm

from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize
import mpl_toolkits
#import folium
#import folium.plugins
from matplotlib import animation,rc
import io
import base64
import itertools
from subprocess import check_output
import missingno as msno
sns.set()


# In[ ]:


import os
print(os.listdir("../input"))
print(os.listdir("../input/soccer"))

print(os.listdir("../input/teams-image"))

pathimage="../input/teams-image/english-premier-league-club-logos-english-premier-league-team-badges-20112012-season-soccer-news-ideas.jpg"
pathimage


# ## Import Data

# In[ ]:


with sqlite3.connect('../input/soccer/database.sqlite') as con:
    countries = pd.read_sql_query("SELECT * from Country", con)
    matches = pd.read_sql_query("SELECT * from Match", con)
    leagues = pd.read_sql_query("SELECT * from League", con)
    teams = pd.read_sql_query("SELECT * from Team", con)
    player = pd.read_sql_query("SELECT * from Player",con)
    player_attributes = pd.read_sql_query("SELECT * from Player_Attributes",con)
    sequence = pd.read_sql_query("SELECT * from sqlite_sequence",con)
    team_attributes = pd.read_sql_query("SELECT * from Team_Attributes",con)


# In[ ]:


print (countries)
countries.info()


# In[ ]:


print (leagues)
leagues.info()


# In[ ]:


matches.head()


# In[ ]:


matches.info()


# In[ ]:


matches.describe().T


# In[ ]:


msno.matrix(matches);


# ## lots of Nulls i will clean it after mergers and keep only important columns 

# In[ ]:


teams.head()
# teams.describe()
# teams.info()


# In[ ]:


teams.info()


# In[ ]:


teams.describe()


# In[ ]:


player.head()


# In[ ]:


player.info()


# In[ ]:


player_attributes.head()


# In[ ]:


player_attributes.info()


# In[ ]:


sequence


# ### Note sequence will not be used in my EDA

# In[ ]:


team_attributes.head()


# In[ ]:


#Merge country and leauge data
countries_leagues = countries.merge(leagues,left_on="id",right_on="id",how="outer")
countries_leagues = countries_leagues.drop("id",axis = 1)
countries_leagues = countries_leagues.rename(columns={'name_x':"country", 'name_y':"league"})


# In[ ]:


matches.head()


# #      take from matches the info we want 
#          subsetting data with necessary columns

# In[ ]:


matches_new = matches[[ 'country_id', 'league_id', 'season', 'stage', 'date',
                   'match_api_id', 'home_team_api_id', 'away_team_api_id',
                    'home_team_goal', 'away_team_goal']]

#matches_new = matches_new.drop("id",axis=1)


# In[ ]:


msno.matrix(matches_new);


# ## matches data is clean now

# In[ ]:


matches_new.head()


# In[ ]:


#merge leauge data with match data
data = matches_new.merge(countries_leagues,left_on="country_id",right_on="country_id",how="outer")
#chech null values
print (data.isnull().sum())
data.head()


# In[ ]:


#Unique values in data
data.nunique()


# # European countries playing football

# In[ ]:


# you have to load or enter your API Key from google map 
# with open("../input/my-key/APIKEY.txt") as f:
#     api_key = f.readline()
#     f.close


# In[ ]:


# !pip install gmaps



# import gmaps
# gmaps.configure(api_key=api_key)


# In[ ]:


countries


# In[ ]:


from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt


# In[ ]:


# !jupyter nbextension enable --py gmaps


# In[ ]:


# europea_coordinates=(50.5, 4.4)
# gmaps.figure(center=europea_coordinates, zoom_level=11)

# locations = [[50.5, 4.469936],[52.355518, -1.174320],[46.227638, 2.213749],[51.165691, 10.451526],[41.871941, 12.567380],
#              [52.132633, 5.291266],[51.919437, 19.145136],[39.399872, -8.224454],[56.490669, -4.202646],[40.463669, -3.749220],
#              [46.818188, 8.227512]]
             
# fig = gmaps.figure()
# marker_locations = locations

# fig = gmaps.figure()
# markers = gmaps.marker_layer(marker_locations)
# fig.add_layer(markers)

# fig.add_layer(gmaps.heatmap_layer(locations))
# fig


# *********since i was not able to see the map here i will save the image *

# ![map.png](attachment:map.png)

# 

# # Matches by League

# In[ ]:


data['league'].value_counts()
data['league'].value_counts().plot.barh(figsize=(8,6),title='Match by league');
plt.gca().invert_yaxis()


# In[ ]:


plt.figure(figsize=(8,8))
ax = sns.countplot(y = data["league"],
                   order=data["league"].value_counts().index,
                   linewidth = 1,
                   edgecolor = "k"*data["league"].nunique()
                 )
for i,j in enumerate(data["league"].value_counts().values):
    ax.text(.7,i,j,weight = "bold")
plt.title("Matches by league")
plt.show()


# ## Home and away goals by league
# 

# In[ ]:


data.groupby("league").agg({"home_team_goal":"sum","away_team_goal":"sum"}).plot(kind="barh",
                                                                                 figsize = (10,8),
                                                                                 edgecolor = "k",
                                                                                 linewidth =1
                                                                                )
plt.title("Home and away goals by league")
plt.legend(loc = "best" , prop = {"size" : 14})
plt.xlabel("total goals")
plt.show()


#  MATCHES PLAYED IN EACH LEAGUE BY SEASON
# ===

# In[ ]:


plt.figure(figsize=(12,12))
sns.countplot(y = data["season"],hue=data["league"],
              palette=["r","g","b","c","lime","m","y","k","gold","orange"])
plt.title("MATCHES PLAYED IN EACH LEAGUE BY SEASON")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()


# In[ ]:


#Merge team data
data = data.merge(teams,left_on="home_team_api_id",right_on="team_api_id",how="left")
data = data.drop(["id","team_api_id",'team_fifa_api_id'],axis = 1)
data = data.rename(columns={ 'team_long_name':"home_team_lname",'team_short_name':"home_team_sname"})
data.columns


# In[ ]:


data = data.merge(teams,left_on="away_team_api_id",right_on="team_api_id",how="left")
data = data.drop(["id","team_api_id",'team_fifa_api_id'],axis = 1)
data = data.rename(columns={ 'team_long_name':"away_team_lname",'team_short_name':"away_team_sname"})
data.columns


# TOP TEAMS BY THEIR HOME & AWAY GOALS
# ==

# In[ ]:


h_t = data.groupby("home_team_lname")["home_team_goal"].sum().reset_index()
a_t = data.groupby("away_team_lname")["away_team_goal"].sum().reset_index()
h_t = h_t.sort_values(by="home_team_goal",ascending= False)
a_t = a_t.sort_values(by="away_team_goal",ascending= False)
plt.figure(figsize=(13,10))
plt.subplot(121)
ax = sns.barplot(y="home_team_lname",x="home_team_goal",
                 data=h_t[:20],palette="summer",
                 linewidth = 1,edgecolor = "k"*20)
plt.ylabel('')
plt.title("top teams by home goals")
for i,j in enumerate(h_t["home_team_goal"][:20]):
    ax.text(.7,i,j,weight = "bold")
plt.subplot(122)
ax = sns.barplot(y="away_team_lname",x="away_team_goal",
                 data=a_t[:20],palette="winter",
                linewidth = 1,edgecolor = "k"*20)
plt.ylabel("")
plt.subplots_adjust(wspace = .4)
plt.title("top teams by away goals")
for i,j in enumerate(a_t["away_team_goal"][:20]):
    ax.text(.7,i,j,weight = "bold")


# In[ ]:


data


# Number of teams in each league
# ==

# In[ ]:


data.groupby('league')['home_team_lname'].nunique()


# In[ ]:


data.groupby('league')['home_team_lname'].nunique().plot.bar();


# I know that there are 20 teams in England Premier League each season so i want to check it
# ==

# In[ ]:


primerleague=data[data['league']=='England Premier League']
primerleague.groupby(['season'])['home_team_lname'].nunique().plot.bar()


# # so this is correct
# ![correct-clipart-1.jpg](attachment:correct-clipart-1.jpg)
# 

# ## !!!  Who won more games in Primer League

# In[ ]:


a=data['home_team_goal']-data['away_team_goal']
data['a']=a
data.head()
data['win']='tie'
data['win']=np.where(data['a']>0,'home','away')
# df['color'] = np.where(df['Set']=='Z', 'green', 'red')
data[['win','a']]

cond=[data['a']==0,data['a']>0,data['a']<0]
choice=['Tie','Home','Away']
data['win'] = np.select(cond, choice)


data[['win','a']].head()


# In[ ]:


data.drop(['a'],axis=1,inplace=True)
data


# In[ ]:


fig, axs = plt.subplots(1, 2)
Pleag=data[data['league']=='England Premier League']
HW=Pleag[Pleag['win']=='Home']
AW=Pleag[Pleag['win']=='Away']
HW.groupby(['home_team_lname'])['stage'].count().sort_values(ascending=False).head(10).plot.bar(figsize=(16,8),ax=axs[0])
AW.groupby(['away_team_lname'])['stage'].count().sort_values(ascending=False).head(10).plot.bar(figsize=(16,8),ax=axs[1],color='red')


# ## and in Spain?

# In[ ]:


fig, axs = plt.subplots(1, 2)
Pleag=data[data['league']=='Spain LIGA BBVA']
HW=Pleag[Pleag['win']=='Home']
AW=Pleag[Pleag['win']=='Away']

# print (HW.groupby(['home_team_lname'])['stage'].count().sort_values(ascending=False))
# print (AW.groupby(['away_team_lname'])['stage'].count().sort_values(ascending=False))

HW.groupby(['home_team_lname'])['stage'].count().sort_values(ascending=False).head(10).plot.bar(figsize=(16,8),ax=axs[0])
AW.groupby(['away_team_lname'])['stage'].count().sort_values(ascending=False).head(10).plot.bar(figsize=(16,8),ax=axs[1],color='red')


# ## and in Europe?

# In[ ]:


fig, axs = plt.subplots(1, 2)
Pleag=data
HW=Pleag[Pleag['win']=='Home']
AW=Pleag[Pleag['win']=='Away']

# print (HW.groupby(['home_team_lname'])['stage'].count().sort_values(ascending=False))
# print (AW.groupby(['away_team_lname'])['stage'].count().sort_values(ascending=False))

HW.groupby(['home_team_lname'])['stage'].count().sort_values(ascending=False).head(10).plot.bar(figsize=(16,8),ax=axs[0])
AW.groupby(['away_team_lname'])['stage'].count().sort_values(ascending=False).head(10).plot.bar(figsize=(16,8),ax=axs[1],color='red')
list=AW.groupby(['away_team_lname'])['stage'].count().sort_values(ascending=False).head(10)
list2=HW.groupby(['home_team_lname'])['stage'].count().sort_values(ascending=False).head(10)
list2.index
list.index
list=(set([i for i in list.index]+[i for i in list2.index]))
len(list)
type(list)


# ## let's be fair
# and show the number of games in the league

# In[ ]:


top=data[data['home_team_lname'].isin(list)]
top.groupby('home_team_lname')['stage'].count().sort_values(ascending=False).plot.bar();


# In[ ]:


pd.DataFrame(top.groupby('home_team_lname')['stage'].count().sort_values(ascending=False))


# ZOOM in England Premier League
# ==
# 
# ### where is it in the world?

# In[ ]:


fig, ax = plt.subplots(figsize=(10,20))
map = Basemap(projection='ortho', 
              lat_0=0, lon_0=0)

map.drawmapboundary(fill_color='aqua')
map.fillcontinents(color='coral',lake_color='aqua')
map.drawcoastlines()
#x, y = m(-0.126240,51.500150 )
#plt.plot(x, y, 'bo', markersize=100)

x, y = map(-0.126240,51.500150 )

map.plot(x, y, marker='D',color='w',markersize=10)

plt.show()
x


# In[ ]:


primerleague


# # what are the teams

# In[ ]:


primerleague['home_team_lname'].value_counts().head(20)
#primerleague.groupby(['season'])['home_team_lname'].unique()
plteams=primerleague['home_team_lname'].value_counts().head(20)
type(plteams)
plteams=pd.DataFrame(plteams)
plteams.drop(['home_team_lname'],axis=1,inplace=True)
#data.drop(['a'],axis=1,inplace=True)
#plteams=plteams.index.tolist()
plteams=['Tottenham Hotspur',
 'Stoke City',
 'Manchester City',
 'Sunderland',
 'Everton',
 'Chelsea',
 'Aston Villa',
 'Manchester United',
 'Arsenal',
 'Liverpool',
 'West Bromwich Albion',
 'West Ham United',
 'Newcastle United',
 'Fulham',
 'Swansea City',
 'Wigan Athletic',
 'Bolton Wanderers',
 'Blackburn Rovers',
 'Southampton',
 'Norwich City']
plteams


# In[ ]:


fig, ax = plt.subplots(figsize=(10,20))
m = Basemap(resolution='l', # c, l, i, h, f or None
            projection='merc',
            lat_0=54.5, lon_0=-4.36,
            llcrnrlon=-6., llcrnrlat= 49.5, urcrnrlon=2., urcrnrlat=55.2)
m.drawmapboundary(fill_color='#46bcec')
m.fillcontinents(color='#f2f2f2',lake_color='#46bcec')

x, y = m(-2.234380, 53.480709)
plt.plot(x, y, 'ok', markersize=5)
plt.text(x, y, ' Manchester', fontsize=12);

x, y = m(-2.977840, 53.410780)
plt.plot(x, y, 'ok', markersize=5)
plt.text(x, y, ' liverpool', fontsize=12);

x, y = m(-1.439740, 53.390630)
plt.plot(x, y, 'ok', markersize=5)
plt.text(x, y, ' Stoke City', fontsize=12);

x, y = m(-0.961370, 53.411460)
plt.plot(x, y, 'ok', markersize=5)
plt.text(x, y, ' Everton', fontsize=12);

x, y = m(-1.863050,52.474730)
plt.plot(x, y, 'ok', markersize=5)
plt.text(x, y, ' Arsenal', fontsize=12);

# x, y = m(-0.126240, 51.500150)
# plt.plot(x, y, 'ok', markersize=5)
# plt.text(x, y, ' London', fontsize=12);

x, y = m( -0.168680,51.487470 )
plt.plot(x, y, 'ok', markersize=5)
plt.text(x, y, ' Chelsea', fontsize=12);

x, y = m( -1.381450,54.904450)
plt.plot(x, y, 'ok', markersize=5)
plt.text(x, y, ' Sunderland', fontsize=12);

x, y = m( -0.075230,51.598550)
plt.plot(x, y, 'ok', markersize=5)
plt.text(x, y, ' Tottenham Hotspur', fontsize=12);

x, y = m( -0.608290,51.848579)
plt.plot(x, y, 'ok', markersize=5)
plt.text(x, y, ' Aston Villa', fontsize=12);

x, y = m( 0.008640,51.534900)
plt.plot(x, y, 'ok', markersize=5)
plt.text(x, y, ' West Ham ', fontsize=12);

x, y = m(-1.612920,54.977840)
plt.plot(x, y, 'ok', markersize=5)
plt.text(x, y, ' Newcastle ', fontsize=12);

x, y = m(-1.403230,50.904970 )
plt.plot(x, y, 'ok', markersize=5)
plt.text(x, y, 'Southampton ', fontsize=12);


x, y = m(-0.126240,51.500150 )
plt.plot(x, y, 'bo', markersize=100)

m.drawcoastlines()
#Manchester 53.480709, -2.234380
#Liverpool 53.410780, -2.977840
# Stoke City',53.390630, -1.439740
# 'Everton', 53.411460, -0.961370
# 'Arsenal'-1.863050,52.474730 
#london -0.126240,51.500150 
#Chelsea', 51.487470, -0.168680
#'Sunderland', , -1.381450,54.904450
#'Tottenham Hotspur',-0.075230,51.598550
#Aston Villa' -0.608290,51.848579
#'West Ham United' 0.008640,51.534900
#'Newcastle United'-1.612920,54.977840 
#Southampton -1.403230,50.904970 


# Betting tool for uk 
# ==

# In[ ]:


#plteams

for i ,j in enumerate(plteams):
    print (i+1 ,j)
    
import matplotlib.image as mpimg
img=mpimg.imread('../input/teams-image/english-premier-league-club-logos-english-premier-league-team-badges-20112012-season-soccer-news-ideas.jpg')
plt.show()




# In[ ]:


text="""Please choose home team  from the list:
1 Tottenham Hotspur
2 Stoke City
3 Manchester City
4 Sunderland
5 Everton
6 Chelsea
7 Aston Villa
8 Manchester United
9 Arsenal
10 Liverpool
11 West Bromwich Albion
12 West Ham United
13 Newcastle United
14 Fulham
15 Swansea City
16 Wigan Athletic
17 Bolton Wanderers
18 Blackburn Rovers
19 Southampton
20 Norwich City\n"""

text1="""Please choose away team  from the list: (above)\n"""



print (text)

a=int(input())
print (text1)
b=int(input(text1))

print (plteams[a-1],"against",plteams[b-1])




# In[ ]:


data.head()
internal=data[(data['home_team_lname']==plteams[a-1]) & (data['away_team_lname']==plteams[b-1])]
H=internal[internal['win']=='Home'].date.count()
A=internal[internal['win']=='Away'].date.count()
T=internal[internal['win']=='Tie'].date.count()
print (H,A,T)
df = pd.DataFrame({'result': [H,A,T]},
                 index=['1', '2', 'X'])

df.plot.pie(y='result', figsize=(5, 5));


# In[ ]:


print (text)
a=int(input())

print (text1)
b=int(input())

print("if home team weans how much you get per each pound?")
c=input()
print ("if away team weans how much you get per each pound?")
d=input()
print("if Tie how much you get per each pound?")
e=input()
print (plteams[a-1],"against",plteams[b-1])
print ("i get {}Eur if {} wins {}Eur if {} wins and {}Eur for Tie".format(c, plteams[a-1],d,plteams[b-1],e))




# In[ ]:


data.head()
internal=data[(data['home_team_lname']==plteams[a-1]) & (data['away_team_lname']==plteams[b-1])]
H=internal[internal['win']=='Home'].date.count()
A=internal[internal['win']=='Away'].date.count()
T=internal[internal['win']=='Tie'].date.count()
print (H,A,T)
df = pd.DataFrame({'result': [H,A,T]},
                 index=['1', '2', 'X'])

df.plot.pie(y='result', figsize=(5, 5));


# In[ ]:


fig, ax = plt.subplots(1, 2)
internal=data[(data['home_team_lname']==plteams[a-1]) & (data['away_team_lname']==plteams[b-1])]
H=internal[internal['win']=='Home'].date.count()
A=internal[internal['win']=='Away'].date.count()
T=internal[internal['win']=='Tie'].date.count()
print (H,A,T)
df = pd.DataFrame({'result': [H,A,T]},
                 index=['1', '2', 'X'])

df.plot.pie(y='result', figsize=(10, 5),ax=ax[0]);
internal=data[(data['home_team_lname']==plteams[a-1]) & (data['away_team_lname']==plteams[b-1])]
H=internal[internal['win']=='Home'].date.count()
A=internal[internal['win']=='Away'].date.count()
T=internal[internal['win']=='Tie'].date.count()
total=H+A+T
H=(H/total)*float(c)
A=(A/total)*float(d)
T=(T/total)*float(e)
print (H,A,T)
df = pd.DataFrame({'result': [H,A,T]},
                 index=['1', '2', 'X'])
df1 = pd.DataFrame({'border': [1,1,1,1,1]},
                 index=['0','1', '2', 'X','d2'])
df1.plot(figsize=(10, 5),ax=ax[1],color='black')
df.plot.bar(y='result', figsize=(10, 5),ax=ax[1],color="red");
#ax.hlines(1,1,3, linestyle='--', color='pink')


# In[ ]:




