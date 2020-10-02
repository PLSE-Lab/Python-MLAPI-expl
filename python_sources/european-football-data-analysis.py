#!/usr/bin/env python
# coding: utf-8

# # The ultimate Soccer database for data analysis and machine learning
# ==
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
# Team line up with squad formation (X, Y coordinates)
# 
# Betting odds from up to 10 providers
# 
# Detailed match events (goal types, possession, corner, cross, fouls, cards etc...) for +10,000 matches

# In[ ]:


import numpy as np
import pandas as pd
import sqlite3
from datetime import timedelta
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
from mpl_toolkits.basemap import Basemap
import folium
import folium.plugins
from matplotlib import animation,rc
import io
import base64
import itertools
from subprocess import check_output


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
lat_long = pd.read_excel("../input/lat-lon-info-cities/latlong.xlsx",sheetname="Sheet1")


# In[ ]:


countries.head()
leagues.head()
matches.head()
teams.head()
player.head()
player_attributes.head()
sequence.head()
team_attributes.head()


# In[ ]:


countries_leagues = countries.merge(leagues,left_on="id",right_on="id",how="outer")
countries_leagues = countries_leagues.drop("id",axis = 1)
countries_leagues = countries_leagues.rename(columns={'name_x':"country", 'name_y':"league"})


# In[ ]:


matches_new = matches[['id', 'country_id', 'league_id', 'season', 'stage', 'date',
                   'match_api_id', 'home_team_api_id', 'away_team_api_id',
                    'home_team_goal', 'away_team_goal']]
matches_new = matches_new.drop("id",axis=1)


# In[ ]:


data = matches_new.merge(countries_leagues,left_on="country_id",right_on="country_id",how="outer")
data.isnull().sum()


# In[ ]:


data.nunique()


# # "MATCHES PLAYED IN COUNTRIES"

# In[ ]:


country_info  = countries_leagues.merge(lat_long,left_on="country",right_on="name",how="left")
country_info = country_info.drop(["country_id","country_y","name"],axis = 1)

m3 = Basemap(projection='ortho', resolution=None, lat_0=50, lon_0=10,urcrnrlat=80,llcrnrlat=-80)

plt.figure(figsize=(12,12))

country = list(country_info["country_x"].unique())
c    = sns.color_palette("Set1",11)
label = country
def function(country,c,label):
    lat = list(country_info[country_info["country_x"] == country].latitude)
    lon = list(country_info[country_info["country_x"] == country].longitude)
    x,y = m3(lon,lat)
    m3.plot(x,y,"go",markersize=15,color=j,alpha=.8,label=i)

for i,j in zip(country,c):
    function(i,j,i)

m3.bluemarble(scale=0.5)
plt.legend(loc="center right",frameon=True,prop={"size":15}).get_frame().set_facecolor("white")
plt.title("MATCHES PLAYED IN COUNTRIES")
plt.show()


# # LEAGUES IN DIFFFERT COUNTRIES

# In[ ]:


#westlimit=-23.8; southlimit=25.8; eastlimit=60.6; northlimit=64.9
m =Basemap(projection="merc",llcrnrlat=35,urcrnrlat=60,llcrnrlon=-12,urcrnrlon=25,lat_ts=20,lat_0=True,lon_0=True)

plt.figure(figsize=(15,10))

m.drawmapboundary(fill_color="skyblue",color="k",linewidth=2)
m.drawcoastlines(linewidth=2)
m.drawcountries(linewidth=2,color="grey")
m.fillcontinents(color="gold",alpha=1,lake_color="b")

leag = list(country_info["league"].unique())
c    = sns.color_palette("Set1",11)
lat = list(country_info[country_info["league"] == leag].latitude)
lon = list(country_info[country_info["league"] == leag].longitude)
x,y  = m(lon,lat) 

def function(leag,c):
    lat = list(country_info[country_info["league"] == leag].latitude)
    lon = list(country_info[country_info["league"] == leag].longitude)
    x,y = m(lon,lat)
    m.plot(x,y,"go",markersize=40,color=j,alpha=.8,linewidth=20)
    
for i,j in zip(leag,c):
    function(i,j)
        
for i,j,k in itertools.zip_longest(x,y,country_info["league"]):
    plt.text(i,j,k,fontsize =10,color="k",horizontalalignment='center',verticalalignment='center',weight="bold")
plt.title("LEAGUES IN DIFFFERT COUNTRIES")
plt.show()


# # COUNT PLOT FOR LEAGUES IN DATA

# In[ ]:


plt.figure(figsize=(8,8))
ax = sns.countplot(y = data["league"],order=data["league"].value_counts().index)
for i,j in enumerate(data["league"].value_counts().values):
    ax.text(.7,i,j,weight = "bold")
plt.title("# COUNT PLOT FOR LEAGUES IN DATA")
plt.show()


# TOTAL HOME AND AWAY GOALS IN EACH LEAGUE
# ==

# In[ ]:


data.groupby("league").agg({"home_team_goal":"sum","away_team_goal":"sum"}).plot(kind="barh",figsize = (10,10))
plt.title("TOTAL HOME AND AWAY GOALS IN EACH LEAGUE")
plt.show()


# In[ ]:


data["date"] = pd.to_datetime(data["date"],format="%Y-%m-%d")
data["year"] = pd.DatetimeIndex(data["date"]).year


# MATCHES PLAYED IN EACH LEAGUE BY SEASON
# ==

# In[ ]:


plt.figure(figsize=(10,10))
sns.countplot(y = data["season"],hue=data["league"],palette=["r","g","b","c","lime","m","y","k","gold","orange"])
plt.title("MATCHES PLAYED IN EACH LEAGUE BY SEASON")
plt.show()


# In[ ]:


data = data.merge(teams,left_on="home_team_api_id",right_on="team_api_id",how="left")


# In[ ]:


data = data.drop(["id","team_api_id",'team_fifa_api_id'],axis = 1)
data = data.rename(columns={ 'team_long_name':"home_team_lname",'team_short_name':"home_team_sname"})
data.columns


# In[ ]:


data = data.merge(teams,left_on="away_team_api_id",right_on="team_api_id",how="left")


# In[ ]:


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
plt.figure(figsize=(13,8))
plt.subplot(121)
ax = sns.barplot(y="home_team_lname",x="home_team_goal",data=h_t[:20],palette="summer")
plt.ylabel('')
plt.title("top teams by home goals")
for i,j in enumerate(h_t["home_team_goal"][:20]):
    ax.text(.7,i,j,weight = "bold")
plt.subplot(122)
ax = sns.barplot(y="away_team_lname",x="away_team_goal",data=a_t[:20],palette="winter")
plt.ylabel("")
plt.subplots_adjust(wspace = .4)
plt.title("top teams by away goals")
for i,j in enumerate(a_t["away_team_goal"][:20]):
    ax.text(.7,i,j,weight = "bold")


# TOTAL GOALS SCORED BY TOP TEAMS 
# ==

# In[ ]:


x = h_t
x = x.rename(columns={'home_team_lname':"team", 'home_team_goal':"goals"})
y = a_t
y = y.rename(columns={'away_team_lname':"team", 'away_team_goal':"goals"})
goals = pd.concat([x,y])
goals = goals.groupby("team")["goals"].sum().reset_index().sort_values(by = "goals",ascending = False)
plt.figure(figsize=(10,12))
ax = sns.barplot(x="goals",y="team",data=goals[:30],palette="rainbow")
for i,j in enumerate(goals["goals"][:30]):
    ax.text(.3,i,j,weight="bold",color = "k",fontsize =12)
plt.title("TOTAL GOALS SCORED BY TOP TEAMS ")
plt.show()


# # MATCHES PLAYED BY TEAMS

# In[ ]:


x = data.groupby("home_team_lname")["match_api_id"].count().reset_index()
x = x.rename(columns={"home_team_lname":"team"})
y = data.groupby("away_team_lname")["match_api_id"].count().reset_index()
y = y.rename(columns={"away_team_lname":"team"})
xy = pd.concat([x,y],axis=0)
match_teams =  xy.groupby("team")["match_api_id"].sum().reset_index().sort_values(by="match_api_id",ascending =False)
match_teams = match_teams.rename(columns={"match_api_id":"matches_played"})
match_teams.head(10)


# NETWORK LAYOUT FOR MATCHES PLAYED BETWEEN TOP SCORERS
# ==

# In[ ]:


ts = list(goals["team"][:50])
v =data[["home_team_lname","away_team_lname"]]
v = v[(v["home_team_lname"].isin(ts)) & (v["away_team_lname"].isin(ts))]
import networkx as nx
g = nx.from_pandas_dataframe(v,"home_team_lname","away_team_lname")
fig = plt.figure(figsize=(11,10))
nx.draw_kamada_kawai(g,with_labels =True,node_size =2500,node_color ="Orangered",alpha=.8)
plt.title("NETWORK LAYOUT FOR MATCHES PLAYED BETWEEN TOP SCORERS")
fig.set_facecolor("ghostwhite")


# DISTRIBUTION OF HOME AND AWAY GOALS
# ==

# In[ ]:


plt.figure(figsize=(8,4))
plt.hist(data["home_team_goal"],color="b",alpha=.3,label="home")
plt.hist(data["away_team_goal"],color="r",alpha = .3,label="away")
plt.legend(loc="best")
plt.title("DISTRIBUTION OF HOME AND AWAY GOALS")
plt.show()


# MATCHES VS GOALS BY TEAMS
# ==

# In[ ]:


x = data.groupby(["home_team_lname","league"]).agg({"match_api_id":"count","home_team_goal":"sum"}).reset_index()
y = data.groupby(["away_team_lname","league"]).agg({"match_api_id":"count","away_team_goal":"sum"}).reset_index()
x = x.rename(columns={'home_team_lname':"team", 'match_api_id':"matches", 'home_team_goal':"goals"})
y = y.rename(columns={'away_team_lname':"team", 'match_api_id':"matches", 'away_team_goal':"goals"})
xy = pd.concat([x,y])
xy = xy.groupby(["team","league"])[["matches","goals"]].sum().reset_index()
xy = xy.sort_values(by="goals",ascending=False)
plt.figure(figsize=(13,6))
c   = ["r","g","b","m","y","yellow","c","orange","grey","lime","white"]
lg = xy["league"].unique()
for i,j,k in itertools.zip_longest(lg,range(len(lg)),c):
    plt.scatter("matches","goals",data=xy[xy["league"] == i],label=[i],s=100,alpha=1,linewidths=1,edgecolors="k",color=k)
    plt.legend(loc="best")
    plt.xlabel("MATCHES")
    plt.ylabel("GOALS SCORED")

plt.title("MATCHES VS GOALS BY TEAMS")
plt.show()


# # MATCHES VS GOALS BY TOP 50 TEAMS# 

# In[ ]:


plt.figure(figsize=(8,10))
plt.scatter(y = xy["team"][:50],x = xy["matches"][:50],s=xy["goals"],alpha=.7,c=sns.color_palette("Blues"),linewidths=1,edgecolors="b")
plt.xticks(rotation = 90)
plt.title("MATCHES VS GOALS BY TOP 50 TEAMS")
plt.show()


# In[ ]:


plt.figure(figsize=(13,5))
plt.subplot(121)
sns.boxplot(y = data["season"],x = data["away_team_goal"],palette="rainbow")
plt.xticks(rotation = 60)
plt.title("HOME GOALS BY SEASON")
plt.subplot(122)
sns.boxplot(y = data["season"],x = data["home_team_goal"],palette="rainbow")
plt.xticks(rotation = 60)
plt.title("AWAY GOALS BY SEASON")
plt.show()


# SUMMARY OF TOTAL GOALS SCORED BY YEAR
# ==

# In[ ]:


data["total_goal"] = data["home_team_goal"]+data["away_team_goal"]
a = data.groupby("season").agg({"total_goal":"sum"})
m = data.groupby("season").agg({"total_goal":"mean"})
s = data.groupby("season").agg({"total_goal":"std"})
x = data.groupby("season").agg({"total_goal":"max"})
xx = a.merge(m,left_index=True,right_index=True,how="left")
yy = s.merge(x,left_index=True,right_index=True,how="left")
x_y = xx.merge(yy,left_index=True,right_index=True,how="left").reset_index()
x_y = x_y.rename(columns={'total_goal_x_x':"goals", 'total_goal_y_x':"mean", 'total_goal_x_y':"std",'total_goal_y_y':"max"})
import itertools
cols = [ 'goals', 'mean', 'std', 'max' ]
length = len(cols)
cs   = ["r","g","b","c"] 
plt.figure(figsize=(15,15))

for i,j,k in itertools.zip_longest(cols,range(length),cs):
    plt.subplot(length,length/length,j+1)
    sns.pointplot(x_y["season"],x_y[i],color=k)
    plt.title(i)
    plt.subplots_adjust(hspace =.3)


# INTERACTION BETWEEN TEAMS
# ==

# In[ ]:


g = nx.from_pandas_dataframe(data,"home_team_sname","away_team_sname")
fig = plt.figure(figsize=(12,10))
nx.draw_kamada_kawai(g,with_labels = True)
plt.title("INTERACTION BETWEEN TEAMS")
fig.set_facecolor("moccasin")


# In[ ]:


def label(data):
    if data["home_team_goal"] > data["away_team_goal"]:
        return data["home_team_lname"]
    elif data["away_team_goal"] > data["home_team_goal"]:
        return data["away_team_lname"]
    elif data["home_team_goal"] == data["away_team_goal"]:
        return "DRAW"


# In[ ]:


data["win"] = data.apply(lambda data:label(data),axis=1)


# In[ ]:


def lab(data):
    if data["home_team_goal"] > data["away_team_goal"]:
        return "HOME TEAM WIN"
    elif data["away_team_goal"] > data["home_team_goal"]:
        return "AWAY TEAM WIN"
    elif data["home_team_goal"] == data["away_team_goal"]:
        return "DRAW"


# In[ ]:


data["outcome_side"] = data.apply(lambda data:lab(data),axis = 1)


# In[ ]:


def labe(data):
    if data["home_team_goal"] < data["away_team_goal"]:
        return data["home_team_lname"]
    elif data["away_team_goal"] < data["home_team_goal"]:
        return data["away_team_lname"]
    elif data["home_team_goal"] == data["away_team_goal"]:
        return "DRAW"
    


# In[ ]:


data["lost"] = data.apply(lambda data:labe(data),axis=1)


# PROPORTION OF GAME OUTCOMES
# ==

# In[ ]:


plt.figure(figsize=(6,6))
data["outcome_side"].value_counts().plot.pie(autopct = "%1.0f%%",colors =sns.color_palette("rainbow",3),wedgeprops = {"linewidth":2,"edgecolor":"white"})
my_circ = plt.Circle((0,0),.7,color = "white")
plt.gca().add_artist(my_circ)
plt.title("PROPORTION OF GAME OUTCOMES")
plt.show()


# # TOP WINNING  & LOSING TEAMS

# In[ ]:


win = data["win"].value_counts()[1:].reset_index()
lost = data["lost"].value_counts()[1:].reset_index()
plt.figure(figsize=(13,14))
plt.subplot(121)
ax = sns.barplot(win["win"][:30],win["index"][:30],palette="magma")
plt.title(" TOP WINNING TEAMS")
for i,j in enumerate(win["win"][:30]):
    ax.text(.7,i,j,color = "white",weight = "bold")
plt.subplot(122)
ax = sns.barplot(lost["lost"][:30],lost["index"][:30],palette="jet_r")
plt.title(" TOP TEAMS that Lost")
plt.subplots_adjust(wspace = .3)
for i,j in enumerate(lost["lost"][:30]):
    ax.text(.7,i,j,color = "black",weight = "bold")


# In[ ]:


f = xy.merge(win,left_on="team",right_on="index",how="left")
f = f.drop("index",axis =1)
f = f.rename(columns={"outcome":"wins"})
f = f.merge(lost,left_on="team",right_on="index",how="left")
f = f.drop("index",axis =1)


# In[ ]:


dr = data[data["outcome_side"] == "DRAW"][["home_team_lname","away_team_lname"]]
l  = dr["home_team_lname"].value_counts().reset_index()
v  = dr["away_team_lname"].value_counts().reset_index()
l  = l.rename(columns={'index':"team", 'home_team_lname':"draw"})
v  = v.rename(columns={'index':"team", 'away_team_lname':"draw"})
lv = pd.concat([l,v])
lv = lv.groupby("team")["draw"].sum().reset_index()
f = f.merge(lv,left_on="team",right_on="team",how ="left")


# # PERFORMANCE BY TOP TEAMS

# In[ ]:


f = f.sort_values(by="goals",ascending=False)
plt.figure(figsize=(14,5))
sns.barplot("team","matches",data=f[:20],color="b",label ="MATCHES PLAYED")
sns.barplot("team","win",data=f[:20],color="g",label ="MATCHES WON")
sns.barplot("team","lost",data=f[:20],color="r",label ="MATCHES LOST")
plt.xticks(rotation = 70)
plt.legend(loc="best")
plt.title("PERFORMANCE BY TOP TEAMS")
plt.show()


# # WIN VS LOST VS DRAW

# In[ ]:


from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(10,10))
ax  = fig.add_subplot(111,projection ="3d")
ax.scatter(f["win"],f["lost"],f["draw"],s=f["matches"]*3,alpha=.4,linewidth =1,edgecolor= "k",c="lime")
ax.set_xlabel("wins")
ax.set_ylabel("lost")
ax.set_zlabel("draw")
plt.title("WIN VS LOST VS DRAW")
plt.show()


# # AREA PLOT FOR MATCH ATTRIBUTES

# In[ ]:


plt.figure(figsize=(13,5))
cols = ["matches","win","lost","draw"]
c    = ["b","orange","lime","m"]
length = len(cols)
for i,j,k in itertools.zip_longest(cols,range(length),c):
    plt.stackplot(f.index,f[i],alpha=.6,color = k,labels=[i])
    plt.axhline(f[i].mean(),color=k,linestyle="dashed",label="mean")
    plt.legend(loc="best")
    plt.title("AREA PLOT FOR MATCH ATTRIBUTES")


# # TOP TEAMS BY LEAGUES

# In[ ]:


x = pd.DataFrame(data.groupby(["league","win"])["win"].count())
x = x.rename(columns={"win":"team"}).reset_index()
x = x.rename(columns={"win":"team","team":"win"})
x = x.sort_values(by="win",ascending=False)
x = x[x["team"] != "DRAW"]
x = x.drop_duplicates(subset=["league"],keep="first")
plt.figure(figsize=(10,5))
ax =sns.barplot(x["win"],x["league"],palette="cool")
for i,j in enumerate(x["team"]):
    ax.text(.7,i,j,weight = "bold",fontsize = 12)
plt.title("TOP TEAMS BY LEAGUES")
plt.show()


# # MATCHES PLAYED IN EACH LEAGUE VS TOTAL GOALS SCORED

# In[ ]:


data.groupby(["league"]).agg({"match_api_id":"count","total_goal":"sum"}).plot(kind="barh",stacked =True,figsize=(8,6))
plt.title("# MATCHES PLAYED IN EACH LEAGUE VS TOTAL GOALS SCORED")
plt.show()


# # PROPORTION OF  MATCHES PLAYED AND GOALS SCORED IN LEAGUES

# In[ ]:


plt.figure(figsize=(14,7))
plt.subplot(121)
data.groupby(["league"]).agg({"match_api_id":"count","total_goal":"sum"})["match_api_id"].plot.pie(colors=sns.color_palette("seismic",10),autopct="%1.0f%%",wedgeprops={"linewidth":2,"edgecolor":"white"})
plt.ylabel("")
my_circ = plt.Circle((0,0),.7,color ="white")
plt.gca().add_artist(my_circ)
plt.title("PROPORTION OF MATCHES PLAYED IN LEAGUES")
plt.subplot(122)
data.groupby(["league"]).agg({"match_api_id":"count","total_goal":"sum"})["total_goal"].plot.pie(colors=sns.color_palette("seismic",10),autopct="%1.0f%%",wedgeprops={"linewidth":2,"edgecolor":"white"})
plt.ylabel("")
my_circ = plt.Circle((0,0),.7,color ="white")
plt.gca().add_artist(my_circ)
plt.title("PROPORTION OF GOALS SCORED IN LEAGUES")
plt.show()


# # TOP TEAMS

# In[ ]:


from wordcloud import WordCloud
import nltk
wrd = data[data["win"] != "DRAW"]["win"].to_frame()
wrd = wrd["win"].value_counts()[wrd["win"].value_counts() > 100].keys().str.replace(" ","")
wrd = pd.DataFrame(wrd)
wc = WordCloud(background_color="black",scale =2,colormap="flag").generate(str(wrd[0]))
plt.figure(figsize=(13,8))
plt.imshow(wc,interpolation="bilinear")
plt.axis("off")
plt.title("TOP TEAMS")
plt.show()


# # GOALS SCORED IN EACH SEASON OF LEAUGES

# In[ ]:


pd.pivot_table(index="season",columns="league",values="total_goal",data=data,aggfunc="sum").plot(kind = "barh",stacked = True,figsize =(12,6),colors =sns.color_palette("rainbow",11))
plt.title("GOALS SCORED IN EACH SEASON OF LEAUGES")
plt.show()


# # HOME GOALS SCORED BY TOP TEAMS BY SEASON

# In[ ]:


i = data["win"].value_counts()[1:25].index
t= pd.pivot_table(index="home_team_lname",columns="season",values="home_team_goal",data=data,aggfunc="sum")
t=t[t.index.isin(i)]
t.plot(kind="barh",stacked=True,figsize=(12,8),colors=sns.color_palette("prism",11))
plt.title("HOME GOALS SCORED BY TOP TEAMS BY SEASON")
plt.show()


# # AWAY GOALS SCORED BY TOP TEAMS BY SEASON# 

# In[ ]:


i = data["win"].value_counts()[1:25].index
t= pd.pivot_table(index="away_team_lname",columns="season",values="away_team_goal",data=data,aggfunc="sum")
t=t[t.index.isin(i)]
t.plot(kind="barh",stacked=True,figsize=(12,8),colors=sns.color_palette("prism",11))
plt.title("HOME GOALS SCORED BY TOP TEAMS BY SEASON")
plt.show()


# # "COUNT OF MATCHES PLAYED BETWEEN TOP TEAMS"

# In[ ]:


i = data["win"].value_counts()[1:50].index
c = data[(data["home_team_lname"].isin(i)) & (data["away_team_lname"].isin(i))]
d = pd.crosstab(c["home_team_lname"],c["away_team_lname"])
plt.figure(figsize=(15,10))
sns.heatmap(d,annot=True,linecolor="k",linewidths=.1,cmap=sns.color_palette("inferno"))
plt.title("COUNT OF MATCHES PLAYED BETWEEN TOP TEAMS")
plt.show()


# #WINNERS OF EACH SEASON BY LEAGUE
# ==

# In[ ]:


nw = data[["season","league","win"]]
nw["team"] = nw["win"]
nw = nw.groupby(["season","league","team"])["win"].count().reset_index().sort_values(by=["season","league","win"],ascending =False)
nw = nw[nw["team"] != "DRAW"]
nw = nw.drop_duplicates(subset=["season","league"],keep="first").sort_values(by=["league","season"],ascending =True)

plt.figure(figsize=(13,25))
plt.subplot(621)
lg = nw[nw["league"] == "Belgium Jupiler League"]
ax = sns.barplot(lg["win"],lg["season"],palette="cool")
for i,j in enumerate(lg["team"]):
    ax.text(.7,i,j,weight = "bold")
plt.title("Belgium Jupiler League")
plt.xlabel("")
plt.ylabel("")

plt.subplot(622)
lg = nw[nw["league"] == "England Premier League"]
ax = sns.barplot(lg["win"],lg["season"],palette="magma")
for i,j in enumerate(lg["team"]):
    ax.text(.7,i,j,weight = "bold",color="white")
plt.title("England Premier League")
plt.xlabel("")
plt.ylabel("")

plt.subplot(623)
lg = nw[nw["league"] == 'Spain LIGA BBVA']
ax = sns.barplot(lg["win"],lg["season"],palette="rainbow")
for i,j in enumerate(lg["team"]):
    ax.text(.7,i,j,weight = "bold")
plt.title('Spain LIGA BBVA')
plt.xlabel("")
plt.ylabel("")

plt.subplot(624)
lg = nw[nw["league"] == 'France Ligue 1']
ax = sns.barplot(lg["win"],lg["season"],palette="summer")
for i,j in enumerate(lg["team"]):
    ax.text(.7,i,j,weight = "bold",color = "white")
plt.title('France Ligue 1')
plt.xlabel("")
plt.ylabel("")

plt.subplot(625)
lg = nw[nw["league"] == 'Germany 1. Bundesliga']
ax = sns.barplot(lg["win"],lg["season"],palette="winter")
for i,j in enumerate(lg["team"]):
    ax.text(.7,i,j,weight = "bold")
plt.title('Germany 1. Bundesliga')
plt.xlabel("")
plt.ylabel("")

plt.subplot(626)
lg = nw[nw["league"] == 'Italy Serie A']
ax = sns.barplot(lg["win"],lg["season"],palette="husl")
for i,j in enumerate(lg["team"]):
    ax.text(.7,i,j,weight = "bold")
plt.title('Italy Serie A')
plt.xlabel("")
plt.ylabel("")
plt.show()


# In[ ]:


plt.figure(figsize=(13,25))
plt.subplot(621)
lg = nw[nw["league"] == 'Netherlands Eredivisie']
ax = sns.barplot(lg["win"],lg["season"],palette="Blues")
for i,j in enumerate(lg["team"]):
    ax.text(.7,i,j,weight = "bold")
plt.title('Netherlands Eredivisie')
plt.xlabel("")
plt.ylabel("")

plt.subplot(622)
lg = nw[nw["league"] == 'Poland Ekstraklasa']
ax = sns.barplot(lg["win"],lg["season"],palette="winter")
for i,j in enumerate(lg["team"]):
    ax.text(.7,i,j,weight = "bold")
plt.title('Poland Ekstraklasa')
plt.xlabel("")
plt.ylabel("")

plt.subplot(623)
lg = nw[nw["league"] == 'Portugal Liga ZON Sagres']
ax = sns.barplot(lg["win"],lg["season"],palette="rainbow")
for i,j in enumerate(lg["team"]):
    ax.text(.7,i,j,weight = "bold")
plt.title('Portugal Liga ZON Sagres')
plt.xlabel("")
plt.ylabel("")

plt.subplot(624)
lg = nw[nw["league"] == 'Scotland Premier League']
ax = sns.barplot(lg["win"],lg["season"],palette="Greens")
for i,j in enumerate(lg["team"]):
    ax.text(.7,i,j,weight = "bold")
plt.title('Scotland Premier League')
plt.xlabel("")
plt.ylabel("")

plt.subplot(625)
lg = nw[nw["league"] == 'Switzerland Super League']
ax = sns.barplot(lg["win"],lg["season"],palette="cool")
for i,j in enumerate(lg["team"]):
    ax.text(.7,i,j,weight = "bold")
plt.title('Switzerland Super League')
plt.xlabel("")
plt.ylabel("")
plt.show()


# # "GOALS SCORED IN LEAGUES"
# 

# In[ ]:


plt.figure(figsize=(13,5))
sns.violinplot(data["league"],data["total_goal"],palette="rainbow")
plt.title("GOALS SCORED IN LEAGUES")
plt.xticks(rotation = 60)
plt.show()


# # TOP LEAGUE WINNERS

# In[ ]:


plt.figure(figsize=(8,10))
ax = sns.countplot(y=nw["team"],order=nw["team"].value_counts().index,palette="plasma")
plt.title("TOP LEAGUE WINNERS")
for i,j in enumerate(nw["team"].value_counts().values):
    ax.text(.2,i,j,color = "white",weight="bold")


# # PLOTTING PLAYER ATTRIBUTES DISPLAYING WHITE NOISE

# In[ ]:


player
player["weight_kg"] = player["weight"] * 0.453592
player["height_m"]  = player["height"] / 100
player["bmi"]       = player["weight_kg"]/(player["height_m"] * player["height_m"])
player["year"]  = pd.DatetimeIndex(player["birthday"]).year
player["age"]   = 2018 - player["year"]

cols  = ["bmi","weight_kg","height_m","age"]
length = len(cols)
c = ["b","r","g","c"]
plt.figure(figsize=(13,13))
for i,j,k in itertools.zip_longest(cols,range(length),c):
    plt.subplot(4,1,j+1)
    player[i].plot(color = k ,linewidth =.2,label = i)
    plt.axhline(player[i].mean(),color = "k",linestyle = "dashed",label="mean")
    plt.legend(loc="best")
    plt.title(i)


# 
# # NORMALLY DISTRIBUTED PLAYER ATTRIBUTES

# In[ ]:


cols  = ["bmi","weight_kg","height_m","age"]
length = len(cols)
c = ["b","r","k","c"]
plt.figure(figsize=(13,10))

for i,j,k in itertools.zip_longest(cols,range(length),c):
    plt.subplot(2,2,j+1)
    sns.distplot(player[i],color=k)
    plt.axvline(player[i].mean(),color = "k",linestyle = "dashed",label="mean")
    plt.legend(loc="best")
    plt.title(i)
    plt.xlabel("")


# In[ ]:


player["year"]  = pd.DatetimeIndex(player["birthday"]).year
player["age"]   = 2018 - player["year"]


# In[ ]:


print ("PLAYER ATTRIBUTES")
print ("===================================================================================================")
print ("Oldest Player is",player.loc[player["age"].idxmax()]["player_name"],"of age ",player["age"].max(),"years")
print ("Youngest Players are",list(player[player["age"]==19]["player_name"]),"of age",player["age"].min(),"years")
print ("Tallest Player is",player.loc[player["height_m"].idxmax()]["player_name"],"of height",player["height_m"].max(),"meters")
print ("Shortest Player is",player.loc[player["height_m"].idxmin()]["player_name"],"of height",player["height_m"].min(),"meters")
print ("Player with highest weight are",list(player[player["weight_kg"] == 110.222856]["player_name"]),"of height",player["weight_kg"].max(),"kilograms")
print ("Player with lowest weight is",player.loc[player["weight_kg"].idxmin()]["player_name"],"of height",player["weight_kg"].min(),"kilograms")
print ("Player with Highest Body Mass Index is",player.loc[player["bmi"].idxmax()]["player_name"],"of",player["bmi"].max(),"kg/m2")
print ("Player with lowest Body Mass Index is",player.loc[player["bmi"].idxmin()]["player_name"],"of",player["bmi"].min(),"kg/m2")


# # Player attributes Summary# 

# In[ ]:


plt.figure(figsize=(13,6))
sns.heatmap(player[["height","weight","weight_kg","age","bmi"]].describe()[1:].transpose(),annot=True,fmt="f",linecolor="white",linewidths=2)
plt.title("Player attributes Summary")
plt.show()


# # CORRELATION BETWEEN VARIABLES

# In[ ]:


correlation = player.corr()
plt.figure(figsize=(13,8))
sns.heatmap(correlation,annot=True,fmt="f",linecolor="k",linewidths=2,cmap =sns.color_palette("Set2"))
plt.title("CORRELATION BETWEEN VARIABLES")
plt.show()


# # "Density Plot between height & weight"# 

# In[ ]:


sns.jointplot(player["height"],player["weight"],kind="kde",color="b")
plt.title("Density Plot between height & weight")
plt.show()


# # PAIR PLOT BETWEEN VARIABLES OF PLAYERS

# In[ ]:


sns.pairplot(player)


# # HEXBIN PLOT FOR AGE VS WEIGHT,HEIGHT,BMI

# In[ ]:


cols = [ 'weight_kg', 'height_m', 'bmi']
length=len(cols)
plt.figure(figsize=(13,5))
for i,j in itertools.zip_longest(cols,range(length)):
    plt.subplot(1,3,j+1)
    plt.hexbin(player["age"],player[i],cmap="hot",gridsize=(15,15))
    plt.xlabel("age")
    plt.ylabel(i)
    plt.colorbar()
    plt.title(i)


# # TREND IN PLAYER NAMES STARTING LETTERS

# In[ ]:


first = pd.DataFrame(player["player_name"].str.split(" ").str[0].str.upper().str[:1].value_counts())
last = pd.DataFrame(player["player_name"].str.split(" ").str[1].str.upper().str[:1].value_counts())
lets = first.merge(last,left_index=True,right_index=True,how="left").reset_index()
lets = lets.rename(columns={"index":"letter","player_name_x":'first_name',"player_name_y":"last_name"}).sort_values(by="letter",ascending = True)
plt.figure(figsize=(13,8))
plt.subplot(211)
sns.barplot(lets["letter"],lets["first_name"],color="b")
plt.ylabel("")
plt.xlabel("")
plt.title("VALUE COUNT FOR FIRST NAME FIRST LETTER")
plt.subplot(212)
sns.barplot(lets["letter"],lets["last_name"],color="r")
plt.ylabel("")
plt.title("VALUE COUNT FOR LAST NAME FIRST LETTER")
plt.show()


# # NETWORK CONNECTION BETWEEN FIRST LETTER OF FIRST & LAST NAMES# 

# In[ ]:


first = pd.DataFrame(player["player_name"].str.split(" ").str[0].str.upper().str[:1])
last = pd.DataFrame(player["player_name"].str.split(" ").str[1].str.upper().str[:1])
x = first.merge(last,left_index=True,right_index=True)
x = x[x["player_name_y"].notnull()]
g = nx.from_pandas_dataframe(x,"player_name_x","player_name_y",create_using=nx.Graph())
fig = plt.figure(figsize=(10,10))
nx.draw_circular(g,with_labels=True,edge_color = "k",node_color="red",node_size=1000)
fig.set_facecolor("lightblue")
plt.title("NETWORK CONNECTION BETWEEN FIRST LETTER OF FIRST & LAST NAMES")
plt.show()


# In[ ]:


player_info = player_attributes.merge(player,left_on="player_api_id",right_on="player_api_id",how="left")


# In[ ]:


i =["id_x","id_y",'player_fifa_api_id_y','height', 'weight', 'weight_kg', 'height_m', 'bmi', 'year','age','birthday']
player_info = player_info[[x for x in player_info.columns if x not in i]]
player_info.columns


# In[ ]:


player_info["date"] = pd.to_datetime(player_info["date"],format="%Y-%m-%d")


# # PLAYERS WHO PLAYED HIGHEST MATCHES

# In[ ]:


ax = player_info["player_name"].value_counts().sort_values()[-20:].plot(kind="barh",figsize=(10,8),color="b",width=.9)
for i,j in enumerate(player_info["player_name"].value_counts().sort_values()[-20:].values):
    ax.text(.7,i,j,weight = "bold",color="white")
ax.set_title("PLAYERS WHO PLAYED HIGHEST MATCHES")
plt.show()


# # WORD CLOUD FOR PLAYER NAMES

# In[ ]:


play = player_info["player_name"].unique()
import nltk
from PIL import Image
img = np.array(Image.open("../input/picture-wrd/z.jpg"))
wc = WordCloud(background_color="black",scale=2,mask=img,colormap="cool",max_words=100000).generate(" ".join(play))
fig = plt.figure(figsize=(15,15))
plt.imshow(wc,interpolation="bilinear")
plt.axis("off")
plt.title("WORD CLOUD FOR PLAYER NAMES")
plt.show()


# # TOP RATED PLAYERS

# In[ ]:


top_rated = player_info[player_info["overall_rating"]  > 88 ]["player_name"].value_counts().index
import nltk
wc = WordCloud(background_color="white",scale=2).generate(" ".join(top_rated))
fig = plt.figure(figsize=(15,8))
plt.imshow(wc,interpolation="bilinear")
plt.axis("off")
plt.title("TOP RATED PLAYERS")
plt.show()


# # PREFERRED FOOT BY ALL PLAYERS# 

# In[ ]:


plt.figure(figsize=(12,6))
plt.subplot(121)
player_info.groupby(["player_api_id","preferred_foot"])["overall_rating"].mean().reset_index()["preferred_foot"].value_counts().plot.pie(autopct = "%1.0f%%",shadow = True,wedgeprops={"linewidth":2,"edgecolor":"white"},colors=["grey","r"],explode=[0,.1],startangle=45)
plt.title("PREFERRED FOOT BY ALL PLAYERS")
plt.subplot(122)
t_f = player_info.groupby(["player_api_id","preferred_foot"])["overall_rating"].mean().reset_index()
t_f[t_f["overall_rating"] > 80]["preferred_foot"].value_counts().plot.pie(autopct = "%1.0f%%",shadow = True,wedgeprops={"linewidth":2,"edgecolor":"white"},colors=["grey","r"],explode=[0,.1],startangle=45)
plt.title("PREFERRED FOOT BY TOP RATED PLAYERS")
plt.show()


# In[ ]:


player_info.columns


# # TOP RATED PLAYERS STATS

# In[ ]:


top_rated = player_info[player_info["overall_rating"]  > 91 ]
top_rated = top_rated[['player_name','player_api_id', 'date', 'overall_rating','potential','finishing','acceleration','ball_control' ,'penalties']]
top_rated = top_rated.groupby("player_name").agg({'overall_rating':"mean",'potential':"mean",'finishing':"mean",'acceleration':"mean",'ball_control':"mean" ,'penalties':"mean"})
top_rated.plot(kind="bar",width=.6,figsize=(15,5),colors=["r","b","lime","gold","c","m"],alpha=.5)
plt.title("TOP PLAYER STATS")
plt.xticks(rotation = 0)
plt.legend(loc ="lower center")
plt.show()


# # PLAYER COMPARATOR

# In[ ]:


idx  = "player_api_id"
cols = ['overall_rating','potential', 'crossing', 'finishing', 'heading_accuracy',
       'short_passing', 'volleys', 'dribbling', 'curve', 'free_kick_accuracy',
       'long_passing', 'ball_control', 'acceleration', 'sprint_speed',
       'agility', 'reactions', 'balance', 'shot_power', 'jumping', 'stamina',
       'strength', 'long_shots', 'aggression', 'interceptions', 'positioning',
       'vision', 'penalties', 'marking', 'standing_tackle', 'sliding_tackle']

def player_comparator(player1,player2):
    
    x1 = player_info[player_info["player_name"] == player1]
    x1 = x1.groupby(["player_name"])[cols].mean()
    
    x2 = player_info[player_info["player_name"] == player2]
    x2 = x2.groupby(["player_name"])[cols].mean()
    
    z  = pd.concat([x1,x2]).transpose().reset_index()
    z  = z.rename(columns={"index":"attributes",player1:player1,player2:player2})
    
    plt.figure(figsize=(13,11))
    plt.subplot(121)
    ax = sns.barplot(y= z["attributes"],x = z[player1],palette="cool")
    plt.title(player1,fontsize = 20)
    plt.ylabel("")
    for i,j in enumerate(round(z[player1],2)):
        ax.text(.7,i,j,weight = "bold")
        
    plt.subplot(122)
    ax = sns.barplot(y= z["attributes"],x = z[player2],palette="cool")
    plt.title(player2,fontsize = 20)
    plt.ylabel("")
    for i,j in enumerate(round(z[player2],2)):
        ax.text(.7,i,j,weight = "bold")
    plt.subplots_adjust(wspace = .4)
        


# # MESSI VS RONALDO

# In[ ]:


player_comparator('Lionel Messi','Cristiano Ronaldo')


# In[ ]:


player_comparator( 'Ronaldinho','Wayne Rooney')


# In[ ]:


player_comparator('Zlatan Ibrahimovic','Cristiano Ronaldo')


# In[ ]:


goal_keeper = player_info[["player_api_id",'gk_diving', 'gk_handling', 'gk_kicking', 'gk_positioning','gk_reflexes', 'player_name',"overall_rating"]]
goal_keeper = goal_keeper[(goal_keeper["gk_diving"]>75) & (goal_keeper["gk_handling"]>75)
                          & (goal_keeper["gk_kicking"]>75)& (goal_keeper["gk_positioning"]>75)
                          & (goal_keeper["gk_reflexes"]>75) ]
goal_keeper = goal_keeper.groupby(["player_api_id","player_name"])[['gk_diving', 'gk_handling', 'gk_kicking',
                                    'gk_positioning', 'gk_reflexes', 'overall_rating']].mean()
goal_keeper = goal_keeper.sort_values(by="overall_rating",ascending =False).reset_index()
goal_keeper.index = goal_keeper["player_name"]


# # TOP GOAL KEEPERS STATS

# In[ ]:


goal_keeper[['gk_diving', 'gk_handling', 'gk_kicking',
       'gk_positioning', 'gk_reflexes']][:8].plot(kind = "bar",figsize=(15,5),color =["r","b","k","lime","yellow"])
plt.xticks(rotation =0)
plt.legend(loc ="lower center")
plt.title("# TOP GOAL KEEPERS STATS")
plt.show()


# In[ ]:


col = ["player_api_id",'overall_rating','potential','crossing', 'finishing', 'heading_accuracy',
       'short_passing', 'volleys', 'dribbling', 'curve', 'free_kick_accuracy',
       'long_passing', 'ball_control', 'acceleration', 'sprint_speed',
       'agility', 'reactions', 'balance', 'shot_power', 'jumping', 'stamina',
       'strength', 'long_shots', 'aggression', 'interceptions', 'positioning',
       'vision', 'penalties', 'marking', 'standing_tackle', 'sliding_tackle',
       'gk_diving', 'gk_handling', 'gk_kicking', 'gk_positioning',
       'gk_reflexes']
x = player_attributes[col]
x = x.groupby("player_api_id")[col].mean()
x = x.drop("player_api_id",axis =1).reset_index().drop("player_api_id",axis=1)


# # scatter plot for overall rating and player attributes

# In[ ]:


col = ['potential','crossing', 'finishing', 'heading_accuracy',
       'short_passing', 'volleys', 'dribbling', 'curve', 'free_kick_accuracy',
       'long_passing', 'ball_control', 'acceleration', 'sprint_speed',
       'agility', 'reactions', 'balance', 'shot_power', 'jumping', 'stamina',
       'strength', 'long_shots', 'aggression', 'interceptions', 'positioning',
       'vision', 'penalties', 'marking', 'standing_tackle', 'sliding_tackle',
       'gk_diving', 'gk_handling', 'gk_kicking', 'gk_positioning',
       'gk_reflexes' ]
length = len(col)
plt.figure(figsize=(15,15))
for i,j in itertools.zip_longest(col,range(length)):
    plt.subplot(5,7,j+1)
    plt.scatter(x["overall_rating"],x[i],s=.01,color="Orange")
    plt.title(i)
    plt.subplots_adjust(hspace =.4)


# # RADAR CHART FOR TOP PLAYERS AND THEIR ATTRIBUTES

# In[ ]:


from math import pi
string = ['Andres Iniesta','Cristiano Ronaldo', 'Lionel Messi','Luis Suarez','Neymar', 'Ronaldinho','Wayne Rooney','Zlatan Ibrahimovic']
play   = player_info[player_info["player_name"].isin(string)]
cols   = ["player_name",'overall_rating','potential', 'crossing', 'finishing', 'heading_accuracy',
       'short_passing', 'volleys', 'dribbling', 'curve', 'free_kick_accuracy',
       'long_passing', 'ball_control', 'acceleration', 'sprint_speed',
       'agility', 'reactions', 'balance', 'shot_power', 'jumping', 'stamina',
       'strength', 'long_shots', 'aggression', 'interceptions', 'positioning',
       'vision', 'penalties', 'standing_tackle', 'sliding_tackle']
play = play[cols]
play = play.groupby("player_name")[cols].mean().reset_index()

num = [0,1,2,3,4,5,6,7]
c = ["r","y","b","c","orange","m","k","lime"]
plt.figure(figsize=(15,15))
for i,j,k in itertools.zip_longest(num,range(len(num)),c):
    plt.subplot(3,3,j+1,projection="polar")
    cats = list(play)[1:]
    N    = len(cats)
    values = play.loc[i].drop("player_name").values.flatten().tolist()
    values += values[:1]
    values
    angles = [n / float(N)*2*pi for n in range(N)]
    angles += angles[:1]
    angles
    
    plt.xticks(angles[:-1],cats,color="k",size=7)
    plt.ylim([0,100])
    plt.plot(angles,values,color=k,linewidth=2,linestyle="solid")
    plt.fill(angles,values,color=k,alpha=0.5)
    plt.title(play["player_name"][i],color="r")
    plt.subplots_adjust(wspace=.4,hspace=.4)


# # PREFFERED FOOT BY ATTACKERS VS DEFENDERS

# In[ ]:


x = player_info[player_info["attacking_work_rate"] == "low"]
x = x.groupby(["player_api_id","player_name","preferred_foot"])["date"].count().reset_index()
plt.figure(figsize=(12,6))
plt.subplot(121)
x["preferred_foot"].value_counts().plot.pie(autopct = "%1.0f%%",shadow = True,wedgeprops={"linewidth":2,"edgecolor":"white"},colors=["grey","orange"],explode=[0,.1],startangle=50)
plt.ylabel("")
plt.title("Preferred foot by players with High attacking rate")

x = player_info[player_info["defensive_work_rate"] == "low"]
x = x.groupby(["player_api_id","player_name","preferred_foot"])["date"].count().reset_index()
plt.subplot(122)
x["preferred_foot"].value_counts().plot.pie(autopct = "%1.0f%%",shadow = True,wedgeprops={"linewidth":2,"edgecolor":"white"},colors=["grey","orange"],explode=[0,.1],startangle=45)
plt.ylabel("")
plt.title("Preferred foot by players with High defensive rate")
plt.show()


# In[ ]:


team_attributes
teams
team_info =  team_attributes.merge(teams,left_on="team_api_id",right_on="team_api_id",how="left")
team_info.head()


# In[ ]:


team_info = team_info.drop(['id_x','id_y', 'team_fifa_api_id_y'],axis=1)


# In[ ]:


team_info["date"] = pd.to_datetime(team_info["date"],format="%Y-%m-%d")


# In[ ]:


columns= team_info.columns
cat_col= columns[columns.str.contains("Class")].tolist()
num_col= [x for x in team_info.columns if x not in columns[columns.str.contains("Class")].tolist()+["team_api_id"]+['team_fifa_api_id_x']+["date"]+['team_long_name']+[ 'team_short_name']]
categorical_team_info = team_info[cat_col+["team_api_id"]+['team_fifa_api_id_x']+["date"]+['team_long_name']+[ 'team_short_name']]
numerical_team_info   = team_info[num_col+["team_api_id"]+['team_fifa_api_id_x']+["date"]+['team_long_name']+[ 'team_short_name']]


# # TOP TEAMS BY TEAM ATTRIBUTES

# In[ ]:


numerical_team_info
n = numerical_team_info.groupby("team_long_name")[num_col].mean().reset_index()
cols = [x for x in n.columns if x not in ["team_long_name"]]
length = len(cols)
plt.figure(figsize=(15,15))
for i,j in itertools.zip_longest(cols,range(length)):
    plt.subplot(length/3,length/3,j+1)
    ax = sns.barplot(i,"team_long_name",data=n.sort_values(by=i,ascending=False)[:7],palette="winter")
    plt.title(i)
    plt.subplots_adjust(wspace = .6,hspace =.3)
    plt.ylabel("")
    for i,j in enumerate(round(n.sort_values(by = i,ascending=False)[i][:7],2)):
        ax.text(.7,i,j,weight = "bold",color="white") 


# # DISTRIBUTION OF TEAM ATTRIBUTES AMONG TEAMS

# In[ ]:


from scipy.stats import mode

c = categorical_team_info.groupby("team_long_name").agg({"buildUpPlaySpeedClass":lambda x:mode(x)[0],
                                                    "buildUpPlayDribblingClass":lambda x:mode(x)[0],
                                                    'buildUpPlayPassingClass':lambda x:mode(x)[0],
                                                    'buildUpPlayPositioningClass':lambda x:mode(x)[0],
                                                    'chanceCreationPassingClass':lambda x:mode(x)[0],
                                                    'chanceCreationCrossingClass':lambda x:mode(x)[0],
                                                     'chanceCreationShootingClass':lambda x:mode(x)[0],
                                                     'chanceCreationPositioningClass':lambda x:mode(x)[0],
                                                     'defencePressureClass':lambda x:mode(x)[0],
                                                     'defenceAggressionClass':lambda x:mode(x)[0],
                                                     'defenceTeamWidthClass':lambda x:mode(x)[0],
                                                     'defenceDefenderLineClass':lambda x:mode(x)[0]}).reset_index()
cat_col
plt.figure(figsize=(15,20))
for i,j in itertools.zip_longest(cat_col,range(len(cat_col))):
    plt.subplot(4,3,j+1)
    plt.pie(c[i].value_counts().values,labels=c[i].value_counts().keys(),wedgeprops={"linewidth":3,"edgecolor":"k"},
           colors=sns.color_palette("Pastel1"),autopct = "%1.0f%%")
    my_circ = plt.Circle((0,0),.7,color="white")
    plt.gca().add_artist(my_circ)
    plt.title(i)
    plt.xlabel("")


# # TEAM COMPARATOR

# In[ ]:


def team_comparator(team1,team2):
    
    team_list = [team1,team2]
    length    = len(team_list)
    cr        = ["b","r"]
    fig = plt.figure(figsize=(15,8))
    plt.subplot(111,projection= "polar")
    
    for i,j,k in itertools.zip_longest(team_list,range(length),cr):
        cats = num_col
        N    = len(cats)
        
        values = n[n["team_long_name"] ==  i][cats].values.flatten().tolist()
        values += values[:1]
        
        angles = [n/float(N)*2*pi for n in range(N)]
        angles += angles[:1]
        
        plt.xticks(angles[:-1],cats,color="k",fontsize=15)
        plt.plot(angles,values,linewidth=3,color=k)
        plt.fill(angles,values,color = k,alpha=.4,label = i)
        plt.legend(loc="upper right",frameon =True,prop={"size":15}).get_frame().set_facecolor("lightgrey")
        fig.set_facecolor("w")
        fig.set_edgecolor("k")
        plt.title("TEAM COMPARATOR",fontsize=30,color="tomato")
        


# In[ ]:


team_comparator("Real Madrid CF","FC Barcelona")


# In[ ]:


team_comparator("Manchester United","Liverpool")


# In[ ]:





# In[ ]:




