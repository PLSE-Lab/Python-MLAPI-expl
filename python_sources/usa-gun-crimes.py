#!/usr/bin/env python
# coding: utf-8

# # USA  GUN CRIMES - (Exploratory Data Analysis)

# In[ ]:


#Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import itertools
import warnings
warnings.filterwarnings("ignore")
from wordcloud import WordCloud
import io
import base64
from matplotlib import rc,animation
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.basemap import Basemap
import folium
import folium.plugins
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls


# # Data- overview

# In[ ]:


data = pd.read_csv(r"../input/gun-violence-data/gun-violence-data_01-2013_03-2018.csv")


# In[ ]:


data.head()


# In[ ]:


print ("columns in data :\n",data.columns)
print ("\n data - shape :",data.shape)


# In[ ]:


#DATE TYPE CONVERSION
data["date"] = pd.to_datetime(data["date"],format="%Y-%m-%d")
data = data.drop(["incident_url","incident_url_fields_missing"],axis=1)


# In[ ]:


#EXTRACTING STRING INFO FROM SOURCES
data["source_url"] = data["source_url"].str.split("//").str[1].str.split("/").str[0]
data["source_url"] = data["source_url"].str.replace("www.|.com|.net|.co|.gov|.org","")
data["sources"]    = data['sources'].str.split("//").str[1].str.split("/").str[0]
data["sources"]    = data["sources"].str.replace("www.|.com|.net|.co|.gov|.org","")


# In[ ]:


#EXTRACTING STRING INFO FROM STOLEN GUN,TYPE,AGE-GROUP
data["gun_stolen"] = data['gun_stolen'].str.replace("Unknown|[0-9]|[|:]|","")
data["gun_type"]   = data["gun_type"].str.replace("Unknown|[0-9]|[|:]|","")
data["participant_age_group"] = data["participant_age_group"].str.replace("[:|+-]|[0-9]","")


# In[ ]:


#EXTRACTING YEAR,MONTH,DAY FROM DATE
data["year"] = pd.DatetimeIndex(data["date"]).year
data["month"]= pd.DatetimeIndex(data["date"]).month
data["day"]  = pd.DatetimeIndex(data["date"]).day
data["month"] = data["month"].map({1:"JAN",2:"FEB",3:"MAR",4:"APR",5:"MAY",6:"JUN",7:"JUL",8:"AUG",9:"SEP",10:"OCT",11:"NOV",12:"DEC"})
data["month_year"] = data["month"]+"-"+data["year"].astype(str)


# In[ ]:


data = data.rename(columns={"n_killed":"killed","n_injured":"injured"})
data = data.sort_values(by="date",ascending=True)


# # NUMBER OF PEOPLE KILLED AND INJURED IN YEARS

# In[ ]:



fig = data.groupby("year")[["killed","injured"]].sum().plot(kind="bar",stacked=True,
                                                            figsize=(13,7),
                                                            linewidth = 1,
                                                            edgecolor = "k"*data["year"].nunique())
plt.grid(True,alpha=.3)
plt.xticks(rotation=0)
fig.set_facecolor("ghostwhite")
plt.legend(prop={"size":20})
plt.title("NUMBER OF PEOPLE KILLED AND INJURED IN YEARS",fontsize = 20)

plt.show()


# # NUMBER OF PEOPLE KILLED AND INJURED IN YEARS# 

# In[ ]:


fatalities_year = data.groupby("year")[["killed","injured"]].sum()
fig = plt.figure(figsize=(13,6))
plt.subplot(121)
plt.pie(fatalities_year["killed"],labels=fatalities_year.index,autopct="%1.0f%%",shadow=True,
        colors=sns.color_palette("Set1",7),wedgeprops={"linewidth":2,"edgecolor":"white"})
my_circ = plt.Circle((0,0),.7,color="w")
plt.gca().add_artist(my_circ)
plt.title("PROPORTION OF KILLINGS BY YEAR",fontsize=15)

plt.subplot(122)
plt.pie(fatalities_year["injured"],labels=fatalities_year.index,autopct="%1.0f%%",shadow=True,
        colors=sns.color_palette("Set1",7),wedgeprops={"linewidth":2,"edgecolor":"white"})
my_circ = plt.Circle((0,0),.7,color="w")
plt.gca().add_artist(my_circ)
plt.title("PROPORTION OF INJURIES BY YEAR",fontsize=15)
fig.set_facecolor("w")


# In[ ]:


print ("Maximum killings happend in the",data.loc[data["killed"].idxmax()]["state"],"state,",data.loc[data["killed"].idxmax()]["city_or_county"],"city",
      "on",data.loc[data["killed"].idxmax()]["date"],"by killing",data["killed"].max(),"and injuring",data["injured"].max())


# # PEOPLE KILLED & INJURED DURING JAN-2013 TO MAR-2018

# In[ ]:


fatalities_monthyear = data.groupby("month_year")[["killed","injured"]].sum().reset_index()
order = data["month_year"].unique().tolist()
fig = plt.figure(figsize=(13,7))
sns.pointplot(fatalities_monthyear["month_year"],fatalities_monthyear["injured"],order=order,color="g",markers="h")
sns.pointplot(fatalities_monthyear["month_year"],fatalities_monthyear["killed"],order=order,color="r",markers="h")
plt.axhline(fatalities_monthyear["killed"].mean(),label="KILLED,MEAN",linestyle="dashed",color="r")
plt.axhline(fatalities_monthyear["injured"].mean(),label="INJURED,MEAN",linestyle="dashed",color="g")
plt.xticks(rotation=90)
fig.set_facecolor("lightgrey")
plt.legend(loc="best",prop={"size":15})
plt.title("PEOPLE KILLED & INJURED DURING JAN-2013 TO MAR-2018",fontsize =20)
plt.grid(True,alpha=.3)
plt.show()


# # TOTAL KILLINGS & INJURIES BY MONTHS

# In[ ]:


fatalities_month = data.groupby("month")[["killed","injured"]].sum().reset_index()
order1 = data["month"].unique().tolist()
fig = plt.figure(figsize=(12,8))
sns.barplot(y = fatalities_month["month"],x = fatalities_month["injured"],
            color="b",order=order1,label="INJURED",alpha=0.5,
            linewidth = 1 ,edgecolor = "k"*data["month"].nunique())
sns.barplot(y = fatalities_month["month"],x = fatalities_month["killed"],
            color="lime",order=order1,label="KILLED",alpha=0.6,
            linewidth = 1 ,edgecolor = "k"*data["month"].nunique())
plt.legend(loc="best",prop={"size":14})
plt.title("TOTAL KILLINGS & INJURIES BY MONTHS ",fontsize=20)
plt.grid(True,alpha=.3)
plt.show()


# # PEOPLE KILLED OVER YEARS IN THEIR RESPECTIVE MONTHS

# In[ ]:


kills_my = pd.pivot_table(index="month",columns="year",data=data,values="killed",aggfunc="sum").fillna(0)

for i in kills_my.columns:
    kills_my[i] = kills_my[i].astype(np.int64)
plt.figure(figsize=(13,6))
fig = sns.heatmap(kills_my.transpose(),cmap="inferno",
                  annot=True,fmt="d",linecolor="white",linewidths=2)
plt.xticks(fontsize =12)
plt.yticks(fontsize =12,rotation=0)
plt.ylabel("YEAR",fontsize=15)
plt.xlabel("MONTH",fontsize=15)
plt.title("PEOPLE KILLED OVER YEARS IN THEIR RESPECTIVE MONTHS",fontsize=15,color="b")
plt.show()


# # PEOPLE INJURED OVER YEARS IN THEIR RESPECTIVE MONTHS

# In[ ]:


injured_my = pd.pivot_table(index="month",columns="year",data=data,values="injured",aggfunc="sum").fillna(0)

for i in injured_my.columns:
    injured_my[i] = injured_my[i].astype(np.int64)
plt.figure(figsize=(13,6))
fig = sns.heatmap(injured_my.transpose(),cmap="magma",
                  annot=True,fmt="d",linecolor="white",linewidths=2)
plt.xticks(fontsize =12)
plt.yticks(fontsize =12,rotation=0)
plt.ylabel("YEAR",fontsize=15)
plt.xlabel("MONTH",fontsize=15)
plt.title("PEOPLE INJURED OVER YEARS IN THEIR RESPECTIVE MONTHS",fontsize=15,color="b")
plt.show()


# # TOP STATES IN KILLINGS AND INJURIES

# In[ ]:


state_kill   = data.groupby("state")["killed"].sum().reset_index().sort_values(by="killed",ascending=False)
state_injury = data.groupby("state")["injured"].sum().reset_index().sort_values(by="injured",ascending=False)

fig = plt.figure(figsize=(12,10))
plt.subplot(121)
ax = sns.barplot("killed","state",
                 data=state_kill[:15],palette="husl",
                 linewidth=1,edgecolor = "k"*15)
plt.ylabel("STATE",fontsize=15)
plt.xlabel("KILLED",fontsize=15)
plt.title("TOP STATES BY DEATHS DUE TO VIOLENCE")
for i,j in enumerate(state_kill["killed"][:15]):
    ax.text(.9,i,j,weight="bold",fontsize=15)
plt.grid(True,alpha= .3)

plt.subplot(122)
ax = sns.barplot("injured","state",
                 data=state_injury[:15],palette="husl",
                linewidth=1,edgecolor = "k"*15)
plt.ylabel("")
plt.xlabel("INJURED",fontsize=15)
plt.subplots_adjust(wspace=.3)
plt.title("TOP STATES BY INJURIES DUE TO VIOLENCE")
for i,j in enumerate(state_injury["injured"][:15]):
    ax.text(.9,i,j,weight="bold",fontsize=15)
plt.grid(True,alpha= .3)
    
fig.set_facecolor("w")


# # YEARLY TOP STATES BY CASUALITIES

# In[ ]:


casualities = data.groupby(["year","state"])[["killed","injured"]].sum().reset_index()
casualities["casualities"] = casualities["killed"] + casualities["injured"]
casualities = casualities.sort_values(by=["year","casualities"],ascending=False)

years  = data["year"].unique().tolist()
length = len(years)

fig = plt.figure(figsize=(13,18))
for i,j in itertools.zip_longest(years,range(length)):
    plt.subplot(3,2,j+1)
    ax = sns.barplot("casualities","state",
                     data=casualities[casualities["year"] == i][:10],palette="cool",
                     linewidth = 1 ,edgecolor = "k"*10)
    plt.title(i,fontsize=20,color="b")
    plt.ylabel("")
    plt.subplots_adjust(wspace=.35,hspace=.4)
    plt.yticks(fontsize=12)
    for i,j in enumerate(casualities[casualities["year"] == i]["casualities"][:10]):
        ax.text(.7,i,j,weight="bold")
    fig.set_facecolor("w")


# # WORD CLOUD FOR STATES IN DATA

# In[ ]:


from PIL import  Image

words = data["state"].value_counts().keys()
wc = WordCloud(max_words=60,scale=5,colormap="magma",background_color="dimgrey").generate(" ".join(words))
plt.figure(figsize=(13,8))
plt.imshow(wc,interpolation="bilinear")
plt.axis("off")
plt.title("STATES")
plt.show()


# # LOCATION MAP FOR GUN VIOLENCE INCIDENTS 

# In[ ]:


map_data=data[data["latitude"].notnull()]
map_data = map_data[["latitude","longitude","state","city_or_county"]]

# m = Basemap(projection="merc",llcrnrlat=20,urcrnrlat=53,
#             llcrnrlon=-130,urcrnrlon=-60,lat_ts=20,resolution='c')
m=Basemap(lat_0=40, lon_0=-100, projection='ortho' )

lat = list(map_data["latitude"].values)
lon = list(map_data["longitude"].values)
x,y = m(lon,lat)
fig = plt.figure(figsize=(12,12))
m.plot(x,y,"go",markersize =.2,color="yellow")
m.drawcountries()
m.drawcoastlines()
m.bluemarble(scale=1)
plt.title("LOCATION MAP FOR GUN VIOLENCE INCIDENTS")
plt.show()


# # CHOROPLETH MAP FOR PEOPLE KILLED IN STATES

# In[ ]:


import os
import folium

state_data = data.groupby("state")[["killed","injured"]].sum().reset_index()
geo = os.path.join(r"../input/usstates/us-states.json")
st = pd.read_excel(r"../input/states/states.xlsx",sheetname="Sheet1")
state_data = state_data.merge(st,left_on="state",right_on="State",how="left")

m1 = folium.Map(location=[37, -102], zoom_start=4,tiles="stamentoner")

m1.choropleth(line_weight=1.2,
    highlight = True,
    geo_data = geo,
    name = "choropleth",
    data = state_data,
    columns = ["Abbreviation","killed"],
    key_on = "feature.id",
    fill_color = "YlOrRd",
    fill_opacity = .7,
    line_opacity = 1,
    legend_name = "NO OF PEOPLE KILLED IN STATES", 
)
folium.LayerControl().add_to(m1)
m1


# # CHOROPLETH MAP FOR PEOPLE INJURED IN STATES# 

# In[ ]:


m2 = folium.Map(location=[37,-102],zoom_start=4,tiles="stamentoner")

m2.choropleth(geo_data=geo,line_weight=1.2,
              highlight =True,
              data=state_data,
              columns=["Abbreviation","injured"],
              fill_color="YlGn",
              fill_opacity=.7,
              key_on="feature.id",
              name="choropleth",
              line_opacity=1,
              legend_name = "NO OF PEOPLE INJURED IN STATES")

folium.LayerControl().add_to(m2)
m2


# # STATE WISE HIGHEST ATTACKED CITIES

# In[ ]:


data["casualities"] = data["killed"] + data["injured"]
statewise = data.groupby(["state","city_or_county"])["casualities"].sum().reset_index()
statewise = statewise.sort_values(by="casualities",ascending=False)
statewise = statewise.drop_duplicates(subset=["state"],keep="first")
fig = plt.figure(figsize=(10,17))
ax = sns.barplot(y = statewise["state"],x = statewise["casualities"],
                 palette=["skyblue"])
for i,j in enumerate(statewise["city_or_county"]):
    ax.text(.7,i,j,fontsize =10)
plt.title("STATE WISE HIGHEST ATTACKED CITIES",fontsize = 20)
plt.xlabel("CASUALITIES",fontsize = 20)
plt.ylabel("STATES",fontsize = 20)
fig.set_facecolor("w")


# # TOTAL CASUALITIES OVER YEAR-MONTHS

# In[ ]:


my_cas = data.groupby("month_year")["casualities"].sum().reset_index()
plt.figure(figsize=(13,7))
fig = sns.barplot(my_cas["month_year"],my_cas["casualities"],
                  order=data["month_year"].unique().tolist(),color="skyblue",alpha=.9,
                  linewidth = .3 , edgecolor = "w"*data["month_year"].nunique())
plt.xticks(rotation =90)
plt.grid(True,alpha = .1)
plt.title("TOTAL CASUALITIES BY YEAR-MONTHS",fontsize=20)
fig.set_facecolor("k")


# # CASUALITIES BY TOP 15 STATES AFFECTED

# In[ ]:


state_data["casualities"] = state_data["killed"] + state_data["injured"]
state_data = state_data.sort_values(by = "casualities" ,ascending=False)
fig = plt.figure(figsize=(13,7))
plt.scatter(state_data["state"][:15],state_data["casualities"][:15],c=state_data["killed"][:15],
            s=state_data["injured"][:15]/3,linewidth=2,edgecolor = "k",alpha=.7)
plt.xticks(rotation = 60)
plt.yticks(np.arange(0,25000,2500))
fig.set_facecolor("w")
plt.xlabel("STATES")
plt.ylabel("CASUALITIES")
plt.colorbar()
plt.grid(True,alpha=.3)
print ("SIZES = INJURED ,COLOR = KILLED")
print ("********************************")
plt.title("CASUALITIES BY TOP 15 STATES")
plt.show()


# # TOP SOURCES FOR NEWS ABOUT GUN VIOLENCE

# In[ ]:


source  = data["source_url"].value_counts().reset_index()
fig = plt.figure(figsize=(10,10))
sns.barplot("source_url","index",
            data=source[:25],palette="Set1",
            linewidth = 1 ,edgecolor = "k"*25)
plt.title("TOP SOURCES FOR NEWS ABOUT GUN VIOLENCE",fontsize=12,color="r")
fig.set_facecolor("w")
plt.ylabel("")
plt.show()


# # WORDCLOUD FOR TOP SOURCES OF NEWS

# In[ ]:


from PIL import Image

wc1 = WordCloud(background_color="white",scale=5,colormap="inferno",max_words=10000).generate(str(source["index"]))

plt.figure(figsize = (13,8))
plt.imshow(wc1,interpolation = "bilinear")
plt.axis("off")
plt.title("")
plt.show()


# # TOP 50 INCIDENTS OF GUN VIOLENCE

# In[ ]:


top_att = data.sort_values(by="casualities",ascending=False)[:50]

top_att = top_att[["date","state","city_or_county","address","killed","injured","casualities","latitude","longitude"]]
top_att["pop"] = ("STATE-"+top_att["state"]+"|COUNTY-"+top_att["city_or_county"] + "|ADDRESS-" +
                  top_att["address"]+"|KILLED-"+top_att["killed"].astype(str)+
                  "|INJURED-"+top_att["injured"].astype(str)+"|CASUALITIES-"+top_att["casualities"].astype(str))
top_att=top_att[top_att["pop"].notnull()]

m4 = folium.Map(location=[37,-102],tiles='Stamen terrain',zoom_start=4)


lat = list(top_att["latitude"])
lon = list(top_att["longitude"])
pop = list(top_att["pop"])
cs = ["red","blue","green"]*17

for i1,i2,i3,i4 in zip(lat,lon,pop,cs):
    
    folium.Marker([i1, i2],
              popup=i3,
              icon=folium.Icon(color=i4,icon="bar-chart", prefix='fa')
             ).add_to(m4)
m4


# # 91% OF TIMES PEOPLE USED SINGLE GUN WHEN THE INCIDENT IS RECORDED

# In[ ]:


n_guns = data[data["n_guns_involved"].notnull()]
n_guns["n_guns_involved"] = n_guns["n_guns_involved"].astype(int)
n_guns = n_guns[["n_guns_involved"]]

def label(n_guns):
    if n_guns["n_guns_involved"] == 1 :
        return "ONE-GUN"
    elif n_guns["n_guns_involved"] > 1 :
        return "GREATER THAN ONE GUN"

n_guns["x"] = n_guns.apply(lambda n_guns:label(n_guns),axis=1)
n_guns["x"].value_counts().plot.pie(figsize=(7,7),autopct ="%1.0f%%",explode = [0,.2],shadow = True,colors=["orange","grey"],startangle =25)
plt.title("NO OF GUNS INVOLVED")
plt.ylabel("")
plt.show()


# ## ** District wise kills & injuries for  Illinois,California,Texas, Florida, Ohio, Pennsylvania,North Carolina, New York, Louisiana**

# In[ ]:


state_district = data.groupby(["state","congressional_district"])["killed","injured","casualities"].sum().reset_index()
state_district["congressional_district"] = state_district["congressional_district"].astype(int)

state = list(state_data["state"][:9])
state_district1 = state_district[state_district["state"].isin(state)]
state_district1["district"] = "District "+state_district1["congressional_district"].astype(str)

length = len(state)

fig = plt.figure(figsize=(13,30))
for i,j in itertools.zip_longest(state,range(length)):
    plt.subplot(length/3,length/3,j+1)
    sns.barplot("injured","district",data=state_district1[state_district1["state"] == i],color = "b",alpha =.7,label = "INJURED")
    sns.barplot("killed","district",data=state_district1[state_district1["state"] == i],color = "r",alpha=.7,label="KILLED")
    plt.legend(loc="best")
    plt.title(i,fontsize = 20)
    plt.subplots_adjust(wspace=.5)
    plt.xlabel("KILLED , INJURED")
    plt.ylabel("")
fig.set_facecolor("w")


# # TREND IN DISTRICT NUMBER BY CASUALITIES

# In[ ]:


state_district["district"] = state_district["congressional_district"].astype(str) +"District "
dist = state_district.groupby("district")[["casualities","killed","injured"]].sum().reset_index()
fig = plt.figure(figsize=(13,7))
order2 = state_district["district"].unique().tolist()
sns.pointplot(dist["district"],dist["killed"],order=order2,markers="o")
sns.pointplot(dist["district"],dist["injured"],order=order2,markers="o",color="orangered")
plt.xticks(rotation =90)
plt.title("TREND IN DISTRICT NUMBER BY CASUALITIES")
plt.grid(True,alpha=.3)
fig.set_facecolor("w")


# # WORDCLOUD FOR TYPES OF GUNS USED

# In[ ]:


gn = data[data["gun_type"].notnull()]
w = WordCloud(background_color="black",colormap="cool",scale=5).generate(" ".join(gn["gun_type"]))
plt.figure(figsize=(13,8))
plt.imshow(w,interpolation="bilinear")
plt.axis("off")
plt.show()


# # TOP CITIES OF EVENT OCCURENCES

# In[ ]:


city_data = data.groupby("city_or_county").agg({"killed":"sum","injured":"sum","casualities":"sum","latitude":"median","longitude":"median"}).reset_index()
city_data1 = city_data.sort_values(by="casualities",ascending=False)[:10]

plt.figure(figsize=(15,8))

m5 = Basemap(projection="merc",llcrnrlon=-130, llcrnrlat=20,urcrnrlon=-60,urcrnrlat=55,lat_0=True,lat_1=True,lat_ts=20)
m5.drawcoastlines(color="black",linewidth=2)
m5.drawcountries(color="black",linestyle="dashed",linewidth=2)
m5.drawstates(color="r",linestyle="dotted",linewidth=1)
m5.fillcontinents(lake_color="aqua",color="bisque")
m5.drawmapboundary(fill_color="aqua")
m5.drawcounties(color="grey",linewidth=.5)

c    = sns.color_palette("Set2",10)
city = list(city_data1["city_or_county"])
lat  = list(city_data1[city_data1["city_or_county"] == city]["latitude"])
lon  = list(city_data1[city_data1["city_or_county"] == city]["longitude"])
cas  = list(city_data1["casualities"])


def function(city,c,cas):

    lat  = list(city_data1[city_data1["city_or_county"] == city]["latitude"])
    lon  = list(city_data1[city_data1["city_or_county"] == city]["longitude"])
    x,y = m5(lon,lat)
    m5.plot(x,y,"go",markersize=k/150,linestyle="none",marker = "o",c= j,alpha=.8,markeredgecolor="black",markeredgewidth = 1,label= i)

for i,j,k in zip(city,c,cas):
    function(i,j,k)
    
# for i,j,k in itertools.zip_longest(x,y,city):
#     plt.text(i,j,k,weight = "bold",horizontalalignment='center',verticalalignment='center',color="k")

plt.legend(loc="best",prop={"size":15}).get_frame().set_facecolor("white")
plt.title("TOP CITIES OF EVENT OCCURENCES")
plt.show()


# # Total Number of people killed and injured for top cities

# In[ ]:


city_data2 = city_data.sort_values(by="casualities",ascending=False)[["city_or_county","killed","injured"]][:30]
plt.figure(figsize=(12,7))
sns.barplot("city_or_county","injured",
            data=city_data2,color="b",label="Killed",
            linewidth = .5 ,edgecolor ="k" * city_data2["city_or_county"].nunique())
sns.barplot("city_or_county","killed",
            data=city_data2,color="r",label="Injured",
           linewidth = .5 ,edgecolor = "k" * city_data2["city_or_county"].nunique())
plt.legend(loc="best",prop = {"size" : 15})
plt.xticks(rotation = 70)
plt.title("Total Number of people killed and injured for top cities")
plt.grid(True,alpha=.3)
plt.show()


# # STATE WISE KILLINGS AND INJURIES

# In[ ]:


from math import pi
from mpl_toolkits.mplot3d import Axes3D

state_ki = data.groupby("state")[["killed","injured","casualities"]].sum().reset_index()

cols = [x for x in state_ki.columns if x not in ["casualities"]+["state"]]
l  = len(cols)
cs = ["b","orange"]

fig = plt.figure(figsize=(12,12))
plt.subplot(111,projection = "polar")
fig.set_facecolor("white")

for i,j,k in itertools.zip_longest(cols,range(l),cs):
    cats = state_ki["state"].tolist()
    length = len(cats)

    values = state_ki[i].tolist()
    values += values[:1]

    angles = [n/float(length)*2*pi for n in range(length)]
    angles += angles[:1]
    
    plt.xticks(angles[:-1],cats,color="k",fontsize = 11)
    plt.plot(angles,values,color=k,label = i)
    plt.fill(angles,values,color=k,alpha=.4)
    plt.legend(loc = "best",prop = {"size":12})
    plt.title("STATE WISE KILLINGS AND INJURIES")


# # CITY COMPARATOR

# In[ ]:


def city_compare(city1,city2):
    
    city_list = [city1,city2]
    compare = pd.pivot_table(data=data[data["city_or_county"].isin(city_list)],
                             columns="city_or_county",index="year",values="casualities",aggfunc="sum").fillna(0)
 
    fig = compare[[city1,city2]].plot(figsize = (13,5),marker='o', markerfacecolor="white", markersize=12, color=["skyblue","orange"], linewidth=5)
    plt.legend(prop={"size":15})
    plt.title("CITY CASUALTIES COMAPARATOR")
    plt.xticks([2012,2013,2014,2015,2016,2017,2018,2019])
    fig.set_facecolor("k")


# In[ ]:


city_compare('Chicago','Baltimore')


# In[ ]:


city_compare('Los Angeles','Las Vegas')


# # TOP LOCATIONS FOR THE EVENTS OF VIOLENCE

# In[ ]:


location = data[data['location_description'].notnull()]["location_description"].value_counts()[:30]
location = pd.DataFrame(location).reset_index()
plt.figure(figsize=(10,10))
sns.barplot(y=location["index"],x=location["location_description"],palette="cool",
            linewidth = .5 , edgecolor = "k"*30 )
plt.ylabel("")
plt.title("TOP LOCATIONS FOR THE EVENTS OF VIOLENCE")
plt.show()


# # WORD CLOUD FOR LOCATION DESCRIPTION

# In[ ]:


location = data[data['location_description'].notnull()]["location_description"].value_counts().keys()
wc2 =WordCloud(scale = 5,background_color="white",colormap="cool").generate(str(data[data["location_description"].notnull()]["location_description"]))
plt.figure(figsize=(13,8))
plt.imshow(wc2,interpolation="bilinear")
plt.axis("off")
plt.title("WORD CLOUD FOR LOCATION DESCRIPTION")
plt.show()


# In[ ]:


plt.figure(figsize=(13,12))
plt.subplot(211)
sns.distplot(data[data["state_house_district"].notnull()]["state_house_district"],color= "b")
plt.title("Distribution of state_house_district")
plt.subplot(212)
sns.distplot(data[data["state_senate_district"].notnull()]["state_senate_district"],color="orangered")
plt.title("Distribution of state_senate_district")
plt.show()


# # Tree Plot for number of people killed across all states

# In[ ]:


import squarify
plt.figure(figsize=(12,10))
squarify.plot(state_data["killed"],label=state_data["state"],color=sns.color_palette("hot"),alpha=.7,linewidth=2,edgecolor="b")
plt.title("Tree Plot for number of people killed across all states")
plt.axis("off")
plt.show()


# # Tree Plot for number of people killed across all states

# In[ ]:


import squarify
plt.figure(figsize=(13,10))
squarify.plot(state_data["injured"],label=state_data["state"],color=sns.color_palette("magma"),alpha=.7,linewidth=2,edgecolor="b")
plt.title("Tree Plot for number of people injured across all states")
plt.axis("off")
plt.show()


# # WORDCLOUD FOR NOTES RECORDED DURING VIOLENT EVENTS

# In[ ]:


from wordcloud import STOPWORDS

notes = data[data["notes"].notnull()]["notes"]
img = np.array(Image.open(r"../input/image-of-us/us.jpg"))

wc3 = WordCloud(stopwords=STOPWORDS,scale=5,mask=img,background_color="black",colormap="cool").generate(str(notes))
plt.figure(figsize=(15,10))
plt.imshow(wc3,interpolation="bilinear")
plt.axis("off")
plt.show()


# In[ ]:


age = data[data["participant_age"].notnull()][["participant_age"]]
age["participant_age"] = age["participant_age"].str.replace("::","-")
age["participant_age"] = age["participant_age"].str.replace(":","-")
age["participant_age"] = age["participant_age"].str.replace("[||]",",")
age = pd.DataFrame(age["participant_age"])
x1 = pd.DataFrame(age["participant_age"].str.split(",").str[0])
x2 = pd.DataFrame(age["participant_age"].str.split(",").str[1])
x3 = pd.DataFrame(age["participant_age"].str.split(",").str[2])
x4 = pd.DataFrame(age["participant_age"].str.split(",").str[3])
x5 = pd.DataFrame(age["participant_age"].str.split(",").str[4])
x6 = pd.DataFrame(age["participant_age"].str.split(",").str[5])
x7 = pd.DataFrame(age["participant_age"].str.split(",").str[6])
x1 = x1[x1["participant_age"].notnull()]
x2 = x2[x2["participant_age"].notnull()]
x3 = x3[x3["participant_age"].notnull()]
x4 = x4[x4["participant_age"].notnull()]
x5 = x5[x5["participant_age"].notnull()]
x6 = x6[x6["participant_age"].notnull()]
x7 = x7[x7["participant_age"].notnull()]

age_dec  = pd.concat([x1,x2,x3,x4,x5,x6,x7],axis = 0)
age_dec["lwr_lmt"] = age_dec["participant_age"].str.split("-").str[0]
age_dec["upr_lmt"] = age_dec["participant_age"].str.split("-").str[1]
age_dec.head()

age_dec= age_dec[age_dec["lwr_lmt"]!='']
age_dec["lwr_lmt"] = age_dec["lwr_lmt"].astype(int)
age_dec["upr_lmt"] = age_dec["upr_lmt"].astype(int)


# # Distribution of age groups of participants

# In[ ]:


age_dec["age_bins"] = pd.cut(age_dec["upr_lmt"],bins=[0,20,35,55,130],labels=["TEEN[0-20]","YOUNG[20-35]","MIDDLE-AGED[35-55]","OLD[>55]"])
plt.figure(figsize=(8,8))
age_dec["age_bins"].value_counts().plot.pie(autopct = "%1.0f%%",shadow =True,startangle = 0,colors = sns.color_palette("prism",5),
                                            wedgeprops = {"linewidth" :3,"edgecolor":"w"})
my_circ = plt.Circle((0,0),.7,color = "white")
plt.gca().add_artist(my_circ)
plt.ylabel("")
plt.title("Distribution of age groups of participants",fontsize=20)
plt.show()


# # Gender Proportion by Participants

# In[ ]:


gender = data["participant_gender"].str.replace("[::0-9|]","").str.upper()
gender = gender.str.replace("FEMALE","F")
gender = gender.str.replace("MALE","M")
gender = pd.DataFrame(gender)
gender = gender[gender["participant_gender"].notnull()]
gender["female"] = gender["participant_gender"].str.count("F")
gender["male"]   = gender["participant_gender"].str.count("M")
size = [sum(gender["female"]),sum(gender["male"])]
plt.figure(figsize=(8,8))
plt.pie(size,labels=["FEMALE","MALE"],shadow=True,autopct="%1.0f%%",wedgeprops={"linewidth":2,"edgecolor":"k"},explode=[.1,0])
plt.title("GENDER PROPORTION BY PARTICIPANTS")
plt.show()


# # Word cloud for participant Names

# In[ ]:


names = data[ 'participant_name'].str.replace("[:|.]","")
names = names.str.replace("[0-9]",",")
names = names[names.notnull()]
wc3 = WordCloud(background_color="black",colormap="rainbow",scale=5).generate(str(names))
plt.figure(figsize=(14,8))
plt.imshow(wc3,interpolation="bilinear")
plt.axis("off")
plt.show()


# # PARTICPANT RELATION TYPE IN VIOLENT EVENTS

# In[ ]:


relation = data['participant_relationship']
relation = relation[relation.notnull()]
relation = relation.str.replace("[:|0-9]"," ").str.upper()
relation1 = pd.DataFrame({"count":[len(relation[relation.str.contains("FAMILY")]),
               len(relation[relation.str.contains("ROBBERY")]),
               len(relation[relation.str.contains("FRIENDS")]),
               len(relation[relation.str.contains("AQUAINTANCE")]),
               len(relation[relation.str.contains("NEIGHBOR")]),
               len(relation[relation.str.contains("INVASION")]),
               len(relation[relation.str.contains("CO-WORKER")]),
               len(relation[relation.str.contains("GANG")]),
               len(relation[relation.str.contains("RANDOM")]),
               len(relation[relation.str.contains("MASS SHOOTING")])],
              "category":["FAMILY","ROBBERY","FRIENDS","AQUAINTANCE","NEIGHBOR","INVASION","CO-WORKER","GANG","RANDOM","MASS SHOOTING"]})
relation1
plt.figure(figsize=(12,7))
sns.barplot("category","count",data=relation1,palette="prism",
            linewidth =1 ,edgecolor = "k" *relation1["category"].nunique())
plt.title("COUNT PLOT FOR PARTICPANT RELATION TYPE IN VIOLENT EVENTS")
plt.grid(True,alpha=.3)
plt.show()


# # PARTICIPANT STATUS DISTRIBUTION

# In[ ]:


data[ 'participant_status'].value_counts()
p_status = data["participant_status"].str.replace("[::0-9|,]","").str.upper()
p_status = p_status.str.replace("ARRESTED","A")
p_status = p_status.str.replace("INJURED","I")
p_status = p_status.str.replace("KILLED","K")
p_status = p_status.str.replace("UNHARMED","U")
p_status = p_status[p_status.notnull()]
p_status = pd.DataFrame(p_status)
p_status["arested"]  = p_status["participant_status"].str.count("A")
p_status["injured"]  = p_status["participant_status"].str.count("I")
p_status["killed"]   = p_status["participant_status"].str.count("K")
p_status["unharmed"] = p_status["participant_status"].str.count("U")

sizes = [sum(p_status["arested"]),sum(p_status["injured"]),sum(p_status["killed"]),sum(p_status["unharmed"])]
plt.figure(figsize=(8,8))
plt.pie(sizes,labels=["ARRESTED","INJURED","KILLED","UNHARMED"],shadow=True,autopct="%1.0f%%",
        colors = sns.color_palette("prism",5), wedgeprops = {"linewidth" :3,"edgecolor":"white"})
my_circ = plt.Circle((0,0),.7,color = "white")
plt.gca().add_artist(my_circ)
plt.title("PARTICIPANT STATUS")
plt.show()


# # Participant type distribution

# In[ ]:


data[ 'participant_type'].value_counts()
p_type = data["participant_type"].str.replace("[::0-9|,]","").str.upper()
p_type = p_type.str.replace("VICTIM","V")
p_type = p_type.str.replace("SUBJECT-SUSPECT","S")
p_type = p_type[p_type.notnull()]
p_type = pd.DataFrame(p_type)
p_type["victim"]  = p_type["participant_type"].str.count("V")
p_type["subject_suspect"]  = p_type["participant_type"].str.count("S")
size = [sum(p_type["victim"]),sum(p_type["subject_suspect"])]
plt.figure(figsize=(8,8))
plt.pie(size,labels=["VICTIM","SUBJECT-SUSPECT"],shadow=True,autopct="%1.0f%%",wedgeprops={"linewidth":2,"edgecolor":"k"},
        colors=["lightgrey","orange"],explode=[.01,0])
plt.title("GENDER PROPORTION BY PARTICIPANTS")
plt.show()


# In[ ]:





# In[ ]:




