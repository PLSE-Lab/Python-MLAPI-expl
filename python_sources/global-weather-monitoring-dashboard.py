#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from plotly.offline import iplot, init_notebook_mode
import plotly.graph_objs as go
import tarfile
import gzip
import re
import os
import datetime as dt


# In[ ]:


init_notebook_mode(connected=True)


# In[ ]:


#stat_num = 1000 # Number of stations to plot for testing
year_num = 20 # Number of past years to consider
extremes_num = 10 # Number of hottest and coldest places to display


# In[ ]:


yearfiles = os.listdir("../input/gsod_all_years/")
yearfiles.sort()
yearfiles = yearfiles[-year_num:]
years = [int(re.findall('\d+',yearfile)[0]) for yearfile in yearfiles]


# In[ ]:


station_loc = pd.read_csv('../input/isd-history.csv')
station_loc = station_loc.replace([0.0, -999.0, -999.9],np.nan)
station_loc = station_loc[pd.notnull(station_loc['LAT']) & pd.notnull(station_loc['LON'])]
station_loc = station_loc[[int(re.findall('^\d{4}', str(end_year))[0])==max(years) for end_year in station_loc['END']]]
station_loc = station_loc[[int(re.findall('^\d{4}', str(beg_year))[0])<=min(years) for beg_year in station_loc['BEGIN']]]


# In[ ]:


station_loc['LBL'] = station_loc[['STATION NAME','STATE','CTRY']].apply(lambda x: x.str.cat(sep=', '), axis=1)
station_loc['ELEV_LBL'] = station_loc['ELEV(M)'].apply(lambda x: 'Elevation: '+str(x)+' m' if ~np.isnan(x) else np.nan)
station_loc['LBL'] = station_loc[['LBL','ELEV_LBL']].apply(lambda x: x.str.cat(sep='<br>'), axis=1)
station_loc = station_loc.drop(['STATION NAME','STATE','ELEV_LBL','ICAO','BEGIN','END'], axis=1)
#station_loc = station_loc.sample(stat_num)


# In[ ]:


df = pd.DataFrame([])
df_day = pd.DataFrame([])

def preprocess_station_file_content(content):
    headers=content.pop(0)
    headers=[headers[ind] for ind in [0,1,2,3,4,8,11,12]]
    for d in range(len(content)):
        content[d]=[content[d][ind] for ind in [0,1,2,3,5,13,17,18]]
    content=pd.DataFrame(content, columns=headers)
    content.rename(columns={'STN---': 'USAF'}, inplace=True)
    content['MAX'] = content['MAX'].apply(lambda x: re.sub("\*$","",x))
    content['MIN'] = content['MIN'].apply(lambda x: re.sub("\*$","",x))
    content[['WBAN','TEMP','DEWP','WDSP','MAX','MIN']] = content[['WBAN','TEMP','DEWP','WDSP','MAX','MIN']].apply(pd.to_numeric)
    content['YEARMODA']=pd.to_datetime(content['YEARMODA'], format='%Y%m%d', errors='ignore')
    content['YEAR']=pd.DatetimeIndex(content['YEARMODA']).year
    content['MONTH']=pd.DatetimeIndex(content['YEARMODA']).month
    content['DAY']=pd.DatetimeIndex(content['YEARMODA']).day
    return content


# In[ ]:


df


# In[ ]:


yearfile = yearfiles[-1]
print(yearfile)
i=0
tar = tarfile.open("../input/gsod_all_years/"+yearfile, "r")
print(len(tar.getmembers()[1:]))
#for member in np.random.choice(tar.getmembers()[1:], size=stat_num, replace=False):
for member in tar.getmembers()[1:]:
    name_parts = re.sub("\.op\.gz$","",re.sub("^\./","",member.name)).split("-")
    usaf = name_parts[0]
    wban = int(name_parts[1])
    if station_loc[(station_loc['USAF']==usaf) & (station_loc['WBAN']==wban)].shape[0]!=0:
        i=i+1
        #if i%(stat_num//10) == 0: print(i)
        f=tar.extractfile(member)
        f=gzip.open(f, 'rb')
        content=[re.sub(" +", ",", line.decode("utf-8")).split(",") for line in f.readlines()]
        content=preprocess_station_file_content(content)
        df_day = df_day.append(content[content['YEARMODA']==content['YEARMODA'].max()])
        content = content.groupby(['USAF','WBAN','YEAR','MONTH']).agg('median').reset_index()
        df = df.append(content)
tar.close()


# In[ ]:


day = df_day['YEARMODA'].max()
df_day = df_day[df_day['YEARMODA']==day]


# In[ ]:


for yearfile in yearfiles[:-1]:
    print(yearfile)
    i=0
    tar = tarfile.open("../input/gsod_all_years/"+yearfile, "r")
    print(len(tar.getmembers()[1:]))
    #for member in np.random.choice(tar.getmembers()[1:], size=stat_num, replace=False):
    for member in tar.getmembers()[1:]:
        name_parts = re.sub("\.op\.gz$","",re.sub("^\./","",member.name)).split("-")
        usaf = name_parts[0]
        wban = int(name_parts[1])
        if station_loc[(station_loc['USAF']==usaf) & (station_loc['WBAN']==wban)].shape[0]!=0:
            i=i+1
            #if i%(stat_num//10) == 0: print(i)
            f=tar.extractfile(member)
            f=gzip.open(f, 'rb')
            content=[re.sub(" +", ",", line.decode("utf-8")).split(",") for line in f.readlines()]
            content=preprocess_station_file_content(content)
            df_day = df_day.append(content[(content['MONTH']==day.month) & (content['DAY']==day.day)])
            content = content.groupby(['USAF','WBAN','YEAR','MONTH']).agg('median').reset_index()
            df = df.append(content)
    tar.close()


# In[ ]:


df_loc = pd.merge(df, station_loc, how='inner', on=['USAF','WBAN'])
df_day_loc = pd.merge(df_day, station_loc, how='inner', on=['USAF','WBAN'])

df_loc['ADD_LBL'] = df_loc['TEMP'].apply(lambda x: 'Temperature: '+str(np.round((x-32)*5/9,1))+' C')
df_loc['LBL'] = df_loc[['LBL','ADD_LBL']].apply(lambda x: x.str.cat(sep='<br>'), axis=1)
df_loc = df_loc.drop('ADD_LBL', axis=1)

df_day_loc['ADD_LBL'] = df_day_loc['TEMP'].apply(lambda x: 'Temperature: '+str(np.round((x-32)*5/9,1))+' C')
df_day_loc['LBL_TRACE'] = df_day_loc['LBL']
df_day_loc['LBL'] = df_day_loc[['LBL','ADD_LBL']].apply(lambda x: x.str.cat(sep='<br>'), axis=1)
df_day_loc = df_day_loc.drop('ADD_LBL', axis=1)


# In[ ]:


extremes = pd.DataFrame([])
extremes = extremes.append(df_day_loc[df_day_loc['YEARMODA']==day].sort_values(by="TEMP", ascending=False).head(extremes_num))
extremes = extremes.append(df_day_loc[df_day_loc['YEARMODA']==day].sort_values(by="TEMP", ascending=False).tail(extremes_num))


# In[ ]:


scl = [0,"rgb(150,0,90)"],[0.125,"rgb(0, 0, 200)"],[0.25,"rgb(0, 25, 255)"],[0.375,"rgb(0, 152, 255)"],[0.5,"rgb(44, 200, 150)"],[0.75,"rgb(255, 234, 0)"],[0.875,"rgb(255, 111, 0)"],[1,"rgb(255, 0, 0)"]


# In[ ]:


data = [ dict(
    type = 'scattergeo',
    text = extremes['LBL'],
    lat = extremes['LAT'],
    lon = extremes['LON'],
    marker = dict(
        color = (extremes['TEMP']-32)*5/9,
        colorscale = scl,
        cmin = -50,
        cmax = 50,
        size = 5,
        colorbar = dict(
            thickness = 10,
            titleside = "right",
            outlinecolor = "rgba(68, 68, 68, 0)",
            tickvals = [-50,-30,-15,0,15,30,50],
            ticks = "outside",
            ticklen = 3,
            ticksuffix = " C",
            showticksuffix = "all"
        )
    )
)]

layout = dict(
    geo = dict(
        scope = 'world',
        showland = True,
        landcolor = "rgb(212, 212, 212)",
        showlakes = True,
        lakecolor = "rgb(255, 255, 255)",
        showsubunits = False,
        #subunitcolor = "rgb(255, 255, 255)",
        showcountries = False,
        #countrycolor = "rgb(255, 255, 255)",
        showcoastlines=False,
        resolution = 110
        ),
    )
fig = dict( data=data, layout=layout )


# Top 10 hottest and top 10 coldest stations in the world on:

# In[ ]:


print(dt.date.strftime(day, '%Y-%m-%d'))


# In[ ]:


iplot(fig)


# In[ ]:


df_day_loc = df_day_loc[(df_day_loc['USAF'].isin(extremes['USAF'])) & (df_day_loc['WBAN'].isin(extremes['WBAN']))]
df_day_loc = df_day_loc.sort_values(by=['LAT','LON','YEAR'])


# In[ ]:


data = [ go.Scatter(
    x = df_day_loc[(df_day_loc['USAF']==stat_code[1][0]) & (df_day_loc['WBAN']==stat_code[1][1])]['YEARMODA'],
    y = (df_day_loc[(df_day_loc['USAF']==stat_code[1][0]) & (df_day_loc['WBAN']==stat_code[1][1])]['TEMP']-32)*5/9,
    mode = 'lines+markers',
    line=dict(color='rgba(212, 212, 212,1)'),
    hoverinfo = 'x+y',
    marker = dict(
        color = (df_day_loc[(df_day_loc['USAF']==stat_code[1][0]) & (df_day_loc['WBAN']==stat_code[1][1])]['TEMP']-32)*5/9,
        colorscale = scl,
        cmin = -50,
        cmax = 50
    ),
    name = df_day_loc[(df_day_loc['USAF']==stat_code[1][0]) & (df_day_loc['WBAN']==stat_code[1][1])]['LBL_TRACE'].values[0],
) for stat_code in df_day_loc[['USAF','WBAN']].drop_duplicates().iterrows()]

layout = go.Layout(
    autosize=False,
    width=2300,
    height=800,
    xaxis=go.layout.XAxis(
        tickvals=df_day_loc['YEARMODA'].drop_duplicates(),
        automargin=True
    )
)

fig = go.Figure(data, layout)


# Past years temperatures at the stations displayed above on the same day

# In[ ]:


iplot(fig)


# In[ ]:


#temp = df_loc.groupby(['USAF','WBAN','LBL','LAT','LON','ELEV(M)','MONTH']).agg('median').reset_index()
#temp = temp.groupby(['USAF','WBAN','LBL','LAT','LON','ELEV(M)']).agg({'TEMP':['min','max'],'DEWP':['min','max'],'WDSP':['min','max'],'MAX':'max','MIN':'min'}).reset_index()
#temp = temp[(temp['TEMP', 'min']>=41)&\
#            (temp['TEMP', 'max']<=77)&\
#            (temp['TEMP', 'max']>=68)&\
#            (temp['MIN', 'min']>=23)&\
#            (temp['MAX', 'max']<=95)&\
#            (temp['DEWP', 'min']>=35)&\
#            (temp['DEWP', 'max']<=60)&\
#            (temp['WDSP', 'max']<=60)]


# In[ ]:


data = [ dict(
    visible = False,
    name = '',
    type = 'scattergeo',
    text = df_loc[(df_loc['MONTH']==month) & (df_loc['YEAR']==year)]['LBL'],
    lat = df_loc[(df_loc['MONTH']==month) & (df_loc['YEAR']==year)]['LAT'],
    lon = df_loc[(df_loc['MONTH']==month) & (df_loc['YEAR']==year)]['LON'],
    marker = dict(
        color = (df_loc[(df_loc['MONTH']==month) & (df_loc['YEAR']==year)]['TEMP']-32)*5/9,
        colorscale = scl,
        cmin = -50,
        cmax = 50,
        opacity = 0.5,
        size = 5,
        colorbar = dict(
            thickness = 10,
            titleside = "right",
            outlinecolor = "rgba(68, 68, 68, 0)",
            tickvals = [-50,-30,-15,0,15,30,50],
            ticks = "outside",
            ticklen = 3,
            ticksuffix = " C",
            showticksuffix = "all"
        )
    )
) for year in [years[-1]] for month in range(1,13)]
data[-1]['visible'] = True

steps = []
for i in range(len(data)):
    step = dict(
        method = 'restyle',  
        args = ['visible', [False] * len(data)],
        label = [str(month)+"."+str(year) for year in [years[-1]] for month in range(1,13)][i]
    )
    step['args'][1][i] = True # Toggle i'th trace to "visible"
    steps.append(step)

sliders = [dict(
    active = len(steps)-1,
    currentvalue = {"prefix": "Month and year: "},
    #pad = {"t": 50},
    steps = steps
)]

layout = dict(
    sliders=sliders,
    geo = dict(
        scope = 'world',
        showland = True,
        landcolor = "rgb(212, 212, 212)",
        showlakes = True,
        lakecolor = "rgb(255, 255, 255)",
        showsubunits = False,
        #subunitcolor = "rgb(255, 255, 255)",
        showcountries = False,
        #countrycolor = "rgb(255, 255, 255)",
        showcoastlines=False,
        resolution = 110
        ),
    )
fig = dict( data=data, layout=layout )


# Average monthly temperatures in:

# In[ ]:


print(dt.date.strftime(day, '%Y'))


# In[ ]:


iplot(fig)


# In[ ]:


month = 2


# In[ ]:


data = [ dict(
    visible = False,
    name = '',
    type = 'scattergeo',
    text = df_loc[(df_loc['MONTH']==month) & (df_loc['YEAR']==year)]['LBL'],
    lat = df_loc[(df_loc['MONTH']==month) & (df_loc['YEAR']==year)]['LAT'],
    lon = df_loc[(df_loc['MONTH']==month) & (df_loc['YEAR']==year)]['LON'],
    marker = dict(
        color = (df_loc[(df_loc['MONTH']==month) & (df_loc['YEAR']==year)]['TEMP']-32)*5/9,
        colorscale = scl,
        cmin = -50,
        cmax = 50,
        opacity = 0.5,
        size = 5,
        colorbar = dict(
            thickness = 10,
            titleside = "right",
            outlinecolor = "rgba(68, 68, 68, 0)",
            tickvals = [-50,-30,-15,0,15,30,50],
            ticks = "outside",
            ticklen = 3,
            ticksuffix = " C",
            showticksuffix = "all"
        )
    )
) for year in years]
data[-1]['visible'] = True

steps = []
for i in range(len(data)):
    step = dict(
        method = 'restyle',  
        args = ['visible', [False] * len(data)],
        label = [str(month)+"."+str(year) for year in years][i]
    )
    step['args'][1][i] = True # Toggle i'th trace to "visible"
    steps.append(step)

sliders = [dict(
    active = len(steps)-1,
    currentvalue = {"prefix": "Month and year: "},
    #pad = {"t": 50},
    steps = steps
)]

layout = dict(
    sliders=sliders,
    geo = dict(
        scope = 'world',
        showland = True,
        landcolor = "rgb(212, 212, 212)",
        showlakes = True,
        lakecolor = "rgb(255, 255, 255)",
        showsubunits = False,
        #subunitcolor = "rgb(255, 255, 255)",
        showcountries = False,
        #countrycolor = "rgb(255, 255, 255)",
        showcoastlines=False,
        resolution = 110
        ),
    )
fig = dict( data=data, layout=layout )


# February temperatures

# In[ ]:


iplot(fig)


# In[ ]:


month = 7


# In[ ]:


data = [ dict(
    visible = False,
    name = '',
    type = 'scattergeo',
    text = df_loc[(df_loc['MONTH']==month) & (df_loc['YEAR']==year)]['LBL'],
    lat = df_loc[(df_loc['MONTH']==month) & (df_loc['YEAR']==year)]['LAT'],
    lon = df_loc[(df_loc['MONTH']==month) & (df_loc['YEAR']==year)]['LON'],
    marker = dict(
        color = (df_loc[(df_loc['MONTH']==month) & (df_loc['YEAR']==year)]['TEMP']-32)*5/9,
        colorscale = scl,
        cmin = -50,
        cmax = 50,
        opacity = 0.5,
        size = 5,
        colorbar = dict(
            thickness = 10,
            titleside = "right",
            outlinecolor = "rgba(68, 68, 68, 0)",
            tickvals = [-50,-30,-15,0,15,30,50],
            ticks = "outside",
            ticklen = 3,
            ticksuffix = " C",
            showticksuffix = "all"
        )
    )
) for year in years]
data[-1]['visible'] = True

steps = []
for i in range(len(data)):
    step = dict(
        method = 'restyle',  
        args = ['visible', [False] * len(data)],
        label = [str(month)+"."+str(year) for year in years][i]
    )
    step['args'][1][i] = True # Toggle i'th trace to "visible"
    steps.append(step)

sliders = [dict(
    active = len(steps)-1,
    currentvalue = {"prefix": "Month and year: "},
    #pad = {"t": 50},
    steps = steps
)]

layout = dict(
    sliders=sliders,
    geo = dict(
        scope = 'world',
        showland = True,
        landcolor = "rgb(212, 212, 212)",
        showlakes = True,
        lakecolor = "rgb(255, 255, 255)",
        showsubunits = False,
        #subunitcolor = "rgb(255, 255, 255)",
        showcountries = False,
        #countrycolor = "rgb(255, 255, 255)",
        showcoastlines=False,
        resolution = 110
        ),
    )
fig = dict( data=data, layout=layout )


# July temperatures

# In[ ]:


iplot(fig)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




