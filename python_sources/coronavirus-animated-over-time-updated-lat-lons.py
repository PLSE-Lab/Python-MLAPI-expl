#!/usr/bin/env python
# coding: utf-8

# Great Kernel created by FBruzzesi.  Updated with new latitudes and longitutdes so kernel can run through Feb 4

# **Remark**: Many great kernels have already been posted. My goal is to explore the data using the Plotly animation feature in scatter and geo plots!
# For the moment this kernel has no further insights nor predictions.

# Load libraries and the dataset.

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
get_ipython().run_line_magic('matplotlib', 'inline')

import plotly.express as px
import plotly.graph_objs as go
from plotly.offline import iplot, init_notebook_mode
import plotly.figure_factory as ff
from plotly import subplots
from plotly.subplots import make_subplots
init_notebook_mode(connected=True)

from datetime import date, datetime, timedelta

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


def resumetable(df):
    print(f"Dataset Shape: {df.shape}")
    summary = pd.DataFrame(df.dtypes, columns=['dtypes'])
    summary = summary.reset_index()
    summary['Name'] = summary['index']
    summary = summary[['Name','dtypes']]
    summary['Missing'] = df.isnull().sum().values    
    summary['Uniques'] = df.nunique().values

    return summary


# In[ ]:


df = pd.read_csv("../input/novel-corona-virus-2019-dataset/2019_nCoV_data.csv",)
resumetable(df)


# Let's rename columns, change datetime to date format, drop rows with (0,0,0) triplets.

# In[ ]:


df.rename(columns={'Last Update': 'LastUpdate',
                   'Province/State': 'PS'},
         inplace=True)
df['Date'] = pd.to_datetime(df['Date']).dt.date

virus_cols=['Confirmed', 'Deaths', 'Recovered']

df = df[df[virus_cols].sum(axis=1)!=0]

resumetable(df)


# We see that there are lots of missing values in the Province/State column, let's fill with Country value if there are no other Province/State, and drop the remaining 2 rows.

# In[ ]:


df.loc[(df['PS'].isnull()) & (df.groupby('Country')['PS'].transform('nunique') == 0), 'PS'] =         df.loc[(df['PS'].isnull()) & (df.groupby('Country')['PS'].transform('nunique') == 0), 'Country'].to_numpy()

df['Country'] = np.where(df['Country']=='Mainland China', 'China', df['Country'])
df.dropna(inplace=True)
resumetable(df)


# Retrieve latitute and longitude for each Country-Province pair using geopy package.

# In[ ]:


#import time
#import geopy
#locator = geopy.Nominatim(user_agent='uagent')
#
#pairs = df[['Country', 'PS']].drop_duplicates().to_numpy()
##d={}
#for p in pairs:
#    if p[0] + ', ' + p[1] not in d:
#        l = p[0] + ', ' + p[1] if p[0]!=p[1] else p[0]
#        location = locator.geocode(l)
#
#        d[l] = [location.latitude, location.longitude]
#        print(l, location.latitude, location.longitude)
#        time.sleep(1)


# Yet, since I cannot make it work on kaggle notebook, here the full list 

# In[ ]:


d = {'Australia, Victoria': [-36.5986096, 144.6780052],
 'Australia, Queensland': [-22.1646782, 144.5844903],
 'Australia, New South Wales': [-31.8759835, 147.2869493],
 'Cambodia': [13.5066394, 104.869423],
 'Canada, Ontario': [50.000678, -86.000977],
 'Canada, British Columbia': [55.001251, -125.002441],
 'China, Anhui': [32.0, 117.0],
 'China, Fujian': [26.5450001, 117.842778],
 'China, Guizhou': [27.0, 107.0],
 'China, Hebei': [39.0000001, 116.0],
 'China, Jiangsu': [33.0000001, 119.9999999],
 'China, Macau': [22.1757605, 113.5514142],
 'China, Ningxia': [37.0000001, 105.9999999],
 'China, Shanxi': [37.0, 112.0],
 'China, Taiwan': [23.9739374, 120.9820179],
 'China, Yunnan': [25.0, 102.0],
 'China, Jilin': [42.9995032, 125.9816054],
 'China, Inner Mongolia': [43.2443242, 114.3251664],
 'China, Qinghai': [35.40709525, 95.95211573241954],
 'China, Tibet': [31.894343149999997, 87.07813712706509],
 'China, Guangxi': [24.0, 109.0],
 'China, Jiangxi': [28.0, 116.0],
 'China, Liaoning': [40.9975197, 122.9955469],
 'China, Shandong': [36.0000001, 118.9999999],
 'China, Gansu': [38.0000001, 101.9999999],
 'China, Heilongjiang': [48.0000047, 127.999992],
 'China, Xinjiang': [41.7574769, 87.16738423046897],
 'China, Shaanxi': [36.0, 109.0],
 'China, Hainan': [19.2000001, 109.5999999],
 'China, Hunan': [27.9995878, 112.009538],
 'China, Tianjin': [39.1235635, 117.1980785],
 'China, Henan': [34.0000001, 113.9999999],
 'China, Sichuan': [30.5000001, 102.4999999],
 'China, Chongqing': [30.05518, 107.8748712],
 'China, Shanghai': [31.2322758, 121.4692071],
 'China, Zhejiang': [29.0000001, 119.9999999],
 'China, Beijing': [39.906217, 116.3912757],
 'China, Guangdong': [23.1357694, 113.1982688],
 'China, Hubei': [31.15172525, 112.87832224656043],
 'Finland': [63.2467777, 25.9209164],
 'France': [46.603354, 1.8883335],
 'Germany, Bavaria': [48.9467562, 11.4038717],
 'Hong Kong': [22.2793278, 114.1628131],
 'India': [22.3511148, 78.6677428],
 'Italy': [42.6384261, 12.674297],
 'Japan': [36.5748441, 139.2394179],
 'Macau': [22.1757605, 113.5514142],
 'Malaysia': [4.5693754, 102.2656823],
 'Nepal': [28.1083929, 84.0917139],
 'Philippines': [12.7503486, 122.7312101],
 'Russia': [64.6863136, 97.7453061],
 'Singapore': [1.357107, 103.8194992],
 'South Korea': [36.5581914, 127.9408564],
 'Spain': [39.3262345, -4.8380649],
 'Sri Lanka': [7.5554942, 80.7137847],
 'Sweden': [59.6749712, 14.5208584],
 'Taiwan': [23.9739374, 120.9820179],
 'Thailand': [14.8971921, 100.83273],
 'UK': [54.7023545, -3.2765753],
 'US, Washington': [38.8948932, -77.0365529],
 'US, Chicago': [41.8755616, -87.6244212],
 'US, Illinois': [40.0796606, -89.4337288],
 'US, Arizona': [34.395342, -111.7632755],
 'US, California': [36.7014631, -118.7559974],
 'United Arab Emirates': [24.0002488, 53.9994829],
 'Vietnam': [13.2904027, 108.4265113],
 'Australia, South Australia': [-30.5343665, 135.6301212],
 'Germany, Hong Kong': [51.49144235, 11.974836546938391],
 'US, Boston, MA': [42.3602534, -71.0582912],
 'US, Los Angeles, CA': [34.0536909, -118.2427666],
 'US, Orange, CA': [29.7742659, -95.3341066],
 'US, Santa Clara, CA': [37.3541132, -121.9551744],
 'US, Seattle, WA': [47.6038321, -122.3300624],
 'US, Tempe, AZ': [33.4255056, -111.9400125],
 'US, Chicago, IL': [41.8755616, -87.6244212],
 'Canada, Toronto, ON': [43.651070, -79.347015],
 'US, San Benito, CA': [26.132576, -97.6311006],
 'Canada, London, ON': [42.983612, -81.249725],
 'Belgium': [50.85045, 4.34878] #using Brussels
    }


# In[ ]:


def coords(row):
    
    k = row['Country'] +', '+ row['PS'] if row['Country'] != row['PS'] else row['Country']
    try:
      row['lat'] = d[k][0]
      row['lon'] = d[k][1]
    except:
        print("Need lat and lon for", k)
    return row

df = df.apply(coords, axis=1)
df.head(10)


# In[ ]:


df = df.groupby(['PS', 'Country', 'Date']).agg({'Confirmed': 'sum',
                                                'Deaths': 'sum',
                                                'Recovered': 'sum',
                                                'lat': 'max',
                                                'lon': 'max'}).reset_index()
df = df[df['Date']>date(2020,1,20)]


# Let's plot the virus spreading in Asia and in the rest of the world over time. Size is related to number of confirmed cases, colorscale depends upon the number of deaths.

# In[ ]:


dates = np.sort(df['Date'].unique())
data = [go.Scattergeo(
            locationmode='country names',
            lon = df.loc[df['Date']==dt, 'lon'],
            lat = df.loc[df['Date']==dt, 'lat'],
            text = df.loc[df['Date']==dt, 'Country'] + ', ' + df.loc[df['Date']==dt, 'PS'] +   '-> Deaths: ' + df.loc[df['Date']==dt, 'Deaths'].astype(str) + ' Confirmed: ' + df.loc[df['Date']==dt,'Confirmed'].astype(str),
            mode = 'markers',
            marker = dict(
                size = (df.loc[df['Date']==dt,'Confirmed'])**(1/2.7)+3,
                opacity = 0.6,
                reversescale = True,
                autocolorscale = False,
                line = dict(
                    width=0.5,
                    color='rgba(0, 0, 0)'
                        ),
                #colorscale='rdgy', #'jet',rdylbu, 'oryel', 
                cmin=0,
                color=df.loc[df['Date']==dt,'Deaths'],
                cmax=df['Deaths'].max(),
                colorbar_title="Number of Deaths"
            )) 
        for dt in dates]


fig = go.Figure(
    data=data[0],
    layout=go.Layout(
        title = {'text': f'Corona Virus spreading in Asia, {dates[0]}',
                                'y':0.98,
                                'x':0.5,
                                'xanchor': 'center',
                                'yanchor': 'top'},
        geo = dict(
            scope='asia',
            projection_type='robinson',
            showland = True,
            landcolor = "rgb(252, 240, 220)",
            showcountries=True,
            showocean=True,
            oceancolor="rgb(219, 245, 255)",
            countrycolor = "rgb(128, 128, 128)",
            lakecolor ="rgb(219, 245, 255)",
            showrivers=True,
            showlakes=True,
            showcoastlines=True,
            countrywidth = 1,
            
            ),
     updatemenus=[dict(
            type="buttons",
            buttons=[dict(label="Play",
                          method="animate",
                          args=[None])])]),
    
    frames=[go.Frame(data=dt, 
                     layout=go.Layout(
                          title={'text': f'Corona Virus spreading in Asia, {date}',
                                'y':0.98,
                                'x':0.5,
                                'xanchor': 'center',
                                'yanchor': 'top'}
                           ))
            for dt,date in zip(data[1:],dates[1:])])

fig.show()


# In[ ]:


dates = np.sort(df['Date'].unique())
data = [go.Scattergeo(
            locationmode='country names',
            lon = df.loc[df['Date']==dt, 'lon'],
            lat = df.loc[df['Date']==dt, 'lat'],
            text = df.loc[df['Date']==dt, 'Country'] + ', ' + df.loc[df['Date']==dt, 'PS'] +   '-> Deaths: ' + df.loc[df['Date']==dt, 'Deaths'].astype(str) + ' Confirmed: ' + df.loc[df['Date']==dt,'Confirmed'].astype(str),
            mode = 'markers',
            marker = dict(
                size = (df.loc[df['Date']==dt,'Confirmed'])**(1/2.7)+3,
                opacity = 0.6,
                reversescale = True,
                autocolorscale = False,
                line = dict(
                    width=0.5,
                    color='rgba(0, 0, 0)'
                        ),
                #colorscale='rdgy', #'jet',rdylbu, 'oryel', 
                cmin=0,
                color=df.loc[df['Date']==dt,'Deaths'],
                cmax=df['Deaths'].max(),
                colorbar_title="Number of Deaths"
            )) 
        for dt in dates]


fig = go.Figure(
    data=data[0],
    layout=go.Layout(
        title = {'text': f'Corona Virus, {dates[0]}',
                                'y':0.98,
                                'x':0.5,
                                'xanchor': 'center',
                                'yanchor': 'top'},
        geo = dict(
            scope='world',
            projection_type='robinson',
            showland = True,
            landcolor = "rgb(252, 240, 220)",
            showcountries=True,
            showocean=True,
            oceancolor="rgb(219, 245, 255)",
            countrycolor = "rgb(128, 128, 128)",
            lakecolor ="rgb(219, 245, 255)",
            showrivers=True,
            showlakes=True,
            showcoastlines=True,
            countrywidth = 1,
            
            ),
     updatemenus=[dict(
            type="buttons",
            buttons=[dict(label="Play",
                          method="animate",
                          args=[None])])]),
    
    frames=[go.Frame(data=dt, 
                     layout=go.Layout(
                          title={'text': f'Corona Virus, {date}',
                                'y':0.98,
                                'x':0.5,
                                'xanchor': 'center',
                                'yanchor': 'top'}
                           ))
            for dt,date in zip(data[1:],dates[1:])])

fig.show()


# Lastly, let's see how number of confirmed, deaths and recovered evolve over time, in China and the rest of the world.
# 
# **Take care**, y-scales are very different!

# In[ ]:


import plotly.express as px
import plotly.graph_objs as go
from plotly.offline import iplot, init_notebook_mode
import plotly.figure_factory as ff
from plotly import subplots
init_notebook_mode(connected=True)

from datetime import timedelta

china=df.loc[df['Country']=='China']
hubei=china.loc[china['PS']=='Hubei']
rest_of_china=china.loc[china['PS']!='Hubei'].groupby('Date').sum().reset_index()

china=china.groupby('Date').sum().reset_index()

agg_df=df.groupby(['Country', 'Date']).sum().reset_index()

rest_df=agg_df.loc[agg_df['Country']!='China'].groupby('Date').sum().reset_index()



dates = np.sort(df['Date'].unique())
dt_range = [np.min(dates)-timedelta(days=1), np.max(dates)+timedelta(days=1)]

# Row 1
frames_hubei = [go.Scatter(x=hubei['Date'],
                           y=hubei.loc[hubei['Date']<=dt, 'Confirmed'],
                           name='Hubei, Confirmed',
                           legendgroup="21") for dt in dates]

frames_rchina = [go.Scatter(x=rest_of_china['Date'],
                           y=rest_of_china.loc[rest_of_china['Date']<=dt, 'Confirmed'],
                           name='Rest of China, Confirmed',
                           legendgroup="21") for dt in dates]


frames_world = [go.Scatter(x=rest_df['Date'],
                           y=rest_df.loc[rest_df['Date']<=dt, 'Confirmed'],
                           name='Rest of the World, Confirmed',
                           legendgroup="22") for dt in dates]


# Row 2
frames_china_d = [go.Scatter(x=china['Date'],
                           y=china.loc[china['Date']<=dt, 'Deaths'],
                           name='China, Deaths',
                           legendgroup="31") for dt in dates]

frames_china_r = [go.Scatter(x=china['Date'],
                           y=china.loc[china['Date']<=dt, 'Recovered'],
                           name='China, Recovered',
                           legendgroup="31") for dt in dates]


frames_world_d = [go.Scatter(x=rest_df['Date'],
                           y=rest_df.loc[rest_df['Date']<=dt, 'Deaths'],
                           name='Rest of World, Deaths',
                           legendgroup="32") for dt in dates]

frames_world_r = [go.Scatter(x=rest_df['Date'],
                           y=rest_df.loc[rest_df['Date']<=dt, 'Recovered'],
                           name='Rest of World, Recovered',
                           legendgroup="32") for dt in dates]




fig = make_subplots(
    rows=2, cols=2,
    specs=[[{}, {}],
           [{}, {}]],
    subplot_titles=("China, Confirmed", 'Rest of the World, Confirmed',
                    "China, Deaths & Recovered", 'Rest of the World, Deaths & Recovered'))


# Row 1: Confirmed
fig.add_trace(frames_hubei[0], row=1, col=1)
fig.add_trace(frames_rchina[0], row=1, col=1)
fig.add_trace(frames_world[0], row=1,col=2)


# Row 2: Deaths & Recovered
fig.add_trace(frames_china_d[0], row=2, col=1)
fig.add_trace(frames_china_r[0], row=2, col=1)
fig.add_trace(frames_world_d[0], row=2,col=2)
fig.add_trace(frames_world_r[0], row=2,col=2)


# Add Layout
fig.update_xaxes(showgrid=False)

fig.update_layout(
        title={
            'text': 'Corona Virus: Confirmed, Deaths & Recovered',
            'y':0.98,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
        height=750,
        legend_orientation="h",
        #legend=dict(x=1, y=0.4),
        xaxis1=dict(range=dt_range, autorange=False),
        yaxis1=dict(range=[-10, hubei['Confirmed'].max()*1.1 ], autorange=False),
        xaxis2=dict(range=dt_range, autorange=False),
        yaxis2=dict(range=[-10, rest_df['Confirmed'].max()*1.1 ], autorange=False),
        xaxis3=dict(range=dt_range, autorange=False),
        yaxis3=dict(range=[-10, np.max([china['Recovered'].max(), china['Deaths'].max()])*1.1 ], autorange=False),
        xaxis4=dict(range=dt_range, autorange=False),
        yaxis4=dict(range=[-0.5, np.max([rest_df['Recovered'].max(), rest_df['Deaths'].max()])*1.1], autorange=False),
        )


frames = [dict(
               name = str(dt),
               data = [frames_hubei[i], frames_rchina[i], frames_world[i],
                       frames_china_d[i], frames_china_r[i],
                       frames_world_d[i], frames_world_r[i]
                       ],
               traces=[0, 1, 2, 3, 4 ,5 ,6, 7]
              ) for i, dt in enumerate(dates)]



updatemenus = [dict(type='buttons',
                    buttons=[dict(label='Play',
                                  method='animate',
                                  args=[[str(dt) for dt in dates[1:]], 
                                         dict(frame=dict(duration=500, redraw=False), 
                                              transition=dict(duration=0),
                                              easing='linear',
                                              fromcurrent=True,
                                              mode='immediate'
                                                                 )])],
                    direction= 'left', 
                    pad=dict(r= 10, t=85), 
                    showactive =True, x= 0.6, y= -0.1, xanchor= 'right', yanchor= 'top')
            ]

sliders = [{'yanchor': 'top',
            'xanchor': 'left', 
            'currentvalue': {'font': {'size': 16}, 'prefix': 'Date: ', 'visible': True, 'xanchor': 'right'},
            'transition': {'duration': 500.0, 'easing': 'linear'},
            'pad': {'b': 10, 't': 50}, 
            'len': 0.9, 'x': 0.1, 'y': -0.2, 
            'steps': [{'args': [[str(dt)], {'frame': {'duration': 500.0, 'easing': 'linear', 'redraw': False},
                                      'transition': {'duration': 0, 'easing': 'linear'}}], 
                       'label': str(dt), 'method': 'animate'} for dt in dates     
                    ]}]



fig.update(frames=frames),
fig.update_layout(updatemenus=updatemenus,
                  sliders=sliders);
fig.show() 

