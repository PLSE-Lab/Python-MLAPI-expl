#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# import the necessary libraries
import numpy as np 
import pandas as pd 

# Visualisation libraries
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set()
from plotly.offline import init_notebook_mode, iplot 
import plotly.graph_objs as go
import plotly.offline as py
import pycountry
py.init_notebook_mode(connected=True)
import folium 
from folium import plugins
from folium.plugins import MarkerCluster


# Graphics in retina format 
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

# Increase the default plot size and set the color scheme
plt.rcParams['figure.figsize'] = 8, 5
#plt.rcParams['image.cmap'] = 'viridis'


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Disable warnings 
import warnings
warnings.filterwarnings('ignore')


# # COVID-19 Visual Analysis
# 
# On 31 December 2019, WHO was informed of a cluster of cases of pneumonia of unknown cause detected in Wuhan City, Hubei Province of China. 2019 Novel Coronavirus (COVID-19) is a virus (more specifically, a coronavirus) identified as the cause of an outbreak of respiratory illness first detected in Wuhan, China.
# 
# In addition to providing care to patients and isolating new cases as they are identified, Chinese public health officials have reported that they remain focused on continued contact tracing, conducting environmental assessments at the wholesale market, and investigations to identify the pathogen causing the outbreak. *(source: WHO)

# In[ ]:


# Reading the dataset
data= pd.read_csv("/kaggle/input/2020-corona-virus-timeseries/COVID-19_geo_timeseries_ver_0311.csv")
data.head()


# In[ ]:


data = data[data.data_source=='jhu']
# Convert Last Update column to datetime64 format
data['update_time'] = pd.to_datetime(data['update_time'])
print(data['update_time'].dtype)
# Extract date from the timestamp
data['update_date'] = data['update_time'].dt.date


# In[ ]:


from matplotlib import pyplot as plt
import plotly.graph_objects as go
from fbprophet import Prophet
import pycountry
import plotly.express as px

df = data[data.data_source=='jhu']
df_agg = df.groupby('update_date').agg({'confirmed_cases':'sum','deaths':'sum','recovered':'sum'}).reset_index()


# In[ ]:


fig = go.Figure()
fig.add_trace(go.Bar(x=df_agg['update_date'],
                y=df_agg['confirmed_cases'],
                name='Confirmed',
                marker_color='blue'
                ))
fig.add_trace(go.Bar(x=df_agg['update_date'],
                y=df_agg['deaths'],
                name='Deaths',
                marker_color='Red'
                ))
fig.add_trace(go.Bar(x=df_agg['update_date'],
                y=df_agg['recovered'],
                name='Recovered',
                marker_color='Green'
                ))

fig.update_layout(
    title='Worldwide Corona Virus Cases - Confirmed, Deaths, Recovered',
    xaxis_tickfont_size=14,
    yaxis=dict(
        title='Number of Cases',
        titlefont_size=16,
        tickfont_size=14,
    ),
    legend=dict(
        x=0,
        y=1.0,
        bgcolor='rgba(255, 255, 255, 0)',
        bordercolor='rgba(255, 255, 255, 0)'
    ),
    barmode='group',
    bargap=0.15, # gap between bars of adjacent location coordinates.
    bargroupgap=0.1 # gap between bars of the same location coordinate.
)
fig.show()


# In[ ]:


# Quick glimpse of the data info
data.info()


# ## Countries affected by the COVID-19

# In[ ]:


# Countries affected
countries = data[(data['country']!='Others') & (data['country']!='Undisclosed')]['country'].unique().tolist()
# Use this print trick to get more readable list output
print(*countries, sep = "\n")
print("\nTotal countries affected by COVID-19: ",len(countries))


# ## Current status worldwide as of Feb 12, 2020

# In[ ]:


# get the latest timestamp
latest_date = data['update_time'].max()
# extract year, month, day from the latest timestamp so we can use it just report the latest data
year = latest_date.year
month = latest_date.month
# adjust for timezone
day = latest_date.day - 1

# Filter to only include the latest day data
from datetime import date
data_latest = data[data['update_time'] > pd.Timestamp(date(year,month,day))]
data_latest.head()


# In[ ]:


# Creating a dataframe with total no of confirmed cases for every country as of the latest available date
affected_country_latest = data_latest.groupby(['country','country_code','region','latitude','longitude','country_flag']).agg({'update_time': np.max}).reset_index()
key = ['country','country_code','region','latitude','longitude','country_flag','update_time']
global_cases = pd.merge(data_latest, affected_country_latest, how='inner', on=key).drop_duplicates().groupby(key).max().sort_values(by=['confirmed_cases'],ascending=False).reset_index()
global_cases.index+=1
global_cases_columns = global_cases.columns.tolist()
global_cases_columns.remove('update_time')
global_cases = global_cases[global_cases_columns]
global_cases


# ## Generating world map

# In[ ]:


shape_url = '/kaggle/input/python-folio-country-boundaries/world-countries.json'
world_geo = shape_url

m = folium.Map(location=[35.86166,104.195397], zoom_start=3,tiles='Stamen Toner')

folium.Choropleth(
    geo_data=world_geo,
    name='choropleth',
    data=global_cases,
    columns=['country', 'confirmed_cases'],
    key_on='feature.properties.name',
    fill_color='OrRd',
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name='Number of Confirmed Cases'
).add_to(m)

for lat, lon, value, name in zip(global_cases['latitude'], global_cases['longitude'], global_cases['confirmed_cases'], global_cases['country']):
    folium.CircleMarker(
        [lat, lon],
        radius=10,
        popup = ('<strong>Country</strong>: ' + str(name).capitalize() + '<br>'
                 '<strong>Confirmed Cases</strong>: ' + str(value) + '<br>'),        
        color='orange',
        fill=True,
        fill_color='orange',
        fill_opacity=0.7
    ).add_to(m)

folium.LayerControl().add_to(m)

m


# # Interactive World Map with Time Steps (All Dates)

# In[ ]:


# Creating a dataframe with total no of confirmed cases for every country for all available dates
key1 = ['country','country_code','region','latitude','longitude','country_flag','update_date']
key2 = ['country','country_code','region','latitude','longitude','country_flag','update_date','confirmed_cases','deaths','recovered']
key3 = ['country','country_code','region','latitude','longitude','country_flag']
df_full = data[data.data_source == 'jhu'][key2].drop_duplicates().groupby(key1).max().reset_index()
# df_full = data[key2].drop_duplicates().groupby(key1).max().sort_values(by=['country','update_date']).groupby(key3).cumsum().sort_values(by=['confirmed_cases','update_date'],ascending=[False,False]).reset_index()
# df_full = df_full.groupby(key1).agg({'confirmed_cases':np.cumsum, 'deaths':np.cumsum ,'recovered':np.cumsum}).reset_index()
df_full[['confirmed_cases','deaths','recovered']] = df_full[['confirmed_cases','deaths','recovered']].fillna(0)
df_full['log_confirmed_cases'] = np.log(df_full['confirmed_cases'])
df_full.sort_values(by=['confirmed_cases','update_date'],ascending=[False,False]).head(10)


# In[ ]:


import plotly
import plotly.graph_objs as go
from datetime import datetime
from datetime import timedelta

scl = [[0.0, '#e7e1ef'],[0.2, '#d4b9da'],[0.4, '#c994c7'], 
       [0.6, '#df65b0'],[0.8, '#dd1c77'],[1.0, '#980043']] # reds

data_slider = []
all_dates = df_full['update_date'].sort_values().unique()
for m,d in zip(pd.DatetimeIndex(all_dates).month,pd.DatetimeIndex(all_dates).day):
    df_selected = df_full[(pd.DatetimeIndex(df_full['update_date']).month==m) & (pd.DatetimeIndex(df_full['update_date']).day==d)]
    df_selected['text'] =   'Date: '+ df_selected['update_date'].astype(str)                             + '<br>' + 'Confirmed Cases: ' + df_selected['confirmed_cases'].astype(str)                             + '<br>' + 'Deaths: '+ df_selected['deaths'].astype(str)                             + '<br>' + 'Recovered: '+ df_selected['recovered'].astype(str)
    data_one_day = dict(
        type='choropleth',
        colorscale = scl,
        autocolorscale=False,
        locations = df_selected['country'].tolist(),
        z = df_selected['log_confirmed_cases'].tolist(),
        locationmode = 'country names',
        text = df_selected['text'],
        colorbar_title = 'Confirmed Cases (Logarithm)'
    )
    data_slider.append(data_one_day)

steps = []
for i in range(len(data_slider)):
    step = dict(method='restyle',
                args=['visible', [False] * len(data_slider)],
                label=(datetime.strptime('2020-01-21','%Y-%m-%d') + timedelta(days=i)).strftime('%Y-%m-%d')
               )
    step['args'][1][i] = True
    steps.append(step)

sliders = [dict(active=0, pad={"t": 1}, steps=steps)]  

lyt = dict(
    geo=dict(scope='world'), 
    sliders=sliders, 
    title_text = 'COVID-19 Trend Analysis (World)' + '<br>' + '(Hover for breakdown)'
)
fig = dict(data=data_slider, layout=lyt)
plotly.offline.iplot(fig)


# # China Visual Analysis by Provinces

# In[ ]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
#Mainland China
key = ['province','country']
China = data_latest[data_latest['country']=='China'].groupby(key).agg({'confirmed_cases':np.max,'deaths':np.max,'recovered':np.max,'update_date':np.max}).fillna(0).reset_index()
China['log_confirmed_cases'] = np.log(China['confirmed_cases'])
China['log_recovered'] = np.log(China['recovered'])
China['log_deaths'] = np.log(China['deaths'])
China['norm_confirmed_cases'] = scaler.fit_transform(China[['confirmed_cases']])
China['norm_recovered'] = scaler.fit_transform(China[['recovered']])
China['norm_deaths'] = scaler.fit_transform(China[['deaths']])
China = China.sort_values(by='confirmed_cases',ascending=False)
China


# ## Confirmed vs Recovered figures of Provinces of China
# Due to the total number of cases in Hubei Province is much larger than the rest provinces, for better visualization, it's not shown here.

# In[ ]:


f, ax = plt.subplots(figsize=(12, 8))

sns.set_color_codes("pastel")

sns.barplot(x="confirmed_cases", y="province", data=China[1:],
            label="confirmed_cases", color="r")

sns.set_color_codes("muted")
sns.barplot(x="recovered", y="province", data=China[1:],
            label="recovered", color="g")

# Add a legend and informative axis label
ax.legend(ncol=2, loc="lower right", frameon=True)
ax.set(xlim=(0, 1500), ylabel="",xlabel="# cases",title="Confirmed vs Recovered")
sns.despine(left=True, bottom=True)


# In[ ]:


# prepare China Mainland data
china_coordinates= pd.read_csv("../input/chinese-cities/china_coordinates.csv")
china_coordinates.rename(columns={'admin':'province'},inplace=True)
china_coordinates = china_coordinates[(china_coordinates.capital == 'admin') | (china_coordinates.capital == 'primary')]
china_merged = China.merge(china_coordinates,on='province', how='left')

key_china = ['province','lat','lng','confirmed_cases','recovered','deaths','log_confirmed_cases','log_recovered','log_deaths','norm_confirmed_cases','norm_recovered','norm_deaths']
china_merged = china_merged[key_china].dropna()
china_merged


# # [Interactive] Heat Map of Confirmed Cases by China's Provinces

# In[ ]:


import json
import branca
latitude = 30.86166
longitude = 114.195397

china_shape_url = '/kaggle/input/china-regions-map/china-provinces.json'

china_confirmed_colorscale = branca.colormap.linear.YlOrRd_09.scale(0, 2000)
china_confirmed_series = china_merged.set_index('province')['confirmed_cases']

def confirmed_style_function(feature):
    china_show = china_confirmed_series.get(str(feature['properties']['NAME_1']), None)
    return {
        'fillOpacity': 0.5,
        'weight': 0,
        'fillColor': '#black' if china_show is None else china_confirmed_colorscale(china_show)
    }

china_confirmed_map = folium.Map(location=[latitude, longitude], zoom_start=4.5,tiles='Stamen Toner')

folium.TopoJson(
    json.load(open(china_shape_url)),
    'objects.CHN_adm1',
    style_function=confirmed_style_function
).add_to(china_confirmed_map)

for lat, lon, rd, value, name in zip(china_merged['lat'], china_merged['lng'], china_merged['log_confirmed_cases'], china_merged['confirmed_cases'], china_merged['province']):
    folium.CircleMarker([lat, lon],
                        radius=rd*4,
                        tooltip = ('Province: ' + str(name).capitalize() + '<br>'
                        'Confirmed Cases: ' + str(f"{int(value):,}") + '<br>'),
                        color='none',
                        fill_color='purple',
                        fill_opacity=0.5 ).add_to(china_confirmed_map)

china_confirmed_map


# # [Interactive] Heat Map of Deceased Cases by China's Provinces

# In[ ]:


china_deceased_colorscale = branca.colormap.linear.PuRd_09.scale(0, 20)
china_deceased_series = china_merged.set_index('province')['deaths']

def deceased_style_function(feature):
    china_show = china_deceased_series.get(str(feature['properties']['NAME_1']), None)
    return {
        'fillOpacity': 0.5,
        'weight': 0,
        'fillColor': '#black' if china_show is None else china_deceased_colorscale(china_show)
    }

china_deceased_map = folium.Map(location=[latitude, longitude], zoom_start=4.5,tiles='Stamen Toner')

folium.TopoJson(
    json.load(open(china_shape_url)),
    'objects.CHN_adm1',
    style_function=deceased_style_function
).add_to(china_deceased_map)

for lat, lon, rd, value, name in zip(china_merged['lat'], china_merged['lng'], china_merged['log_deaths'], china_merged['deaths'], china_merged['province']):
    folium.CircleMarker([lat, lon],
                        radius=rd*4,
                        tooltip = ('Province: ' + str(name).capitalize() + '<br>'
                        'Deaths: ' + str(f"{int(value):,}") + '<br>'),
                        color='red',
                        fill_color='black',
                        fill_opacity=0.5 ).add_to(china_deceased_map)
china_deceased_map


# # [Interactive] Heat Map of Recovered Cases by China's Provinces

# In[ ]:


china_recovered_colorscale = branca.colormap.linear.YlGn_09.scale(0, 500)
china_recovered_series = china_merged.set_index('province')['recovered']

def recovered_style_function(feature):
    china_show = china_recovered_series.get(str(feature['properties']['NAME_1']), None)
    return {
        'fillOpacity': 0.5,
        'weight': 0,
        'fillColor': '#black' if china_show is None else china_recovered_colorscale(china_show)
    }

china_recovered_map = folium.Map(location=[latitude, longitude], zoom_start=4.5,tiles='Stamen Toner')

folium.TopoJson(
    json.load(open(china_shape_url)),
    'objects.CHN_adm1',
    style_function=recovered_style_function
).add_to(china_recovered_map)

for lat, lon, rd, value, name in zip(china_merged['lat'], china_merged['lng'], china_merged['log_recovered'], china_merged['recovered'], china_merged['province']):
    folium.CircleMarker([lat, lon],
                        radius=rd*4,
                        tooltip = ('Province: ' + str(name) + '<br>'
                        'Recovered: ' + str(f"{int(value):,}") + '<br>'),
                        color='none',
                        fill_color='#6baed6',
                        fill_opacity=0.5 ).add_to(china_recovered_map)
china_recovered_map


# In[ ]:




