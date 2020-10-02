#!/usr/bin/env python
# coding: utf-8

# ## Introduction
# This kernel is about data exploration and storytelling of the air pollution in Seoul.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from bokeh.io import output_file,show,output_notebook,push_notebook
from bokeh.plotting import figure
from bokeh.models import BoxAnnotation
from bokeh.models.widgets import Tabs,Panel
from bokeh.models.formatters import DatetimeTickFormatter
output_notebook()


# In[ ]:


df = pd.read_csv('../input/air-pollution-in-seoul/AirPollutionSeoul/Measurement_summary.csv')
df_item = pd.read_csv('../input/air-pollution-in-seoul/AirPollutionSeoul/Original Data/Measurement_item_info.csv')
df_measure = pd.read_csv('../input/air-pollution-in-seoul/AirPollutionSeoul/Original Data/Measurement_info.csv')
df_station = pd.read_csv('../input/air-pollution-in-seoul/AirPollutionSeoul/Original Data/Measurement_station_info.csv')

# Change measurement date to datetime type, and separate them
# Set indexing on station code, date, and time
df['Measurement date'] = pd.to_datetime(df['Measurement date'])
df['date'] = [d.date() for d in df['Measurement date']]
df['time'] = [d.time() for d in df['Measurement date']]

df_measure['Measurement date'] = pd.to_datetime(df_measure['Measurement date'])

item_codes = {
    1:'SO2', 
    3:'NO2', 
    5:'CO',
    6:'O3',
    8:'PM10',
    9:'PM2.5'
}

# Mapping item code
df_measure['pollutant'] = df_measure['Item code'].apply(lambda x: item_codes[x]) 


# ## Understanding the Data
# - **This data measures six pollutants.** (SO2, NO2, CO, O3, PM10, PM2.5).
# - **Data were measured every hour between 2017 and 2019 in 25 districts in Seoul.** Consistent timeframe makes the data easier to interpret.
# - **Not all data was recorded perfectly.** Some of them was either recorded with abnormalities or not recorded. We can approximate the original data by doing some either preprocessing, or approximation by prediction models.
# - **The Lower, The Better.** Pollution is a thing that can be measured by how worse it is or how safe it is. So how we read the pollution measurement data?  We can look at the `Measurement_info.csv` file, the less the value of the pollutant, the least pollution is happening.
# - **Item code indicates which measured pollutant.** (1 -> SO2; 3 -> NO2; 5 -> CO; 6 -> O3; 8 -> PM10; 9 -> PM2.5)

# ## What time of the day is the most pollution occurs?
# 
# We know that these measuring pollution instruments have some abnormality conditions. In this case, I only use the datas that recorded with no abnormalities to give the real condition of the pollution. By using the instrument status (0 status) on the measurement info, we can filter the datas that are normally recorded.
# 
# 

# In[ ]:


pollutant_cols = ['SO2', 'NO2', 'CO', 'O3', 'PM10', 'PM2.5']
def filter_normal(df_1, df_2, col):
    return df_2[df_2['pollutant'] == col].merge(df_1[['Measurement date', 'Station code', col]], on=['Measurement date', 'Station code'])

df_pollutant = {}
for c in pollutant_cols:
    df_merged = filter_normal(df, df_measure[df_measure['Instrument status'] == 0], c)
    df_merged['date'] = df_merged['Measurement date'].dt.date
    df_merged['time'] = df_merged['Measurement date'].dt.time
    df_pollutant[c] = df_merged.copy()


# These are the average value in 24 hours (of all time) of each pollutant in each measurement station.

# In[ ]:


# Tabs of mean in each station
units = {
    'SO2':'ppm', 
    'NO2':'ppm', 
    'CO':'ppm',
    'O3':'ppm',
    'PM10':'Mircrogram/m3',
    'PM2.5':'Mircrogram/m3'
}

lp = []
for c in pollutant_cols:
    p = figure(plot_width=700, plot_height=400)
    dt = df_pollutant[c].groupby(['Station code', 'time']).mean().reset_index()
    for x in df_station['Station code'].unique():
        dtemp = dt[dt['Station code'] == x]
        p.line(dtemp['time'], dtemp['Average value'], line_width=1)
    
    label = df_item[df_item['Item name'] == c]
    box = BoxAnnotation(top=float(label['Good(Blue)']), fill_alpha=0.1, fill_color='gray')
    p.add_layout(box)
    box = BoxAnnotation(bottom=float(label['Good(Blue)']), top=float(label['Normal(Green)']), fill_alpha=0.1, fill_color='blue')
    p.add_layout(box)
    box = BoxAnnotation(bottom=float(label['Normal(Green)']), top=float(label['Bad(Yellow)']), fill_alpha=0.1, fill_color='green')
    p.add_layout(box)
    box = BoxAnnotation(bottom=float(label['Bad(Yellow)']), top=float(label['Very bad(Red)']), fill_alpha=0.1, fill_color='yellow')
    p.add_layout(box)
    box = BoxAnnotation(bottom=float(label['Very bad(Red)']), fill_alpha=0.1, fill_color='red')
    p.add_layout(box)

    p.xaxis.formatter = DatetimeTickFormatter(hours='%Hh')
    p.xaxis.axis_label = 'Time (h)'
    p.yaxis.axis_label = units[c]

    tab = Panel(child=p, title=c)
    lp.append(tab)
    
tabs = Tabs(tabs=lp)
show(tabs)


# Yes, I agree that drawing 100+ lines at a single plot is ugly. But by doing this, it helped us to see a pattern on the plots. It seems that every measurement have the similar function added by various offset. I decided to calculate the mean of those line, so there will be a single line that represents the pollution. 

# In[ ]:


# Tabs of overall means
lp = []
for c in pollutant_cols:
    p = figure(plot_width=700, plot_height=400, x_axis_type='datetime')
    dt = df_pollutant[c].groupby(['time']).mean().reset_index()
    p.line(dt['time'], dt['Average value'], line_width=1)
    
    label = df_item[df_item['Item name'] == c]
    box = BoxAnnotation(top=float(label['Good(Blue)']), fill_alpha=0.1, fill_color='gray')
    p.add_layout(box)
    box = BoxAnnotation(bottom=float(label['Good(Blue)']), top=float(label['Normal(Green)']), fill_alpha=0.1, fill_color='blue')
    p.add_layout(box)
    box = BoxAnnotation(bottom=float(label['Normal(Green)']), top=float(label['Bad(Yellow)']), fill_alpha=0.1, fill_color='green')
    p.add_layout(box)
    box = BoxAnnotation(bottom=float(label['Bad(Yellow)']), top=float(label['Very bad(Red)']), fill_alpha=0.1, fill_color='yellow')
    p.add_layout(box)
    box = BoxAnnotation(bottom=float(label['Very bad(Red)']), fill_alpha=0.1, fill_color='red')
    p.add_layout(box)
    
    p.xaxis.formatter = DatetimeTickFormatter(hours='%Hh')
    p.xaxis.axis_label = 'Time (h)'
    p.yaxis.axis_label = units[c]
    
    tab = Panel(child=p, title=c)
    lp.append(tab)
    
tabs = Tabs(tabs=lp)
show(tabs)


# Although all the plots seems to tell us that they have either low value or none (gray fill), there is a specific time of the day that the value of the pollution will changes. For example, the pollutant SO2 will start increasing at the morning drastically (about 6 a.m.) and will be reaching it's peak at 10, then slowly decreasing. This makes sense beacuse SO2 are produced by the emission of the fossil fuel.
# And the other plots continues..

# ## The Bottom of the Notebook
# 
# In the end, i want to say again that this notebook provides the storytelling of the air pollution data in Seoul. Big thanks to Kaggle and [bappe](https://www.kaggle.com/bappekim) for providing the data.
# I will update this kernel if I find another story either by myself or other kernels on the kaggle community.
# I hope you learn something useful by reading this notebook.
# 
# Thank you.
