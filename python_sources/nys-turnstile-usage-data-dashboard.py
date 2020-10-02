#!/usr/bin/env python
# coding: utf-8

# # NYS Turnstile Usage Data 2018 Interactive Dashboard
# ## Introduction
# This Notebook puts into practice the Dashboarding Tutorial created by kaggler [Rachael Tatman](https://www.kaggle.com/rtatman), which is found [here](https://www.kaggle.com/rtatman/dashboarding-with-notebooks-day-1?utm_medium=email&utm_source=intercom&utm_campaign=dashboarding-event).
# 
# The NYS Turnstile Usage Data 2018 is a public dataset that collects entry/exit register values for individual control areas of Subway stations in the New York Metropolitan Area. The chart and map created below show the passenger activity of the three most busy stations during a 90-day period, ending on the last date the data was  updated.

# In[ ]:


# PART 1
# Choosing a dataset and exploring relevant data to include in dashboard
# Loading the NYS Turnstile Data
import pandas as pd 
import numpy as np

turnstile_df = pd.read_csv(r'../input/nys-turnstile-usage-data/turnstile-usage-data-2018.csv')
turnstile_df.head()
#Explore shape
#print('Dataset has',turnstile_df.shape[0],'Rows and',turnstile_df.shape[1],'Columns.')
#Choose most recent values
date_filter = turnstile_df['Date'] > str(pd.to_datetime(turnstile_df['Date']).max() -
                                         pd.Timedelta(days=90))#filter last 90 days from current time. 
turnstile_df = turnstile_df[date_filter]
# turnstile_df.head(10)
#Sorting the Linename values alphabetically. 
turnstile_df['Line Name'] = turnstile_df['Line Name'].apply(lambda x: ''.join(sorted(x)))
# turnstile_df.head(10)


# In[ ]:


#Take out relevant features from dataframe and build a dictionary.
from collections import defaultdict #import so you can append values to the list value in dict.

temp_d = defaultdict(list) #Dictionary (C_A, unit, scp, station, linename, date)=>Entries list
for row in turnstile_df.itertuples():
    C_A, unit, scp, station, linename, date = row[1], row[2], row[3], row[4],                                                ''.join(sorted(row[5])), row[7]
    entries = row[10]
    k = (C_A, unit, scp, station, linename, date)
    temp_d[k].append(entries)
    
turnstile_d = {}    
for key, value in temp_d.items():
    entry = abs(max(value) - min(value)) #Correct for turnstiles that count backwards
    turnstile_d[key] = entry
    
clean_df = pd.DataFrame.from_dict(turnstile_d, orient='index')
clean_df.rename(columns = {0:'Entries'}, inplace=True)
# clean_df.head()

#Create a new dataframe using the dictionary previously built.
turnstile_df = pd.DataFrame(columns=[])
turnstile_df['C/A'] = [row[0][0] for row in clean_df.itertuples()]
turnstile_df['Unit'] = [row[0][1] for row in clean_df.itertuples()]
turnstile_df['SCP'] = [row[0][2] for row in clean_df.itertuples()]
turnstile_df['Station'] = [row[0][3] for row in clean_df.itertuples()]
turnstile_df['Linename'] = [row[0][4] for row in clean_df.itertuples()]
turnstile_df['Date'] = [row[0][5] for row in clean_df.itertuples()]
turnstile_df['Entries'] = [row[1] for row in clean_df.itertuples()]


# In[ ]:


#Turnstiles work in crazy ways, sometimes counting backwards, sometimes resetting, etc.
#This behavior produces a lot of incorrect values in the data, and this gives us unwanted outliers.
#Next, the outliers will be removed:
def delete_outliers(df, iters=7): #iters = number of outliers to delete
    """The iter number needs to be calibrated carefully. First try iterating 1 time, 
    check if there are outliers left, then try iterating 2 times and so on, 
    until there are no oultliers left"""
    for i in range(iters): #Each iteration will remove an outlier
        to_delete_rows = df.loc[df.groupby(["Station","Linename"])['Entries'].idxmax()]
        to_delete_indices = list(to_delete_rows.index.values)
        df.drop(to_delete_indices, inplace=True)
    return df

no_outliers_turnstile_df = delete_outliers(turnstile_df)
no_outliers_turnstile_df['Entries'].max()

#Sort the stations with the most entries.
no_outliers_turnstile_df.groupby('Station').sum().sort_values('Entries',ascending=False).head()


# In[ ]:


#Exploratory Plotting 
#This plot will also show if the outliers were properly removed.
import matplotlib.pyplot as plt

#Pick the three most busy stations and filter them.
filter_34stPenn = no_outliers_turnstile_df['Station'] == '34 ST-PENN STA'
filter_GRDcntrl = no_outliers_turnstile_df['Station'] == 'GRD CNTRL-42 ST'
filter_34stHerald = no_outliers_turnstile_df['Station'] == '34 ST-HERALD SQ'
stations_df = no_outliers_turnstile_df[filter_34stPenn | filter_GRDcntrl | filter_34stHerald]

stations_pivot_df = stations_df.pivot_table(index='Date',
                                            columns='Station',
                                            values='Entries',
                                            aggfunc=sum)

ax = stations_pivot_df.plot.line(figsize=(12,5))

ax.set_title('Entries Comparison Among The Three Busiest Stations')


# In[ ]:


get_ipython().system('pip install chart-studio')


# In[ ]:


# PART 2
# How to create effective dashboards in notebooks
#The Plotly library will be used for interactive visualizations :
# import plotly
import chart_studio.plotly as py
import plotly.graph_objs as go

# these two lines are what allow your code to show up in a notebook!
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode()

# specify that we want a scatter plot with date on the x axis and # of entries on the y axis
# a sactter plot was set for each train station
station1 = go.Scatter(x=stations_pivot_df.index, y=stations_pivot_df['34 ST-PENN STA'], 
                      name='34 ST-PENN STA', line=dict(color=('rgb(50, 205, 50)')), 
                      mode='lines+markers')
station2 = go.Scatter(x=stations_pivot_df.index, y=stations_pivot_df['GRD CNTRL-42 ST'], 
                      name='GRD CNTRL-42 ST', line=dict(color=('rgb(51, 153, 255)')), 
                      mode='lines+markers')
station3 = go.Scatter(x=stations_pivot_df.index, y=stations_pivot_df['34 ST-HERALD SQ'], 
                      name='34 ST-HERALD SQ', line=dict(color=('rgb(255, 165, 0)')), 
                      mode='lines+markers')

#Set data variable as a list to introduce in fig.
data = [station1,station2,station3]

# specify the layout of our figure
layout = dict(title = "Daily Station Entries for Three Months",
              xaxis= dict(title= 'Date',ticklen= 5,zeroline= False), 
              yaxis = dict(title = 'Entries',
                           tickformat=',',
                          hoverformat=','))#tickformat modifies number format

# create and show our figure
fig = dict(data = data, layout = layout)
iplot(fig)


# In[ ]:


locations = [(40.7488979,-73.9887886),(40.7503222,-73.9908707),(40.7521646,-73.9771532)] 
colors = ['rgb(255, 165, 0)','rgb(50, 205, 50)','rgb(51, 153, 255)']
scale = 110000

positions = ['top right','top left','top left']

# df['text'] = '<b>'+df['station']+', \n90 day count: '+df['cnt_str']+'</b>'
access = open('../input/access/access.txt','r')
mapbox_access_token = access.read()

data = [0,0,0]
for i in range(len(stations_pivot_df.columns)):
    st = dict(type='scattermapbox',
            lat=[locations[i][0]],
            lon=[locations[i][1]],
            mode='markers+text',
            name=stations_pivot_df.columns[i],
            marker=dict(
                size=stations_pivot_df[stations_pivot_df.columns[i]].sum()/scale,
                color=colors[i],
                opacity=0.65
            ),
            text='90-day count: \n'+
              str('{:,.0f}'.format(stations_pivot_df[stations_pivot_df.columns[i]].sum())),
            textposition=positions[i], 
            textfont=dict(
            size=13.5,
            color=colors[i]
        ))
    data[i] = st


layout = go.Layout(#Customize size of map with width=pixels and height=pixels when no autosize
    title="New York Stations with Highest Turnstile Count in a 90-Day Period",
    autosize=True,
    hovermode='closest',
    mapbox=dict(
        accesstoken=mapbox_access_token,
        bearing=0,
        center=dict(
            lat=40.75,
            lon=-73.985
        ),
        pitch=0, #Change map view, 0 means top view.
        zoom=13.5,
        style='dark'
    ),
)

fig = dict(data=data, layout=layout)

iplot(fig)

