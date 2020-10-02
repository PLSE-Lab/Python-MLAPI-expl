#!/usr/bin/env python
# coding: utf-8

# # Current Seattle Crisis Dashboard

# ![Seattle](http://dthomas.mathematical.guru/seattle.jpg)

# In[ ]:


get_ipython().system('pip install chart_studio')


# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import plotly
import chart_studio.plotly as py

import plotly.graph_objects as go

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)    #THIS LINE IS MOST IMPORTANT AS THIS WILL DISPLAY PLOT ON 
#NOTEBOOK WHILE KERNEL IS RUNNING


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory


import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


data = pd.read_csv("../input/crisis-data.csv")
print('Dimensions: ', data.shape)
print('Unique IDs: ', data['Template ID'].nunique())
print('Data from ', min(data['Reported Date']), ' to ', max(data['Reported Date']))
data.head()


# In[ ]:


print('Data is current as of ', max(data['Reported Date']))


# In[ ]:


data.loc[data.Beat == "L2","Lon"] = -122.320
data.loc[data.Beat == "M1","Lon"] = -122.310
data.loc[data.Beat == "K3","Lon"] = -122.323
data.loc[data.Beat == "K2","Lon"] = -122.327
data.loc[data.Beat == "K1","Lon"] = -122.327
data.loc[data.Beat == "M2","Lon"] = -122.329
data.loc[data.Beat == "M3","Lon"] = -122.329
data.loc[data.Beat == "L1","Lon"] = -122.314
data.loc[data.Beat == "L3","Lon"] = -122.299
data.loc[data.Beat == "N2","Lon"] = -122.327
data.loc[data.Beat == "N3","Lon"] = -122.327
data.loc[data.Beat == "N1","Lon"] = -122.364
data.loc[data.Beat == "J1","Lon"] = -122.386
data.loc[data.Beat == "J2","Lon"] = -122.380
data.loc[data.Beat == "J3","Lon"] = -122.337
data.loc[data.Beat == "B3","Lon"] = -122.334
data.loc[data.Beat == "B2","Lon"] = -122.354
data.loc[data.Beat == "B1","Lon"] = -122.391
data.loc[data.Beat == "U1","Lon"] = -122.316
data.loc[data.Beat == "U2","Lon"] = -122.308
data.loc[data.Beat == "U3","Lon"] = -122.291
data.loc[data.Beat == "W1","Lon"] = -122.377
data.loc[data.Beat == "W2","Lon"] = -122.385
data.loc[data.Beat == "W3","Lon"] = -122.381
data.loc[data.Beat == "F1","Lon"] = -122.361
data.loc[data.Beat == "F2","Lon"] = -122.361
data.loc[data.Beat == "F3","Lon"] = -122.337
data.loc[data.Beat == "O1","Lon"] = -122.336
data.loc[data.Beat == "O2","Lon"] = -122.334
data.loc[data.Beat == "O3","Lon"] = -122.320
data.loc[data.Beat == "S1","Lon"] = -122.296
data.loc[data.Beat == "S2","Lon"] = -122.288
data.loc[data.Beat == "S3","Lon"] = -122.268
data.loc[data.Beat == "R1","Lon"] = -122.304
data.loc[data.Beat == "R2","Lon"] = -122.306
data.loc[data.Beat == "R3","Lon"] = -122.289
data.loc[data.Beat == "C1","Lon"] = -122.213
data.loc[data.Beat == "C2","Lon"] = -122.305
data.loc[data.Beat == "C3","Lon"] = -122.291
data.loc[data.Beat == "E1","Lon"] = -122.317
data.loc[data.Beat == "E2","Lon"] = -122.313
data.loc[data.Beat == "E3","Lon"] = -122.316
data.loc[data.Beat == "G1","Lon"] = -122.313
data.loc[data.Beat == "G2","Lon"] = -122.297
data.loc[data.Beat == "G3","Lon"] = -122.298
data.loc[data.Beat == "Q1","Lon"] = -122.407
data.loc[data.Beat == "Q2","Lon"] = -122.365
data.loc[data.Beat == "Q3","Lon"] = -122.364
data.loc[data.Beat == "D1","Lon"] = -122.364
data.loc[data.Beat == "D2","Lon"] = -122.355
data.loc[data.Beat == "D3","Lon"] = -122.343


# In[ ]:


data.loc[data.Beat == "L2","Lat"] = 47.695
data.loc[data.Beat == "M1","Lat"] = 47.600
data.loc[data.Beat == "K3","Lat"] = 47.598
data.loc[data.Beat == "K2","Lat"] = 47.586
data.loc[data.Beat == "K1","Lat"] = 47.603
data.loc[data.Beat == "M2","Lat"] = 47.611
data.loc[data.Beat == "M3","Lat"] = 47.610
data.loc[data.Beat == "L1","Lat"] = 47.722
data.loc[data.Beat == "L3","Lat"] = 47.713
data.loc[data.Beat == "N2","Lat"] = 47.719
data.loc[data.Beat == "N3","Lat"] = 47.705
data.loc[data.Beat == "N1","Lat"] = 47.714
data.loc[data.Beat == "J1","Lat"] = 47.693
data.loc[data.Beat == "J2","Lat"] = 47.682
data.loc[data.Beat == "J3","Lat"] = 47.679
data.loc[data.Beat == "B3","Lat"] = 47.661
data.loc[data.Beat == "B2","Lat"] = 47.662
data.loc[data.Beat == "B1","Lat"] = 47.668
data.loc[data.Beat == "U1","Lat"] = 47.695
data.loc[data.Beat == "U2","Lat"] = 47.657
data.loc[data.Beat == "U3","Lat"] = 47.671
data.loc[data.Beat == "W1","Lat"] = 47.574
data.loc[data.Beat == "W2","Lat"] = 47.559
data.loc[data.Beat == "W3","Lat"] = 47.517
data.loc[data.Beat == "F1","Lat"] = 47.541
data.loc[data.Beat == "F2","Lat"] = 47.528
data.loc[data.Beat == "F3","Lat"] = 47.527
data.loc[data.Beat == "O1","Lat"] = 47.582
data.loc[data.Beat == "O2","Lat"] = 47.559
data.loc[data.Beat == "O3","Lat"] = 47.542
data.loc[data.Beat == "S1","Lat"] = 47.539
data.loc[data.Beat == "S2","Lat"] = 47.538
data.loc[data.Beat == "S3","Lat"] = 47.517
data.loc[data.Beat == "R1","Lat"] = 47.566
data.loc[data.Beat == "R2","Lat"] = 47.565
data.loc[data.Beat == "R3","Lat"] = 47.552
data.loc[data.Beat == "C1","Lat"] = 47.622
data.loc[data.Beat == "C2","Lat"] = 47.630
data.loc[data.Beat == "C3","Lat"] = 47.621
data.loc[data.Beat == "E1","Lat"] = 47.620
data.loc[data.Beat == "E2","Lat"] = 47.613
data.loc[data.Beat == "E3","Lat"] = 47.610
data.loc[data.Beat == "G1","Lat"] = 47.606
data.loc[data.Beat == "G2","Lat"] = 47.604
data.loc[data.Beat == "G3","Lat"] = 47.593
data.loc[data.Beat == "Q1","Lat"] = 47.650
data.loc[data.Beat == "Q2","Lat"] = 47.639
data.loc[data.Beat == "Q3","Lat"] = 47.623
data.loc[data.Beat == "D1","Lat"] = 47.617
data.loc[data.Beat == "D2","Lat"] = 47.618
data.loc[data.Beat == "D3","Lat"] = 47.621


# In[ ]:


df_date = data.tail(100)


# In[ ]:


df = pd.read_csv("../input/crisis-data.csv", parse_dates=['Reported Date', 'Occurred Date / Time'])
df.head(30)


# In[ ]:


# how big is the dataset?
print("Num rows: {0}".format(df.shape[0]))
# what range of dates do we have?
print("Min date: {0} | Max date: {1}".format(df['Reported Date'].min(), df['Reported Date'].max()))


# In[ ]:


# clean up bogus dates
df = df[df['Reported Date'].dt.year > 2000].copy()
print("Min date: {0} | Max date: {1}".format(df['Reported Date'].min(), df['Reported Date'].max()))


# In[ ]:


dfg = df[['Reported Date','Template ID']].groupby('Reported Date').count()
dfg.rename({'Template ID':'Incidents'},axis=1,inplace=True)


# In[ ]:


dfg.plot.line(
    figsize=(12,5), 
    colormap='tab20',
    title="Incidents per day",
    legend=False
)


# In[ ]:


datadf = [go.Scatter(x=dfg.index.tolist(), y=dfg['Incidents'])]

# specify the layout of our figure
layout = dict(title = "Number of Incidents Per Day",
              xaxis= dict(title= 'Date',ticklen= 5,zeroline= False))

# create and show our figure
fig = dict(data = datadf, layout = layout)
iplot(fig)


# In[ ]:


dfp_precinct = pd.pivot_table(
    data=df[['Reported Date', 'Precinct']],
    index='Reported Date',
    columns=['Precinct'],
    aggfunc=len,
)
dfp_precinct.iloc[-30::,:].plot.bar(
    stacked=True, 
    figsize=(12,5), 
    colormap='tab20',
    title="Incidents per day by precint (last 30 days)"
).legend(loc='center left', bbox_to_anchor=(1, 0.5))


# In[ ]:


dfp_precinct.iloc[-30::,:].div(dfp_precinct.iloc[-30::,:].sum(axis=1), axis=0).plot.bar(
    stacked=True, 
    figsize=(12,5), 
    colormap='tab20',
    title="Incidents per day by precint (last 30 days)"
).legend(loc='center left', bbox_to_anchor=(1, 0.5))


# In[ ]:


datar = [
    go.Bar(
        x=dfp_precinct.index.tolist(), 
        y=dfp_precinct[col],
        name=col
    ) for col in dfp_precinct.columns
]

# updatemenus = list([
#     dict(active=-1,
#          buttons=list([   
#             dict(label = 'High',
#                  method = 'update',
#                  args = [{'visible': [True, True, False, False]},
#                          {'title': 'Yahoo High',
#                           'annotations': high_annotations}]),
#             dict(label = 'Low',
#                  method = 'update',
#                  args = [{'visible': [False, False, True, True]},
#                          {'title': 'Yahoo Low',
#                           'annotations': low_annotations}]),
#             dict(label = 'Both',
#                  method = 'update',
#                  args = [{'visible': [True, True, True, True]},
#                          {'title': 'Yahoo',
#                           'annotations': high_annotations+low_annotations}]),
#             dict(label = 'Reset',
#                  method = 'update',
#                  args = [{'visible': [True, False, True, False]},
#                          {'title': 'Yahoo',
#                           'annotations': []}])
#         ]),
#     )
# ])

# specify the layout of our figure
layout = dict(
    title = "Number of Incidents Per Day",
    xaxis= dict(
        title= 'Date',
        ticklen= 5,
        zeroline= False,
        rangeselector=dict(
            buttons=list([
                dict(count=1,
                     label='1m',
                     step='month',
                     stepmode='backward'),
                dict(count=6,
                     label='6m',
                     step='month',
                     stepmode='backward'),
                dict(count=1,
                    label='YTD',
                    step='year',
                    stepmode='todate'),
                dict(count=1,
                    label='1y',
                    step='year',
                    stepmode='backward'),
                dict(step='all')
            ])
        ),
        rangeslider=dict(
            visible = True
        ),
        type='date',
    ),
    barmode='stack',
)

# create and show our figure
fig = dict(data = datar, layout = layout)
iplot(fig)


# In[ ]:


dfp_sector = pd.pivot_table(
    data=df[['Reported Date', 'Sector']],
    index='Reported Date',
    columns=['Sector'],
    aggfunc=len,
)

dfp_sector.iloc[-30::,:].plot.bar(
    stacked=True, 
    figsize=(12,5), 
    colormap='tab20',
    title="Incidents per day by sector (last 30 days)"
).legend(loc='center left', bbox_to_anchor=(1, 0.5))


# In[ ]:


dfp_sector.iloc[-14::,:].div(dfp_sector.iloc[-14::,:].sum(axis=1), axis=0).plot.bar(
    stacked=True, 
    figsize=(12,5), 
    colormap='tab20',
    title="Incidents per day by sector (last 14 days)"
).legend(loc='center left', bbox_to_anchor=(1, 0.5))


# In[ ]:


df_trend = pd.DataFrame(index=df['Sector'].sort_values().unique()).iloc[0:-1,:]
df_trend = df_trend.join(
    pd.Series(data=dfp_sector.iloc[-30::,:].T.mean(axis=1), name='last30')
)
df_trend = df_trend.join(
    pd.Series(data=dfp_sector.iloc[-3::,:].T.mean(axis=1), name='last3')
)
df_trend['trend'] = df_trend['last3'] - df_trend['last30'] 
df_trend['trend'].sort_values().plot.bar(title="Average daily incidents per sector over last three days vs last month", figsize=(12,5), colormap='tab20')


# In[ ]:


data2 = [
    go.Bar(
        x=df_trend.index.tolist(), 
        y=df_trend['trend'],
        marker={
            'color':['red' if x>0 else 'green' for x in  df_trend['trend']]
        }
    )
]

# specify the layout of our figure
layout = dict(title = "Average daily incidents over last three days vs last month",
              xaxis= dict(title= 'Sector',ticklen= 5,zeroline= False))

# create and show our figure
fig = dict(data = data2, layout = layout)
iplot(fig)


# In[ ]:


# Delete rows with bad date data
data = data.loc[data['Reported Date'] != '1900-01-01',:]


# In[ ]:


# Set column types
data['Reported Date'] = pd.to_datetime(data['Reported Date'])
data['Precinct'] = pd.Categorical(data['Precinct'])


# In[ ]:


data['Date'] = pd.to_datetime(data['Reported Date']) - pd.to_timedelta(7, unit='d')
fig, ax = plt.subplots(figsize=(14, 8))
data.groupby([pd.Grouper(key='Date', freq='W-MON'), 'Precinct'])['Template ID'].nunique().reset_index(level=1).last("5M").reset_index().pivot(index='Date', columns='Precinct', values= 'Template ID').plot(ax=ax)
plt.title('Events by week by precinct')
plt.show()


# In[ ]:


fig, ax = plt.subplots(figsize=(14, 10))
(data[['Template ID', 'Disposition']].groupby(['Disposition'])['Template ID'].nunique().sort_values()).plot.barh(ax=ax)
plt.title('Events by Disposition')
plt.show()


# In[ ]:




# Creating a dataframe with call count by call type
precinct = data["Officer Precinct Desc"].value_counts(normalize=False, sort=True, ascending=False, bins=None, dropna=True)

# Creating a dataframe with top 5 call types
top_5_precinct = precinct.head(5)

# Plotting a bar chart for top 5 call types
top_5_precinct.plot(kind = 'bar')


# In[ ]:


# Plotting a bar chart for top Precinct
# converting the top 5 medical incidents series to dataframe and assign columns names 
top5Pre_df = pd.DataFrame({'CallType':top_5_precinct.index.tolist(), 'IncidentCount':top_5_precinct.values})

data_top5Pre = [go.Bar(x=top5Pre_df["CallType"], y=top5Pre_df["IncidentCount"])]

# specify the layout of figure
layout_top5Pre = dict(title = "Top 5 Call Types by Precinct",
              xaxis= dict(title= 'Call Type',ticklen= 10,zeroline= False))

# create and show figure
fig_top5Pre = dict(data = data_top5Pre, layout = layout_top5Pre)
iplot(fig_top5Pre)


# In[ ]:


callType = data["Initial Call Type"].value_counts(normalize=False, sort=True, ascending=False, bins=None, dropna=True)
top_5_callType = callType.head(5)
top_5_callType.plot(kind = "bar")


# In[ ]:



# Plotting a bar chart for top 5 call types
# converting the top 5 medical incidents series to dataframe and assign columns names 
top5_df = pd.DataFrame({'CallType':top_5_callType.index.tolist(), 'IncidentCount':top_5_callType.values})

data_top5 = [go.Bar(x=top5_df["CallType"], y=top5_df["IncidentCount"])]

# specify the layout of figure
layout_top5 = dict(title = "Top 5 Call Types by Incidents",
              xaxis= dict(title= 'Call Type',ticklen= 10,zeroline= False))

# create and show figure
fig_top5 = dict(data = data_top5, layout = layout_top5)
iplot(fig_top5)


# In[ ]:





# In[ ]:


df_date = df_date[["Lat", "Lon","Disposition"]]
df_date = df_date.dropna()
df_date.head()


# In[ ]:


df_date=df_date.reset_index(drop=True)


# In[ ]:


df_date.info()


# ### A map of Seattle pinpointing the most recent locations of the last 100 incidents throughout the city.

# In[ ]:


import folium

SEA_COORDINATES = (47.6, -122.3)
# for speed purposes
MAX_RECORDS = 90
  
# create empty map zoomed in on San Francisco
map = folium.Map(location=SEA_COORDINATES, zoom_start=12) 

# add a marker for every record in the filtered data, use a clustered view
for each in df_date[0:MAX_RECORDS].iterrows():
    folium.Marker([each[1]["Lat"],
                   each[1]["Lon"]],
                  popup=each[1]["Disposition"]).add_to(map)
    
display(map)

