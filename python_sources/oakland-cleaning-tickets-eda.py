#!/usr/bin/env python
# coding: utf-8

# # Table of Contents
# -------------------------------------
# 1. [ Introduction ](#1)
# 2. [Packages Import](#2)
# 3. [Data Import and Preparation (Cleaning + Adding Features + Data Integrity)](#3)
# 4. [Analysis](#4)

# # Introduction <a id="1"></a>
# -----------------------------------------------------------

# > Oakland is the largest city and the county seat of Alameda County, California, United States. A major West Coast port city, Oakland is the largest city in the East Bay region of the San Francisco Bay Area, the third largest city overall in the San Francisco Bay Area, the eighth most populated city in California, and the 45th largest city in the United States. With a population of 412,040 as of 2016, it serves as a trade center for the San Francisco Bay Area; its Port of Oakland is the busiest port in the San Francisco Bay, the entirety of Northern California, and the fifth busiest in the United States of America. An act to incorporate the city was passed on May 4, 1852, and incorporation was later approved on March 25, 1854, which officially made Oakland a city. Oakland is a charter city.
# Oakland's territory covers what was once a mosaic of California coastal terrace prairie, oak woodland, and north coastal scrub. Its land served as a rich resource when its hillside oak and redwood timber were logged to build San Francisco.Oakland's fertile flatland soils helped it become a prolific agricultural region. In the late 1860s, Oakland was selected as the western terminal of the Transcontinental Railroad. Following the 1906 San Francisco earthquake, many San Francisco citizens moved to Oakland, enlarging the city's population, increasing its housing stock and improving its infrastructure. It continued to grow in the 20th century with its busy port, shipyards, and a thriving.
# ## Oakland on the Map
# ![on us map](https://www.worldatlas.com/img/locator/city/031/22331-oakland-locator-map.jpg)
# ![meow](http://art-en-provence.com/wp-content/uploads/2018/07/Dadabecbee-Pictures-In-Gallery-Map-Of-Oakland-California.jpg)
# ## Skyline
# ![city skyline](https://upload.wikimedia.org/wikipedia/commons/thumb/d/d5/OAKLAND%2C_CA%2C_USA_-_Skyline_and_Bridge.JPG/1280px-OAKLAND%2C_CA%2C_USA_-_Skyline_and_Bridge.JPG)
# 
# 
# # So What is the Data All About?
# 
# ### The data contains all the parking tickets given to car owners who parked their car in an illegal spot and blocked the street sweeping vehicles.
# 
# ### The fine currently stands on 66 $
# 
# 
# ![meow5](http://www.parksomerville.com/images/showcase-sweepers.jpg)
# 

# # Packages Import <a id="2"></a>
# ----------------------------

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import ast
import os
import plotly
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.figure_factory as ff
import plotly.graph_objs as go
from plotly.graph_objs import *
from geopy.geocoders import Nominatim
import os
import re
import random
import matplotlib.pyplot as plt
import seaborn as sns
init_notebook_mode()
print('Files are:\n\t' + '\t\n\t'.join(os.listdir("../input")))

# Any results you write to the current directory are saved as output.


# # Data Import and Preparation (Cleaning+ + Adding Features + Data Integrity) <a id="3"></a>
# ----------------------------------------

# ## Import Data

# In[ ]:


cleaning = pd.read_csv('../input/prr-9545-street-sweeping-2013-2015-05.08.2015.csv')


# ## Sneak Peak at the Data

# In[ ]:


cleaning.head()


# ## Data Integrity

# In[ ]:


print("Number of NaN Values in Each Column:\n===================================================")
cleaning.isna().sum()


# ## Adding Features

# In[ ]:


def num_to_day(x):
    return ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][x]
def num_to_month(x):
    return ["January","February","March","April","May","June","July","August","September","October","November","December"][x - 1]
        

def map_x(x):
    if not pd.isna(x):
        splited_hour = int(x.split(':')[0])
        if splited_hour < 6:
            return "00AM-6AM"
        if splited_hour < 12 and splited_hour > 6:
            return "6AM-12PM"
        if splited_hour >= 12 and splited_hour < 18:
            return "12PM-6PM"
        if splited_hour > 18:
            return "6PM-00AM"
    else:
        return None
    
def map_street(x):
    splited = x.split(' ')
    if len(splited)  < 2:
        return "WALKER"
    else:
        return splited[1]
    
def prep_data(df):
    df.set_index('Citation Number', inplace=True)
    df["Citation_Date"] = pd.to_datetime(df['Citation Date'], infer_datetime_format=True)
    df["latitude"] = df["Location 1"].apply(lambda x:float(ast.literal_eval(x)["latitude"]) if not pd.isna(x) else None)
    df["longitude"] = df["Location 1"].apply(lambda x:float(ast.literal_eval(x)["longitude"]) if not pd.isna(x) else None)
    df["time_of_the_day"] = df["Citation Time"].apply(map_x)
    df["street"] = df["Location"].apply(map_street)
    df["year"] = df["Citation_Date"].apply(lambda x:x.year)
    df["month_of_year"] = df["Citation_Date"].apply(lambda x:num_to_month(x.month))
    df["day_of_month"] = df["Citation_Date"].apply(lambda x:x.day)
    df["day_of_week"] = df["Citation_Date"].apply(lambda x:num_to_day(x.weekday()))
    df.drop(["Citation Date", "Location 1"], inplace=True, axis=1)
    return df


# ### Prepare the Data for Work

# In[ ]:


cleaning = prep_data(cleaning)


# In[ ]:


records_per_year = cleaning.groupby(by="year", axis=0).count()
print("The records are from 2013 to 2015(8th of May)\n===============================================")
for x in range(2013, 2016):
    print("\n Year {}: \n\t{:,} Tickets".format(x, records_per_year.loc[x]["Location"]))


# ### New Data After Adding New Features

# In[ ]:


cleaning.head(5)


# # Analysis <a id="4"></a>
# -------------------------------------

# In[ ]:


trace = go.Histogram(x=cleaning["Issued Amount"], xbins=dict(start=np.min(cleaning["Issued Amount"]), size=0.75, end=np.max(cleaning["Issued Amount"])),
                   marker=dict(color='rgb(75, 150, 25)'))

layout = go.Layout(
    title="Issued Amount Frequency Counts"
)

fig = go.Figure(data=[trace], layout=layout)
iplot(fig, filename='histogram-freq-counts-larger-bins')


# In[ ]:


lst_tuples = list(zip(cleaning["street"].values, list(cleaning["street"].groupby(by=cleaning["street"].values, axis=0).count())))
lst_tuples.sort(key=lambda tup: tup[1], reverse=True)
streets_names = [x[0] for x in lst_tuples]
count_streets = [x[1] for x in lst_tuples]
trace1 = {
  "y": count_streets[:50], 
  "x": streets_names[:50], 
  "marker": {"color": "rgb(100, 100, 5)"}, 
  "type": "bar"
}
layout = {
  "title": "Streets Violation Frequency", 
  "xaxis": {
    "tickfont": {"size": 12}, 
    "title": "<br><br><br>Street",
    "tickangle": 45
  }, 
  "yaxis": {
    "title": "Frequency <br>", 
    "titlefont": {"size": 12}
  }
}
fig = Figure(data=[trace1], layout=layout)
iplot(fig, filename='Where Do Most Tickets are Given ? (TOP 50 Streets)')


# In[ ]:


fig, ax = plt.subplots(nrows=3, ncols=1)
plt.subplots_adjust(left=0, right=2.2, top=3, bottom=0)
i = 0
for row in ax:
        year = 2013 + i
        streets = np.sort(cleaning[cleaning["year"] == year].groupby(by='street', as_index=False, axis=0).count().nlargest(12,'Location')["street"].values, axis=-1, kind='mergesort')
        row.set_title(str(year))
        sns.countplot(data=cleaning[cleaning["street"].isin(streets)], x="street",hue="time_of_the_day", palette="Set1", ax=row, order=streets)
        i+=1


# In[ ]:


fig, ax = plt.subplots(nrows=3, ncols=3)
plt.subplots_adjust(left=0, right=3, top=7, bottom=0)
i_list = 0
cleaning_years = [x[x["year"] == y] for x, y in zip([cleaning, cleaning, cleaning], [2013, 2014, 2015])]
for i, row in enumerate(ax):
    for j, col in enumerate(row):
        year_string = str(2013 + i)
        if j == 1:
            month_or_day = 'Day of The Month'
            title = year_string+ '\n' + month_or_day +'\n Tickets Count'
            col.set_title(title)
            col.set_xticklabels(col.get_xticklabels(), rotation=90)
            sns.countplot(data=cleaning_years[i_list], x="day_of_month" ,palette="Set1", ax=col)
        elif j == 2:
            month_or_day = 'Month of The Year'
            title = year_string + '\n' + month_or_day +'\n Tickets Count'
            col.set_title(title)
            col.set_xticklabels(col.get_xticklabels(), rotation=45)
            sns.countplot(data=cleaning_years[i_list], x="month_of_year",palette="Set1", ax=col, order=["January","February","March","April","May","June","July","August","September","October","November","December"])
        else:
            month_or_day = 'Day of The Week'
            title = year_string+ '\n' + month_or_day +'\n Tickets Count'
            col.set_title(title)
            col.set_xticklabels(col.get_xticklabels(), rotation=25)
            sns.countplot(data=cleaning_years[i_list], x="day_of_week" ,palette="Set1", ax=col, order=[ "Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"])
            
    i_list += 1


# In[ ]:


g = sns.catplot(data=cleaning,kind="count", x="day_of_week",col="year", hue="time_of_the_day", palette='magma', order=[ "Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"])
g.set_xticklabels(rotation=25)


# In[ ]:


temp = cleaning.groupby(by="street").count()
temp2 = cleaning.groupby(by="street").mean()
streets = cleaning.groupby(by="street", as_index=False).count().nlargest(25000, "year")["street"].values
tuples = []
for street in streets:
    tuples.append((street, temp.loc[street]["year"], cleaning[cleaning["street"] == street].iloc[0]["latitude"], cleaning[cleaning["street"] == street].iloc[0]["longitude"]))
data = [
    go.Scattermapbox(
        lat=[i[2] for i in tuples],
        lon=[i[3] for i in tuples],
        mode='markers',
        name="Tickets",
        marker=dict(
            size = (np.array([i[1] for i in tuples]) / 1000) + 20,
            color='rgb(135, 14, 87)',
            opacity=0.5
        ),
        text= ["{:,} Tickets, Street: {} ".format(x[1], x[0]) for x in tuples]
    )
]

layout = go.Layout(
    autosize=True,
    hovermode='closest',
    title="Tickets Map", 
    mapbox=dict(
        accesstoken="pk.eyJ1Ijoic3luY3VzaCIsImEiOiJjam05aTEyNHUwMDNnM3JscjRvODFuMDY1In0.Iw54eGGxr-h70qh86bMFjA",
        bearing=0,
        center=dict(
            lat=37.8044,
            lon=-122.2711
        ),
        pitch=0,
        zoom=10
    ),
)

fig = dict(data=data, layout=layout)
iplot(fig, filename='Multiple Mapbox')


# In[ ]:


ax = sns.barplot(x="year", y="Issued Amount", data=cleaning.groupby(by="year", axis=0, as_index=False).sum(), palette='magma')
ax.set_title("Oakland Revenue from Cleaning Tickets")
ax.set_ylabel("Revenue (in $)")
ax.set_yscale('log')
print("Oakland Municipality Made \n")
sumi = 0
for x, y in zip(cleaning.groupby(by="year", axis=0)["Issued Amount"].sum(), ["2013", "2014", "2015"]):
    print("\t{:,} $ in {} from Cleaning Tickets\n".format(x, y))
    sumi += x 
print("Oakland Municipality Made {:,} $ in Total".format(sumi))

