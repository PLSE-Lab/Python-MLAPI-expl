#!/usr/bin/env python
# coding: utf-8

#  # Earthquakes in Turkey between 1910-2017

# In this project I want to show earthquakes in Turkey between 1910-2017. We can see effects, severity level, some features of earthquakes, type of earthquakes in terms of technical locution. I am gonna explain all of these types of term such as Mw, Md, Ms...

# In[ ]:


#Let's start with adding libraries I will use

import numpy as np 
import pandas as pd 
import seaborn as sns

# plotly
import chart_studio.plotly as py
from plotly.offline import init_notebook_mode, iplot, plot
import plotly as py
init_notebook_mode(connected=True)
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.templates.default = "plotly_dark"
from plotly.subplots import make_subplots

# word cloud library
from wordcloud import WordCloud

# matplotlib
import matplotlib.pyplot as plt

from collections import Counter

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


# Upload Data
earthquake = pd.read_csv("../input/earthquake.csv")
earthquake.head()


# # > Data Summary

# Biggest magnitude value in specified magnitude values (MD, ML, Mw, Ms and Mb).
# 
# Magnitude types (MD: Duration, ML: Local, Mw: Moment, Ms: Surface wave, Mb: Body-wave). 0.0 (zero) means no calculation for that type of magnitude.
# 
# Higher magnitude (xM).
# 
# Richter : The severity of the earthquake
# 
# Time Dependent Size (Md)
# 
# Surface Wave Size (Ms)
# 
# Object Wave Size (Mb)
# 
# Moment Size (Mw)
# 
# Mw size type is the most reliable than others. In the scientific world, if the moment magnitude can be calculated for an earthquake, it is considered that other types of magnitude are not needed. So I will use this type for the comment.
# 

# In[ ]:


earthquake.info()


# In[ ]:


Richter_mean = earthquake.Richter.mean()
earthquake["Severity_level"] = ["Heavy" if Richter_mean < each else "Harmless" for each in earthquake.Richter]

xm_mean = earthquake.xm.mean()
earthquake["Xm_level"] = ["High" if xm_mean < each else "Low" for each in earthquake.xm]

Depth_mean = 60
earthquake["Depth_level"] = ["Deep" if Depth_mean < each else "Shallow" for each in earthquake.Depth]

print(Richter_mean)
print(xm_mean)
print(Depth_mean)

earthquake.head()


# In[ ]:


# I  want to see how many different Direction we have.
print(earthquake.Direction.unique())


# > What is the difference between Magnitud and Violence?
# 
# A measure of the energy released at the source of the magnitude earthquake; violence is a measure of the effects of the earthquake on buildings and people.
# 
# > What is the Magnitude of the Earthquake?
# 
# The magnitude of the earthquake is defined as the logarithm of the amplitude of the earthquake waves on the seismograph recorded over a certain period of time. (Richter-ML, mb, MS, MW)

# Then I am gonna search some relationship between these datas. First of all we can see realtionship between all variables with heatmap.
# 

# In[ ]:


#Heatmap
data1 = earthquake[["Depth","xm","md","Richter","mw","ms","mb"]]

data1.corr()
f,ax = plt.subplots(figsize=(15,15))
sns.heatmap(data1.corr(), annot=True, linewidths=.5, fmt=".1f",ax=ax)
plt.show()


# So we can say there is some relationship as;
# 
# Depth has positive relationship with MB(0.3),MS(0.3),MW(0.2),Richter(0.2),XM(0.3) but there is no relation with MD(0).
# 
# The biggset corr in these varies is the between XM and MW (0.8), MS(0.7), MB(0.6).
# 

# # > Depth Effect
# The depth of the earthquake is very important. The above effect decreases as the earthquake deepens. Now we can see that with box plot.
# 
# Earthquakes with a depth of 0-60 km are considered as shallow earthquakes. Earthquakes with depths of 70-300 km in the ground are medium depths. Deep earthquakes are earthquakes that are more than 300 meters deep. earthquakes in Turkey, which is usually between 0-60 km and depths are shallow earthquakes.

# In[ ]:


earthquake.boxplot(column="Richter",by="Depth_level")
plt.show()


# In[ ]:


earthquake.boxplot(column="mw",by="Depth_level")
plt.show()


# # > Types of earthquake intensity
# 
# Great if MW >= 8
# 
# Major if MW between 7 - 7.9
# 
# Strong if MW between 6 - 6.9
# 
# Moderate if MW between 5 - 5.9
# 
# Light if MW between 4 - 4.9
# 
# Minor if MW between 3 - 3.9
# 
# Very Minor <= 3

# In[ ]:


# I added new column as Violence and it will show violence level.
violence = 5 
earthquake["Violence"] = ["Strong" if violence < each else "Soft" for each in earthquake.mw]
earthquake.head()


# In[ ]:


print(earthquake["Violence"].value_counts(dropna =False))


# As we can see from the last print,
# 
# There is 22806 Soft Earthquake (MW lower than 5)
# 
# 1201 Strong Earthquake (MW greater than 5).

# In[ ]:


earthquake['Date'] =pd.to_datetime(earthquake['Date'])


# In[ ]:


# I will continue with Strong Earthquake (mw>5), so I filtered as Strong_earthquake data.
strong_earthquake = earthquake[(earthquake.Violence=="Strong")]
strong_earthquake.head()


# In[ ]:


# MW vs Richter of each state
# visualize
f,ax1 = plt.subplots(figsize =(35,10))
sns.pointplot(x='City',y='mw',data=earthquake,color='lime',alpha=0.8)
sns.pointplot(x='City',y='Richter',data=earthquake,color='red',alpha=0.8)
plt.text(40,0.6,'MW',color='red',fontsize = 17,style = 'italic')
plt.text(40,0.55,'Richter',color='lime',fontsize = 18,style = 'italic')
plt.xlabel('City',fontsize = 15,color='blue')
plt.ylabel('Values',fontsize = 15,color='blue')
plt.title('MW  VS  Richter',fontsize = 20,color='blue')
plt.grid()


# In[ ]:


g = sns.jointplot(earthquake.mw, earthquake.Richter, kind="kde", size=7)

plt.show()


# We can say  there is really positive relationship between these two variables. We can use both of them when we are searching about eartquake.
# 

# In[ ]:


city = earthquake.City.value_counts()
plt.figure(figsize=(10,7))
sns.barplot(x=city[:12].index,y=city[:12].values)
plt.xticks(rotation=45)
plt.title('Most dangerous cities',color = 'blue',fontsize=15)


# In[ ]:


earthquake.head()


# # > Now, we can see these earthquakes in map untill now.
# 

# In[ ]:


years = [str(each) for each in list(earthquake.Year.unique())]  # str unique years
types = ['Soft', 'Strong']
custom_colors = {
    'Soft': 'rgb(34, 139, 34)',
    'Strong': 'rgb(167, 34, 0)'
}
# make figure
figure = {
    'data': [],
    'layout': {},
    'frames': []
}

figure['layout']['geo'] = dict(showframe=False, showland=True, showcoastlines=True, showcountries=True,
               countrywidth=1, 
              landcolor = 'rgb(217, 217, 217)',
              subunitwidth=1,
              showlakes = True,
              lakecolor = 'rgb(255, 255, 255)',
              countrycolor="rgb(5, 5, 5)")
figure['layout']['hovermode'] = 'closest'
figure['layout']['sliders'] = {
    'args': [
        'transition', {
            'duration': 400,
            'easing': 'cubic-in-out'
        }
    ],
    'initialValue': '1910',
    'plotlycommand': 'animate',
    'values': years,
    'visible': True
}
figure['layout']['updatemenus'] = [
    {
        'buttons': [
            {
                'args': [None, {'frame': {'duration': 500, 'redraw': False},
                         'fromcurrent': True, 'transition': {'duration': 300, 'easing': 'quadratic-in-out'}}],
                'label': 'Play',
                'method': 'animate'
            },
            {
                'args': [[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate',
                'transition': {'duration': 0}}],
                'label': 'Pause',
                'method': 'animate'
            }
        ],
        'direction': 'left',
        'pad': {'r': 10, 't': 87},
        'showactive': False,
        'type': 'buttons',
        'x': 0.1,
        'xanchor': 'right',
        'y': 0,
        'yanchor': 'top'
    }
]

sliders_dict = {
    'active': 0,
    'yanchor': 'top',
    'xanchor': 'left',
    'currentvalue': {
        'font': {'size': 20},
        'prefix': 'Year:',
        'visible': True,
        'xanchor': 'right'
    },
    'transition': {'duration': 300, 'easing': 'cubic-in-out'},
    'pad': {'b': 10, 't': 50},
    'len': 0.9,
    'x': 0.1,
    'y': 0,
    'steps': []
}

# make data
year = 1910
for ty in types:
    dataset_by_year = earthquake[earthquake['Year'] == year]
    dataset_by_year_and_cont = dataset_by_year[dataset_by_year['Violence'] == ty]
    
    data_dict = dict(
    type='scattergeo',
    lon = earthquake['Long'],
    lat = earthquake['Lat'],
    hoverinfo = 'text',
    text = ty,
    mode = 'markers',
    marker=dict(
        sizemode = 'area',
        sizeref = 1,
        size= 10 ,
        line = dict(width=1,color = "white"),
        color = custom_colors[ty],
        opacity = 0.7),
)
    figure['data'].append(data_dict)
    
# make frames
for year in years:
    frame = {'data': [], 'name': str(year)}
    for ty in types:
        dataset_by_year = earthquake[earthquake['Year'] == int(year)]
        dataset_by_year_and_cont = dataset_by_year[dataset_by_year['Violence'] == ty]

        data_dict = dict(
                type='scattergeo',
                lon = dataset_by_year_and_cont['Long'],
                lat = dataset_by_year_and_cont['Lat'],
                hoverinfo = 'text',
                text = ty,
                mode = 'markers',
                marker=dict(
                    sizemode = 'area',
                    sizeref = 1,
                    size= 10 ,
                    line = dict(width=1,color = "white"),
                    color = custom_colors[ty],
                    opacity = 0.7),
                name = ty
            )
        frame['data'].append(data_dict)

    figure['frames'].append(frame)
    slider_step = {'args': [
        [year],
        {'frame': {'duration': 300, 'redraw': False},
         'mode': 'immediate',
       'transition': {'duration': 300}}
     ],
     'label': year,
     'method': 'animate'}
    sliders_dict['steps'].append(slider_step)


figure["layout"]["autosize"]= True
figure["layout"]["title"] = "Earthquake"       

figure['layout']['sliders'] = [sliders_dict]

iplot(figure)


# When you zoom in you can see earthquakes in Turkey between 1910-2017.
# 
# When you click green and red buttons you can change format. If you click red button you will see only Soft earthquakes (lower than 5 mw).
# 
# In the lower of the map you can see time period.
