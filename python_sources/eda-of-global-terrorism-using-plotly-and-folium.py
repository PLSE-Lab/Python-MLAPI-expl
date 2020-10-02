#!/usr/bin/env python
# coding: utf-8

# # **This notebook is because of the awesome kagglers that have tried to showcase their creativity. Hence this is my version of the EDA and my first try in Plotly and Folium. Comments and Criticisms are welcomed. please upvote if you found it useful.**

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# # **Credits to I,Coder whose kernel is a top hit. I have used the same methods to select the columns**

# In[ ]:


terror_df = pd.read_csv('../input/gtd/globalterrorismdb_0617dist.csv',encoding='ISO-8859-1')
terror_df.rename(columns={'iyear':'Year','imonth':'Month','iday':'Day','country_txt':'Country','provstate':'state','region_txt':'Region','attacktype1_txt':'AttackType','target1':'Target','nkill':'Killed','nwound':'Wounded','summary':'Summary','gname':'Group','targtype1_txt':'Target_type','weaptype1_txt':'Weapon_type','motive':'Motive'},inplace=True)


# In[ ]:


terror_df=terror_df[['Year','Month','Day','Country','state','Region','city','latitude','longitude','AttackType','Killed','Wounded','Target','Summary','Group','Target_type','Weapon_type','Motive']]
terror_df.head()


# In[ ]:


terror_df.isnull().sum()


# In[ ]:


terror_df.info()


# # **Destructive Features**

# In[ ]:


print("Country with the most attacks:",terror_df['Country'].value_counts().idxmax())
print("City with the most attacks:",terror_df['city'].value_counts().index[1]) #as first entry is 'unknown'
print("Region with the most attacks:",terror_df['Region'].value_counts().idxmax())
print("Year with the most attacks:",terror_df['Year'].value_counts().idxmax())
print("Month with the most attacks:",terror_df['Month'].value_counts().idxmax())
print("Group with the most attacks:",terror_df['Group'].value_counts().index[1])
print("Most Attack Types:",terror_df['AttackType'].value_counts().idxmax())


#  # **Terrorist Activities by Region in each Year through Area Plot**

# In[ ]:


pd.crosstab(terror_df.Year, terror_df.Region).plot(kind='area',figsize=(15,6))
plt.title('Terrorist Activities by Region in each Year')
plt.ylabel('Number of Attacks')
plt.show()


# # **Number of Terrorist Activities each Year**

# In[ ]:


plt.subplots(figsize=(15,6))
sns.countplot('Year',data=terror_df,palette='RdYlGn_r',edgecolor=sns.color_palette("YlOrBr", 10))
plt.xticks(rotation=90)
plt.title('Number Of Terrorist Activities Each Year')
plt.show()


# # **History of the Worst Terror Attacks in Heatmap using Plotly**

# In[ ]:


terror_df['Wounded'] = terror_df['Wounded'].fillna(0).astype(int)
terror_df['Killed'] = terror_df['Killed'].fillna(0).astype(int)
terror_df['casualities'] = terror_df['Killed'] + terror_df['Wounded']


# ## **Values are sorted by the top 40 worst terror attacks as to keep the heatmap simple and easy to visualize**

# In[ ]:


terror_df1 = terror_df.sort_values(by='casualities',ascending=False)[:40]


# In[ ]:


heat=terror_df1.pivot_table(index='Country',columns='Year',values='casualities')
heat.fillna(0,inplace=True)


# In[ ]:


import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
colorscale = [[0, '#edf8fb'], [.3, '#00BFFF'],  [.6, '#8856a7'],  [1, '#810f7c']]
heatmap = go.Heatmap(z=heat.as_matrix(), x=heat.columns, y=heat.index, colorscale=colorscale)
data = [heatmap]
layout = go.Layout(
    title='Top 40 Worst Terror Attacks in History from 1982 to 2016',
    xaxis = dict(ticks='', nticks=20),
    yaxis = dict(ticks='')
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='heatmap',show_link=False)


# In[ ]:


terror_df.Country.value_counts()[:15]


# # **Top Countries affected by Terror Attacks**

# In[ ]:


plt.subplots(figsize=(15,6))
sns.barplot(terror_df['Country'].value_counts()[:15].index,terror_df['Country'].value_counts()[:15].values,palette='Blues_d')
plt.title('Top Countries Affected')
plt.xlabel('Countries')
plt.ylabel('Count')
plt.xticks(rotation= 90)
plt.show()


# # **The Big Four**
# ## **Now to visualize the top four countries that have suffered the most using bubble charts in Plotly**

# In[ ]:


terror_bubble_df = terror_df[(terror_df['Group'] != 'Unknown') & (terror_df['casualities'] > 50)]
terror_bubble_df.head()


# In[ ]:


terror_bubble_df = terror_bubble_df.sort_values(['Region', 'Country'])


# # **It is best to always check for null values and drop the features that are not needed. Atleast it can execute much faster.** 

# In[ ]:


terror_bubble_df.isnull().sum()


# In[ ]:


terror_bubble_df = terror_bubble_df.drop(['latitude','longitude','Summary','Motive','Target'],axis=1)


# In[ ]:


terror_bubble_df = terror_bubble_df.dropna(subset=['city'])


# In[ ]:


terror_bubble_df.isnull().sum()


# # **Iraq, Pakistan, Afghanistan and India have suffered the most number of terror attacks. So here I have used an Interactive Bubble chart to highlight their timeline with details like City, Terrorist Group, Number of Casualities and Year. And ofcourse, I have used the bubble size according to the casualities suffered** 

# In[ ]:


hover_text = []
for index, row in terror_bubble_df.iterrows():
    hover_text.append(('City: {city}<br>'+
                      'Group: {group}<br>'+
                      'casualities: {casualities}<br>'+
                      'Year: {year}').format(city=row['city'],
                                            group=row['Group'],
                                            casualities=row['casualities'],
                                            year=row['Year']))
terror_bubble_df['text'] = hover_text


# In[ ]:


trace0 = go.Scatter(
    x=terror_bubble_df['Year'][terror_bubble_df['Country'] == 'Iraq'],
    y=terror_bubble_df['casualities'][terror_bubble_df['Country'] == 'Iraq'],
    mode='markers',
    name='Iraq',
    text=terror_bubble_df['text'][terror_bubble_df['Country'] == 'Iraq'],
    marker=dict(
        symbol='circle',
        sizemode='area',
        size=terror_bubble_df['casualities'][terror_bubble_df['Country'] == 'Iraq'],
        line=dict(
            width=2
        ),
    )
)
trace1 = go.Scatter(
    x=terror_bubble_df['Year'][terror_bubble_df['Country'] == 'Pakistan'],
    y=terror_bubble_df['casualities'][terror_bubble_df['Country'] == 'Pakistan'],
    mode='markers',
    name='Pakistan',
    text=terror_bubble_df['text'][terror_bubble_df['Country'] == 'Pakistan'],
    marker=dict(
        symbol='circle',
        sizemode='area',
        size=terror_bubble_df['casualities'][terror_bubble_df['Country'] == 'Pakistan'],
        line=dict(
            width=2
        ),
    )
)
trace2 = go.Scatter(
    x=terror_bubble_df['Year'][terror_bubble_df['Country'] == 'Afghanistan'],
    y=terror_bubble_df['casualities'][terror_bubble_df['Country'] == 'Afghanistan'],
    mode='markers',
    name='Afghanistan',
    text=terror_bubble_df['text'][terror_bubble_df['Country'] == 'Afghanistan'],
    marker=dict(
        symbol='circle',
        sizemode='area',
        size=terror_bubble_df['casualities'][terror_bubble_df['Country'] == 'Afghanistan'],
        line=dict(
            width=2
        ),
    )
)
trace3 = go.Scatter(
    x=terror_bubble_df['Year'][terror_bubble_df['Country'] == 'India'],
    y=terror_bubble_df['casualities'][terror_bubble_df['Country'] == 'India'],
    mode='markers',
    name='India',
    text=terror_bubble_df['text'][terror_bubble_df['Country'] == 'India'],
    marker=dict(
        symbol='circle',
        sizemode='area',
        size=terror_bubble_df['casualities'][terror_bubble_df['Country'] == 'India'],
        line=dict(
            width=2
        ),
    )
)


# In[ ]:


data = [trace0, trace1, trace2, trace3]
layout = go.Layout(
         title = 'The Big Four',
         xaxis = dict(
             title = 'Year',
             #type = 'log',
             range = [1976,2016],
             tickmode = 'auto',
             nticks = 30,
             showline = True,
             showgrid = False
             ),
         yaxis = dict(
             title = 'Casualities',
             type = 'log',
             range = [1.8,3.6],
             tickmode = 'auto',
             nticks = 40,
             showline = True,
             showgrid = False),
         paper_bgcolor='rgb(243, 243, 243)',
         plot_bgcolor='rgb(243, 243, 243)',
         )


# In[ ]:


fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='Terrorism Bubble')


# # **Now let us check out which terrorist organizations have carried out their operations in each country. A value count would give us the terrorist organizations that have carried out the most attacks. we have indexed from 1 as to negate the value of 'Unknown'** 

# In[ ]:


terror_df.Group.value_counts()[1:15]


# In[ ]:


test = terror_df[terror_df.Group.isin(['Shining Path (SL)','Taliban','Islamic State of Iraq and the Levant (ISIL)'])]


# In[ ]:


test.Country.unique()


# # **Now let us plot the presence of these terrorist organizations on a world map using Folium. Through this, we would be able to know the organizations that have carried out their operations on some of the countries**

# In[ ]:


import folium
from folium.plugins import MarkerCluster


# In[ ]:


terror_df_group = terror_df.dropna(subset=['latitude','longitude'])


# ## **It makes sense now to have only unique rows of Country and Group as multiple groups  can operate on a single Country. But it is highly unlikely in most of the countries. But still there are exceptions.** 

# In[ ]:


terror_df_group = terror_df_group.drop_duplicates(subset=['Country','Group'])


# ## **We now use only the top 8 terrorist organizations sorted based on the number of attacks worldwide for the sake of rendering.** 

# In[ ]:


terrorist_groups = terror_df.Group.value_counts()[1:8].index.tolist()


# In[ ]:


terror_df_group = terror_df_group.loc[terror_df_group.Group.isin(terrorist_groups)]


# In[ ]:


terror_df_group.Group.unique()


# In[ ]:


m = folium.Map(location=[20, 0], tiles="Mapbox Bright", zoom_start=2)
for i in range(0,len(terror_df_group)):
    folium.Marker([terror_df_group.iloc[i]['latitude'],terror_df_group.iloc[i]['longitude']], 
                  popup='Group:{}<br>Country:{}'.format(terror_df_group.iloc[i]['Group'], 
                  terror_df_group.iloc[i]['Country'])).add_to(m)
 # Save it as html
m.save('Terrorist_Organizations_in_Country.html')


# In[ ]:


m


# ## **The Above map looks untidy even though it can be zoomed in to view the Country in question. Hence in the next chart, I have used Folium's Marker Cluster to cluster these icons. This makes it visually pleasing and highly interactive.**  

# In[ ]:


m1 = folium.Map(location=[20, 0], tiles="Stamenterrain", zoom_start=2)
marker_cluster = MarkerCluster(
    name='clustered icons',
    overlay=True,
    control=False,
    icon_create_function=None
)
for i in range(0,len(terror_df_group)):
    marker=folium.Marker([terror_df_group.iloc[i]['latitude'],terror_df_group.iloc[i]['longitude']]) 
    popup='Group:{}<br>Country:{}'.format(terror_df_group.iloc[i]['Group'],
                                          terror_df_group.iloc[i]['Country'])
    folium.Popup(popup).add_to(marker)
    marker_cluster.add_child(marker)
marker_cluster.add_to(m1)
folium.TileLayer('openstreetmap').add_to(m1)
folium.TileLayer('Mapbox Bright').add_to(m1)
folium.TileLayer('cartodbdark_matter').add_to(m1)
folium.TileLayer('stamentoner').add_to(m1)
folium.LayerControl().add_to(m1)
m1.save('Terrorist_Organizations_in_Country_cluster.html')


# ## **The Layer Control at the top right corner makes it easy to use any kind of Map Tile.**

# In[ ]:


m1


# # **A Choropleth map of the countries that have suffered the most casualities from 1970 to 2016**

# In[ ]:


import json
import os
world_geo = os.path.join('../input/worldcountries1','world-countries.json')


# In[ ]:


terror_df_world = terror_df[['Country','casualities']]


# In[ ]:


terror_df_world = terror_df_world.groupby(['Country'])['casualities'].sum().sort_values(ascending=False) .reset_index()


# In[ ]:


terror_df_world.head()


# In[ ]:


terror_df_world.loc[terror_df_world['Country']== 'United States']


# In[ ]:


m2 = folium.Map(location=[0, 0],zoom_start=2,tiles='Mapbox Bright')


# In[ ]:


m2.choropleth(
    geo_data=world_geo,
    name='choropleth',
    data=terror_df_world,
    columns=['Country','casualities'],
    key_on='feature.properties.name',
    threshold_scale=[0,8000, 21000, 48000, 75000, 200600],
    fill_color='YlOrRd',
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name='Casualities in Numbers from 1970 to 2016'
)
folium.LayerControl().add_to(m2)
 
# Save to html
m2.save('chloropleth_world.html')


# In[ ]:


m2

