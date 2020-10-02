#!/usr/bin/env python
# coding: utf-8

# **Terror Attacks in India : A Political Anaysis** 
# 
# ![](http://hmedia1.santabanta.com/full1/SantaBanta%20Special/Terrorism/terrorism-3h.jpg)

# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np
import seaborn as sns
import pandas as pd
pd.options.mode.chained_assignment = None
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import plotly.plotly as py
import plotly.graph_objs as go
from plotly import tools
import io
import base64
import folium
import folium.plugins
from IPython.display import HTML, display
import warnings
warnings.filterwarnings('ignore')
from scipy.misc import imread
import codecs
from subprocess import check_output
from plotly.offline import iplot, init_notebook_mode
init_notebook_mode()
import imageio

#generate the base64 encoding for the image file 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


terror = pd.read_csv('../input/globalterrorismdb_0718dist.csv', encoding='ISO-8859-1',
                          usecols=[0, 1, 2, 3, 8, 11, 13, 14, 35, 84, 100, 103])
terror=pd.read_csv('../input/globalterrorismdb_0718dist.csv',encoding='ISO-8859-1')
terror.rename(columns={'iyear':'Year','imonth':'Month','iday':'Day','country_txt':'Country','region_txt':'Region','attacktype1_txt':'AttackType','target1':'Target','nkill':'Killed','nwound':'Wounded','summary':'Summary','gname':'Group','targtype1_txt':'Target_type','weaptype1_txt':'Weapon_type','motive':'Motive'},inplace=True)
terror=terror[['Year','Month','Day','Country','Region','city','latitude','longitude','AttackType','Killed','Wounded','Target','Summary','Group','Target_type','Weapon_type','Motive']]
terror['casualities']=terror['Killed']+terror['Wounded']
terror['Killed'] = terror['Killed'].fillna(0).astype(int)
terror['Wounded'] = terror['Wounded'].fillna(0).astype(int)
terror.head(3)


# In[ ]:


def year (start,end):
    terror_i = terror[(terror.Country == 'India') &
                         (terror.longitude > 0) &
                          (terror.Year >= start) & (terror.Year <= end)]
    terror_i['Day'][terror_i.Day == 0] = 1
    terror_i['date'] = pd.to_datetime(terror_i[['Day', 'Month', 'Year']])
    terror_i = terror_i[['date','Year','Month','Day','Country','Region','city','latitude','longitude','AttackType','Killed','Wounded','Target','Summary','Group','Target_type','Weapon_type','Motive']]

    terror_i = terror_i.sort_values(['Killed', 'Wounded'], ascending = False)
    terror_i = terror_i.drop_duplicates(['date', 'latitude', 'longitude', 'Killed'])
    terror_i['text'] = terror_i['date'].dt.strftime('%B %-d, %Y') + '<br>' +                        terror_i['city'].astype(str) + '  '+                     terror_i['Killed'].astype(str) + ' Killed, ' +                     terror_i['Wounded'].astype(str) + ' Injured'
    return (terror_i)


# In[ ]:


terror_india = year(1975,2017)

fatality = dict(
           type = 'scattergeo',
           locationmode = 'ISO-3',
           lon = terror_india[terror_india.Killed > 0]['longitude'],
           lat = terror_india[terror_india.Killed > 0]['latitude'],
           text = terror_india[terror_india.Killed > 0]['text'],
           mode = 'markers',
           name = 'Fatalities',
           hoverinfo = 'text+name',
           marker = dict(
               size = terror_india[terror_india.Killed > 0]['Killed'] ** 0.255 * 8,
               opacity = 0.95,
               color = 'rgb(240, 140, 45)')
           )
injury = dict(
         type = 'scattergeo',
         locationmode = 'ISO-3',
         lon = terror_india[terror_india.Killed == 0]['longitude'],
         lat = terror_india[terror_india.Killed == 0]['latitude'],
         text = terror_india[terror_india.Killed == 0]['text'],
         mode = 'markers',
         name = 'Injuries',
         hoverinfo = 'text+name',
         marker = dict(
             size = (terror_india[terror_india.Killed == 0]['Wounded'] + 1) ** 0.245 * 8,
             opacity = 0.85,
             color = 'rgb(20, 150, 187)')
         )
layout = go.Layout(
    title = 'Terrorist Attacks in India from 1975-2017',
     showlegend = True,
         legend = dict(
             x = 0.85, y = 0.4
         ),
     geo = dict(
             scope = 'asia',
             #projection = dict(type = 'India'),
             lonaxis = dict( range= [ 65.0 ,100.0] ),
             lataxis = dict( range= [ 0.0,40.0 ] ),
             projection=dict( type = 'mercator'),
             showland = True,
             landcolor = 'rgb(250, 250, 250)',
             subunitwidth = 1,
             subunitcolor = 'rgb(217, 217, 217)',
             countrywidth = 1,
             countrycolor = 'rgb(217, 217, 217)',
             showlakes = True,
             lakecolor = 'rgb(255, 255, 255)'),
     width=1000,
    height=800
             
         )
data = [fatality, injury]
figure = dict(data = data,layout = layout)
iplot(figure)


# In[ ]:


ind_groups=terror_india['Group'].value_counts()[1:11].index
ind_groups=terror_india[terror_india['Group'].isin(ind_groups)]

plt.subplots(figsize=(15,6))
sns.countplot(y='Group',data=ind_groups)
plt.title("Top Terrorist Groups")
plt.xticks(rotation=90)
plt.show()

plt.subplots(figsize=(15,6))
sns.countplot(y='Target_type',data=terror_india)
plt.title('Favorite Targets')
plt.xticks(rotation=90)
plt.show()


# In[ ]:


plt.subplots(figsize=(15,6))
sns.countplot(x='Year',data=terror_india,palette='RdYlGn_r',edgecolor=sns.color_palette('dark',7))
plt.xticks(rotation=90)
plt.title('Number Of Terrorist Activities Each Year')
plt.show()
plt.subplots(figsize=(15,6))
sns.countplot('AttackType',data=terror_india,palette='inferno',order=terror['AttackType'].value_counts().index)
plt.xticks(rotation=90)
plt.title('Attacking Methods by Terrorists')
plt.show()
plt.subplots(figsize=(15,6))
ind_groups=terror_india['Group'].value_counts()[1:25].index
ind_groups=terror_india[terror_india['Group'].isin(ind_groups)]
sns.countplot(y='Group',data=ind_groups)
plt.title("Top Terrorist Groups")
plt.xticks(rotation=90)
plt.show()
plt.subplots(figsize=(15,6))
sns.countplot(y='Target_type',data=terror_india)
plt.title('Favorite Targets')
plt.xticks(rotation=90)
plt.show()
import nltk
from wordcloud import WordCloud, STOPWORDS
motive=terror_india['Motive'].str.lower().str.replace(r'\|', ' ').str.cat(sep=' ')
words=nltk.tokenize.word_tokenize(motive)
word_dist = nltk.FreqDist(words)
stopwords = nltk.corpus.stopwords.words('english')
words_except_stop_dist = nltk.FreqDist(w for w in words if w not in stopwords) 
wordcloud = WordCloud(stopwords=STOPWORDS,background_color='black').generate(" ".join(words_except_stop_dist))
plt.imshow(wordcloud)
fig=plt.gcf()
fig.set_size_inches(20,6)
plt.axis('off')
plt.show()


# In[ ]:


terror_india = year(1977,1979)

fatality = dict(
           type = 'scattergeo',
           locationmode = 'ISO-3',
           lon = terror_india[terror_india.Killed > 0]['longitude'],
           lat = terror_india[terror_india.Killed > 0]['latitude'],
           text = terror_india[terror_india.Killed > 0]['text'],
           mode = 'markers',
           name = 'Fatalities',
           hoverinfo = 'text+name',
           marker = dict(
               size = terror_india[terror_india.Killed > 0]['Killed'] ** 0.255 * 8,
               opacity = 0.95,
               color = 'rgb(240, 140, 45)')
           )
injury = dict(
         type = 'scattergeo',
         locationmode = 'ISO-3',
         lon = terror_india[terror_india.Killed == 0]['longitude'],
         lat = terror_india[terror_india.Killed == 0]['latitude'],
         text = terror_india[terror_india.Killed == 0]['text'],
         mode = 'markers',
         name = 'Injuries',
         hoverinfo = 'text+name',
         marker = dict(
             size = (terror_india[terror_india.Killed == 0]['Wounded'] + 1) ** 0.245 * 8,
             opacity = 0.85,
             color = 'rgb(20, 150, 187)')
         )
layout = go.Layout(
    title = 'Terrorist Attacks in India from 1977-1979',
     showlegend = True,
         legend = dict(
             x = 0.85, y = 0.4
         ),
     geo = dict(
             scope = 'asia',
             #projection = dict(type = 'India'),
             lonaxis = dict( range= [ 65.0 ,100.0] ),
             lataxis = dict( range= [ 0.0,40.0 ] ),
             projection=dict( type = 'mercator'),
             showland = True,
             landcolor = 'rgb(250, 250, 250)',
             subunitwidth = 1,
             subunitcolor = 'rgb(217, 217, 217)',
             countrywidth = 1,
             countrycolor = 'rgb(217, 217, 217)',
             showlakes = True,
             lakecolor = 'rgb(255, 255, 255)'),
     width=1000,
    height=800
             
         )
data = [fatality, injury]
figure = dict(data = data,layout = layout)
iplot(figure)


# In[ ]:


plt.subplots(figsize=(15,6))
sns.countplot(x='Year',data=terror_india,palette='RdYlGn_r',edgecolor=sns.color_palette('dark',7))
plt.xticks(rotation=90)
plt.title('Number Of Terrorist Activities Each Year')
plt.show()

plt.subplots(figsize=(15,6))
sns.countplot(y='AttackType',data=terror_india)
plt.title('Attack Type')
plt.xticks(rotation=90)
plt.show()

plt.subplots(figsize=(15,6))
ind_groups=terror_india['Group'].value_counts()[1:25].index
ind_groups=terror_india[terror_india['Group'].isin(ind_groups)]
sns.countplot(y='Group',data=ind_groups)
plt.title("Top Terrorist Groups")
plt.xticks(rotation=90)
plt.show()

plt.subplots(figsize=(15,6))
sns.countplot(y='Target_type',data=terror_india)
plt.title('Favorite Targets')
plt.xticks(rotation=90)
plt.show()

plt.subplots(figsize=(15,6))
sns.countplot(y='Weapon_type',data=terror_india)
plt.title('Weapon Used')
plt.xticks(rotation=90)
plt.show()


# In[ ]:


terror_india = year(1980,1989)

fatality = dict(
           type = 'scattergeo',
           locationmode = 'ISO-3',
           lon = terror_india[terror_india.Killed > 0]['longitude'],
           lat = terror_india[terror_india.Killed > 0]['latitude'],
           text = terror_india[terror_india.Killed > 0]['text'],
           mode = 'markers',
           name = 'Fatalities',
           hoverinfo = 'text+name',
           marker = dict(
               size = terror_india[terror_india.Killed > 0]['Killed'] ** 0.255 * 8,
               opacity = 0.95,
               color = 'rgb(240, 140, 45)')
           )
injury = dict(
         type = 'scattergeo',
         locationmode = 'ISO-3',
         lon = terror_india[terror_india.Killed == 0]['longitude'],
         lat = terror_india[terror_india.Killed == 0]['latitude'],
         text = terror_india[terror_india.Killed == 0]['text'],
         mode = 'markers',
         name = 'Injuries',
         hoverinfo = 'text+name',
         marker = dict(
             size = (terror_india[terror_india.Killed == 0]['Wounded'] + 1) ** 0.245 * 8,
             opacity = 0.85,
             color = 'rgb(20, 150, 187)')
         )
layout = go.Layout(
    title = 'Terrorist Attacks in India from 1980-1989',
     showlegend = True,
         legend = dict(
             x = 0.85, y = 0.4
         ),
     geo = dict(
             scope = 'asia',
             #projection = dict(type = 'India'),
             lonaxis = dict( range= [ 65.0 ,100.0] ),
             lataxis = dict( range= [ 0.0,40.0 ] ),
             projection=dict( type = 'mercator'),
             showland = True,
             landcolor = 'rgb(250, 250, 250)',
             subunitwidth = 1,
             subunitcolor = 'rgb(217, 217, 217)',
             countrywidth = 1,
             countrycolor = 'rgb(217, 217, 217)',
             showlakes = True,
             lakecolor = 'rgb(255, 255, 255)'),
    width=1000,
    height=800
             
         )
data = [fatality, injury]
figure = dict(data = data,layout = layout)
iplot(figure)


# In[ ]:


plt.subplots(figsize=(15,6))
sns.countplot(x='Year',data=terror_india,palette='RdYlGn_r',edgecolor=sns.color_palette('dark',7))
plt.xticks(rotation=90)
plt.title('Number Of Terrorist Activities Each Year')
plt.show()
plt.subplots(figsize=(15,6))
sns.countplot('AttackType',data=terror_india,palette='inferno',order=terror['AttackType'].value_counts().index)
plt.xticks(rotation=90)
plt.title('Attacking Methods by Terrorists')
plt.show()
plt.subplots(figsize=(15,6))
ind_groups=terror_india['Group'].value_counts()[1:25].index
ind_groups=terror_india[terror_india['Group'].isin(ind_groups)]
sns.countplot(y='Group',data=ind_groups)
plt.title("Top Terrorist Groups")
plt.xticks(rotation=90)
plt.show()
plt.subplots(figsize=(15,6))
sns.countplot(y='Target_type',data=terror_india)
plt.title('Favorite Targets')
plt.xticks(rotation=90)
plt.show()


# In[ ]:


terror_india = year(1990,1997)



fatality = dict(
           type = 'scattergeo',
           locationmode = 'ISO-3',
           lon = terror_india[terror_india.Killed > 0]['longitude'],
           lat = terror_india[terror_india.Killed > 0]['latitude'],
           text = terror_india[terror_india.Killed > 0]['text'],
           mode = 'markers',
           name = 'Fatalities',
           hoverinfo = 'text+name',
           marker = dict(
               size = terror_india[terror_india.Killed > 0]['Killed'] ** 0.255 * 8,
               opacity = 0.95,
               color = 'rgb(240, 140, 45)')
           )
injury = dict(
         type = 'scattergeo',
         locationmode = 'ISO-3',
         lon = terror_india[terror_india.Killed == 0]['longitude'],
         lat = terror_india[terror_india.Killed == 0]['latitude'],
         text = terror_india[terror_india.Killed == 0]['text'],
         mode = 'markers',
         name = 'Injuries',
         hoverinfo = 'text+name',
         marker = dict(
             size = (terror_india[terror_india.Killed == 0]['Wounded'] + 1) ** 0.245 * 8,
             opacity = 0.85,
             color = 'rgb(20, 150, 187)')
         )
layout = go.Layout(
    title = 'Terrorist Attacks in India under INC-Janata Dal-Samajwadi Governments (1989-1998)',
     showlegend = True,
         legend = dict(
             x = 0.85, y = 0.4
         ),
     geo = dict(
             scope = 'asia',
             #projection = dict(type = 'India'),
             lonaxis = dict( range= [ 65.0 ,100.0] ),
             lataxis = dict( range= [ 0.0,40.0 ] ),
             projection=dict( type = 'mercator'),
             showland = True,
             landcolor = 'rgb(250, 250, 250)',
             subunitwidth = 1,
             subunitcolor = 'rgb(217, 217, 217)',
             countrywidth = 1,
             countrycolor = 'rgb(217, 217, 217)',
             showlakes = True,
             lakecolor = 'rgb(255, 255, 255)'),
    width=1000,
    height=800
             
         )
data = [fatality, injury]
figure = dict(data = data,layout = layout)

iplot(figure)


# In[ ]:


plt.subplots(figsize=(15,6))
sns.countplot(x='Year',data=terror_india,palette='RdYlGn_r',edgecolor=sns.color_palette('dark',7))
plt.xticks(rotation=90)
plt.title('Number Of Terrorist Activities Each Year')
plt.show()
plt.subplots(figsize=(15,6))
sns.countplot('AttackType',data=terror_india,palette='inferno',order=terror['AttackType'].value_counts().index)
plt.xticks(rotation=90)
plt.title('Attacking Methods by Terrorists')
plt.show()
plt.subplots(figsize=(15,6))
ind_groups=terror_india['Group'].value_counts()[1:25].index
ind_groups=terror_india[terror_india['Group'].isin(ind_groups)]
sns.countplot(y='Group',data=ind_groups)
plt.title("Top Terrorist Groups")
plt.xticks(rotation=90)
plt.show()
plt.subplots(figsize=(15,6))
sns.countplot(y='Target_type',data=terror_india)
plt.title('Favorite Targets')
plt.xticks(rotation=90)
plt.show()
import nltk
from wordcloud import WordCloud, STOPWORDS
motive=terror_india['Motive'].str.lower().str.replace(r'\|', ' ').str.cat(sep=' ')
words=nltk.tokenize.word_tokenize(motive)
word_dist = nltk.FreqDist(words)
stopwords = nltk.corpus.stopwords.words('english')
words_except_stop_dist = nltk.FreqDist(w for w in words if w not in stopwords) 
wordcloud = WordCloud(stopwords=STOPWORDS,background_color='black').generate(" ".join(words_except_stop_dist))
plt.imshow(wordcloud)
fig=plt.gcf()
fig.set_size_inches(10,6)
plt.axis('off')
plt.show()


# In[ ]:


terror_india = year(1998,2003)

fatality = dict(
           type = 'scattergeo',
           locationmode = 'ISO-3',
           lon = terror_india[terror_india.Killed > 0]['longitude'],
           lat = terror_india[terror_india.Killed > 0]['latitude'],
           text = terror_india[terror_india.Killed > 0]['text'],
           mode = 'markers',
           name = 'Fatalities',
           hoverinfo = 'text+name',
           marker = dict(
               size = terror_india[terror_india.Killed > 0]['Killed'] ** 0.255 * 8,
               opacity = 0.95,
               color = 'rgb(240, 140, 45)')
           )
injury = dict(
         type = 'scattergeo',
         locationmode = 'ISO-3',
         lon = terror_india[terror_india.Killed == 0]['longitude'],
         lat = terror_india[terror_india.Killed == 0]['latitude'],
         text = terror_india[terror_india.Killed == 0]['text'],
         mode = 'markers',
         name = 'Injuries',
         hoverinfo = 'text+name',
         marker = dict(
             size = (terror_india[terror_india.Killed == 0]['Wounded'] + 1) ** 0.245 * 8,
             opacity = 0.85,
             color = 'rgb(20, 150, 187)')
         )
layout = go.Layout(
    title = 'Terrorist Attacks in India under NDA Government (1998-2003)',
     showlegend = True,
         legend = dict(
             x = 0.85, y = 0.4
         ),
     geo = dict(
             scope = 'asia',
             #projection = dict(type = 'India'),
             lonaxis = dict( range= [ 65.0 ,100.0] ),
             lataxis = dict( range= [ 0.0,40.0 ] ),
             projection=dict( type = 'mercator'),
             showland = True,
             landcolor = 'rgb(250, 250, 250)',
             subunitwidth = 1,
             subunitcolor = 'rgb(217, 217, 217)',
             countrywidth = 1,
             countrycolor = 'rgb(217, 217, 217)',
             showlakes = True,
             lakecolor = 'rgb(255, 255, 255)'),
    width=1000,
    height=800
             
         )
data = [fatality, injury]
figure = dict(data = data,layout = layout)

iplot(figure)


# In[ ]:


plt.subplots(figsize=(15,6))
sns.countplot(x='Year',data=terror_india,palette='RdYlGn_r',edgecolor=sns.color_palette('dark',7))
plt.xticks(rotation=90)
plt.title('Number Of Terrorist Activities Each Year')
plt.show()
plt.subplots(figsize=(15,6))
sns.countplot('AttackType',data=terror_india,palette='inferno',order=terror['AttackType'].value_counts().index)
plt.xticks(rotation=90)
plt.title('Attacking Methods by Terrorists')
plt.show()
plt.subplots(figsize=(15,6))
ind_groups=terror_india['Group'].value_counts()[1:25].index
ind_groups=terror_india[terror_india['Group'].isin(ind_groups)]
sns.countplot(y='Group',data=ind_groups)
plt.title("Top Terrorist Groups")
plt.xticks(rotation=90)
plt.show()
plt.subplots(figsize=(15,6))
sns.countplot(y='Target_type',data=terror_india)
plt.title('Favorite Targets')
plt.xticks(rotation=90)
plt.show()
import nltk
from wordcloud import WordCloud, STOPWORDS
motive=terror_india['Motive'].str.lower().str.replace(r'\|', ' ').str.cat(sep=' ')
words=nltk.tokenize.word_tokenize(motive)
word_dist = nltk.FreqDist(words)
stopwords = nltk.corpus.stopwords.words('english')
words_except_stop_dist = nltk.FreqDist(w for w in words if w not in stopwords) 
wordcloud = WordCloud(stopwords=STOPWORDS,background_color='black').generate(" ".join(words_except_stop_dist))
plt.imshow(wordcloud)
fig=plt.gcf()
fig.set_size_inches(10,6)
plt.axis('off')
plt.show()


# In[ ]:


terror_india = year(2003,2014)

fatality = dict(
           type = 'scattergeo',
           locationmode = 'ISO-3',
           lon = terror_india[terror_india.Killed > 0]['longitude'],
           lat = terror_india[terror_india.Killed > 0]['latitude'],
           text = terror_india[terror_india.Killed > 0]['text'],
           mode = 'markers',
           name = 'Fatalities',
           hoverinfo = 'text+name',
           marker = dict(
               size = terror_india[terror_india.Killed > 0]['Killed'] ** 0.255 * 8,
               opacity = 0.95,
               color = 'rgb(240, 140, 45)')
           )
injury = dict(
         type = 'scattergeo',
         locationmode = 'ISO-3',
         lon = terror_india[terror_india.Killed == 0]['longitude'],
         lat = terror_india[terror_india.Killed == 0]['latitude'],
         text = terror_india[terror_india.Killed == 0]['text'],
         mode = 'markers',
         name = 'Injuries',
         hoverinfo = 'text+name',
         marker = dict(
             size = (terror_india[terror_india.Killed == 0]['Wounded'] + 1) ** 0.245 * 8,
             opacity = 0.85,
             color = 'rgb(20, 150, 187)')
         )
layout = go.Layout(
    title = 'Terrorist Attacks in India under UPA Government (2003-2014)',
     showlegend = True,
         legend = dict(
             x = 0.85, y = 0.4
         ),
     geo = dict(
             scope = 'asia',
             #projection = dict(type = 'India'),
             lonaxis = dict( range= [ 65.0 ,100.0] ),
             lataxis = dict( range= [ 0.0,40.0 ] ),
             projection=dict( type = 'mercator'),
             showland = True,
             landcolor = 'rgb(250, 250, 250)',
             subunitwidth = 1,
             subunitcolor = 'rgb(217, 217, 217)',
             countrywidth = 1,
             countrycolor = 'rgb(217, 217, 217)',
             showlakes = True,
             lakecolor = 'rgb(255, 255, 255)'),
    width=1000,
    height=800
             
         )
data = [fatality, injury]
figure = dict(data = data,layout = layout)

iplot(figure)


# In[ ]:


plt.subplots(figsize=(15,6))
sns.countplot(x='Year',data=terror_india,palette='RdYlGn_r',edgecolor=sns.color_palette('dark',7))
plt.xticks(rotation=90)
plt.title('Number Of Terrorist Activities Each Year')
plt.show()
plt.subplots(figsize=(15,6))
sns.countplot('AttackType',data=terror_india,palette='inferno',order=terror['AttackType'].value_counts().index)
plt.xticks(rotation=90)
plt.title('Attacking Methods by Terrorists')
plt.show()
plt.subplots(figsize=(15,6))
ind_groups=terror_india['Group'].value_counts()[1:25].index
ind_groups=terror_india[terror_india['Group'].isin(ind_groups)]
sns.countplot(y='Group',data=ind_groups)
plt.title("Top Terrorist Groups")
plt.xticks(rotation=90)
plt.show()
plt.subplots(figsize=(15,6))
sns.countplot(y='Target_type',data=terror_india)
plt.title('Favorite Targets')
plt.xticks(rotation=90)
plt.show()
import nltk
from wordcloud import WordCloud, STOPWORDS
motive=terror_india['Motive'].str.lower().str.replace(r'\|', ' ').str.cat(sep=' ')
words=nltk.tokenize.word_tokenize(motive)
word_dist = nltk.FreqDist(words)
stopwords = nltk.corpus.stopwords.words('english')
words_except_stop_dist = nltk.FreqDist(w for w in words if w not in stopwords) 
wordcloud = WordCloud(stopwords=STOPWORDS,background_color='black').generate(" ".join(words_except_stop_dist))
plt.imshow(wordcloud)
fig=plt.gcf()
fig.set_size_inches(10,6)
plt.axis('off')
plt.show()


# In[ ]:


terror_india = year(2014,2017)

fatality = dict(
           type = 'scattergeo',
           locationmode = 'ISO-3',
           lon = terror_india[terror_india.Killed > 0]['longitude'],
           lat = terror_india[terror_india.Killed > 0]['latitude'],
           text = terror_india[terror_india.Killed > 0]['text'],
           mode = 'markers',
           name = 'Fatalities',
           hoverinfo = 'text+name',
           marker = dict(
               size = terror_india[terror_india.Killed > 0]['Killed'] ** 0.255 * 8,
               opacity = 0.95,
               color = 'rgb(240, 140, 45)')
           )
injury = dict(
         type = 'scattergeo',
         locationmode = 'ISO-3',
         lon = terror_india[terror_india.Killed == 0]['longitude'],
         lat = terror_india[terror_india.Killed == 0]['latitude'],
         text = terror_india[terror_india.Killed == 0]['text'],
         mode = 'markers',
         name = 'Injuries',
         hoverinfo = 'text+name',
         marker = dict(
             size = (terror_india[terror_india.Killed == 0]['Wounded'] + 1) ** 0.245 * 8,
             opacity = 0.85,
             color = 'rgb(20, 150, 187)')
         )
layout = go.Layout(
    title = 'Terrorist Attacks in India under BJP Government (2014-2017)',
     showlegend = True,
         legend = dict(
             x = 0.85, y = 0.4
         ),
     geo = dict(
             scope = 'asia',
             #projection = dict(type = 'India'),
             lonaxis = dict( range= [ 65.0 ,100.0] ),
             lataxis = dict( range= [ 0.0,40.0 ] ),
             projection=dict( type = 'mercator'),
             showland = True,
             landcolor = 'rgb(250, 250, 250)',
             subunitwidth = 1,
             subunitcolor = 'rgb(217, 217, 217)',
             countrywidth = 1,
             countrycolor = 'rgb(217, 217, 217)',
             showlakes = True,
             lakecolor = 'rgb(255, 255, 255)'),
    width=1000,
    height=800
             
         )
data = [fatality, injury]
figure = dict(data = data,layout = layout)

iplot(figure)



# In[ ]:


plt.subplots(figsize=(15,6))
sns.countplot(x='Year',data=terror_india,palette='RdYlGn_r',edgecolor=sns.color_palette('dark',7))
plt.xticks(rotation=90)
plt.title('Number Of Terrorist Activities Each Year')
plt.show()
plt.subplots(figsize=(15,6))
sns.countplot('AttackType',data=terror_india,palette='inferno',order=terror['AttackType'].value_counts().index)
plt.xticks(rotation=90)
plt.title('Attacking Methods by Terrorists')
plt.show()
plt.subplots(figsize=(15,6))
ind_groups=terror_india['Group'].value_counts()[1:25].index
ind_groups=terror_india[terror_india['Group'].isin(ind_groups)]
sns.countplot(y='Group',data=ind_groups)
plt.title("Top Terrorist Groups")
plt.xticks(rotation=90)
plt.show()
plt.subplots(figsize=(15,6))
sns.countplot(y='Target_type',data=terror_india)
plt.title('Favorite Targets')
plt.xticks(rotation=90)
plt.show()
import nltk
from wordcloud import WordCloud, STOPWORDS
motive=terror_india['Motive'].str.lower().str.replace(r'\|', ' ').str.cat(sep=' ')
words=nltk.tokenize.word_tokenize(motive)
word_dist = nltk.FreqDist(words)
stopwords = nltk.corpus.stopwords.words('english')
words_except_stop_dist = nltk.FreqDist(w for w in words if w not in stopwords) 
wordcloud = WordCloud(stopwords=STOPWORDS,background_color='black').generate(" ".join(words_except_stop_dist))
plt.imshow(wordcloud)
fig=plt.gcf()
fig.set_size_inches(10,6)
plt.axis('off')
plt.show()


# In[ ]:




