#!/usr/bin/env python
# coding: utf-8

# # Animated EDA and Tweets Analysis
# 
# **Remark**: Many great kernels have already been posted. My goal is to explore the data using the *Plotly* animation feature in scatter and geo plots!
# 
# **Update**: I recently gathered some tweets following the *coronavirus* hashtag and trying to analyze them.
# 
# For the moment this kernel has no predictions.
# 
# * [EDA](#eda)
#     - [nCoV in Asia](#asia)
#     - [nCoV in the World](#world)
#     - [Confirmed/Deaths/Recovered over Time](#scatter)
# * [Tweets Analysis](#tweets)
#     - [Sentiment Distribution](#sentiment)
#     - [WordCloud](#wordcloud)
#     - [Hashtags](#hashtags)

# <a id="eda"></a>
# # (Geographic) EDA

# Load libraries and the dataset.

# In[ ]:


import pandas as pd
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)

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
import time

from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer

from wordcloud import WordCloud
from collections import Counter

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


df = pd.read_csv("../input/novel-corona-virus-2019-dataset/covid_19_data.csv",)
resumetable(df)


# Let's rename columns, change datetime to date format, drop rows with (0,0,0) triplets.

# In[ ]:


df.rename(columns={'Last Update': 'LastUpdate',
                   'ObservationDate':'Date',
                   'Country/Region': 'Country',
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


# Retrieve latitute and longitude for each Country-Province pair using the time series dataset.
# (Remark, previously I was using the geopy package). 
# 

# In[ ]:


usecols=['Province/State', 'Country/Region', 'Lat', 'Long']
path= '../input/novel-corona-virus-2019-dataset/time_series_covid_19_'
csvs=['confirmed.csv', 'deaths.csv', 'recovered.csv']

coords_df = pd.concat([pd.read_csv(path + csv, usecols=usecols) for csv in csvs])

coords_df.rename(columns={'Country/Region': 'Country',
                          'Province/State': 'PS'}, 
                inplace=True)


# In[ ]:


coords_df.loc[(coords_df['PS'].isnull()) & (coords_df.groupby('Country')['PS'].transform('nunique') == 0), 'PS'] =    coords_df.loc[(coords_df['PS'].isnull()) & (coords_df.groupby('Country')['PS'].transform('nunique') == 0), 'Country'].to_numpy()

coords_df['Country'] = np.where(coords_df['Country']=='Mainland China', 'China', coords_df['Country'])


coords_df = coords_df.drop_duplicates()
df = pd.merge(df, coords_df, on=['Country', 'PS'], how='left')
df


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

#def coords(row):
#    
#    k = row['Country'] +', '+ row['PS'] if row['Country'] != row['PS'] else row['Country']
#    row['lat'] = d[k][0]
#    row['lon'] = d[k][1]
#    return row
#
#df = df.apply(coords, axis=1)
#df.head(10)


# In[ ]:


df = df.groupby(['PS', 'Country', 'Date']).agg({'Confirmed': 'sum',
                                                'Deaths': 'sum',
                                                'Recovered': 'sum',
                                                'Lat': 'max',
                                                'Long': 'max'}).reset_index()
df = df[df['Date']>date(2020,1,20)]


# Let's plot the virus spreading in Asia and in the rest of the world over time. 
# * Size is proportional to number of confirmed cases.
# * Colorscale depends upon the number of deaths.

# <a id="asia"></a>
# ### Asia Scattergeo

# In[ ]:


dates = np.sort(df['Date'].unique())
data = [go.Scattergeo(
            locationmode='country names',
            lon = df.loc[df['Date']==dt, 'Long'],
            lat = df.loc[df['Date']==dt, 'Lat'],
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


# <a id="world"></a>
# ### World Scattergeo

# In[ ]:


dates = np.sort(df['Date'].unique())
data = [go.Scattergeo(
            locationmode='country names',
            lon = df.loc[df['Date']==dt, 'Long'],
            lat = df.loc[df['Date']==dt, 'Lat'],
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


# <a id="scatter"></a>
# ### Confirmed/Deaths/Recovered over Time
# 
# Also let's check how number of confirmed, deaths and recovered evolve over time, in China and the rest of the world.
# 
# **Take care**, y-scales are very different!

# In[ ]:


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
        height=820,
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


# We can see that China's kinda hit inflection point in mid February, on the other hand the rest of the world just began the exponential phase. 

# <a id="tweets"></a>
# # Tweets Analysis
# 
# I tried to retrieve some tweets with the *coronavirus* hashtag during the last day, you can find the dataset among my inputs.
# The csv has already filtered out all the retweets.
# 
# ### Load data and Clean Text
# Preprocess each tweet, removing some patterns as *http*, *https*, *@[..]* and others..

# In[ ]:


df_tweets = pd.read_csv("../input/tweets/nCoV_tweets.csv", index_col=0)
df_tweets.rename(columns={'txt': 'tweets',
                         'dt':'date'}, inplace=True)

import re
def tweet_parser(text, pattern_regex):
    
    for pr in pattern_regex:
        text = re.sub(pr, ' ', text)
        
    return text.strip()

pattern_regex = ['\n', '\t', ':', ',', ';', '\.', '"', "''", 
                 '@.*?\s+', 'RT.*?\s+', 'http.*?\s+', 'https.*?\s+']

df_tweets['tidy_tweets'] = df_tweets.apply(lambda r: tweet_parser(r['tweets'], pattern_regex), axis=1)
df_tweets['date'] = pd.to_datetime(df_tweets['date']).dt.date

df_tweets.head()


# Let's use TextBlob library to infer tweet sentiments, and later categorize them into *Negative*, *Neutral* and *Positive*.

# In[ ]:


df_tweets['sentiment'] = df_tweets.apply(lambda r: TextBlob(r['tidy_tweets']).sentiment.polarity, axis=1)
df_tweets['sent_adj'] = np.where(df_tweets['sentiment']<0, 'Negative', np.where(df_tweets['sentiment']>0, 'Positive', 'Neutral'))
df_tweets['sent_adj'] = df_tweets['sent_adj'].astype('category')
sizes = df_tweets.groupby('sent_adj').size()

df_tweets.head()


# <a id="sentiment"></a>
# ### Raw Sentiment Distribution and Adjusted Sentiment Histogram

# In[ ]:


fig = ff.create_distplot([df_tweets['sentiment']], group_labels = ['sentiment'], bin_size=[.05], colors=['indianred'])
fig.update_layout(
        title={'text': 'Sentiment Distribution',
               'y':0.95, 'x':0.5,
               'xanchor': 'center', 'yanchor': 'top'},
        showlegend=False)

fig.show()


# In[ ]:


fig = go.Figure()
fig.add_trace(
    go.Bar(x=sizes.index,
           y=sizes.values,
           opacity=0.9,
           text = sizes.values,
           textposition='outside',
           marker={'color':'indianred'}
                   ))
fig.update_layout(
      title={'text': 'Sentiment Adjusted Histogram',
             'y':0.95, 'x':0.5,
            'xanchor': 'center', 'yanchor': 'top'},
       showlegend=False,
       xaxis_title_text='Sentiment',
        yaxis_title_text='Count',
    bargap=0.3)

fig.show()


# <a id="wordcloud"></a>
# ### Tweets WordCloud

# In[ ]:


def render_wordcloud(df, sent='Positive'):
    
    color = {'Positive': 'Set2', 'Negative': 'RdGy', 'Neutral': 'Accent_r'}
    
    words = ' '.join([text for text in df.loc[df['sent_adj']==sent, 'tidy_tweets']])
    
    wordcloud = WordCloud(width=800, height=500, 
                          background_color='black',
                          max_font_size=100, 
                          relative_scaling=0.1, 
                          colormap=color[sent]).generate(words)

    plt.figure(figsize=(14, 10))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.title(sent + ' Wordcloud', fontsize=20)
    plt.axis('off')
    plt.show()
    


# In[ ]:


for s in ['Positive', 'Negative', 'Neutral']:
    render_wordcloud(df_tweets, s)


#  At least at first sight, all wordclouds look pretty similar, and I cannot see huge differences in the words displayed.
#  Let's see if there are differences among the hashtags used.
#  
#  <a id="hashtags"></a>
# 
#  ### Hashtag Analysis 
#  Let's check hashtags counts, both overall and split by sentiment. From the hashtags I am excluding the #coronavirus one, since it's my tweet research key. 

# In[ ]:


def get_hashtag(series):
    s = series.str.lower()
    s = s.str.extractall(r'(\#\w*)')[0].value_counts()
    return pd.DataFrame(data={'hashtag': s.index,
                              'count': s.values})

def get_hashtag_by_sent(df):
    d={}
    for s in df['sent_adj'].unique():
        tmp = get_hashtag(df.loc[df['sent_adj']==s, 'tidy_tweets'])
        d[s] = tmp[(tmp['hashtag'].str.len()>2) & (~tmp['hashtag'].str.contains('coronavirus'))]
    return d

all_hashtag = get_hashtag(df_tweets['tidy_tweets'])
all_hashtag = all_hashtag[(all_hashtag['hashtag'].str.len()>2) & (~all_hashtag['hashtag'].str.contains('coronavirus'))]

d = get_hashtag_by_sent(df_tweets)


# In[ ]:


fig = make_subplots(
    rows=2, cols=3,
    specs=[[{"colspan": 3}, None, None],
           [{},{},{}]],
    subplot_titles=('Overall Most Frequent Hashtags',
                    'Positive Hashtags', 'Neutral Hashtags', 'Negative Hashtags' )
)

fig.add_trace(
     go.Bar(
         x=all_hashtag.loc[:20,'hashtag'].to_numpy(),
         y=all_hashtag.loc[:20, 'count'].to_numpy(),
         opacity=0.8,
         orientation='v'),
    row=1, col=1)

for i, k in enumerate(['Positive', 'Negative', 'Neutral']):
    
    fig.add_trace(
         go.Bar(x =  d[k].loc[:10,'hashtag'].to_numpy(),
               y =d[k].loc[:10, 'count'].to_numpy(),
               opacity=0.8,
               orientation='v'),
        row=2, col=i+1)

fig.update_layout(
      title={'text': 'Most Frequent Hashtags',
             'y':1, 'x':0.5,
            'xanchor': 'center', 'yanchor': 'top'},
      height = 1000,
    showlegend=False
    )
fig.show()


# As in the case of wordclouds, all *sentiments* seems to share the same hashtags.
# 
# If you got this far, please let me know what are your thoughts and feedbacks.
