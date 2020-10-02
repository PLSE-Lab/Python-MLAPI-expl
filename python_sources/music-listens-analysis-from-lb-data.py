#!/usr/bin/env python
# coding: utf-8

# # Music Listens Analysis from ListenBrainz data.
# You can add your music listening history at [ListenBrainz](https://listenbrainz.org/). This project is an open source and open data project about which you can know more from the website. They allow you to export your listens history in an JSON format. I've used my listens history to get some cool stats of my data.
# 
# I've used [Plotly](https://plot.ly) for plotting graphs which makes the graphs interactive. I've used [pandasql](https://pypi.org/project/pandasql/) to query the DataFrame
# which makes it easier to execute SQL queries on the DataFrame. The queries are in no way the most efficient ones to do the job but with around 2000 listens this works fine.

# # Contents:
# * [Import required libraries](#-1)
# * [Load JSON data and convert it into a DataFrame](#0)
# * [View a small sample of data](#-2)
# * [Top 10 Listens](#1)
# * [Top 10 Artists](#2)
# * [Top 10 Releases](#3)
# * [Daily Listens Count](#4)
# * [Daily Listening Timings](#5)
# * [Top 10 Tracks Listening Timeline](#6)
# * [Top 10 Artists Listening Timeline](#7)
# * [Top 10 Releases Listening Timeline](#8)
# * [Conclusion](#9)

# # Import required libraries <a class="anchor" id="-1"></a>

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import json # load JSON data
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import plotly.graph_objs as go
import os

from datetime import datetime
from pandasql import sqldf # run SQL queries on dataframe
pysql = lambda q: sqldf(q, globals())

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)


# # Load JSON data and convert it into a DataFrame <a class="anchor" id="0"></a>
# - Load JSON data.
# - Retrive only coulmns that are relevant to us.
# - Create a pandas DataFrame for easy processing.

# In[ ]:


with open('../input/kartikeyaSh_lb-2019-03-20.json') as f:
    listens_json = json.load(f)
columns=['timestamp', 'artist_name', 'track_name', 'release_name']
data = [[listen.get(column) for column in columns ] for listen in listens_json]
data = [[idx] + listen for idx, listen in enumerate(data)]
data = pd.DataFrame(data, columns=['id'] + columns)
data['date_time'] = pd.to_datetime(data['timestamp'], unit='s')
# Convert time to my timezone
data['date_time'] = data['date_time'] + pd.Timedelta('5 hour 30 min')
data.info()


# # View a small sample of data <a class="anchor" id="-2"></a>
# 

# In[ ]:


data.head()


# # Top 10 Listens <a class="anchor" id="1"></a>
# * Get top 10 listens based on the number of times I've listened to those.

# In[ ]:


top_tracks = pysql("""
SELECT
    id, track_name, count(track_name) AS track_count , artist_name
FROM
    data
GROUP BY
    track_name
ORDER BY
    count(track_name) DESC
LIMIT 10
""")

track_artist_name = ["{0}<br>By: {1}".format(track, artist) for (track, artist) in zip(top_tracks['track_name'], top_tracks['artist_name'])]

graph = [go.Bar(
            x=top_tracks['track_name'],
            y=top_tracks['track_count'],
            text=track_artist_name,
            marker=dict(
                color='rgb(158,202,225)',
                line=dict(
                    color='rgb(8,48,107)',
                    width=1.5,
                )
            ),
            opacity=0.7,
)]

layout = go.Layout(
    title='Top 10 Listens',
    plot_bgcolor='azure',
    paper_bgcolor='azure',
    titlefont=dict(
            size=20,
            color='black'
    ),
    xaxis=dict(
        title='Tracks',
        titlefont=dict(
            size=16,
            color='black'
        ),
        tickfont=dict(
            size=12,
            color='gray'
        )
    ),
    yaxis=dict(
        title='Listen Count',
        titlefont=dict(
            size=16,
            color='black'
        ),
        tickfont=dict(
            size=12,
            color='black'
        )
    ),
)

fig = go.Figure(data=graph, layout=layout)

iplot(fig)


# # Top 10 Artists <a class="anchor" id="2"></a>
# * Get top 10 artists based on the number of times I've listened to those.

# In[ ]:


top_artists = pysql("""
SELECT
    id, artist_name, count(artist_name) AS artist_count 
FROM
    data
GROUP BY
    artist_name
ORDER BY
    count(artist_name) DESC
LIMIT 10
""")

graph = [go.Bar(
            x=top_artists['artist_name'],
            y=top_artists['artist_count'],
            text=top_artists['artist_name'],
            marker=dict(
                color='purple',
                line=dict(
                    color='blue',
                    width=1.5,
                )
            ),
            opacity=0.6,
)]

layout = go.Layout(
    title='Top 10 Artists',
    plot_bgcolor='pink',
    paper_bgcolor='pink',
    titlefont=dict(
            size=20,
            color='black'
    ),
    xaxis=dict(
        title='Artists',
        titlefont=dict(
            size=16,
            color='black'
        ),
        tickfont=dict(
            size=12,
            color='gray'
        )
    ),
    yaxis=dict(
        title='Listen Count',
        titlefont=dict(
            size=16,
            color='black'
        ),
        tickfont=dict(
            size=12,
            color='black'
        )
    ),
)

fig = go.Figure(data=graph, layout=layout)

iplot(fig)


# # Top 10 Releases <a class="anchor" id="3"></a>
# * Get top 10 Releases based on the number of times I've listened to those.

# In[ ]:


top_releases = pysql("""
SELECT
    id, release_name, count(release_name) AS release_count 
FROM
    data
WHERE
    release_name IS NOT NULL
GROUP BY
    release_name
ORDER BY
    count(release_name) DESC
LIMIT 10
""")

graph = [go.Bar(
            x=top_releases['release_name'],
            y=top_releases['release_count'],
            text=top_releases['release_name'],
            marker=dict(
                color='seagreen',
                line=dict(
                    color='green',
                    width=1.5,
                )
            ),
            opacity=0.6,
)]

layout = go.Layout(
    title='Top 10 Releases',
    plot_bgcolor='lightgreen',
    paper_bgcolor='lightgreen',
    titlefont=dict(
            size=20,
            color='black'
    ),
    xaxis=dict(
        title='Releases',
        titlefont=dict(
            size=16,
            color='black'
        ),
        tickfont=dict(
            size=12,
            color='gray'
        )
    ),
    yaxis=dict(
        title='Listen Count',
        titlefont=dict(
            size=16,
            color='black'
        ),
        tickfont=dict(
            size=12,
            color='black'
        )
    ),
)

fig = go.Figure(data=graph, layout=layout)

iplot(fig)


# # Daily Listens Count <a class="anchor" id="4"></a>
# * Get the number of songs I listened each day.

# In[ ]:


daily_listen_count =  pysql("""
SELECT
    strftime('%d-%m-%Y', date_time) as date, count(strftime('%d-%m-%Y', date_time)) AS listen_count
FROM
    data
GROUP BY
    strftime('%d-%m-%Y', date_time)
ORDER BY
    strftime('%Y-%m-%d', date_time)
""")

ts_min = daily_listen_count['date'].min()
ts_max = daily_listen_count['date'].max()

graph = [go.Scatter(
            x=daily_listen_count['date'],
            y=daily_listen_count['listen_count'],
            text=daily_listen_count['date'],
            marker=dict(
                color='blue',
                line=dict(
                    color='blue',
                    width=4,
                )
            ),
            opacity=0.6,
)]

layout = go.Layout(
    title='Daily Listens Count',
    plot_bgcolor='azure',
    paper_bgcolor='azure',
    titlefont=dict(
            size=20,
            color='black'
    ),
    xaxis=dict(
        title='Timeline',
        showticklabels=False,
        titlefont=dict(
            size=16,
            color='black'
        ),
        tickfont=dict(
            size=12,
            color='gray'
        )
        
    ),
    yaxis=dict(
        title='Listens Count',
        showgrid=True,
        titlefont=dict(
            size=16,
            color='black'
        ),
        tickfont=dict(
            size=12,
            color='black'
        )
    ),
)

fig = go.Figure(data=graph, layout=layout)
iplot(fig)


# # Daily Listening Timings <a class="anchor" id="5"></a>
# * Get the daily timings on which I listen to music.
# 
# * This one is the most interesting graph in this entire notebook. As can be seen from the graph I listen to music a lot more aroud 10PM-5AM and 2PM-4PM. This can be seen as the time in which I'm not busy. This graph will vary a lot for different users and can be a measure on how busy a person is around which time of the day.

# In[ ]:


daily_timing =  pysql("""
SELECT
    strftime('%H', date_time) as hour, count(strftime('%H', date_time)) AS listen_count
FROM
    data
GROUP BY
    strftime('%H', date_time)
ORDER BY
    strftime('%H', date_time)
""")

graph = [go.Bar(
            x=daily_timing['hour'],
            y=daily_timing['listen_count'],
            marker=dict(
                color='blue',
                line=dict(
                    color='mediumorchid',
                    width=2.5,
                )
            ),
            opacity=0.6,
)]

layout = go.Layout(
    title='Daily Listening timings',
    plot_bgcolor='lightgreen',
    paper_bgcolor='lightgreen',
    titlefont=dict(
            size=20,
            color='black'
    ),
    xaxis=dict(
        title='Hours(24hrs format)',
        titlefont=dict(
            size=16,
            color='black'
        ),
        tickfont=dict(
            size=12,
            color='gray'
        ),
    ),
    yaxis=dict(
        title='Listens Count',
        showgrid=True,
        titlefont=dict(
            size=16,
            color='black'
        ),
        tickfont=dict(
            size=12,
            color='black'
        )
    ),
)

fig = go.Figure(data=graph, layout=layout)
iplot(fig)


# # Top 10 Tracks Listening Timeline <a class="anchor" id="6"></a>
# * Graphs of how many times I listened to a track on a particular day.

# In[ ]:


top_track_listen_timeline = []
for track in top_tracks['track_name']:
    if "'" in track:
        track = track.replace("'", "''")
    top_track_listen_timeline.append(pysql("""
        SELECT
            track_name, strftime('%Y-%m-%d', date_time) as date, timestamp,
            count(strftime('%d-%m-%Y', date_time)) AS listen_count
        FROM
            data
        WHERE
            track_name = '{0}'
        GROUP BY
            strftime('%d-%m-%Y', date_time)
        ORDER BY
            strftime('%Y-%m-%d', date_time)
    """.format(track))
    )

for idx, track in enumerate(top_tracks['track_name']):
    graph = [go.Scatter(
                x=top_track_listen_timeline[idx]['date'],
                y=top_track_listen_timeline[idx]['listen_count'],
                marker=dict(
                    color='blue',
                ),
                opacity=0.6,
    )]

    layout = go.Layout(
        title=track,
        plot_bgcolor='azure',
        paper_bgcolor='azure',
        titlefont=dict(
                size=20,
                color='black'
        ),
        xaxis=dict(
            title='Timeline',
            zerolinewidth=4,
            titlefont=dict(
                size=16,
                color='black'
            ),
            tickfont=dict(
                size=12,
                color='black'
            )

        ),
        yaxis=dict(
            title='Listens Count',
            showgrid=True,
            titlefont=dict(
                size=16,
                color='black'
            ),
            tickfont=dict(
                size=12,
                color='black'
            )
        ),
    )

    fig = go.Figure(data=graph, layout=layout)
    iplot(fig)


# # Top 10 Artists Listening Timeline <a class="anchor" id="7"></a>
# * Graphs of how many times I listened to an artist on a particular day.

# In[ ]:


top_artist_listen_timeline = []
for artist in top_artists['artist_name']:
    if "'" in artist:
        artist = artist.replace("'", "''")
    top_artist_listen_timeline.append(pysql("""
        SELECT
            artist_name, strftime('%Y-%m-%d', date_time) as date, timestamp,
            count(strftime('%d-%m-%Y', date_time)) AS listen_count
        FROM
            data
        WHERE
            artist_name = '{0}'
        GROUP BY
            strftime('%d-%m-%Y', date_time)
        ORDER BY
            strftime('%Y-%m-%d', date_time)
    """.format(artist))
    )

for idx, artist in enumerate(top_artists['artist_name']):
    graph = [go.Scatter(
                hoverinfo='x+y+text',
                x=top_artist_listen_timeline[idx]['date'],
                y=top_artist_listen_timeline[idx]['listen_count'],
                text=artist,
                marker=dict(
                    color='red',
                ),
                opacity=0.6,
    )]

    layout = go.Layout(
        title=artist,
        plot_bgcolor='lightpink',
        paper_bgcolor='lightpink',
        titlefont=dict(
                size=20,
                color='black'
        ),
        xaxis=dict(
            title='Timeline',
            zerolinewidth=4,
            titlefont=dict(
                size=16,
                color='black'
            ),
            tickfont=dict(
                size=12,
                color='black'
            )

        ),
        yaxis=dict(
            title='Listens Count',
            showgrid=True,
            titlefont=dict(
                size=16,
                color='black'
            ),
            tickfont=dict(
                size=12,
                color='black'
            )
        ),
    )

    fig = go.Figure(data=graph, layout=layout)
    iplot(fig)


# # Top 10 Releases Listening Timeline <a class="anchor" id="8"></a>
# * Graphs of how many times I listened to a release on a particular day.
# * An interesting one here is timeline of release "100 violin masterworks". Which makes to the top 10 despite of the fact that I've listened it on only two days.

# In[ ]:


top_release_listen_timeline = []
for release in top_releases['release_name']:
    if "'" in release:
        release = release.replace("'", "''")
    top_release_listen_timeline.append(pysql("""
        SELECT
            release_name, strftime('%Y-%m-%d', date_time) as date, timestamp,
            count(strftime('%d-%m-%Y', date_time)) AS listen_count
        FROM
            data
        WHERE
            release_name = '{0}'
        GROUP BY
            strftime('%d-%m-%Y', date_time)
        ORDER BY
            strftime('%Y-%m-%d', date_time)
    """.format(release)
    ))

for idx, release in enumerate(top_releases['release_name']):
    graph = [go.Scatter(
                x=top_release_listen_timeline[idx]['date'],
                y=top_release_listen_timeline[idx]['listen_count'],
                marker=dict(
                    color='blue',
                ),
                opacity=0.6,
    )]

    layout = go.Layout(
        title=release,
        plot_bgcolor='lightgreen',
        paper_bgcolor='lightgreen',
        titlefont=dict(
                size=20,
                color='black'
        ),
        xaxis=dict(
            title='Timeline',
            zerolinewidth=4,
            titlefont=dict(
                size=16,
                color='black'
            ),
            tickfont=dict(
                size=12,
                color='black'
            )

        ),
        yaxis=dict(
            title='Listens Count',
            showgrid=True,
            titlefont=dict(
                size=16,
                color='black'
            ),
            tickfont=dict(
                size=12,
                color='black'
            )
        ),
    )

    fig = go.Figure(data=graph, layout=layout)
    iplot(fig)


# # Conclusion  <a class="anchor" id="9"></a>
# With just the timestamp, track, artist, and release we can have this type of stats. But the data contains much more information than that.
# 
# We have MSIDs(MessyBrainzIDentifier) in the data which are associated with unclean metadata submitted by users (to know more about it checkout [here](https://messybrainz.org/)).
# 
# We also have MBIDs(MusicBrainzIDentifier) for some data which are associated with clean metadata (to know more about it checkout [here](https://musicbrainz.org/doc/MusicBrainz_Identifier)). Using the wide varity of information from [MusicBrainz](https://musicbrainz.org/) we can have access to a lot more stats.
# 
# From MBIDs we can get  Acoustic information of a recording (to know more go to [AcousticBrainz](https://acousticbrainz.org/)), we can get information about genres and moods of the song with which we can get stats like:
# 
# - Most listened genres
# - Daily mood based on listen history
