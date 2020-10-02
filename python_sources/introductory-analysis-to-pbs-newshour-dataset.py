#!/usr/bin/env python
# coding: utf-8

# <a id="toc"></a>
# 
# # <u>Table of Contents</u>
# 
# ### Part I: Data Overview  
# 1.) [Background](#background)  
# 2.) [Setup](#setup)  
# &nbsp;&nbsp;&nbsp;&nbsp; 2.1.) [Standard Imports](#imports)   
# &nbsp;&nbsp;&nbsp;&nbsp; 2.2.) [Visualization Imports](#imports)   
# &nbsp;&nbsp;&nbsp;&nbsp; 2.3.) [Helpers](#helpers)   
# &nbsp;&nbsp;&nbsp;&nbsp; 2.4.) [Load data](#load)   
# 3.) [General Overview](#general)  
# &nbsp;&nbsp;&nbsp;&nbsp; 3.1.) [Timezone](#timezone)   
# &nbsp;&nbsp;&nbsp;&nbsp; 3.2.) [Oldest Transcript](#oldest)   
# &nbsp;&nbsp;&nbsp;&nbsp; 3.3.) [5 Oldest Stories](#old_5)   
# &nbsp;&nbsp;&nbsp;&nbsp; 3.4.) [Date spread](#date_spread)   
# &nbsp;&nbsp;&nbsp;&nbsp; 3.5.) [Earliest interview by person](#earliest_interview)   
# &nbsp;&nbsp;&nbsp;&nbsp; 3.6.) [Total words spoken](#speaker_total_words)   
# 4.) [Trends](#trends)  
# &nbsp;&nbsp;&nbsp;&nbsp; 4.1.) [Topic Popularity](#topic_popularity)  

# ---
# <a id="background"></a>
# 
# # [^](#toc) <u>Background</u>
# 
# PBS Newshour is an American daily news program founded in 1975.  The program spans 1hr on the weekdays and 30min on the weekends and it covers domestic and international news.
# 
# This notebook is a very basic introduction to PBS Newshour's dataset.  There is more to be done and I hope to do it soon!

# ---
# <a id="setup"></a>
# 
# # [^](#toc) <u>Setup</u>
# 
# Below I import some libraries and create helper functions

# <a id="imports"></a>
# 
# ### [^](#toc) Standard imports

# In[ ]:


### Standard imports
import pandas as pd
import numpy as np
pd.options.display.max_columns = 50

### Time imports
import datetime
import time

# Counter
from collections import Counter

# Operator
import operator

# Regular Expressions
import re

# Directory helper
import glob

# Language processing import
import nltk

# Random
import random

# Progress bar
from tqdm import tqdm

### Removes warnings that occassionally show in imports
import warnings
warnings.filterwarnings('ignore')


# <a id="vis_imports"></a>
# 
# ### [^](#toc) Visualization imports

# In[ ]:


### Standard imports
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set()

### Altair
import altair as alt
alt.renderers.enable('notebook')

### Plotly
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
import plotly.plotly as py
from plotly import tools
init_notebook_mode(connected=True)

# WordCloud
from wordcloud import WordCloud

# Folium
import folium


# <a id="helpers"></a>
# 
# ### [^](#toc) Helpers

# In[ ]:


# A short hand way to plot most bar graphs
def pretty_bar(data, ax, xlabel=None, ylabel=None, title=None, int_text=False, x=None, y=None):
    
    if x is None:
        x = data.values
    if y is None:
        y = data.index
    
    # Plots the data
    fig = sns.barplot(x, y, ax=ax)
    
    # Places text for each value in data
    for i, v in enumerate(x):
        
        # Decides whether the text should be rounded or left as floats
        if int_text:
            ax.text(0, i, int(v), color='k', fontsize=14)
        else:
            ax.text(0, i, round(v, 3), color='k', fontsize=14)
     
    ### Labels plot
    ylabel != None and fig.set(ylabel=ylabel)
    xlabel != None and fig.set(xlabel=xlabel)
    title != None and fig.set(title=title)

def pretty_transcript(transcript, limit_output=0):
    for i, speaker in enumerate(transcript):
        if limit_output and i > limit_output:
            print("  (...)")
            break
        print(color.UNDERLINE, speaker[0] + ":", color.END)
        for txt in speaker[1:]:
            print("\n\n   ".join(txt))
        print()
    
def get_trend(series, ROLLING_WINDOW=16):
    trend = series.rolling(
        window=ROLLING_WINDOW,
        center=True, min_periods=1).mean()

    trend = trend.rolling(
        window=ROLLING_WINDOW // 2,
        center=True, min_periods=1).mean()

    trend = trend.rolling(
        window=ROLLING_WINDOW // 4,
        center=True, min_periods=1).mean()
    return trend
    
### Used to style Python print statements
class color:
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


# <a id="load"></a>
# 
# ### [^](#toc) Load data
# 
# I'm working with PBS Newshour's clean dataset.  I cleaned it myself and the cleaning is up to my standards :)

# In[ ]:


pbs = pd.read_json("../input/PBS-newhour-clean.json")
pbs = pbs.sort_values("Date")

pbs.Story.fillna("", inplace=True)

pbs["Year"]  = pbs.Date.map(lambda x: x.year)
pbs["Month"] = pbs.Date.map(lambda x: x.month)

print("Shape of pbs:", pbs.shape)
pbs.head()


# <a id="general"></a>
# 
# # [^](#toc) <u>General Overview</u>

# <a id="timezone"></a>
# 
# ### [^](#toc) Timezone
# 
# We can see the timezone is always EDT.  The program is hosted in Arlington, Virginia and New York City so this makes sense.

# In[ ]:


pbs.Timezone.value_counts()


# <a id="oldest_clip"></a>
# 
# ### [^](#toc) Oldest Clip
# 
# The oddest news clip is Robert MacNeil's and Jim Lehrer's coverage of Nixon's Watergate scandal.  This story aired back in 1973 and won MacNeil and Lehrer both an Emmy.
# 
# The coverage was so highly praised that it led to the creation of The MacNeil/Lehrer Report, the predecessor of PBS Newshour.

# In[ ]:


temp = pbs.iloc[0]

print(temp.Title)
print(temp.URL)


# <a id="oldest_transcript"></a>
# 
# ### [^](#toc) Oldest Transcript
# 
# The oldest complete transcript on PBS's website is an interview with Fidel Castro in February of 1985.

# In[ ]:


temp = pbs[pbs.Transcript.map(lambda x: x != [])].iloc[0]

print(f"{color.BOLD}{temp.Date}{color.END}")
print(f"{color.BOLD}{temp.Title}{color.END}")
print()
pretty_transcript(temp.Transcript, limit_output=2)


# <a id="old_5"></a>
# 
# ### [^](#toc) 5 Oldest Stories
# 
# It looks like PBS Newshour's archive have a lot of gaps early in it's development

# In[ ]:


for i in range(5):
    print(pbs.iloc[i].Date)
    print(pbs.iloc[i].Story)
    print()


# <a id="date_spread"></a>
# 
# ### [^](#toc) Date spread
# 
# The activity starts around March 2011, so we have 7 years of history to analyze

# In[ ]:


temp = (pbs
        .assign(n=0)
        .set_index("Date")
        .groupby(pd.Grouper(freq="M"))
        .n
        .apply(len)
        .sort_index()
)

trace = go.Scatter(
        x=temp.index,
        y=temp.values,
    )

layout = go.Layout(
    title = "Number of transcripts available over time",
    yaxis=dict(title="Number of transcripts"),
    xaxis=dict(title="Date"),
)



fig = go.Figure(data=[trace], layout=layout)
iplot(fig)


# <a id="earliest_interview"></a>
# 
# ### [^](#toc) Earliest interview by person
# 
# It's only 7 years, but I think it's amazing just looking back 7 years.  So much has changed.  In another sense, not much has changed.
# 
# The earliest mention of Donald Trump is in 2011 when he was demanding Obama for his birth certificate.  During that segment he is considering running for office. ([link](https://www.pbs.org/newshour/show/with-birth-certificate-release-obama-urges-shift-in-national-dialogue)).  This is tangetial, but this [clip](https://www.pbs.org/newshour/show/with-birth-certificate-release-obama-urges-shift-in-national-dialogue) also features PBS' Jim Lehrer 40 years earlier.
# 
# The earliest mention of Bernie Sanders is him weighing in on the 2011 Debt Ceiling negotitions ([link](https://www.pbs.org/newshour/show/debt-deal-stalemate-spills-into-weekend-for-obama-congress)).  He warns that the burden will fall on the working class.

# In[ ]:


### These are just examples
pois = {0: "BERNIE SANDERS",
        1: "VLADIMIR PUTIN",
        2: "DONALD TRUMP",
        3: "JUDY WOODRUFF",
        4: "BEN CARSON",
        5: "STEPHEN COLBERT",
        6: "HILLARY CLINTON",
        7: "JOHN F. KENNEDY",
        8: "ANGELA MERKEL",
        9: "JEFF BEZOS",
        10: "XI JINPING"
}

poi = pois[2]

print("Showing results for:", poi)
pbs[pbs.Speakers.map(lambda x: poi in x)].head(3)


# <a id="speaker_total_words"></a>
# 
# ### [^](#toc) Total words spoken
# 
# Before doing any analysis, it's important to check that we have enough data to form meaningful conclusions. We can see from figure 2 the number of words spoken by each famous person. Besides Angela Merkel, it looks like we have plenty of data to work with!

# In[ ]:


pois = ["BERNIE SANDERS", "DONALD TRUMP", "HILLARY CLINTON",
        "BARACK OBAMA", "MITT ROMNEY", "ANGELA MERKEL",
        "JOSEPH BIDEN", "MIKE PENCE"]

def get_num_articles(df, poi):
    num_articles = len(df[df.Speakers.map(lambda x: poi in x)])
    return num_articles

def get_num_words(df, poi):
    speaker_text = list()
    transcripts  = df[df.Speakers.map(lambda x: poi in x)].Transcript.values
    num_words    = 0
    
    for transcript in transcripts:
        for person in transcript:
            if person[0] == poi:
                for txt in person[1]:
                    num_words += len(txt.split(" "))
    return num_words

articles, words = list(), list()

for poi in pois:
    num_articles = get_num_articles(pbs, poi)
    num_words    = get_num_words(pbs, poi)
    
    articles.append(num_articles)
    words.append(num_words)

trace1 = go.Bar(
    x=pois,
    y=articles,
    name='Total articles'
)
trace2 = go.Bar(
    x=pois,
    y=words,
    name='Total words'
)

data = [trace1, trace2]
layout = go.Layout(
    barmode='group'
)

fig = go.Figure(data=data, layout=layout)
iplot(fig);


# ### [^](#toc) Most Popular Speakers
# 
# The plot below counts the number of times a speaker has appeared in an article and plots the top counts.
# 
# The top speakers are mostly employees of PBS Newshour, so it makes sense they'd top the list.  However besides them and the filler names like "Man" and "Woman", we have Barack Obama, Donald Trump, Hillary Clinton, Mitch McConnell, Mitt Romney, John Boehner, and John Kerry.  All of which are very important politicians

# In[ ]:


persons = pbs.Speakers.map(list).sum()

freq = sorted(Counter(persons).items(), key=operator.itemgetter(1), reverse=True)
x, y = list(zip(*freq[:25]))

fig, ax = plt.subplots(1, 1, figsize=(14, 14))
temp = pd.Series(list(y), index=list(x))
pretty_bar(temp, ax, title="Top Speakers", xlabel="Number of Transcripts");


# ---
# <a id="trends"></a>
# 
# # [^](#toc) Trends

# <a id="topic_popularity"></a>
# 
# ### [^](#toc) Topic Popularity
# 
# This shows the popularity of a word for a given month.  I measure the fraction of time a word is used for a particular story, then take the average value for a given month.
# 
# To look at the topic of a topic, multiple moving averages are performed to smooth out fluctuations.
# 
# There seems to be an increasing trend talking about immigration and racism.  Interestingly, PBS has no mention of racism until 2013.

# In[ ]:


LIMIT_TIME = True
topics     = ["Obama", "Trump", "Clinton", "Bush", "Immigration", "Congress", "Racism"]

def topic_popularity(topic):
    def popularity_helper(transcript):
        transcript = list(map(lambda x: x[1][0], transcript))
        transcript = (" ".join(transcript).lower()).split(" ")
        N          = len(transcript)
        counts     = Counter(transcript)
        return (counts[topic.lower()] / N) * 100
    return popularity_helper

if LIMIT_TIME:
    temp = pbs[pbs.Year > 2010]
else:
    temp = pbs

datas = []
for topic in tqdm(topics):
    temp["Temp"] = (
                temp[temp.Transcript.map(lambda x: x != [])]
                    .Transcript
                    .map(topic_popularity(topic))
                )

    data = (temp
         .set_index("Date")
         .groupby(pd.Grouper(freq="M"))
         .Temp
         .apply(np.mean)
    )

    trend = get_trend(data, ROLLING_WINDOW=12)

    datas.append((topic, data, trend))

traces = []

for topic, data, _ in datas:
    traces.append(go.Scatter(
                            x=data.index,
                            y=data.values,
                            name=f"{topic} - actual"
                        ))
    
for topic, _, trend in datas:
    traces.append(go.Scatter(
                            x=trend.index,
                            y=trend.values, 
                            name=f"{topic} - trend"
                        ))
buttons = []

for i, topic in enumerate(topics):
    visibility = [i==j for j in range(len(topics))]
    button = dict(
                 label =  topic,
                 method = 'update',
                 args = [{'visible': visibility},
                     {'title': f"'{topic}' usage over time" }])
    buttons.append(button)

updatemenus = list([
    dict(active=-1,
         x=-0.15,
         buttons=buttons
    )
])

layout = dict(title='Topic popularity', 
              updatemenus=updatemenus,
                xaxis=dict(title='Date'),
                yaxis=dict(title='Percent of words')
             )

fig = dict(data=traces, layout=layout)
fig['layout'].update(height=800, width=800)

iplot(fig)


# In[ ]:




