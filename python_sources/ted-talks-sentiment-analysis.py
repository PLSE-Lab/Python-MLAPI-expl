#!/usr/bin/env python
# coding: utf-8

# We will read both datasets and organise them inside a big pandas DataFrame.
# 
# There are some columns that are not being used in this Jupyter notebook, but I have some plans of what I will do with them: they are here just to remeber me what I want to do. 
# 
# This notebook aims to analyse the Data by the tags that were assigned to each video. This ways, we will plot:
# - Views x Tags, which analyse how many views each tag has received;
# - Sentiments x Tags,  which will tell us which are the topics (tags) that make people more positive.

# In[ ]:


import pandas as pd
from afinn import Afinn
import numpy as np
import re
from nltk.corpus import stopwords
afinn = Afinn()

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

from random import randint as rdint


# In[ ]:


dfTED = pd.read_csv('ted_main.csv')
dfTranscripts = pd.read_csv('transcripts.csv')


# In[ ]:


dfTED = dfTED.get(['main_speaker','title','tags','views', 'url'])
dfTED = dfTED.merge(dfTranscripts, how = 'inner', sort = ['views'])


# Now we have created the DataFrame with the columns: 
# - Main_speaker: tells us the name of who talked;
# - Title: has the tittle of the talk, thus, what has more importance within the talk;
# - Tags: the main point of this analysis, it has a string starting with '[', finishing with ']' and has the tags between ' " ' just like "technology";
# - Views: has the amount of views that each video has;
# - Url: has the url for the talk (will be used so we can judge sentiments within the comments;
# - Transcripts: has the transcript of the talk, it's where the sentiment analysis is based on.

# In[ ]:


def clean_n_get(text):
    
    """This function receives a text, clean it and rate the sentiments in it"""
    
    instance = re.sub("[^a-zA-Z]", " ", text).lower().split()
    stops = set(stopwords.words("english"))
    
    cleaned_text = [w for w in instance if w not in stops]
    sentiments = [afinn.score(x) for x in cleaned_text]
    
    return sum(sentiments)

def random_color():
    
    """This function creates an random RGB color string"""
    
    color = str()
    letters = ['A','B','C','D','E','F','0','1','2','3','4','5','6','7','8','9']
    for _ in range(6):
        color += letters[rdint(0,15)]
    return '#' + color


# In[ ]:


dfTED['sentiments'] = dfTED['transcript'].apply(clean_n_get)
# This cell creates the setiments columns, enabling us to create the plot afterwards


# In[ ]:


dfTED.head(6) # This is what the DataFrame looks like


# In[ ]:


dfTED['views'] = dfTED['views'].div(10 ** 6) 
# I am dividng the views columns by 10^6 so it will be more understandable in the plot


# The below cell will create two series. They have, as an index, the tags; and as values numbers, which corresponds to the sum of views and sentiments, respectively.

# In[ ]:


sViews = pd.Series()
sSentiments = pd.Series()
for i, element in enumerate(dfTED['tags']):
    lista = element.strip()[1:-1].split(",")
    for el in lista:
        tag = el.split("'")[1]
        sViews[tag] = sViews.get(tag, 0) + dfTED['views'][i]
        sSentiments[tag] = sSentiments.get(tag, 0) + dfTED['sentiments'][i]
sSentiments = sSentiments.div(sSentiments.max())


# In[ ]:


sViews['TEDx'] = 0
sSentiments['TEDx'] = 0

# we zero 'TEDx' because it appears in many talks but isn't a proper tag

sViews.sort_values(ascending = False, inplace = True)
sSentiments.sort_values(ascending = False, inplace = True)


# Below we set some variables like the size of the text and which color we will use

# In[ ]:


qtdTags = 20
axisSIZE = 2.5 * qtdTags
labelSIZE = 1.2 * axisSIZE
titleSIZE = 1.6 * axisSIZE
COLOR1 = list()
COLOR2 = list()
for _ in range(qtdTags):
    COLOR1.append(random_color())
    COLOR2.append(random_color())


# In[ ]:


fig, (axes1, axes2) = plt.subplots(nrows = 2, ncols = 1, figsize = (3.2 * qtdTags, 3.2 * qtdTags))

dataViews = sViews.head(qtdTags)
dataSentiments = sSentiments.head(qtdTags)

dataViews.plot.bar(ax = axes1, color = COLOR1)
dataSentiments.plot.bar(ax = axes2, color = COLOR2)

x1, y1 = dataViews.index, dataWords
x2, y2 = dataSentiments.index, dataSentiments

axes1.set_title("Tags x Views", fontsize = titleSIZE)
axes1.set_ylabel('Views (10^6)', fontsize = labelSIZE)
axes1.set_xlabel('Tags', fontsize = labelSIZE)
axes1.tick_params(labelsize = axisSIZE)

axes2.set_title("Tags X Sentiments",fontsize = titleSIZE)
axes2.set_ylabel('Sentiments', fontsize = labelSIZE)
axes2.set_xlabel('Tags', fontsize = labelSIZE)
axes2.tick_params(labelsize = axisSIZE)

fig.tight_layout()


# In[ ]:


# fig.savefig('TED_plots.png')
# This saves the plot above


# In[ ]:




