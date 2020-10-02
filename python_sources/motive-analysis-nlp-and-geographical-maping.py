#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# This is my second kernel. In this notebook i would like to try to visualize the motives behind terorism between 1970 and 2016. there are 4 parts that i have done in this Notebook, they are:
# 
#     1. Uploading the data and taking the relevant atributes
#     2. Analyze the motives behind terorism
#     3. Preparing the data for visualizing in map
#     4. Visualize the number of terorism in every country in the map
# 
# 
# Before jumping to the main part, all of the relevant library

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import collections

pd.options.display.max_columns = 999

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

import warnings
warnings.filterwarnings('ignore')


# # 1. Uploading the data and taking the relevant atributes
# 
# For this project, only Country and Motive column are used 

# In[ ]:


terror=pd.read_csv('../input/globalterrorismdb_0617dist.csv',encoding='ISO-8859-1')
terror.rename(columns={'country_txt':'Country','motive':'Motive'},inplace=True)
#terror=terror[['Country','Motive']]
#terror['casualities']=terror['Killed']+terror['Wounded']
terror.head(3)


# # 2. Basic analysis
# ## 2.1 Number of terorist every year

# In[ ]:


plt.subplots(figsize=(15,6))
sns.countplot('Year',data=terror,palette='RdYlGn_r',edgecolor=sns.color_palette('dark',7))
plt.xticks(rotation=90)
plt.title('Number Of Terrorist Activities Each Year')
plt.show()


# 

# #     2. Analyze the motives behind terorism
# 
# For this we will use NLTK for Natural Language Processing. The reason for using NLP is because if we simply take the count of words and make a wordcloud. by using NLTK, we can aldo filter out the unimportant words such as 'the', 'is', etc and find other important words.
# 

# In[ ]:


import nltk
from wordcloud import WordCloud, STOPWORDS
motive=terror['Motive'].str.lower().str.replace(r'\|', ' ').str.cat(sep=' ')
words=nltk.tokenize.word_tokenize(motive)
word_dist = nltk.FreqDist(words)
stopwords = nltk.corpus.stopwords.words('english')
words_except_stop_dist = nltk.FreqDist(w for w in words if w not in stopwords) 
wordcloud = WordCloud(stopwords=STOPWORDS).generate(" ".join(words_except_stop_dist))
plt.imshow(wordcloud)
fig=plt.gcf()
fig.set_size_inches(10,16)
plt.axis('off')
plt.show(wordcloud)


# by seeing figure above it can be seen that the highest word count is 'al' which is the part of the name for some terorist group such as al-qaeda. the other words that catching attention is 'anti' and 'pro'

# # 3. Preparing the data for visualizing in map
# 
# grouping country coulumn is done for getting the total number of terorism in every country. afterwards, i retrive 3 digit country code data from iso3166 library and merge it with country group data based on the country column.

# In[ ]:


terror = terror.replace("United Kingdom","United Kingdom of Great Britain and Northern Ireland")

df = pd.DataFrame(terror.groupby('Country')['Country'].count())
df.columns = ['count']
df.index.names = ['Country']
df = df.reset_index()
df.head(10)


# In[ ]:


from iso3166 import countries
import iso3166
#countries.get(dftotal['Country'])
countlist= pd.DataFrame(iso3166.countries_by_alpha3).T.reset_index()

countlist = countlist[[0,2]]
countlist.rename(columns={0:'Country',2:'code'},inplace=True)
countlist.head(10)


# In[ ]:


dftotal = pd.merge(df, countlist, on=['Country', 'Country'])
dftotal.head(10)


# # 4. Visualize the number of terorism in every country in the map

# In[ ]:


data = dict(type='choropleth',
            locations=dftotal['code'],
            text=dftotal['Country'],
            z=dftotal['count'],
            ) 

layout = dict(
    title = 'the spread of terrorist activity in the world from year 1970 to year 2016',
    geo = dict(
        showframe = False,
        showcoastlines = False,
        projection = dict(
            type = 'Mercator'
        )
    )
)


choromap = go.Figure(data=[data], layout=layout)
py.iplot( choromap, filename='d3' )

