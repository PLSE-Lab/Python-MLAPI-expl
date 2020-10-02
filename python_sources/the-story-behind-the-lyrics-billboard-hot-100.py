#!/usr/bin/env python
# coding: utf-8

# This notebook tells the most interesting things, using the NLTK library, about the lyrics songs of the Billboard Hot 100 ranking, between 1965 and 2015.
# 
# Billboard has published a Year-End Hot 100 every December since 1958. Chart rankings are based on sales (physical and digital), radio play, and online streaming in the United States.
# 
# **A little history...**
# 
# At the start of the rock era in 1955, three such charts existed:
# 
# * **Best Sellers in Stores**, established in 1940. This chart ranked the biggest selling singles in retail stores, as reported by merchants surveyed throughout the country (20 to 50 positions).
# * **Most Played by Jockeys**, it ranked the most played songs on United States radio stations, as reported by radio disc jockeys and radio stations (20 to 25 positions).
# * **Most Played in Jukeboxes**, this was one of the main outlets of measuring song popularity with the younger generation of music listeners, as many radio stations resisted adding rock and roll music to their playlists for many years.
# 
# On the week ending November 12, 1955, Billboard published The Top 100 for the first time. The Top 100 combined all aspects of a single's performance (sales, airplay and jukebox activity), based on a point system that typically gave sales (purchases) more weight than radio airplay. The Best Sellers In Stores, Most Played by Jockeys and Most Played in Jukeboxes charts continued to be published concurrently with the new Top 100 chart.
# 
# The Billboard Hot 100 is still the standard by which a song's popularity is measured in the United States. The Hot 100 is ranked by radio airplay audience impressions as measured by Nielsen BDS, sales data compiled by Nielsen Soundscan (both at retail and digitally) and streaming activity provided by online music sources.
# 
# Source: https://en.wikipedia.org/wiki/Billboard_Hot_100
# 

# ### <font color='gold'>**1. Libraries and data**</font>

# In[ ]:


import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import bokeh.io
#from bokeh.charts import Donut, HeatMap, Histogram, Line, Scatter, show, output_notebook, output_file
from bokeh.plotting import figure
import string
import gensim.models.word2vec as w2v
import multiprocessing
import os
import re
import sklearn
import pprint
import seaborn as sns
import wordcloud
get_ipython().run_line_magic('matplotlib', 'inline')
stop = stopwords.words('english')
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
data = pd.read_csv("../input/billboard_lyrics_1964-2015.csv",encoding='latin-1').dropna()


# ### <font color='gold'>**2. Data**</font>

# In[ ]:


#Count data
len(data)


# In[ ]:


#Show first line of dataset
data.head(1)


# ### <font color='gold'>**3. Cleaning Data Set**</font>
# 
# * Stop Words removal
# * POS tag - Lyrics only Noun and Adj

# In[ ]:


data['Cleaned_Lyrics'] = data['Lyrics'].str.lower().str.split()
data['Cleaned_Lyrics'] = data['Cleaned_Lyrics'].apply(lambda x : [item for item in x if item not in stop])

from nltk.tag import pos_tag
def Getting_NN(sentence):
   sentence=sentence.lower()
   cleaned=' '.join([w for w in sentence.split() if not w in stop]) 
   cleaned=' '.join([w for w , pos in pos_tag(cleaned.split()) if (pos == 'NN'  )])
   cleaned=cleaned.strip()
   return cleaned
data['pos_tag_NN']= data['Lyrics'].apply(lambda x:Getting_NN(x))

def Getting_JJR(sentence):
   sentence=sentence.lower()
   cleaned=' '.join([w for w in sentence.split() if not w in stop]) 
   cleaned=' '.join([w for w , pos in pos_tag(cleaned.split()) if (pos == 'JJR'  )])
   cleaned=cleaned.strip()
   return cleaned

data['pos_tag_JJR']= data['Lyrics'].apply(lambda x:Getting_JJR(x))


# ### <font color='gold'>**4. vocabulary adjustments**</font>
# 
# * Vocabulary count
# * Nouns in lyrics count
# * Adj in lyrics count
# 
# 

# In[ ]:


data['ly_count'] = data['Lyrics'].str.split(" ").str.len()
data['NN_count'] = data['pos_tag_NN'].str.split(" ").str.len()
data['JJR_count'] = data['pos_tag_JJR'].str.split(" ").str.len()
data.head(1)


# ### <font color='gold'>**5. Number of songs by Artist**<font>

# In[ ]:


Artist_Count = data.Artist.value_counts()[:25]
Artist_Count


# In[ ]:


plt.figure(figsize=(13,9))
plt.title("Maximum Lyrics Count By Artist",fontsize=20)
data['Artist'].value_counts()[:30].plot('bar',color='purple')


# **Madonna is the artist with most appearances on the Billboard hot 100 according to the lyrics! To date, she has had 12 songs in No. 1 hits, 38 in the top 10 and 57 songs in the history of the ranking.**

# ### <font color='gold'>**6. Vocabulary in lyrics**<font>

# In[ ]:


Total_year_count = data.groupby(['Year'])['ly_count'].sum()
plt.figure(figsize=(12,9))
plt.title("Lyrics Vocabulary Count Vs Year",fontsize=20)
Total_year_count.plot(kind='line',color="Red")


# In[ ]:


plt.figure(figsize=(13,9))
plt.title("Vocabulary Count In Lyrics vs year",fontsize=20)
Total_year_count.plot(kind='bar',label="Lyrics count vs year",color = 'slateblue')


# **During the first decade of 2000, the greatest amount of vocabulary was used, with around 50,000 words.  year with the most vocabulary was 2003**

# ### <font color='gold'>**7. Nouns and Adjectives**<font>

# In[ ]:


Noun_year_count = data.groupby(['Year'])['NN_count'].sum()
plt.figure(figsize=(13,9))
plt.title("Artist Noun Usage In Lyrics",fontsize=20)
Noun_year_count.plot(label='Noun count vs year',kind='line',color = 'c')


# In[ ]:


Adj_year_count = data.groupby(['Year'])['JJR_count'].sum()
plt.figure(figsize=(13,9))
plt.title("Artist Adjective Usage Count In Lyrics",fontsize=20)
Adj_year_count.plot(label='Adjective count vs year',kind='line',color = 'r')


# ** The use of nouns goes hand in hand with the use of vocabulary, however, the use of adjectives is fluctuating but ascending.**

# ### <font color='gold'>**8. The Love**<font>

# In[ ]:


plt.figure(figsize=(12,5))
plt.title("Max Repeated Words In Lyrics",fontsize=20)
words = pd.Series(' '.join(data['Cleaned_Lyrics'].astype(str)).lower().split(" ")).value_counts()[:500]
words.plot(color='darkcyan')


# In[ ]:


allwords = ' '.join(data['Song']).lower().replace('c', '')
cloud = wordcloud.WordCloud(background_color='black',
                            max_font_size=100,
                            width=1000,
                            height=500,
                            max_words=100,
                            relative_scaling=.5).generate(allwords)
plt.figure(figsize=(15,5))
plt.axis('off')

plt.imshow(cloud);


# **Love is still the great theme. **

# ![image.png](attachment:image.png)

# **According to CNN, what makes a listener melt away with a song can make the sound of a modem look like another. And almost any song can become a love song if it connects with an emotional memory of someone we love or have loved. That is the reason for love to be a success behind the lists.**

# ### <font color='gold'>**9. The best bigram and Tigramas **<font>

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
text = data['Lyrics']
#define vectorizer parameters
tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=20000,
                                 min_df=3, stop_words='english',
                                 use_idf=True,ngram_range=(2,3))
tfidf_matrix = tfidf_vectorizer.fit_transform(text) 
print(tfidf_matrix.shape)
terms = tfidf_vectorizer.get_feature_names()


# In[ ]:


#Bigrams

from sklearn.feature_extraction.text import CountVectorizer
word_vectorizer = CountVectorizer(ngram_range=(2,2),max_features=25,stop_words='english',analyzer ='word',strip_accents='ascii')
bigrams=word_vectorizer.fit(data['Lyrics']).vocabulary_
bigrams


# **Youre gona... where??, and Yeah yeah are the most repeated bigramas in the lists.**

# In[ ]:


#Tigrams

tri = CountVectorizer(ngram_range=(3,3),max_features=25,stop_words='english',analyzer ='word',strip_accents='ascii')
trigrams=tri.fit(data['Lyrics']).vocabulary_
trigrams


# **Yeah yeah yeah and Ya Ya ya, are the most repeated tigrams. I think artists like exclamations**
