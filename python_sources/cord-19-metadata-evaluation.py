#!/usr/bin/env python
# coding: utf-8

# # Evaluation of Metadata

# *Based on/Forked from https://www.kaggle.com/paultimothymooney/most-common-words-in-the-cord-19-dataset*

# [CORD-19](https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge) is a resource of over 24,000 scholarly articles, including over 12,000 with full text, about COVID-19 and the coronavirus group. 

# In[ ]:


# init
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS
from collections import Counter
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import time
import warnings 
warnings.filterwarnings('ignore')


# In[ ]:


# define some functions
def count_ngrams(dataframe, column, begin_ngram, end_ngram):
    # adapted from https://stackoverflow.com/questions/36572221/how-to-find-ngram-frequency-of-a-column-in-a-pandas-dataframe
    word_vectorizer = CountVectorizer(ngram_range=(begin_ngram,end_ngram), analyzer='word')
    sparse_matrix = word_vectorizer.fit_transform(df[column].dropna())
    frequencies = sum(sparse_matrix).toarray()[0]
    most_common = pd.DataFrame(frequencies, 
                               index=word_vectorizer.get_feature_names(), 
                               columns=['frequency']).sort_values('frequency',ascending=False)
    most_common['ngram'] = most_common.index
    most_common.reset_index()
    return most_common

def word_cloud_function(df, column, number_of_words):
    # adapted from https://www.kaggle.com/benhamner/most-common-forum-topic-words
    topic_words = [ z.lower() for y in
                       [ x.split() for x in df[column] if isinstance(x, str)]
                       for z in y]
    word_count_dict = dict(Counter(topic_words))
    popular_words = sorted(word_count_dict, key = word_count_dict.get, reverse = True)
    popular_words_nonstop = [w for w in popular_words if w not in stopwords.words("english")]
    word_string=str(popular_words_nonstop)
    wordcloud = WordCloud(stopwords=STOPWORDS,
                          background_color='white',
                          max_words=number_of_words,
                          width=1000,height=1000,
                         ).generate(word_string)
    plt.clf()
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()

def word_bar_graph_function(df, column, title, nvals=50):
    # adapted from https://www.kaggle.com/benhamner/most-common-forum-topic-words
    topic_words = [ z.lower() for y in
                       [ x.split() for x in df[column] if isinstance(x, str)]
                       for z in y]
    word_count_dict = dict(Counter(topic_words))
    popular_words = sorted(word_count_dict, key = word_count_dict.get, reverse = True)
    popular_words_nonstop = [w for w in popular_words if w not in stopwords.words("english")]
    plt.barh(range(nvals), [word_count_dict[w] for w in reversed(popular_words_nonstop[0:nvals])])
    plt.yticks([x + 0.5 for x in range(nvals)], reversed(popular_words_nonstop[0:nvals]))
    plt.title(title)
    plt.show()


# ### Import data

# In[ ]:


# load metadata
t1 = time.time()
df = pd.read_csv('../input/CORD-19-research-challenge/metadata.csv') # adjust to change in data
t2 = time.time()
print('Elapsed time:', t2-t1)


# # First glance

# In[ ]:


df.head()


# In[ ]:


df.describe(include='all')


# In[ ]:


df.journal.value_counts()


# In[ ]:


# plot top 10 only
df.journal.value_counts()[0:10].plot(kind='bar')
plt.grid()
plt.show()


# In[ ]:


df.source_x.value_counts().plot(kind='bar')
plt.grid()
plt.show()


# In[ ]:


df.publish_time.value_counts()


# Ok, this seems to need a little bit of cleaning up for systematic evaluation.

# In[ ]:


df.license.value_counts()


# # Evaluate titles

# In[ ]:


# show example
df.title[0]


# In[ ]:


# show example
df.title[1]


# In[ ]:


# show most frequent words in titles
plt.figure(figsize=(10,10))
word_bar_graph_function(df,column='title', 
                        title='Most common words in the TITLES of the papers in the CORD-19 dataset',
                        nvals=20)


# In[ ]:


# evaluate 3-grams (takes some time)
t1 = time.time()
three_gram = count_ngrams(df,'title',3,3)
t2 = time.time()
print('Elapsed time:', t2-t1)


# In[ ]:


three_gram[0:20]


# In[ ]:


# plot most frequent 3-grams
fig = px.bar(three_gram.sort_values('frequency',ascending=False)[0:10], 
             x="frequency", 
             y="ngram",
             title='Top Ten 3-Grams in TITLES of Papers in CORD-19 Dataset',
             orientation='h')
fig.show()


# In[ ]:


# evaluate bigrams (takes some time)
t1 = time.time()
bi_gram = count_ngrams(df,'title',2,2)
t2 = time.time()
print('Elapsed time:', t2-t1)


# In[ ]:


bi_gram[0:20]


# In[ ]:


# plot most frequent bigrams
fig = px.bar(bi_gram.sort_values('frequency',ascending=False)[2:12], 
             x="frequency", 
             y="ngram",
             title='Top Ten relevant bigrams in TITLES of Papers in CORD-19 Dataset',
             orientation='h')
fig.show()


# In[ ]:


# word cloud
plt.figure(figsize=(10,10))
word_cloud_function(df,column='title',number_of_words=50000)


# # Search for specific keyword in titles

# In[ ]:


def word_finder(i_word, i_text):
    found = str(i_text).find(i_word)
    if found == -1:
        result = 0
    else:
        result = 1
    return result


# In[ ]:


# define keyword
my_keyword = 'enzyme'


# In[ ]:


# partial function for mapping
word_indicator_partial = lambda text: word_finder(my_keyword, text)
# build indicator vector (0/1) of hits
keyword_indicator = np.asarray(list(map(word_indicator_partial, df.title)))


# In[ ]:


# number of hits
print('Number of hits for keyword <', my_keyword, '> : ', keyword_indicator.sum())


# In[ ]:


# add index vector as additional column
df['selection'] = keyword_indicator


# In[ ]:


# select only hits from data frame
df_hits = df[df['selection']==1]


# In[ ]:


# show results
df_hits


# In[ ]:


# store result in CSV file
df_hits.to_csv('demo_keyword_search.csv')


# # Evaluate abstracts

# In[ ]:


# show example
df.abstract[3]


# In[ ]:


# show most frequent words in abstracts
plt.figure(figsize=(10,10))
word_bar_graph_function(df,column='abstract',
                        title='Most common words in the ABSTRACTS of the papers in the CORD-19 dataset',
                        nvals=20)


# In[ ]:


# word cloud
plt.figure(figsize=(10,10))
word_cloud_function(df,column='abstract',number_of_words=50000)


# In[ ]:


# evaluate 3-grams (takes some time)
t1 = time.time()
three_gram_abs = count_ngrams(df,'abstract',3,3)
t2 = time.time()
print('Elapsed time:', t2-t1)


# In[ ]:


three_gram_abs[0:20]


# In[ ]:


# plot most frequent 3-grams
fig = px.bar(three_gram_abs.sort_values('frequency',ascending=False)[0:10], 
             x="frequency", 
             y="ngram",
             title='Top Ten 3-Grams in ABSTRACTS of Papers in CORD-19 Dataset',
             orientation='h')
fig.show()


# In[ ]:


# evaluate bigrams (takes some time)
t1 = time.time()
bi_gram_abs = count_ngrams(df,'abstract',2,2)
t2 = time.time()
print('Elapsed time:', t2-t1)


# In[ ]:


bi_gram_abs[0:50]

