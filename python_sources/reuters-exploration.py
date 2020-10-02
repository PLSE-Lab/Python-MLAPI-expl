#!/usr/bin/env python
# coding: utf-8

# In this notebook, we are going to do three things basically and they are:  
# 
#     1) Basic exploration - how the news stack up against each other during different time periods  
#     
#     2) Topic modeling - To cluster the news headlines into a particular group based on their content  
#     
#     3) Search engine - Develop a basic functional search engine on top of the data   
#     
# <i>Lets begin</i>

# ## Basic exploration

# In[ ]:



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import nltk
import gensim
from gensim import *
import re
from nltk.stem import WordNetLemmatizer
import os
wnl = WordNetLemmatizer()
from collections import defaultdict
import operator
import datetime


# There are 7 years of reuters news data from 2011 to 2017. Lets read in the datasets 

# In[ ]:


reuters_11 = pd.read_csv('../input/reuters-newswire-2011.csv')
reuters_12 = pd.read_csv('../input/reuters-newswire-2012.csv')
reuters_13 = pd.read_csv('../input/reuters-newswire-2013.csv')
reuters_14 = pd.read_csv('../input/reuters-newswire-2014.csv')
reuters_15 = pd.read_csv('../input/reuters-newswire-2014.csv')
reuters_16 = pd.read_csv('../input/reuters-newswire-2015.csv')
reuters_17 = pd.read_csv('../input/reuters-newswire-2017.csv')


# In[ ]:


print (reuters_11.shape)
print (reuters_12.shape)
print (reuters_13.shape)
print (reuters_14.shape)
print (reuters_15.shape)
print (reuters_16.shape)
print (reuters_17.shape)


# This is a lot of data

# In[ ]:


reuters_11.head()


# There are only two features time of publishing and the headline

# In[ ]:


df_ls = get_ipython().run_line_magic('who_ls', 'DataFrame')
df_ls


# In[ ]:


df_ls = [reuters_11, reuters_12, reuters_13, reuters_14, reuters_15, reuters_16, reuters_17]
years = [2011, 2012, 2013, 2014, 2015, 2016, 2017]
count_headlines = [df.shape[0] for df in df_ls]


# * Lets try to plot the distribution of the number of headlines published year by year

# In[ ]:


plt.rcParams['figure.figsize'] = [15, 4]
sns.barplot(x = years, y = count_headlines)


# 2011, 2012 saw the highest number of headlines

# In[ ]:


#lets merge all the dataframes to a single dataframe
reuters = pd.DataFrame()
for df in df_ls:
    reuters = reuters.append(df, ignore_index=True)

reuters.shape


# In[ ]:


for df in df_ls:
    del df


# In[ ]:


reuters = reuters[~reuters['headline_text'].isnull()]


# In[ ]:


reuters['year'] = reuters['publish_time'].apply(str).apply(lambda x : int(x[:4]))
reuters['month'] = reuters['publish_time'].apply(str).apply(lambda x : int(x[4:6]))
grouped = reuters.groupby('month').size().reset_index()
grouped.columns = ['month', 'count']
grouped


# In[ ]:


plt.rcParams['figure.figsize'] = [15, 4]
sns.barplot(x = 'month', y = 'count', data = grouped)


# The months show a peak in the number of headlines at around the month of May and then falls again

# In[ ]:


reuters['day'] = reuters['publish_time'].apply(str).apply(lambda x : int(x[6:8]))
grouped = reuters.groupby('day').size().reset_index()
grouped.columns = ['day', 'count']
plt.rcParams['figure.figsize'] = [15, 4]
sns.barplot(x = 'day', y = 'count', data = grouped)


# The distribution in the number of headlines day wise does not show any marked difference.

# In[ ]:


reuters['hour'] = reuters['publish_time'].apply(str).apply(lambda x : int(x[8:10]))
grouped = reuters.groupby('hour').size().reset_index()
grouped.columns = ['hour', 'count']
plt.rcParams['figure.figsize'] = [15, 4]
sns.barplot(x = 'hour', y = 'count', data = grouped)


# The number of headlines show a peak at the 8th and 9th hour. That is when people normally would like their newsfeeds to come in

# Let us plot the distribution of yearwise weekday number of headlines distribution

# In[ ]:


def get_date(year, month, day):
    date_text = datetime.date(year, month, day)
    return date_text.strftime("%A")

reuters['wd'] = reuters[['year', 'month', 'day']].apply(lambda x: get_date(*x), axis = 1)


# In[ ]:


reuters.head()


# In[ ]:


grouped = reuters.groupby('wd').size().reset_index()
grouped.columns = ['wd', 'count']
plt.rcParams['figure.figsize'] = [15, 4]
sns.barplot(x = 'wd', y = 'count', data = grouped, order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])


# Surprisingly Saturday and Sunday show a decline in the number of news headlines. Regarding the weekdays the highest is around the midpoint mark of a week

# In[ ]:


reuters['count_words'] = reuters['headline_text'].apply(str).apply(len)
reuters.head()


# In[ ]:


grouped = reuters.groupby('year')['count_words'].mean().reset_index()
grouped.columns = ['year', 'count_words']
plt.rcParams['figure.figsize'] = [15, 4]
grouped['year'] = grouped['year'].astype('category')
grouped = grouped.sort_values(by = 'count_words', ascending=False)
sns.barplot(y = 'year', x = 'count_words', data = grouped, order = grouped['year'])


# The news headlines were normally longer from 2013 to 2015

# Lets see which months of each year dominated in terms of number of headlines generated

# In[ ]:


grouped = reuters.groupby(['year', 'month']).size().reset_index()
grouped.columns = ['year', 'month', 'count']
plt.rcParams['figure.figsize'] = [18, 6]
sns.barplot(x = 'year', y = "count", hue = "month", data = grouped)


# In[ ]:


grouped_1 = grouped.groupby('year')['count'].max().reset_index()
grouped_2 = pd.merge(grouped_1, grouped, on = ["count", "year"], how = 'inner')
grouped_2


# ## Topic Modeling LDA
# Lets run an LDA model to allocate the news headlines into topics. 
# 
# An LDA model basically assumes that documents are made from a collection of topics and that topics are made from a collection of words. So when we will be creating an LDA model in this case we will actually allocate these news headlines to various topics which would then be represented by a group of words.  
# 
# The association of a news headline to a topic would be represented the probability generated by the LDA model. 
# 
# By generating the LDA model we will be able to cluster news headlines into various groups. Also if we want to find out which kind of news dominated reuters during a certain period then we can easily do that by getting the dominating topic during that time period.  
# 
# A few pointers before creating the model
# 
#     1. first lemmatize the words
#     2. remove stopwords and other irrevelant words
#     3. remove words whose frequency is very low
#     
# The library that will be used for creating the model is gensim. 
# Model is created after creating the dictionary (which matches words with their token ids) and the corpus (which is nothing but  kinda document term matrix)

# In[ ]:


reuters.head()


# In[ ]:


get_ipython().run_cell_magic('time', '', "stoplist = set(nltk.corpus.stopwords.words('english'))")


# In[ ]:


#first of all give corpus_ids to everyone
reuters['corpus_id'] = [x for x in range(len(reuters))]


# In[ ]:


reuters.shape


# In[ ]:


get_ipython().run_cell_magic('time', '', "def iter_documents(cursor):\n    count = 0\n    for corp, news in cursor[['corpus_id', 'headline_text']].values:\n        \n        count += 1\n        if count % 1000000 == 0:\n            print (str(count) + ' done')\n        \n        document_clean = re.sub('[^a-zA-Z0-9]', ' ', news )\n        \n        tokens = [w for w in gensim.utils.tokenize(document_clean, lower = True)]\n        tokens = [wnl.lemmatize(wnl.lemmatize(w, pos = 'n'), pos = 'v') for w in tokens]\n        tokens = [w for w in tokens if w not in stoplist]\n        tokens = [w for w in tokens if len(w) > 3]\n        \n        yield corp, tokens\n        \n        \nclass MyCorpus(object):\n    def __init__(self, top_dir):  \n        self.top_dir = top_dir\n        self.dictionary = gensim.corpora.Dictionary(tokens for corp, tokens in iter_documents(top_dir))\n        self.dictionary.filter_extremes(no_below=60, no_above = 0.5)\n        self.dictionary.compactify()\n        self.dictionary.save('kaggle_reuters_news.dict')\n    def __iter__(self): \n        self.corpusID=[]\n        for corp, tokens in iter_documents(self.top_dir):\n            self.corpusID.append(corp)\n            yield self.dictionary.doc2bow(tokens)\n            \ncorpus = MyCorpus(reuters)\n\ncorpora.MmCorpus.serialize('kaggle_reuters_news.mm', corpus)")


# In[ ]:


corpus = corpora.MmCorpus('kaggle_reuters_news.mm')
dictionary= corpora.Dictionary.load('kaggle_reuters_news.dict')


# In[ ]:


len(dictionary)


# In[ ]:


get_ipython().run_cell_magic('time', '', "passes=1\ntopics = 100\nsaveas = 'kaggle_lda_model_reuters'\nlda_model =  models.ldamodel.LdaModel(corpus=corpus, num_topics=topics, id2word=dictionary, \\\n                                                   chunksize=100000, passes=passes, alpha ='symmetric', \\\n                                                   eta=None, decay = 0.5, offset=1.0, iterations=100, \\\n                                                   gamma_threshold=0.001)\n#lda_model.save(saveas)")


# In[ ]:



#lda_model = models.ldamodel.LdaModel.load('kaggle_lda_model_reuters')


# We have created the lda model with 100 topics with 1 pass (the number of times the entire corpus is run through the model) with 100 iterations (the number of iterations in 1 pass)

# In[ ]:


print ('We have made an LDA model of {} topics'.format(lda_model.num_topics))
print ('We have made an LDA model of {} terms'.format(lda_model.num_terms)) # the total size of the dictionary


# We will see what these topics are made up of. We will print the first 5 topics

# In[ ]:


for i in range(5):
    print (i, lda_model.print_topic(i))
    print ('\n')


# Lets add a column in the reuters dataframe containing the top 3 topics for all the news_headlines

# In[ ]:


def get_topics(corp):
    topics = lda_model.get_document_topics(corpus[corp])
    top_three_topics = [topic[0] for topic in topics][:3]
    return top_three_topics


# In[ ]:


get_ipython().run_cell_magic('time', '', "topics_list = []\nfor i, corp in enumerate(reuters['corpus_id']):\n    topics_list.append(get_topics(corp))\n    \n    if i%1000000 == 0:\n        print (str(i) + ' done')\n    \nreuters['topics'] = topics_list")


# In[ ]:


#reuters.to_pickle('reuters_topics.pkl')


# In[ ]:


#reuters = pd.read_pickle('reuters_topics.pkl')


# In[ ]:


reuters.head()


# We will write a function which will plot the top 5 most talked about topics in a given month of the year and we will plot what the top topic is all about

# In[ ]:


def plot_top_topics(month, year):
    df = reuters[(reuters['year'] == year)&(reuters['month'] == month)]
    df = df.groupby(['year', 'month'])['topics'].apply(list).reset_index()
    df_topics = df['topics'].values
    topics_list = []
    for topic_list in df['topics'].values:
        for x in topic_list:
            for y in x:
                topics_list.append(y)
    d = defaultdict(int)
    for x in topics_list:
        d[x] += 1 
    d = sorted(d.items(), key=operator.itemgetter(1), reverse = True)[:5]
    
    plt.rc('ytick', labelsize = 15) 
    fig, ax = plt.subplots(ncols=2,nrows=1,figsize=(16,4))
    topics = ['topic_'+ str(x[0]) for x in d]
    topics_count = [x[1] for x in d]
    sns.barplot(y = topics, x = topics_count, ax = ax[0])
    
    top_topic = d[0][0]
    
    top_topic_dist = lda_model.get_topic_terms(top_topic)
    words = [dictionary[word_prob[0]] for word_prob in top_topic_dist]
    probs = [word_prob[1] for word_prob in top_topic_dist]
    sns.barplot(y = words, x = probs, ax = ax[1])


# In[ ]:


get_ipython().run_cell_magic('time', '', 'plot_top_topics(4, 2013)')


# In this way you can get a fair bit of idea about what kind of news were dominant in a particular month of a year

# ## Search engine -LSI  
#   Lets build a search engine on top of it. We will use LSI to build it on top of a tfidf corpus. Common wisdom says LSI works better on top of tfidf corpus. Due to lack of space in kaggle's server, the below code could not be run but a search engine can be developed by the below code

# In[ ]:


#tfidf = models.TfidfModel(corpus)
#tfidf_corpus = tfidf[corpus]


# In[ ]:



#lsi_model = models.lsimodel.LsiModel(tfidf_corpus, id2word=dictionary, num_topics=100, \
                                 #chunksize= 100000, power_iters=2)


# In[ ]:


#lsi_model.save('kaggle_reuters_lsi.model')


# In[ ]:



#lsi_model = models.lsimodel.LsiModel.load('kaggle_reuters_lsi.model')


# In[ ]:




#index = similarities.Similarity(output_prefix = 'index', corpus = lsi_model[tfidf_corpus], num_best = 25, num_features=len(dictionary))
#index.save('kaggle_reuters_lsi_similarity_index.index')


# In[ ]:


#index = similarities.MatrixSimilarity.load('kaggle_reuters_lsi_similarity_index.index')


# In[ ]:


#def get_similar_news(text):
    #tokens = text.lower().split()
    #tokens = [wnl.lemmatize(wnl.lemmatize(w, pos = 'n'), pos = 'v') for w in tokens]
    #vec_text = dictionary.doc2bow(tokens)
    #vec_lsi = lsi_model[vec_text]
    #index_lsi = index[vec_lsi]
    #similar_docs = []
    #for corp, score in index_lsi:
        #similar_docs.append((reuters.iloc[corp]['headline_text']))
    #return similar_docs


# In[ ]:



#get_similar_news('software')


# In[ ]:



#get_similar_news('sports in qatar')


# And thus we have built a somewhat functional search engine, may not be as sophisticated as google but it will do the job for now

# In[ ]:




