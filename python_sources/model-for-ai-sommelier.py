#!/usr/bin/env python
# coding: utf-8

# This notebook hopes to identify a wine type by it's description. By using Latent Dirichlet Allocation (LDA) on <br/>
# 130,000 wine descriptions that experts used to describe wine varieties are used to make <br/>
# a distribution of n wine types (here I use 30 types).
# 
# Copied much of this code from this great kernel [about Tweets from Elon Musk](https://www.kaggle.com/errearanhas/topic-modelling-lda-on-elon-tweets).
# 
# Goals:
# * Produce a LDA model that groups wine descriptions into 30 Types - *done Frame 3*
# * Display Types as bubbles showing 'distance' between Types -* done Frame 6*
# * Use reverse inference on the LDA model to assign each description to most likely Type. - *done (see DataFrame dv)*
# * Determine sentiment of each wine description and use it to rate each wine variety. -* done see dv*
# * List the wine varieties along with wine Type. - *done see DataFrame dv*
# * Build a TextBox to intake user Input: "Describe the wine you want." Determine most likely Types. *done*
# * Build a recommendation engine to show the best varieties in the most likely Type. 

# In[ ]:



import os
import pandas as pd
import numpy as np
from operator import itemgetter
from gensim import corpora, models, similarities
import gensim
import logging
from collections import OrderedDict
import tempfile
from nltk.corpus import stopwords
from string import punctuation
import pyLDAvis.gensim

#TEMP_FOLDER = tempfile.gettempdir()
#print('Folder "{}" will be used to save temporary dictionary and corpus.'.format(TEMP_FOLDER))

df = pd.DataFrame()
df = pd.read_csv("../input/winemag-data-130k-v2.csv")
df.head(3)


# In[ ]:


# put all the wine descriptions into an numpy array called corpus
corpus=[]
a=[]
for i in range(len(df['description'])):
        a=df['description'][i]
        corpus.append(a)
corpus[0:2]


# In[ ]:


# remove common words and tokenize

stoplist = stopwords.words('english') + list(punctuation)

texts = [[word for word in str(document).lower().split() if word not in stoplist] for document in corpus]
dictionary = corpora.Dictionary(texts)

corpus = [dictionary.doc2bow(text) for text in texts]
#corpora.MmCorpus.serialize(os.path.join(TEMP_FOLDER, 'wine.mm'), corpus)  # store to disk, for later use

tfidf = models.TfidfModel(corpus) # step 1 -- initialize a model
corpus_tfidf = tfidf[corpus]  # step 2 -- use the model to transform vectors
total_topics = 30
# now using the vectorized corpus learn a LDA model
lda = models.LdaModel(corpus, id2word=dictionary, num_topics=total_topics)
corpus_lda = lda[corpus_tfidf] # create a double wrapper over the original corpus: bow->tfidf->fold-in-lsi

#Show first 5 important words in the first 3 topics:
lda.show_topics(3,5)


# In[ ]:


data_lda = {i: OrderedDict(lda.show_topic(i,30)) for i in range(total_topics)}
#data_lda
df_lda = pd.DataFrame(data_lda)
print(df_lda.shape)
df_lda = df_lda.fillna(0).T
print(df_lda.shape)
df_lda.head(5)


# Here is a wonderful interactive presentation of the 30 discovered "Topics".
# Notice the bar chart of the probability of words in given Topics.

# In[ ]:


pyLDAvis.enable_notebook()
panel = pyLDAvis.gensim.prepare(lda, corpus_lda, dictionary, mds='tsne')
panel


# In[ ]:


row = 10007
print(df.loc[row,'variety'])
dt = lda[corpus[row]]
dt


# Above the 10,007th sommelier's wine review is on a Chardonnay.
# It fits best in Topic 5 (36%) and then Topic 28 (24%)

# In[ ]:



dv = pd.DataFrame()
dv['Variety'] = df['variety']
for row in range(len(df)):
    dv.loc[row,'Likely_Topic'] = max(lda[corpus[row]], key = itemgetter(1))[0]
dv.head(5)


# In[ ]:


dv['Price'] = df['price']
dv['Title'] = df['title']
dv['Points'] = df['points']
dv['Vineyard'] = df['winery']
dv.head(15)


# Now add the Sentiment of each wine review.
# Wine Score = Sentiment * Points

# In[ ]:


from textblob import TextBlob

dv['tb_Sentiment'] = 0
for row in range(len(df)):
    blob = TextBlob(df.loc[row,'description'])
    for sentence in blob.sentences:
        dv.loc[row,'tb_Sentiment'] += (sentence.sentiment.polarity)
dv.head(5)


# ## Make a Wine Suggestion text entry box

# In[ ]:


description = ['brisk but sweet german wine. not too sweet']
texts = [word for word in str(description).lower().split() if word not in stoplist]
desc_v = dictionary.doc2bow(texts)
suggestion_types = lda[desc_v]
print(description)
suggestion_types


#  The short description was picked as Topic 7 (which is dominated with Reisling), which is a good guess.<br/>
#  There is a 75.8 % likelihood that this Topic is correct! I like it.
#  
#  Try it yourself by changing the text for description in line 1 above and re-run.</br>
#  I am working on giving each topic a lable which maybe the most frequent variety per Topic.</br>
#  
#  Work in progress.....

# In[ ]:


dv['Score'] = dv['tb_Sentiment']*10 + dv['Points']
t = pd.DataFrame()
t = dv.groupby(['Likely_Topic', 'Variety'])['Score'].agg([('Ave', 'mean'), ('Count', 'count')])
t = t.reset_index()
tt = t.loc[t['Count'].idxmax()] 
tt                                                                                                                                    


# In[ ]:





# In[ ]:




