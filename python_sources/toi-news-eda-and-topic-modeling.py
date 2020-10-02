#!/usr/bin/env python
# coding: utf-8

# ### imports

# In[ ]:


import os
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm.auto import tqdm as tq

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)


# ## Given Data

# In[ ]:


root = "/kaggle/input/toi-2018-news-articles/data/"
art_list = os.listdir(root)
print(f"total articles {len(art_list)}")
print("sample titles: ")
print(*art_list[:3], sep='\n')


# In[ ]:


contents = [open(root+i).read() for i in tq(art_list, leave=False)]
print("sample content :")
print(contents[10])


# ## Data Preparation

# ### dataframe of title and content

# In[ ]:


df = pd.DataFrame(list(zip(art_list, contents)), columns=['title', 'content'])
df.head()


# ### get the date_part and tag from title

# In[ ]:


df['date'] = df.title.apply(lambda x: int(x.split('_')[0]))
df['tag'] = df.title.apply(lambda x: ("_".join(x.split('-')[0].split('_')[1:-1])))
df.head()


# ### get the year month day from the date part

# In[ ]:


months = [i for i in range(1, 13)]
days = [i for i in range(1, 32)]

def convert(x):
    x = str(x)
    splits = [(int(x[:k]), int(x[k:])) for k in range(1, len(x))]
    for i, j in splits:
        if i in months and j in days: 
            return i, j

df['year'] = df.title.apply(lambda x: int(x.split('_')[0][:4]))
rest = df.title.apply(lambda x: int(x.split('_')[0][4:]))
a = rest.apply(convert)
df['month'] = [i[0] for i in a]
df['day'] = [i[1] for i in a]
df.head()


# ### get the headline from title and cleanup the content

# In[ ]:


df['headline'] = df.title.apply(lambda x: (x.split('-')[0].split('_')[-1] + '-' + '-'.join(x.split('-')[1:])).replace("-", " ")[:-3])
df['content'] = df.content.apply(lambda x: x[1:-1])
df.head()


# ### try getting location from first few words of content

# In[ ]:


def get_loc(x):
    p = x.split(':')[0]
    if len(p.split(" ")) < 6:
        return p
    elif len(p.split('()')[0]) < 30:
        return p.split(',')[0]
    return ""

df['loc'] = df['content'].apply(get_loc)
df.head()


# ### final dataframe

# In[ ]:


df = df[['date','year','month' ,'day', 'tag', 'loc', 'headline', 'title', 'content']]
df.head()


# ## EDA

# ### Year

# In[ ]:


print(f"Data from only one year is present : df.year.unique() = {df.year.unique()}")


# ### Month

# In[ ]:


fig, ax = plt.subplots(1,1, figsize=(9, 5))
sns.countplot(ax=ax, x="month", data=df)
plt.show()


# ### Day

# In[ ]:


fig, ax = plt.subplots(1,1, figsize=(9, 5))
sns.countplot(ax=ax, x="day", data=df)
plt.show()


# ### Tags

# In[ ]:


fig, ax = plt.subplots(1,1, figsize=(8, 10))
sns.countplot(ax=ax, 
              y="tag", 
              data=df, 
              order=list(df.tag.value_counts().sort_values(ascending=False).index)[:20], 
              orient='h')
plt.title("Most Popular tags")
plt.show()


# In[ ]:


fig, ax = plt.subplots(1,1, figsize=(8, 10))
sns.countplot(ax=ax, 
              y="tag", 
              data=df, 
              order=list(df.tag.value_counts().sort_values(ascending=True).index)[1:20], 
              orient='h')
plt.title("least Popular tags")
plt.show()


# ### Location

# In[ ]:


fig, ax = plt.subplots(1,1, figsize=(8, 10))
sns.countplot(ax=ax, 
              y="loc", 
              data=df, 
              order=list(df["loc"].value_counts().sort_values(ascending=False).index)[:20], 
              orient='h')
plt.title("Most Popular Locations")
plt.show()


# In[ ]:


fig, ax = plt.subplots(1,1, figsize=(8, 10))
sns.countplot(ax=ax, 
              y="loc", 
              data=df, 
              order=list(df['loc'].value_counts().sort_values(ascending=True).index)[:20], 
              orient='h')
plt.title("least Popular locations")
plt.show()


# ### Headline

# ### sample headlines

# In[ ]:


print(*list(df['headline'].sample(n=10, random_state=1)), sep='\n\n')


# In[ ]:


fig, ax = plt.subplots(1, 1, figsize=(9, 6))
sns.distplot(df['headline'].apply(lambda x: len(x)).values)
plt.title("charecter length of headlines")
plt.show()


# ### Title wordcloud

# In[ ]:


from wordcloud import WordCloud, STOPWORDS

wordcloud = WordCloud(
    width = 800, 
    height = 800, 
    background_color ='white', 
    stopwords = set(STOPWORDS), 
    min_font_size = 10)

wc_img = wordcloud.generate(' '.join(df.headline))
plt.figure(figsize=(8, 8), facecolor = None) 
plt.imshow(wc_img) 
plt.axis("off")
plt.tight_layout(pad=0) 
plt.show()


# ### Content

# In[ ]:


fig, ax = plt.subplots(1, 1, figsize=(9, 6))
sns.distplot(df['content'].apply(lambda x: len(x)).values)
plt.title("charecter length of content")
plt.show()


# ### Content Wordcloud

# In[ ]:


from wordcloud import WordCloud, STOPWORDS

wordcloud = WordCloud(
    width = 800, 
    height = 800, 
    background_color ='white', 
    stopwords = set(STOPWORDS), 
    min_font_size = 10)

wc_img = wordcloud.generate(' '.join(df.sample(1000).content))
plt.figure(figsize=(8, 8), facecolor = None) 
plt.imshow(wc_img) 
plt.axis("off")
plt.tight_layout(pad=0) 
plt.show()


# In[ ]:


df.to_csv("data.csv", index=False)


# ## Topic Modeling with LDA
# ref : [link](https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/)

# In[ ]:


import re
from pprint import pprint

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# spacy for lemmatization
import spacy

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
from nltk.corpus import stopwords


# In[ ]:


stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])


# In[ ]:


# Convert to list
#df = df.sample(5000)
data = df.content.values.tolist()

pprint(data[0])


# In[ ]:


def sent_to_words(sentence):
        return gensim.utils.simple_preprocess(str(sentence), deacc=True)  # deacc=True removes punctuations

data_words = [sent_to_words(i) for i in tq(data)]

print(data_words[0])


# In[ ]:


# # Build the bigram and trigram models
# bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
# trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  

# # Faster way to get a sentence clubbed as a trigram/bigram
# bigram_mod = gensim.models.phrases.Phraser(bigram)
# trigram_mod = gensim.models.phrases.Phraser(trigram)

# # See trigram example
# print(trigram_mod[bigram_mod[data_words[0]]])

# def make_bigrams(texts):
#     return [bigram_mod[doc] for doc in texts]

# def make_trigrams(texts):
#     return [trigram_mod[bigram_mod[doc]] for doc in texts]


# In[ ]:


# Define functions for stopwords, bigrams, trigrams and lemmatization
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in tq(texts)]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in tq(texts):
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out


# In[ ]:


# Remove Stop Words
data_words = remove_stopwords(data_words)

# Form Bigrams
#data_words = make_bigrams(data_words)

# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
nlp = spacy.load('en', disable=['parser', 'ner'])

# Do lemmatization keeping only noun, adj, vb, adv
data_lemmatized = lemmatization(data_words, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

print(data_lemmatized[0])


# In[ ]:


# Create Dictionary
id2word = corpora.Dictionary(data_lemmatized)

# Create Corpus
texts = data_lemmatized

# Term Document Frequency
# Convert document into the bag-of-words (BoW) format = 
# list of (token_id, token_count) tuples.
corpus = [id2word.doc2bow(text) for text in texts]

# View
print(corpus[0])


# In[ ]:


# Build LDA model
NUM_TOPICS = 6
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=NUM_TOPICS, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)


# In[ ]:


# Compute Perplexity
# a measure of how good the model is. lower the better.
print('Perplexity: ', lda_model.log_perplexity(corpus))  

# Compute Coherence Score
coherence_model_lda = CoherenceModel(model=lda_model, 
                                     texts=data_lemmatized, 
                                     dictionary=id2word, 
                                     coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('Coherence Score: ', coherence_lda)


# **Print different topics with the weights of constituent words**

# In[ ]:


# Print the Keyword in the topics
for i, j in lda_model.print_topics():
    print("-"*80, "\n", i)
    pprint(j)


# In[ ]:


doc_lda = list(lda_model.get_document_topics(corpus))
print(*doc_lda[:5], sep='\n'+('-'*80)+'\n')


# In[ ]:


res = np.zeros((len(doc_lda), NUM_TOPICS))
for num, i in enumerate(doc_lda):
    for p, q in i:
        res[num, p-1] = q

df["topic"] = np.argmax(res, axis=1)+1
for i in range(1, 1+NUM_TOPICS):
    df["score_"+str(i)] = res[:,i-1]
df


# In[ ]:


fig, ax = plt.subplots(1,1, figsize=(9, 5))
sns.countplot(ax=ax, x="topic", data=df)
plt.show()


# In[ ]:


df.to_csv("final.csv", index=False)

