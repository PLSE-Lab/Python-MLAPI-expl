#!/usr/bin/env python
# coding: utf-8

# # Tutorial: Dictionary
# This is a basic guide to efficiently training a Dictionary on the English Wikipedia dump using Gensim.

# In[ ]:


from gensim.matutils import sparse2full 
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from multiprocessing import Pool
from tqdm import tqdm
import sqlite3 as sql
import pandas as pd
import numpy as np
import logging
import time
import re

db = '../input/english-wikipedia-articles-20170820-sqlite/enwiki-20170820.db'
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


# Let's start by creating a helper function to pull data from the database and examine some output.

# In[ ]:


def get_query(select, db=db):
    '''
    1. Connects to SQLite database (db)
    2. Executes select statement
    3. Return results and column names
    
    Input: 'select * from analytics limit 2'
    Output: ([(1, 2, 3)], ['col_1', 'col_2', 'col_3'])
    '''
    with sql.connect(db) as conn:
        c = conn.cursor()
        c.execute(select)
        col_names = [str(name[0]).lower() for name in c.description]
    return c.fetchall(), col_names


# In[ ]:


select = '''select * from articles limit 5'''
data, cols = get_query(select)
df = pd.DataFrame(data, columns=cols)
df


# As we can see, we have four fields: article_id, title, section_title, and section_text. Each wikipedia article is broken into sections, and to reconstitute them, we'll need to grab all section_text associated with a single article_id and combine them. Let's create a function to do just that.

# In[ ]:


def get_article_text(article_id):
    '''
    1. Construct select statement
    2. Retrieve all section_texts associated with article_id
    3. Join section_texts into a single string (article_text)
    4. Return article_text
    
    Input: 100
    Output: ['the','austroasiatic','languages','in',...]
    '''
    select = '''select section_text from articles where article_id=%d''' % article_id
    docs, _ = get_query(select)
    docs = [doc[0] for doc in docs]
    doc = '\n'.join(docs)
    return doc


# In[ ]:


article_text = get_article_text(0)
print(article_text)


# Now that we can get retrieve data from the database, let's start building a dictionary. Dictionaries in Gensim are built on top of the high-performance [containers](https://docs.python.org/3.6/library/collections.html) module found in base python. Essentially, we want to assign an integer ID to each unique word, and keep track of how many times that word comes up across a collection of documents (i.e., corpus). Let's create a function to split our article text into tokens.

# In[ ]:


def tokenize(text, lower=True):
    '''
    1. Strips apostrophes
    2. Searches for all alpha tokens (exception for underscore)
    3. Return list of tokens

    Input: 'The 3 dogs jumped over Scott's tent!'
    Output: ['the', 'dogs', 'jumped', 'over', 'scotts', 'tent']
    '''
    text = re.sub("'", "", text)
    if lower:
        tokens = re.findall('''[a-z_]+''', text.lower())
    else:
        tokens = re.findall('''[A-Za-z_]''', text)
    return tokens


# In[ ]:


tokens = tokenize(article_text)
print(tokens[:5])


# In[ ]:


len(tokens)


# Now that we have a way to get tokens, let's create a dictionary. Once imported, all you have to do to create a dictionary is to instantiate an object and feed it an interator/iterable. Gensim assuming a nested list of lists structure, where each item in the sublist is a string token.

# In[ ]:


dictionary = Dictionary([tokens])


# That's all well and good, but we want more than just one document in our corpus! Let's create a wrapper function to combine the retrieval and tokenization step, then try a few ways of creating a dictionary.

# In[ ]:


def get_article_tokens(article_id):
    '''
    1. Construct select statement
    2. Retrieve all section_texts associated with article_id
    3. Join section_texts into a single string (article_text)
    4. Tokenize article_text
    5. Return list of tokens
    
    Input: 100
    Output: ['the','austroasiatic','languages','in',...]
    '''
    select = '''select section_text from articles where article_id=%d''' % article_id
    docs, _ = get_query(select)
    docs = [doc[0] for doc in docs]
    doc = '\n'.join(docs)
    tokens = tokenize(doc)
    return tokens


# In[ ]:


# First, we need to grab all article_ids from the database
select = '''select distinct article_id from articles'''
article_ids, _ = get_query(select)
article_ids = [article_id[0] for article_id in article_ids]


# In[ ]:


len(article_ids)


# Let's start by loading all documents (from a random sample) into memory, then building a dictionary.

# In[ ]:


start = time.time()
# Grab a random sample of 10K articles and read into memory
sample_ids = np.random.choice(article_ids, size=10000, replace=False)
docs = []
for sample_id in tqdm(sample_ids):
    docs.append(get_article_tokens(sample_id))
# Train dictionary
dictionary = Dictionary(docs)
end = time.time()
print('Time to train dictionary from in-memory sample: %0.2fs' % (end - start))


# Now let's try it *from a generator*.

# In[ ]:


start = time.time()
# Grab a random sample of 10K articles and set up a generator
sample_ids = np.random.choice(article_ids, size=10000, replace=False)
docs = (get_article_tokens(sample_id) for sample_id in sample_ids)
# Train dictionary
dictionary = Dictionary(docs)
end = time.time()
print('Time to train dictionary from generator: %0.2fs' % (end - start))


# Now let's create a *processing pool* to retrieve and preprocess documents.

# In[ ]:


start = time.time()
# Grab a random sample of 10K articles and set up a pooled-process generator
sample_ids = np.random.choice(article_ids, size=10000, replace=False)
with Pool(processes=4, maxtasksperchild=2048) as pool:
    docs = pool.imap_unordered(get_article_tokens, sample_ids)
    dictionary = Dictionary(docs)
end = time.time()
print('Time to train dictionary from pooled-process generator: %0.2fs' % (end - start))


# For various reasons, it's actually better to define an *iterable* when working with Gensim. Here is a simple template.

# In[ ]:


class Corpus():
    def __init__(self, article_ids):
        self.article_ids = article_ids
        self.len = len(article_ids)

    def __iter__(self):
        article_ids = np.random.choice(self.article_ids, self.len, replace=False)
        with Pool(processes=4, maxtasksperchild=2048) as pool:
            docs = pool.imap_unordered(get_article_tokens, article_ids)
            for doc in docs:
                yield doc

    def __len__(self):
        return self.len


# So if we want to train a dictionary on the entire dataset, we can simple create an iterable of all article_ids and feed it into a Dictionary.
# 
# `dictionary = Dictionary(Corpus(article_ids))`
# 
# Feel free to try it, but I've already pre-trained a Dictionary on this dataset, so we'll just load it into memory and explore how to work with it.

# In[ ]:


dictionary = Dictionary.load('../input/english-wikipedia-articles-20170820-models/enwiki_2017_08_20.dict')


# In[ ]:


len(dictionary.dfs)


# In[ ]:


pd.value_counts(list(dictionary.dfs.values()), normalize=True)


# In[ ]:


dictionary.filter_extremes(no_below=200, no_above=0.5, keep_n=10000000000)


# In[ ]:


dictionary[0]


# In[ ]:


tokens = get_article_tokens(0)
dictionary.doc2bow(tokens)


# In[ ]:


tfidf = TfidfModel(dictionary=dictionary)


# In[ ]:


x = tfidf[dictionary.doc2bow(tokens)]


# In[ ]:


from gensim.matutils import sparse2full

sparse2full(x, len(dictionary.dfs))


# In[ ]:


sparse2full(x, len(dictionary.dfs)).shape


# In[ ]:




