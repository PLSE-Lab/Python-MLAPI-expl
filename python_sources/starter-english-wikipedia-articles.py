#!/usr/bin/env python
# coding: utf-8

# ## Introduction
# This kernel is intended to provide guidance on both structure and access of the database. Let's start by importing all the necessary libraries and defining some simple functions.

# In[ ]:


import sqlite3 as sql
import pandas as pd
import re

db = '../input/enwiki-20170820.db'


# In[ ]:


def get_query(select, db=db):
    '''Executes a select statement and returns results and column/field names.'''
    with sql.connect(db) as conn:
        c = conn.cursor()
        c.execute(select)
        col_names = [str(name[0]).lower() for name in c.description]
    return c.fetchall(), col_names

def tokenize(text, lower=True):
    '''Simple tokenizer. Will split text on word boundaries, eliminating apostrophes and retaining alpha tokens with an exception for underscores.'''
    text = re.sub("'", "", text)
    if lower:
        tokens = re.findall('''[a-z_]+''', text.lower())
    else:
        tokens = re.findall('''[A-Za-z_]''', text)
    return tokens

def get_article(article_id):
    '''Returns tokens from a given article id. Pulls, joins, and tokenizes section text from a given article id.'''
    select = '''select section_text from articles where article_id=%d''' % article_id
    docs, _ = get_query(select)
    docs = [doc[0] for doc in docs]
    doc = ' '.join(docs)
    tokens = tokenize(doc)
    return tokens


# Now let's explore the structure of the database by pulling the first few rows. Note that we could use pandas' built-in functionality to read sql queries directly, but I want to use our shiny functions!

# In[ ]:


select = '''select * from articles limit 10'''
data, cols = get_query(select)
df = pd.DataFrame(data, columns=cols)
df


# As you can see, each article is broken into sections and stored on a seperate row. Let's combine the sections from a single article and tokenize the resulting text using the get_article() function.

# In[ ]:


tokens = get_article(0)
print(tokens[:100])


# ## Conclusion
# This gives you everything you need to start exploring the data! To create you own version of this notebook, click the blue "Edit Notebook" button at the top of the kernel. This will create a copy of the code and environment for you to edit. Delete, modify, and add code as you please. Happy Kaggling!

# In[ ]:




