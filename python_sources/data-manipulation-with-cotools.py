#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install cord-19-tools==0.0.7')


# In[ ]:


import cotools
from cotools import text, texts, abstract, abstracts
import spacy
import os
import sys


data = '/kaggle/input/CORD-19-research-challenge/2020-03-13'
os.listdir(data)


# # Loading the data
# First, lets go ahead and load in our data with `cotools.download`. This will get us fresh data if we want it

# In[ ]:


downloaded=True
if not downloaded:
    cotools.download("data")
#os.listdir('data')


# ## Dataset loading
# 
# Next we load in one of our datasets. We will load in the common use subset as a `cotools.Paperset`, which is a lazy file loader (it just indexes the directory, and upon being indexed actually loads the data)

# In[ ]:


comm_use = cotools.Paperset(f'{data}/comm_use_subset/comm_use_subset')
print(abstract(comm_use[0]))
print('-'*60)
print(text(comm_use[0]))


# Now lets take a look more closely at our dataset. First, lets look at the keys, so we can navigate about the data more easily:

# In[ ]:


keys = comm_use.apply(lambda x: list(x.keys()))

print(set(sum(keys, [])))


# Lets see if we can focus exactly on the subset of the data that discusses COVID-19, first by naively searching for the term **novel coronavirus**, using the `text` function:

# In[ ]:


covid_papers = [x for x in comm_use if any( c in text(x).lower() for c in ['covid', 'novel coronavirus'])]
# covid_papers = [x for x in comm_use if 'covid' in text(x).lower()]
len(covid_papers)


# We can use the text and abstract functions similarly to gra

# In[ ]:


covid_text = [text(x) for x in covid_papers]
covid_abstracts = [abstract(x) for x in covid_papers]


# In[ ]:


covid_abstracts


# More to come tomorrow!
