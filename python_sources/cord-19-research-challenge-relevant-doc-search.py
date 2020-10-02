#!/usr/bin/env python
# coding: utf-8

# **Task: What has been published about medical care?**
# 
# This is my first NLP kernel. I try to help fighting the corona virus even with my little skill. Hope I can help out by working on some basic tasks:
# 
# Search relevant papers based on meta data title / abstract:

# In[ ]:


import numpy as np
import pandas as pd
import spacy

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        if filename.endswith(".json"):
            break
        print(os.path.join(dirname, filename))


# Task Details
# 
# What has been published about medical care? What has been published concerning surge capacity and nursing homes? What has been published concerning efforts to inform allocation of scarce resources? What do we know about personal protective equipment? What has been published concerning alternative methods to advise on disease management? What has been published concerning processes of care? What do we know about the clinical characterization and management of the virus?
# 
# Specifically, we want to know what the literature reports about:
# 
# *     **Resources to support skilled nursing facilities and long term care facilities.** -->First Task atm
#     
# *     **Mobilization of surge medical staff to address shortages in overwhelmed communities** --> Second Task
# 
# 
# Currently working on:
# 
# 1. Looking for papers with relevant tokens in meta data title: ["support", "care facilities", "nurse facilities",]
# 
#     --> I search the "title" and "abstract" column for specific terms to collect relevant publications. After that I could have a deeper look at the found papers and make some suggestions on how to support nursing / care facilities
# 
# 

# In[ ]:


#You can change these terms for your own paper search:
terms = ["support", "care facilities", "nurse facilities",]


# In[ ]:


meta = pd.read_csv("/kaggle/input/CORD-19-research-challenge/2020-03-13/all_sources_metadata_2020-03-13.csv")
meta.head()


# In[ ]:


#Looking for NaN's in meta["title"]:
print(f'There are {meta["title"].isna().sum()} NaN values in meta[title]')

#Looking for NaN's in meta["abstract"]:
print(f'There are {meta["abstract"].isna().sum()} NaN values in meta[abstract]')
print( "-" * 50)

#Check for rows where "title" and "abstract" is NaN:
na = meta["title"].isna() & meta["abstract"].isna() 

#select only the rows where at least the title is there:
   
meta_clean = meta[~na]

print("\n\nData after cleaning:")
print( "-" * 50)
print(f'There are {meta_clean["title"].isna().sum()} NaN values in meta_clean[title]')
print(f'There are {meta_clean["abstract"].isna().sum()} NaN values in meta_clean[abstract]')


# In[ ]:


#Import NLP librarie spacy
import spacy
from spacy.matcher import PhraseMatcher
from collections import defaultdict

#Generate model
nlp = spacy.load('en')
matcher = PhraseMatcher(nlp.vocab, attr='LOWER')

term_search = defaultdict(list)


# In[ ]:


#iterate over doc titels in meta data:

patterns = [nlp(text) for text in terms]
matcher.add("SearchTerms", None, *patterns)

for idx, meta in meta_clean.iterrows():
    doc = nlp(meta.title)
    matches = matcher(doc)

    found_items = set([doc[match[1]:match[2]] for match in matches])

    for item in found_items:
        term_search[str(item).lower()].append(meta.title)


# In[ ]:


for term in terms:
    print(f'{len(term_search[term])} papers found for term -{term}-')


# There are no doc's found for the term "nurse facilities" but 12 for "care facilities":

# In[ ]:


term_search["care facilities"]


# I will continuously update the kernel. Please comment if you have any suggestions on how to make the search more efficient.
