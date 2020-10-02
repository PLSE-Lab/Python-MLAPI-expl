#!/usr/bin/env python
# coding: utf-8

# ## Data Prep Acknowledgements
# 
# All of the data prep was taken from this notebook: https://www.kaggle.com/xhlulu/cord-19-eda-parse-json-and-generate-clean-csv. Thank you, xhlulu!
# 
# ## Goal
# 
# The goal in this notebook is to take the cleaned data frames, index them, and strategically search them to find the answers to the task questions.

# In[ ]:


import os
import json
from pprint import pprint
from copy import deepcopy

import numpy as np
import pandas as pd
from tqdm.notebook import tqdm


# In[ ]:


def format_name(author):
    middle_name = " ".join(author['middle'])
    
    if author['middle']:
        return " ".join([author['first'], middle_name, author['last']])
    else:
        return " ".join([author['first'], author['last']])


def format_affiliation(affiliation):
    text = []
    location = affiliation.get('location')
    if location:
        text.extend(list(affiliation['location'].values()))
    
    institution = affiliation.get('institution')
    if institution:
        text = [institution] + text
    return ", ".join(text)

def format_authors(authors, with_affiliation=False):
    name_ls = []
    
    for author in authors:
        name = format_name(author)
        if with_affiliation:
            affiliation = format_affiliation(author['affiliation'])
            if affiliation:
                name_ls.append(f"{name} ({affiliation})")
            else:
                name_ls.append(name)
        else:
            name_ls.append(name)
    
    return ", ".join(name_ls)

def format_body(body_text):
    texts = [(di['section'], di['text']) for di in body_text]
    texts_di = {di['section']: "" for di in body_text}
    
    for section, text in texts:
        texts_di[section] += text

    body = ""

    for section, text in texts_di.items():
        body += section
        body += "\n\n"
        body += text
        body += "\n\n"
    
    return body

def format_bib(bibs):
    if type(bibs) == dict:
        bibs = list(bibs.values())
    bibs = deepcopy(bibs)
    formatted = []
    
    for bib in bibs:
        bib['authors'] = format_authors(
            bib['authors'], 
            with_affiliation=False
        )
        formatted_ls = [str(bib[k]) for k in ['title', 'authors', 'venue', 'year']]
        formatted.append(", ".join(formatted_ls))

    return "; ".join(formatted)


# In[ ]:


def load_files(dirname):
    filenames = os.listdir(dirname)
    raw_files = []

    for filename in tqdm(filenames):
        filename = dirname + filename
        file = json.load(open(filename, 'rb'))
        raw_files.append(file)
    
    return raw_files

def generate_clean_df(all_files):
    cleaned_files = []
    
    for file in tqdm(all_files):
        features = [
            file['paper_id'],
            file['metadata']['title'],
            format_authors(file['metadata']['authors']),
            format_authors(file['metadata']['authors'], 
                           with_affiliation=True),
            format_body(file['abstract']),
            format_body(file['body_text']),
            format_bib(file['bib_entries']),
            file['metadata']['authors'],
            file['bib_entries']
        ]

        cleaned_files.append(features)

    col_names = ['paper_id', 'title', 'authors',
                 'affiliations', 'abstract', 'text', 
                 'bibliography','raw_authors','raw_bibliography']

    clean_df = pd.DataFrame(cleaned_files, columns=col_names)
    clean_df.head()
    
    return clean_df


# In[ ]:


biorxiv_dir = '/kaggle/input/CORD-19-research-challenge/2020-03-13/biorxiv_medrxiv/biorxiv_medrxiv/'
filenames = os.listdir(biorxiv_dir)
print("Number of articles retrieved from biorxiv:", len(filenames))


# In[ ]:


all_files = []

for filename in filenames:
    filename = biorxiv_dir + filename
    file = json.load(open(filename, 'rb'))
    all_files.append(file)


# In[ ]:


texts = [(di['section'], di['text']) for di in file['body_text']]
texts_di = {di['section']: "" for di in file['body_text']}
for section, text in texts:
    texts_di[section] += text


# In[ ]:


body = ""

for section, text in texts_di.items():
    body += section
    body += "\n\n"
    body += text
    body += "\n\n"


# In[ ]:


authors = all_files[0]['metadata']['authors']


# In[ ]:


authors = all_files[4]['metadata']['authors']
print("Formatting without affiliation:")
print(format_authors(authors, with_affiliation=False))
print("\nFormatting with affiliation:")
print(format_authors(authors, with_affiliation=True))


# In[ ]:


cleaned_files = []

for file in tqdm(all_files):
    features = [
        file['paper_id'],
        file['metadata']['title'],
        format_authors(file['metadata']['authors']),
        format_authors(file['metadata']['authors'], 
                       with_affiliation=True),
        format_body(file['abstract']),
        format_body(file['body_text']),
        format_bib(file['bib_entries']),
        file['metadata']['authors'],
        file['bib_entries']
    ]
    
    cleaned_files.append(features)


# In[ ]:


col_names = [
    'paper_id', 
    'title', 
    'authors',
    'affiliations', 
    'abstract', 
    'text', 
    'bibliography',
    'raw_authors',
    'raw_bibliography'
]

clean_df = pd.DataFrame(cleaned_files, columns=col_names)
clean_df.head()


# In[ ]:


#clean_df.to_csv('biorxiv_clean.csv', index=False)


# ## PMC: Generate CSV

# In[ ]:


pmc_dir = '/kaggle/input/CORD-19-research-challenge/2020-03-13/pmc_custom_license/pmc_custom_license/'
pmc_files = load_files(pmc_dir)
pmc_df = generate_clean_df(pmc_files)
pmc_df.head()


# In[ ]:


#pmc_df.to_csv('clean_pmc.csv', index=False)


# ## Commercial Use: Generate CSV

# In[ ]:


comm_dir = '/kaggle/input/CORD-19-research-challenge/2020-03-13/comm_use_subset/comm_use_subset/'
comm_files = load_files(comm_dir)
comm_df = generate_clean_df(comm_files)
comm_df.head()


# In[ ]:


#comm_df.to_csv('clean_comm_use.csv', index=False)


# ## Non-commercial Use: Generate CSV

# In[ ]:


noncomm_dir = '/kaggle/input/CORD-19-research-challenge/2020-03-13/noncomm_use_subset/noncomm_use_subset/'
noncomm_files = load_files(noncomm_dir)
noncomm_df = generate_clean_df(noncomm_files)
noncomm_df.head()


# In[ ]:


get_ipython().system('pip install Whoosh')


# ## Indexing and Searching through the data
# 
# I'll be using the library [Whoosh](https://whoosh.readthedocs.io/en/latest/highlight.html) to index and process the data.

# In[ ]:


from whoosh.index import create_in
from whoosh.fields import *


# # Indexing
# 
# Before we can begin looking into the data, it's best to index it for faster and more efficient querying.
# 
# First we need to define a schema that represents what we'll be storing.  For now, I just want to have the paper_id, title, abstract, and main text of the papers.

# In[ ]:


schema = Schema(paper_id = TEXT(stored=True), title=TEXT(stored=True), abstract = TEXT(stored = True), content = TEXT(stored = True))


# Next, we create an indexiing object that is defined with our schema.

# In[ ]:


ix = create_in(".", schema)


# Now we need to make a writer to store our indexed data.

# In[ ]:


writer = ix.writer()


# Before we index the data, let's make it easier to iterate over.

# In[ ]:


papers = [clean_df, pmc_df, comm_df, noncomm_df]
#merged_papers = pd.concat(papers)


# Now we'll iterate over all of the papers and add them to our index.

# In[ ]:


for paper_set in papers:
    for index, row in paper_set.iterrows():
        writer.add_document(paper_id = row['paper_id'],
                            title    = row['title'],
                            abstract = row['abstract'],
                            content  = row['text']
                           )
writer.commit()


# Let's import the parsing and querying tools.

# In[ ]:


from whoosh.qparser import QueryParser
from whoosh.query import *


# In[ ]:


searcher = ix.searcher()
proposed_time_strings = []
with ix.searcher() as searcher:
    parser = QueryParser("content", ix.schema)
    querystring = u"human incubation AND(content:period OR content:time) AND(content:corona) AND(content:virus)"# AND(content:months OR content:weeks OR content:days OR content:hours OR content:minutes OR content:seconds)"
    myquery = parser.parse(querystring)
    print(myquery)
    results = searcher.search(myquery)
    results.fragmenter.surround = 500
    results.fragmenter.maxchars = 1500
    print(len(results))
    for res in results:
        print(res['title'])
        #if(str.find(sub,start,end))
        res_strings = res.highlights("content", top = 5)
        print(res_strings)
        print("END__________________________________________________")
        #if(res_strings.find("days") or res_strings.find("minutes") or res_strings.find("hours") or res_strings.find("seconds") or res_strings.find("weeks")):
        proposed_time_strings.append(res_strings)
        #proposed_times.append([int(s) for s in res_strings.split() if s.isdigit()])
    
    
        


# ## How long is the incubation period of the virus?
# 
# From searching through the literature we find the following points:
# 
# *The history and epidemiology of Middle East respiratory syndrome corona virus* reports that the median incubation period for MERS-CoV is 5.2-12 days.
# 
# *Simulating the infected population and spread trend of 2019-nCov under different policy by EIR model* reports that "Early predictions for incubation time are between 2 and 14 days, based on data from similar coronaviruses, with the 95th percentile of the distribution at 12.5 days [1] .
# 
# *Clinical characteristics of 82 death cases with COVID-19* states the following: [4] [5] [6] [7] Generally, the incubation period of COVID-19 was 3 to 7 days.
# NOTE: I have left the citation numbers for further searching -- this paper seems to have the citations at the beggining of sentences?
# 
# *Association of Population migration and Coronavirus Disease 2019 epidemic control* says 7 days: However, during the 24 hours around the announcement of travel ban in Wuhan, a sudden large inflow from Wuhan was observed due to the fleeing population. This would explain the burst of new cases on January 31, despite the launch of Level 1 Response. The time lag of 7 days was consistent with the average incubation period reported [23] .
# 
# 
# These values were extracted manually from the highlights of the queries. This process could be expedited with regexp queries

# In[ ]:




