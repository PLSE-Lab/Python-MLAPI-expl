#!/usr/bin/env python
# coding: utf-8

# The goal of this notebook is do two things:
# 1. Some general, basic exploration of the dataset to become familiar with columns and potential issues. I include functions for loading the data as well
# 2. Use cosine similarity with the scispacy package to find articles related to the task questions

# In[ ]:


get_ipython().system('pip install scispacy')
get_ipython().system('pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.4/en_ner_bc5cdr_md-0.2.4.tar.gz')


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import json
import gc
from glob import glob

import scispacy
import spacy
import en_ner_bc5cdr_md

from sklearn.metrics.pairwise import cosine_similarity

pd.set_option('max_columns', 100)
from IPython.core.display import display, HTML
display(HTML('<style>.container { width:100% !important; }</style>'))


# # Load the datasets and combine them.
# ## Heavily based off the work by xhulu

# In[ ]:


def format_ref(r):
    return '\n'.join(['{0}-{1}:{2}'.format(x['start'],x['end'],x['ref_id']) for x in r])

def format_author(author):    
    return " ".join([author['first'], " ".join(author['middle']), author['last']])

def json_reader(file):
    #takes a json file, processes the body, ref, and bib data into a dataframe
    with open(file) as f:
        j = json.load(f)
        
    #format the body text so the sections are clear, but it's easy to view the whole thing
    body_text = '\n\n'.join(['<section {}> '.format(n) + x['section'] + '\n\n' + x['text'] for n,x in enumerate(j['body_text'])])
    ref_spans = '\n\n'.join(['<section {}> '.format(n) + x['section'] + '\n\n' + format_ref(x['ref_spans']) for n,x in enumerate(j['body_text'])])
    cite_spans = '\n\n'.join(['<section {}> '.format(n) + x['section'] + '\n\n' + format_ref(x['cite_spans']) for n,x in enumerate(j['body_text'])])
    
    #format references in a similar way
    ref_data = '\n\n'.join([k + '\n\n' + v['text'] + '\n\nlatex- {}'.format(v['latex']) for k,v in j['ref_entries'].items()])

    #put the bibliography together, and format the authors
    for k in j['bib_entries']:
        j['bib_entries'][k]['author_list'] = ', '.join([format_author(a) for a in (j['bib_entries'][k]['authors'])])

    bib_keys = ['ref_id', 'title', 'author_list', 'year', 'venue', 'volume', 'issn', 'pages', 'other_ids']
    bib_data = '\n\n'.join([', '.join([str(x[k]) for k in bib_keys]) for _,x in j['bib_entries'].items()])

    df = pd.DataFrame(index=[0], data={'body_text':body_text, 
                                            'cite_spans':cite_spans, 
                                            'ref_spans':ref_spans,
                                            'ref_data': ref_data,
                                            'bib_data': bib_data,
                                            'paper_id': j['paper_id']})
    
    return df


def parse_folder(data_folder):
    filelist = glob('/kaggle/input/CORD-19-research-challenge/{0}/{0}/*'.format(data_folder))
    filelist.sort()
    print('{} has {} files'.format(data_folder, len(filelist)))

    df_ls=[]
    for n,file in enumerate(filelist):
        if n%1000==0:
            print(n,file[-46:])
        df = json_reader(file)
        df_ls.append(df)
    return pd.concat(df_ls)


# In[ ]:


#go through each of the four folders of json files and put everything into one dataframe
#takes around 3-4min to complete
df_ls = []
for folder in ['comm_use_subset', 'noncomm_use_subset', 'custom_license', 'biorxiv_medrxiv']:
    t = parse_folder(folder)
    df_ls.append(t)
df = pd.concat(df_ls)


# In[ ]:


meta = pd.read_csv('/kaggle/input/CORD-19-research-challenge/metadata.csv')
meta.rename(columns={'sha':'paper_id'}, inplace=True)
df = meta.merge(df, on='paper_id', how='left')


# In[ ]:


df.shape


# In[ ]:


df.head(3)


# # Data Cleaning
# ### Duplicate papers

# In[ ]:


df.shape


# In[ ]:


df.drop_duplicates(['abstract', 'body_text', 'ref_data'], inplace=True)
df.shape


# #  Basic Stats

# ### Paper text is missing ~28%
# ### Abstract, Title, and Author mostly complete

# In[ ]:


df.isna().sum(axis=0).sort_values(ascending=False) / df.shape[0]


# ### Pubmed makes up more of the abstracts
# ### But Elsevier makes up more of the text

# In[ ]:


df.source_x.value_counts(normalize=True)


# In[ ]:


df[df['paper_id'].notna()].source_x.value_counts(normalize=True)


# ### License Distribution

# In[ ]:


df.license.value_counts(normalize=True)


# ### Journal Distribution

# In[ ]:


df.journal.value_counts(normalize=True).head(10)


# # Publish Time

# ### Papers date back to 1950s

# In[ ]:


df.publish_time.value_counts(normalize=True).head(10)


# In[ ]:


pub_year = df['publish_time'].dropna().apply(lambda x: x[:4]).value_counts().sort_index()
plt.bar(pub_year.index, np.log10(pub_year.values))
plt.xticks([i for i in range(pub_year.shape[0]) if i%8==0]);
plt.title('Log Plot of All Publications Per Year');
plt.ylabel('Log10');


# ### CZI and Biorxiv papers are only available for recent publications

# In[ ]:


pub_year = df[df['source_x'] == 'biorxiv']['publish_time'].dropna().apply(lambda x: x[:4]).value_counts().sort_index()
plt.bar(pub_year.index, np.log10(pub_year.values))

plt.xticks([i for i in range(pub_year.shape[0]) if i%1==0]);
plt.title('BIORXIV Log Plot of Publications Per Year');
plt.ylabel('Log10');


# In[ ]:


pub_year = df[df['source_x'] == 'CZI']['publish_time'].dropna().apply(lambda x: x[:4]).value_counts().sort_index()
plt.bar(pub_year.index, np.log10(pub_year.values))

plt.xticks([i for i in range(pub_year.shape[0]) if i%1==0]);
plt.title('BIORXIV Log Plot of Publications Per Year');
plt.ylabel('Log10');


# ###  Medrxiv papers don't have publish times

# In[ ]:


#All MEDRXIV PAPERS ARE MISSING MOST PUBLISH TIMES
df[df['source_x'] == 'medrxiv']['publish_time'].notna().sum() / df[df['source_x'] =='medrxiv'].shape[0]


# # Covid References

# ### Only ~20% of papers are about Covid-19!

# In[ ]:


((df['abstract'].dropna().str.contains('corona', case=False)) | (df['abstract'].dropna().str.contains('COVID', case=False))).sum() / df.shape[0]


# ### References to Promising Drugs

# In[ ]:


df[df['abstract'].fillna('').str.contains('chloroquine', case=False)].shape[0]


# In[ ]:


df[df['abstract'].fillna('').str.contains('favipiravir', case=False)].shape[0]


# In[ ]:


df[df['abstract'].fillna('').str.contains('lopinavir', case=False)].shape[0]


# In[ ]:


df[df['abstract'].fillna('').str.contains('ritonavir', case=False)].shape[0]


# In[ ]:


df[df['abstract'].fillna('').str.contains('convalescent plasma', case=False)].shape[0]


# In[ ]:


df[df['abstract'].fillna('').str.contains('passive antibody', case=False)].shape[0]


# # Word counts

# ### Some abstracts have more than 10k words!

# In[ ]:


df['abstract'].dropna().apply(lambda x: len(x.split(' '))).describe()


# ### Some body text is less than 1000 words

# In[ ]:


df['body_text'].dropna().apply(lambda x: len(x.split(' '))).describe()


# ### Long Abstract Example

# In[ ]:


long_idx = df[df['abstract'].apply(lambda x: len(x.split(' ')) if isinstance(x,str) else 0) > 10000]['paper_id'].values[0]


# In[ ]:


df[df['paper_id'] == long_idx][['paper_id', 'source_x','title', 'license','abstract','publish_time', 'authors','journal','has_full_text']]


# In[ ]:


' '.join(df[df['paper_id'] == long_idx]['abstract'].values[0].split(' ')[:100])


# In[ ]:


' '.join(df[df['paper_id'] == long_idx]['abstract'].values[0].split(' ')[-100:])


# # Basic Text Analysis

# In[ ]:


df.dropna(subset=['abstract'], inplace=True)
df.drop(columns='body_text', inplace=True)


# In[ ]:


df.shape


# In[ ]:


gc.collect()


# In[ ]:


#load the scispacy model relevant to diseases
nlp = spacy.load('en_ner_bc5cdr_md')


# In[ ]:


nlp.vocab.length


# In[ ]:


#each word comes with an embedding
nlp(df.iloc[0]['abstract'])[0].vector[:10]


# In[ ]:


def get_doc_vec(tokens):
    #combine word embeddings from a document into a single document vector
    #filter out any stop words like 'the', and remove any punction/numbers
    w_all = np.zeros(tokens[0].vector.shape)
    n=0
    for w in tokens:
        if (not w.is_stop) and (len(w)>1) and (not w.is_punct) and (not w.is_digit):
            w_all += w.vector
            n+=1
    return (w_all / n) if n>0 else np.zeros(tokens[0].vector.shape)


# In[ ]:


#takes a long time, load from file
vector_dict={}
for n,row in df.iterrows():
    if n%500==0:
        print(n)
    if len(row['abstract']) > 0:
        vector_dict[row['paper_id']] = get_doc_vec(nlp(row['abstract']))


# In[ ]:


vec_vals = list(vector_dict.values())
vec_vals = [v for v in vec_vals if all(v==0)==False]


# In[ ]:


pd.to_pickle(vec_vals, 'vec_vals.pkl')


# In[ ]:


len(vec_vals)


# In[ ]:


q_vec = [get_doc_vec(nlp('What do we know about COVID-19 risk factors?'))]


# In[ ]:


target_sims = cosine_similarity(vec_vals, q_vec)


# In[ ]:


target_sims.shape


# In[ ]:


q_series = pd.Series(dict(zip(vector_dict.keys(), target_sims)))


# In[ ]:


closet_papers = q_series.sort_values(ascending=False).head(10).index.tolist()


# In[ ]:


pd.set_option('max_colwidth',200)


# In[ ]:


df[df['paper_id'].isin(closet_papers)][['title','abstract']]


# Not bad!
# 
# Some article titles like '**Super-spreaders and the rate of transmission of the SARS virus**', '**Bioethical Implications of Globalization: An International Consortium Project of the European Commission**', and '**Infectious Disease Prevalence and Factors Associated with Upper Respiratory Infection in Cats Following Relocation**' all sound promising. 
# 
# Though I guess it's debatable how relevant the cat article really is.

# In[ ]:




