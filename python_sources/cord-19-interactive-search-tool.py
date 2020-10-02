#!/usr/bin/env python
# coding: utf-8

# # Interactive search tool for notebooks
# This notebook builds a simple, general purpose, search tool based on word embeddings. 
# The embeddings are trained, and document / paragraph vectors calculated in a separate notebook. This notebook focuses on the interactive search tool using widgets.

# ## Load word vector model and pre-calculated paragraph vectors 

# In[ ]:


# Install Facebooks similarity search library
get_ipython().system('pip install faiss-cpu')


# In[ ]:


import numpy as np 
import pandas as pd 
import os
import re
from unidecode import unidecode
import pickle
import gensim 
from IPython.display import display, HTML
from faiss import read_index
from ipywidgets import widgets, interact, Layout, Dropdown, Label, HBox, VBox, interactive_output
from ipywidgets import HTML as widgetHTML
import seaborn as sns
cm = sns.light_palette("green", as_cmap=True)


# In[ ]:


DATA_PATH = '/kaggle/input/CORD-19-research-challenge'
PRE_PROCESSED_PATH = '/kaggle/input/cord-19-interactive-word2vec-paragraph-search'
EMBEDDING_DIMS = 300

all_data = pd.read_pickle(os.path.join(PRE_PROCESSED_PATH, 'CORD_19_all_papers.pkl'))
vocab = pickle.load(open(os.path.join(PRE_PROCESSED_PATH, 'covid_vocab_frequencies.pkl'), 'rb'))
model = gensim.models.Word2Vec.load(os.path.join(PRE_PROCESSED_PATH, 'covid_w2v'))
# paragraph_vectors = pickle.load(open(os.path.join(PRE_PROCESSED_PATH, 'all_para_vectors.pkl'), 'rb'))
index_cosine = read_index(os.path.join(PRE_PROCESSED_PATH, 'index_cosine.faiss'))


# ### Various helper functions

# In[ ]:


# Regex used for cleaning and tokenisation
space = re.compile('\s+')
reference = re.compile(r'[\(\[]\d+(, ?\d+)?[\)\]]')
links = re.compile(r'https?://\S+')
sentence  = re.compile(r'(\S{3,})[.!?]\s')
hyphen_1 = re.compile(r'([A-Z0-9])\-(.)')
hyphen_2 = re.compile(r'(.)\-([A-Z0-9])')


PAT_ALPHABETIC = re.compile(r'(((?![\d])\w)+)', re.UNICODE) # from gensim - removes digits - keeps only other alpha numeric and tokenises on everything else
PAT_ALL = re.compile(r'((\d+(,\d+)*(\.\d+)?)+|([\w_])+)', re.UNICODE) # Includes digits - tokenises on space and non alphanumeric characters (except _)

def clean_text(text):
    text = text.replace('\t', ' ').replace('\n', ' ')
    text = sentence.sub(r'\1 _SENT_ ', text)
    text = text.replace('doi:', ' http://a.website.com/')
    text = unidecode(text) # converts any non-unicode characters to a unicode equivalent
    text = hyphen_1.sub(r'\1\2', text)
    text = hyphen_2.sub(r'\1\2', text)
    text = links.sub(' ', text)
    text = reference.sub(' ', text)
    text = space.sub(' ', text)

    return text.strip()

def fetch_tokens(text, reg_pattern):
    for match in reg_pattern.finditer(text):
        yield match.group()

def tokenise(text, remove_stop=False, lowercase=False, include_digits=True):
    text = clean_text(text)
    
    if lowercase:
        text = text.lower()
    
    if include_digits:
        tokens = list(fetch_tokens(text, reg_pattern=PAT_ALL))
    else:
        tokens = list(fetch_tokens(text, reg_pattern=PAT_ALPHABETIC))
            
    if remove_stop:
        return ' '.join([x for x in tokens if x.lower() not in stopWords])
    else:
        return ' '.join(tokens)

def fetch_vector(words):
    tokens = tokenise(words).split()

    if len(tokens) > 0:
        myvectors = np.zeros((1, EMBEDDING_DIMS), dtype=np.float32)
        myvectors[0, :] = calc_vector(tokens)

        # normalize
        myvectors = myvectors / np.linalg.norm(myvectors, axis=1, keepdims=True)
        return myvectors

def calc_vector(tokens):
    vec = np.zeros((EMBEDDING_DIMS,), dtype=np.float32)
    for word in tokens:
        found=False

        if word not in model.wv.vocab:
            if word.title() in model.wv.vocab:
                word = word.title()
                found = True
            elif word.lower() in model.wv.vocab:
                word = word.lower()
                found = True
            elif word.upper() in model.wv.vocab:
                word = word.upper()
                found = True
        else:
            found = True

        if found:
            vec += model.wv.get_vector(word).astype(np.float32) / np.log(5 + vocab[word])

#         normalise
    vec = vec / np.linalg.norm(vec, axis=0, keepdims=True)
    return vec

def fetch_similar(positive=[], negative=[], k=10):
    label = ';'.join(positive)
    if negative:
        label += ' - ' + ';'.join(negative)
    return pd.DataFrame([(x, y, vocab[x]) for x,y in model.wv.most_similar(positive, negative, topn=k)], columns=[label, 'relevance', 'frequency'])

def view_similar(positive='Beijing,Paris', negative='China', top_k=10):
    if negative=='':
        negative = []
    else:
        negative = negative.split(',')
    return fetch_similar(positive=positive.split(','),
                        negative=negative,
                        k=top_k)


# ## Explore word similarities

# In[ ]:


# common cold is to sore throat, blocked or runny nose as COVID19 is to ?
interact(view_similar, positive='COVID19,SARSCoV2,NCP,runny,nose,sore,throat,sneezing,rhinorrhea', negative='common,cold');


# ## Search the CORD-19 document set

# In[ ]:


# This is the main search function
def search_db(words, k=10, covid_only=True, date_range=[1957,2020], paper_select='All'):
    vec = fetch_vector(words)
    cm = sns.light_palette("green", as_cmap=True)

    if paper_select=='All':
        D, I = index_cosine.search(vec, k+500)
        tmp = all_data.loc[I[0], ['title', 'para_context', 'para_text', 'publish_year', 'doi', 'id']].copy()
        tmp['Relevance'] = np.round(D[0], 2)

        if covid_only:
            # We will inlcude documents that have a blank title
            tmp = tmp.loc[tmp.title.fillna('covid').str.lower().str.contains('covid|sars\-cov|novel coronavirus')]

        if date_range[0]>1957 or date_range[1]<2020:
            # Also include documents missing the published date
            tmp = tmp.loc[(tmp.publish_year=='0') | ((tmp.publish_year>=str(date_range[0])) & (tmp.publish_year<=str(date_range[1])))]
        if len(tmp) > 0:
            tmp = tmp.loc[:, ['Relevance', 'para_context', 'title', 'publish_year', 'doi', 'id']].iloc[:k].copy()
            paper.options = [('All', 'All')] + [(y, x) for x,y in tmp[['title', 'id']].groupby('id').first().itertuples()]
            tmp['title'] = tmp.apply(lambda x: f'<a href="http://doi.org/{x.doi}" target="_blank">{x.title}</a>', axis=1)
            tmp['para_context'] = tmp['para_context'].apply(lambda x: x.replace('_STARTREF_', '<p><i>').replace('_ENDREF_', '</i></p>'))
            
            tmp = tmp.reset_index(drop=True)[['Relevance', 'para_context', 'title', 'publish_year']]                            .rename(columns={'para_context': 'Body text', 'title': 'Paper title', 'publish_year': 'Published'})
            display(tmp.style.background_gradient(cmap=cm))
        else:
            print('No results found')
    else:
        D, I = index_cosine.search(vec, index_cosine.ntotal)
#         print(paper_select)
        tmp = all_data.loc[all_data.id==paper_select, ['para_num', 'title', 'section', 'para_text', 'publish_year', 'doi', 'id']].copy()
        display(HBox([Label('Link to paper:', layout=Layout(width='20%')),
                      widgetHTML(f'<a href="http://doi.org/{tmp.doi.iloc[0]}" target="_blank">{tmp.title.iloc[0]}</a>')]))
        tmp['Relevance'] = 0
        # fetch relevance scores for paragraphs in this document
        scores = [(x,y) for x,y in zip(I[0], np.round(D[0], 2)) if x in tmp.index]
        tmp.loc[[x[0] for x in scores], 'Relevance'] = [x[1] for x in scores]
        tmp['para_text'] = tmp['para_text'].apply(lambda x: '<p>' + x.replace('_STARTREF_', '<p><i>').replace('_ENDREF_', '</i></p>') + '</p>')
        tmp = tmp[['Relevance', 'section', 'para_text', 'para_num']].rename(columns={'para_text': 'Body text'}).set_index('para_num')
        display(tmp.style.background_gradient(cmap=cm))


# In[ ]:


# These are the interactive widgets for the search tool
paper = Dropdown(options=[('All', 'All')], 
                layout=Layout(width='80%'))

word_widget = widgets.Text(
    value='COVID19 symptoms',
    placeholder='?',
    disabled=False,
    layout=Layout(width='80%')
)

date_range_widget = widgets.IntRangeSlider(
        value=[1957, 2020],
        min=1957,
        max=2020,
        step=1,
        description='Publish date:',
        disabled=False,
        continuous_update=False,
)

num_results = widgets.IntSlider(value=10,
                                min=5,
                                max=50,
                                step=1,
                                description='# Results',
                                continuous_update=False,
                                orientation='horizontal')

covid_filter = widgets.Checkbox(value=True,
                               description='Show only COVID articles')


# In[ ]:


# This creates the layout of the widgets
search_tool = VBox([
    HBox([Label('Search phrase:', layout=Layout(width='20%')), word_widget]),
    HBox([Label('Filters:', layout=Layout(width='20%')), num_results, date_range_widget, covid_filter]),
    HBox([Label('Select paper to view full text:', layout=Layout(width='20%')), paper])
])


# In[ ]:


# Next we link the widgets to the search_db method
output = interactive_output(search_db, {'words':word_widget, 'date_range':date_range_widget, 'paper_select':paper, 'k':num_results, 'covid_only':covid_filter})


# In[ ]:


# Make sure we can see very long paragraphs when we display a pandas dataframe
pd.options.display.max_colwidth = 3000


# In[ ]:


# Run the search tool
display(search_tool, output);


# In[ ]:




