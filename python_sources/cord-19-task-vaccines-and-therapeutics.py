#!/usr/bin/env python
# coding: utf-8

# ### (Note) Corpus and embedding in this notebook are created from CORD-19 dataset Version 12

# # My approach
# ### My approach for this contest is to find similarity sentence in CORD-19 of what medical doctor or researcher would like to know from papers. To find similar sentence, overall steps are roughly below.
# 
# 1. Create Custom Sentence Transformer Model using Allen AI SciBert model.
# 2. Extract Corpus from CORD-19 datasets.
# 3. Create Sentence Embeddings.
# 4. Find Similar sentences.

# # 1. Create Custom Sentence Transformer Model using Allen AI SciBert model.
# To create custom sentence transformer model, I use to train [Sentence Transfer Model](https://github.com/UKPLab/sentence-transformers) created by [Ubiquitous Knowledge Processing Lab.](https://www.informatik.tu-darmstadt.de/ukp/ukp_home/index.en.jsp) It is required pre-trained transformer model. I use [Allen AI SciBert Model.](https://github.com/allenai/scibert)
# 
# *(Note) Omit source code to train custom model in this NoteBook. *
# 

# # 2. Extract Corpus from CORD-19 datasets.
# In CORD-19 datasets, paper's contents are extracted into JSON format including title, abstract text, body texts and other meta datas. I extract title, abstract text, body text from all of these JSON files. Then I extract sentences from these text. To split into sentences, I use  [SciSpacy](https://allenai.github.io/scispacy/) model which is containing [spaCy](https://spacy.io/) models for processing biomedical, scientific or clinical text. I use `en_core_sci_lg` model to split into sentences. 
# 
# *(Note) Omit source code to extract corpus from CORD-19 datasets.*

# # 3. Create Sentence Embeddings.
# Create sentence unit embeddings using custom transformer model which is created in step 1. 
# 
# *(Note) Omit source code to create sentence embeddings.*

# # 4. Find Similar Sentences for the contest!
# Using sentence embeddings, calculate cosine similarity with queries. 

# # Setup embedder

# In[ ]:


get_ipython().system('pip install -U sentence-transformers')
from sentence_transformers import SentenceTransformer
import pickle
import sklearn
import numpy as np


# In[ ]:


embedder = SentenceTransformer('../input/cord19-for-v12/My_Model_and_Corpus_20200506/Trained_Model/training_stsbenchmark_continue_training-nli_allenai-scibert_scivocab_uncased-2020-04-24_07-41-01')


# # Define to find URL and path in the datasets

# In[ ]:


with open('../input/cord19-for-v12/My_Model_and_Corpus_20200506/Corpus/Extract_Text/cord-19_title.pickle', 'rb') as f:
    titles = pickle.load(f)
with open('../input/cord19-for-v12/My_Model_and_Corpus_20200506/Corpus/Extract_Text/cord-19_abstract_text.pickle', 'rb') as f:
    abstracts = pickle.load(f)
with open('../input/cord19-for-v12/My_Model_and_Corpus_20200506/Corpus/Extract_Text/cord-19_paths.pickle', 'rb') as f:
    paths = pickle.load(f)
with open('../input/cord19-for-v12/My_Model_and_Corpus_20200506/Corpus/Extract_Text/cord-19_urls.pickle', 'rb') as f:
    urls = pickle.load(f)

# get reference url and path
def str_url(url):
    
    url_list = []
    for u in url:
        if type(u) == str:
            url_list.append(u)
        elif type(u) == np.ndarray:
            for i in range(len(u)):
                if str(u[i]) == 'nan':
                    url_list.append('No URL found in metadata.csv')
                else:
                    url_list.append(str(u[i]))
    
    return url_list
    

def get_url_path_from_title(text):
    
    text = text
    ix = [i for i, title in enumerate(titles) if text in title]
    path = [paths[ix[n]] for n in range(len(ix))]
    url = [urls[ix[n]] for n in range(len(ix))]
    
    return ' '.join(path), ' '.join(str_url(url))

def get_url_path_from_abstract(text):
    
    text = text
    ix = [i for i, abst in enumerate(abstracts) if text in abst]
    path = [paths[ix[n]] for n in range(len(ix))]
    url = [urls[ix[n]] for n in range(len(ix))]
    
    return ' '.join(path), ' '.join(str_url(url))


# # Calculate Similarity

# In[ ]:


top_n = 10
def get_cosine(corpus_type, queries, query_embeddings, corpus, corpus_embeddings):
# Calculate cosine similarity.
    for query, query_embedding in zip(queries, query_embeddings):
        distances = sklearn.metrics.pairwise.cosine_similarity([query_embedding], corpus_embeddings, "cosine")[0]
        print('\nQuery: {}'.format(query))
        print('Top {} similarity sentences out of {} sentences'.format(top_n, len(distances)))
        for i in range(top_n):
            distance_i = np.sort(distances)[::-1][i]
            corpus_i = corpus[np.argsort(distances)[::-1][i]]
            corpus_i = corpus_i.strip()
            if corpus_type == 'TITLE':
                path, url = get_url_path_from_title(corpus_i)
            elif corpus_type == 'ABSTRACT':
                path, url = get_url_path_from_abstract(corpus_i)                
            print('----- no.{} -----'.format(i+1))
            print('(Similarity:{:.2f})'.format(distance_i), corpus_i)
            print('Article Path: {}\nArticle URL: {}'.format(path, url))   


# # What do we know about vaccines and therapeutics?
# ## Task Details
# What do we know about vaccines and therapeutics? What has been published concerning research and development and evaluation efforts of vaccines and therapeutics?
# 
# Specifically, we want to know what the literature reports about:
# 
# 1. Effectiveness of drugs being developed and tried to treat COVID-19 patients.
# 2. Clinical and bench trials to investigate less common viral inhibitors against COVID-19 such as naproxen, clarithromycin, and minocyclinethat that may exert effects on viral replication.
# 3. Methods evaluating potential complication of Antibody-Dependent Enhancement (ADE) in vaccine recipients.
# 4. Exploration of use of best animal models and their predictive value for a human vaccine.
# 5. Capabilities to discover a therapeutic (not vaccine) for the disease, and clinical effectiveness studies to discover therapeutics, to include antiviral agents.
# 6. Alternative models to aid decision makers in determining how to prioritize and distribute scarce, newly proven therapeutics as production ramps up. This could include identifying approaches for expanding production capacity to ensure equitable and timely distribution to populations in need.
# 7. Efforts targeted at a universal coronavirus vaccine.
# 8. Efforts to develop animal models and standardize challenge studies
# 9. Efforts to develop prophylaxis clinical studies and prioritize in healthcare workers
# 10. Approaches to evaluate risk for enhanced disease after vaccination
# 11. Assays to evaluate vaccine immune response and process development for vaccines, alongside suitable animal models [in conjunction with therapeutics]

# In[ ]:


# Queries sentences
queries = ['Effectiveness of drugs being developed and tried to treat COVID-19 patients.',
           'Clinical and bench trials to investigate less common viral inhibitors against COVID-19 such as naproxen, clarithromycin, and minocyclinethat that may exert effects on viral replication.',
           'Methods evaluating potential complication of Antibody-Dependent Enhancement (ADE) in vaccine recipients.',
           'Exploration of use of best animal models and their predictive value for a human vaccine.',
           'Capabilities to discover a therapeutic (not vaccine) for the disease, and clinical effectiveness studies to discover therapeutics, to include antiviral agents.',
           'Alternative models to aid decision makers in determining how to prioritize and distribute scarce, newly proven therapeutics as production ramps up. This could include identifying approaches for expanding production capacity to ensure equitable and timely distribution to populations in need.',
           'Efforts targeted at a universal coronavirus vaccine.',
           'Efforts to develop animal models and standardize challenge studies',
           'Efforts to develop prophylaxis clinical studies and prioritize in healthcare workers',
           'Approaches to evaluate risk for enhanced disease after vaccination',
           'Assays to evaluate vaccine immune response and process development for vaccines, alongside suitable animal models in conjunction with therapeutics'
          ]

# Create Queries Embeddings
query_embeddings = embedder.encode(queries)


# # Title Similarity for Query

# In[ ]:


# Read Corpus
with open('../input/cord19-for-v12/My_Model_and_Corpus_20200506/Corpus/Corpus_and_Embeddings/corpus_title.pickle', 'rb') as f:
    corpus = pickle.load(f)

# Read Corpus Embedding
with open('../input/cord19-for-v12/My_Model_and_Corpus_20200506/Corpus/Corpus_and_Embeddings/corpus_title_emb.pickle', 'rb') as f:
    corpus_embeddings = pickle.load(f)

get_cosine('TITLE', queries, query_embeddings, corpus, corpus_embeddings)


# # Abstract Similarity for Query

# In[ ]:


# Read Corpus
with open('../input/cord19-for-v12/My_Model_and_Corpus_20200506/Corpus/Corpus_and_Embeddings/corpus_abstract.pickle', 'rb') as f:
    corpus = pickle.load(f)

# Read Corpus Embedding
with open('../input/cord19-for-v12/My_Model_and_Corpus_20200506/Corpus/Corpus_and_Embeddings/corpus_abstract_emb.pickle', 'rb') as f:
    corpus_embeddings = pickle.load(f)

get_cosine('ABSTRACT', queries, query_embeddings, corpus, corpus_embeddings)


# # Conclusion of this NoteBook
# With using sentence transformer model, we could find similar sentences from large number of papers. I hope this sentences and source paper could help all medical doctors and researchers fighting against COVID-19.
# Next steps to enhance this notebook below. 
# 1. Create similarity sentences for other queries which is targeted in WHO.
# 2. Create body text embedding.
# 3. Find similarity with body text. 
# 4. Find target article then use question and answer pipeline. 
# 
# Hiroshi Sasaki  
# sasaki@macnica.net
