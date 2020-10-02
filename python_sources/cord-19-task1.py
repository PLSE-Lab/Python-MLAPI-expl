#!/usr/bin/env python
# coding: utf-8

# # What is known about transmission, incubation, and environmental stability?

# # Task Details
# ## What is known about transmission, incubation, and environmental stability? What do we know about natural history, transmission, and diagnostics for the virus? What have we learned about infection prevention and control?
# 
# ### Specifically, we want to know what the literature reports about:
# 
# 1. Range of incubation periods for the disease in humans (and how this varies across age and health status) and how long individuals are contagious, even after recovery.
# 2. Prevalence of asymptomatic shedding and transmission (e.g., particularly children).
# 3. Seasonality of transmission.
# 4. Physical science of the coronavirus (e.g., charge distribution, adhesion to hydrophilic/phobic surfaces, environmental survival to inform decontamination efforts for affected areas and provide information about viral shedding).
# 5. Persistence and stability on a multitude of substrates and sources (e.g., nasal discharge, sputum, urine, fecal matter, blood).
# 6. Persistence of virus on surfaces of different materials (e,g., copper, stainless steel, plastic).
# 7. Natural history of the virus and shedding of it from an infected person
# 8. Implementation of diagnostics and products to improve clinical processes
# 9. Disease models, including animal models for infection, disease and transmission
# 10. Tools and studies to monitor phenotypic change and potential adaptation of the virus
# 11. Immune response and immunity
# 12. Effectiveness of movement control strategies to prevent secondary transmission in health care and community settings
# 13. Effectiveness of personal protective equipment (PPE) and its usefulness to reduce risk of transmission in health care and community settings
# 14. Role of the environment in transmission

# # My approach
# ### My approach for CORD-19 is to find similarity sentence of what medical doctor or researcher would like to know from papers. To find similar sentence, overall steps are roughly below.
# 
# 1. Create Custom Sentence Transformer Model using Allen AI SciBert model.
# 2. Extract Corpus from CORD-19 datasets.
# 3. Create Sentence Embeddings.
# 4. Find Similar sentences.

# # 1. Create Custom Sentence Transformer Model using Allen AI SciBert model.
# To create custom sentence transformer model, I use to train [Sentence Transfer Model](https://github.com/UKPLab/sentence-transformers) created by [Ubiquitous Knowledge Processing Lab.](https://www.informatik.tu-darmstadt.de/ukp/ukp_home/index.en.jsp) It is required pre-trained transformer model. I use [Allen AI SciBert Model.](https://github.com/allenai/scibert)
# 
# *(Note) Omit source code in this notebook to train custom model in this NoteBook. *

# # 2. Extract Corpus from CORD-19 datasets.
# In CORD-19 datasets, paper's contents are extracted into JSON format including title, abstract text, body texts and other meta datas. I extract title, abstract text, body text from all of these JSON files. Then I extract sentences from these text. To split into sentences, I use  [SciSpacy](https://allenai.github.io/scispacy/) model which is containing [spaCy](https://spacy.io/) models for processing biomedical, scientific or clinical text. I use `en_core_sci_lg` model to split into sentences. 
# 
# *(Note) Omit source code in this notebook to extract corpus from CORD-19 datasets.*

# # 3. Create Sentence Embeddings.
# Create sentence unit embeddings using custom transformer model which is created in step 1. 
# 
# *(Note) Omit source code in this notebook to create sentence embeddings.*

# # 4. Find Similar Sentences
# Using sentence embeddings, calculate cosine similarity with queries. 

# In[ ]:


get_ipython().system('pip install -U sentence-transformers')
from sentence_transformers import SentenceTransformer
import pickle
import sklearn
import numpy as np


# In[ ]:


embedder = SentenceTransformer('/kaggle/input/cord19-model-and-emb/My_Model_and_Corpus/Trained_Model/training_stsbenchmark_continue_training-nli_allenai-scibert_scivocab_uncased-2020-04-24_07-41-01')


# In[ ]:


# Queries sentences
queries = ['What is known about COVID-19 transmission, incubation, and environmental stability?', 
           'What do we know about natural history, transmission, and diagnostics for the COVID-19 virus?',
           'What have we learned about COVID-19 infection prevention and control?',
           'Range of incubation periods for the COVID-19 disease in humans (and how this varies across age and health status) and how long individuals are contagious, even after recovery.',
           'COVID-19 prevalence of asymptomatic shedding and transmission (e.g., particularly children).',
           'COVID-19 seasonality of transmission.',
           'Physical science of the COVID-19 coronavirus (e.g., charge distribution, adhesion to hydrophilic/phobic surfaces, environmental survival to inform decontamination efforts for affected areas and provide information about viral shedding).',
           'COVID-19 coronavirus persistence and stability on a multitude of substrates and sources (e.g., nasal discharge, sputum, urine, fecal matter, blood).',
           'Persistence of COVID-19 coronavirus on surfaces of different materials (e,g., copper, stainless steel, plastic).',
           'Natural history of the COVID-19 virus and shedding of it from an infected person.',
           'Implementation of diagnostics and products to improve clinical processes for COVID-19.',
           'COVID-19 disease models, including animal models for infection, disease and transmission.',
           'Tools and studies to monitor phenotypic change and potential adaptation of the COVID-19 coronavirus.',
           'COVID-19 immune response and immunity.',
           'Effectiveness of movement control strategies to prevent secondary transmission in health care and community settings.',
           'Effectiveness of personal protective equipment (PPE) and its usefulness to reduce risk of transmission in health care and community settings.',
           'Role of the environment in transmission.']

# Create Queries Embeddings
query_embeddings = embedder.encode(queries)


# In[ ]:


# Number of similarity sentences from top
top_n = 10


# # Title Similarity

# In[ ]:


# Read Corpus
with open('/kaggle/input/cord19-model-and-emb/My_Model_and_Corpus/Corpus/Corpus_and_Embeddings/corpus_title.pickle', 'rb') as f:
    corpus = pickle.load(f)

# Read Corpus Embedding
with open('/kaggle/input/cord19-model-and-emb/My_Model_and_Corpus/Corpus/Corpus_and_Embeddings/corpus_title_emb.pickle', 'rb') as f:
    corpus_embeddings = pickle.load(f)


# In[ ]:


# Calculate cosine similarity.
for query, query_embedding in zip(queries, query_embeddings):
    distances = sklearn.metrics.pairwise.cosine_similarity([query_embedding], corpus_embeddings, "cosine")[0]
    print('\nQuery: {}'.format(query))
    print('Top {} similarity sentences out of {} sentences'.format(top_n, len(distances)))
    for i in range(top_n):
        print('(Similarity:{:.2f})'.format(np.sort(distances)[::-1][i]), corpus[np.argsort(distances)[::-1][i]])


# # Abstract Similarity

# In[ ]:


# Read Corpus
with open('/kaggle/input/cord19-model-and-emb/My_Model_and_Corpus/Corpus/Corpus_and_Embeddings/corpus_abstract.pickle', 'rb') as f:
    corpus = pickle.load(f)

# Read Corpus Embedding
with open('/kaggle/input/cord19-model-and-emb/My_Model_and_Corpus/Corpus/Corpus_and_Embeddings/corpus_abstract_emb.pickle', 'rb') as f:
    corpus_embeddings = pickle.load(f)


# In[ ]:


# Calculate cosine similarity.
for query, query_embedding in zip(queries, query_embeddings):
    distances = sklearn.metrics.pairwise.cosine_similarity([query_embedding], corpus_embeddings, "cosine")[0]
    print('\nQuery: {}'.format(query))
    print('Top {} similarity sentences out of {} sentences'.format(top_n, len(distances)))
    for i in range(top_n):
        print('(Similarity:{:.2f})'.format(np.sort(distances)[::-1][i]), corpus[np.argsort(distances)[::-1][i]])


# # Conclusion of this NoteBook
# With using sentence transformer model, we could find similar sentences from large number of papers. I hope this sentences and source paper could help all medical doctors and researchers fighting against COVID-19.
# Next steps to enhance this notebook below. 
# 1. Create similarity sentences for other queries which is targeted in WHO.
# 2. Create body text embedding.
# 3. Find similarity with body text. 
# 4. Sentence link to source paper.
