#!/usr/bin/env python
# coding: utf-8

# # Exploratory - Keyword extraction
# 
# ## Objective 
# Get an overview of the document content, at what extent a keyword based search engine may help to find relevant articles. 
# 
# This is a first step to provide an exploratory interface. So far, only metadata is used.
# 
# ## Process 
# - Use sklearn to explore common keywords
# - Visualize keywords
# 
# 
# ## Conclusions 

# In[ ]:


import pickle, nltk, os, json, re
import pandas as pd
import numpy as np

import matplotlib as plt

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler
from collections import Counter
nltk.download('stopwords')


# In[ ]:


pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', 40)
pd.set_option('display.max_rows', None)


# ## 1. Loading metadata

# In[ ]:


def load_metadata(metadata_file):
    df = pd.read_csv(metadata_file,
                 dtype={'Microsoft Academic Paper ID': str,
                        'pubmed_id': str, 
                        "title":str,
                        "absract":str,
                        "WHO #Covidence": str})
    print(f'Loaded metadata with {len(df)} records')
    return df


# In[ ]:


def isNaN(string):
    return string != string
print('hello', isNaN('hello'))
print(np.nan,isNaN(np.nan))


# In[ ]:


METADATA_FILE = '/kaggle/input/CORD-19-research-challenge/metadata.csv'

metadata = load_metadata(METADATA_FILE)


# In[ ]:


def add_calculated_columns(df):
    df['has_title'] = df.title.apply(lambda x: not isNaN(x) and x != 'TOC')
    df['has_abstract'] = df.abstract.apply(lambda x: not isNaN(x) and x != 'TOC')
    return df


# In[ ]:


metadata = add_calculated_columns(metadata)
metadata.head()


# ## 2. Preprocessing
# 
# - Clean articles that do not have metadata
# - Use title and abstract for extracting keywords 

# In[ ]:


df = metadata[metadata['has_title'] | metadata['has_abstract']].copy()
df.shape


# In[ ]:


def join_text(title, abstract, empty_text = ""):
    try:
        str_abstract = '' if (isNaN(abstract)) else abstract
        str_title = '' if (isNaN(title)) else title
        return (str_title + ' ' + str_abstract).strip()
    except: 
        return empty_text
    
print(join_text(df.iloc[0].title, 'Hola'))
print(join_text(df.iloc[0].title, df.iloc[0].abstract))
print(join_text(df.iloc[0].title, np.nan))
print(join_text(np.nan, df.iloc[0].title))
print(join_text(np.nan, np.nan))


# In[ ]:


df['text'] = df.apply(lambda x: join_text(x.title,x.abstract,""), axis = 1)


# ## Build a document - term matrix

# In[ ]:


N_VOCAB = 20000


# In[ ]:


get_ipython().run_line_magic('time', '')
vectorizer = CountVectorizer(max_features=N_VOCAB, 
                        lowercase=True,
                        stop_words=nltk.corpus.stopwords.words('english'))
X_doc_terms = vectorizer.fit_transform(df.text)


# In[ ]:


def showFreqTerms(X_train_counts, vectorizer, ):
    sum_words = X_train_counts.sum(axis = 0)

    term_freq = [(word,sum_words[0,idx]) for word,idx in vectorizer.vocabulary_.items()]
    return sorted(term_freq, key = lambda x:x[1], reverse = True)


# In[ ]:


terms = showFreqTerms(X_doc_terms, vectorizer)
terms_df = pd.DataFrame(terms, columns = ["term", "frequency"])


# In[ ]:


terms_df


# In[ ]:


terms_df.shape


# ## 4. Visualize common terms in the collection

# In[ ]:


plt.rcParams["figure.figsize"] = (6,10)

terms_df[0:50].set_index("term").sort_values("frequency").plot(kind="barh")


# - Common terms are what you could expect from a bio/health collection based on COVID-19 keyword selection: 
#      - Generic medical terms: virus, diseases, simptoms, proteins, expression, etc
#      - Specific aliases to COVID-19 and other related virus
#      - Some broad health terms: health
#      - Some common terms that maybe could be added to a stopword list (in a search engine): new, old, one, two 

# In[ ]:


vectorizer.vocabulary_


# In[ ]:


feature_names = vectorizer.get_feature_names()
feature_names[120:140]


# In[ ]:


feature_names[-30:]


# **TODO**: 
#    - Some preprocessing to remove numbers may help
#    - Removing non unicode
#    - Perform smarter keyword extraction

# ## 5. Extracting the keywords from each text

# In[ ]:


def get_keywords(X, vectorizer):
    feature_names = vectorizer.get_feature_names()
    keywords = []
    for i in range(X.shape[0]):
        pos_list = X[i]
        doc_keywords = []
        for pos in pos_list.indices:
            token = feature_names[pos]
            doc_keywords.append(token)
        keywords.append(doc_keywords)
        
    return keywords


# In[ ]:


keywords = get_keywords(X_doc_terms,vectorizer)

#adding the new column
df['keywords'] = keywords

df.head()


# In[ ]:


df.head()


# ## 6. Build a toy search-engine

# In[ ]:


def find(query_set,keyword_list):
    return len(query_set.intersection(set(keyword_list))) > 0


# In[ ]:


get_ipython().run_line_magic('time', '')

query_set = set(['induced','vaccines'])

df_induced = df[df.keywords.apply(lambda x:  find(query_set, x))]


# In[ ]:


def find_sub_dataframe(df, query_set):
    df_result = df[df.keywords.apply(lambda x:  find(query_set, x))]
    return df_result


# In[ ]:


def extract_term_df(df):
    c = Counter()
    for d in df.keywords:
        c.update(d)
 
    return pd.DataFrame.from_dict(data=c, orient='index', columns=['doc_freq'])


# In[ ]:


def plot_df_terms(df_terms, k = 50 ):
    df_terms.sort_values('doc_freq')[-k:].plot(kind='barh')


# In[ ]:


def show_results(df,query_set):
    show_cols = ['title','journal','url','abstract','keywords']

    df_results = find_sub_dataframe(df, query_set)
    df_terms = extract_term_df(df_results)
    
    print(f"Query terms: {query_set}")
    print(f"Number of results: {df_results.shape[0]}")
    print(f"Number of terms: {df_terms.shape[0]}")
    
    print("Results")
    display(df_results[show_cols].head())
    
    print("Co-occurence terms")
    plot_df_terms(df_terms)


# In[ ]:


pd.set_option('display.max_colwidth', -1)


# In[ ]:


query_set = set(['induced','vaccines'])
show_cols = ['title','journal','url','keywords']

show_results(df,query_set)


# ## 7. CORD-19 Questions
# 
# Range of incubation periods for the disease in humans (and how this varies across age and health status) and how long individuals are contagious, even after recovery.

# In[ ]:


questions = [
 "Range of incubation periods for the disease in humans (and how this varies across age and health status) and how long individuals are contagious, even after recovery.",
 "Prevalence of asymptomatic shedding and transmission (e.g., particularly children).",
 "Seasonality of transmission.",
 "Physical science of the coronavirus (e.g., charge distribution, adhesion to hydrophilic/phobic surfaces, environmental survival to inform decontamination efforts for affected areas and provide information about viral shedding).",
 "Persistence and stability on a multitude of substrates and sources (e.g., nasal discharge, sputum, urine, fecal matter, blood).",
 "Persistence of virus on surfaces of different materials (e,g., copper, stainless steel, plastic).",
 "Natural history of the virus and shedding of it from an infected person",
 "Implementation of diagnostics and products to improve clinical processes",
 "Disease models, including animal models for infection, disease and transmission",
 "Tools and studies to monitor phenotypic change and potential adaptation of the virus",
 "Immune response and immunity",
 "Effectiveness of movement control strategies to prevent secondary transmission in health care and community settings",
 "Effectiveness of personal protective equipment (PPE) and its usefulness to reduce risk of transmission in health care and community settings",
 "Role of the environment in transmission"   
]


# In[ ]:


df_queries = pd.DataFrame(data=questions, columns = ['question'])


# In[ ]:


question_X = vectorizer.transform(df_queries.question)


# In[ ]:


df_queries['keywords'] = get_keywords(question_X,vectorizer)


# In[ ]:


df_queries


# In[ ]:


import matplotlib.pyplot as plt

for i in range(df_queries.shape[0]):
    print(df_queries.loc[i].question)
    show_results(df,set(df_queries.loc[i].keywords)) 
    plt.show()

