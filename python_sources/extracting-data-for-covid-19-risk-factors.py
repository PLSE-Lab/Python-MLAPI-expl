#!/usr/bin/env python
# coding: utf-8

# # **Extracting data for COVID-19 risk factors  ** <hr/> 
# 

# **This notebook has been used to provide insights into the ongoing struggle against this infectious disease through various natural language processing (NLP) techniques. The rapid growth of new literature on coronaviruses is making it extremely difficult for the medical research community  to keep up with the recent updates. As a result, there is a huge demand for these approaches in order to help the scientific research community.
# 
# Text data is not random, but has linguistic properties, which makes it very understandable for other people and computer-processable <hr/>
# 

# **Methodology**: 
# 1. Due to the huge size of the data and our specific main goal concerning Covid-19, we extracted a number of abstracts that openly studied Covid-19 and its derivatives.
# 2. Tokenization and data preprocessing such as stemming, lemmatization and removing stop words was done using sciSpacy on the abstracts. [scispaCy](https://allenai.github.io/scispacy/) is a Python package containing spaCy models for processing biomedical, scientific or clinical text.
# 3. The abstracts were embedded into a TF-IDF Model to calculate the TF-IDF vectors. 
# 4. The cosine similarity of a **dynamic user query** was calculated against each of the abtracts. 
# 5. The next step is to sort the most similar papers to the given query and display the 10 most relevant papers. 
# 
# This allows the medical research community, governments, and decision-makers to rapidly consult the latest findings and discoveries in a given knowledge area. **Also the dynamic search queries will help researchers to search for anything related to Covid-19 **. Links to the full-text research paper are also embedded and directly clickable in the output dataframe. <hr/>

# **Pros:**
# 1. An efficient query answering system that returns the most relevant literature.
# 1. Easy and Rapid Access to the Latest Findings in a Given knowledge area.
# 
# 
# **Cons:**
# 
# 1. An abstract of a research paper is a only partial description.
# 1. Reduced scope: 10% research papers were analyzed as they were the Covid-19 related research papers.
# 
# <hr/>

# **Research Goal**
# 
# Main goal of this research is to analyze the data and find Risk Factors of COVID-19 <hr/>

# # Installations
# Installing the full spaCy pipeline for biomedical data with a large vocabulary and 600k word vectors. Other smaller and similar models are available also. Check [sciSpacy documentation](https://allenai.github.io/scispacy/) for more details

# In[ ]:


get_ipython().system('pip install scispacy')
get_ipython().system('pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.4/en_core_sci_lg-0.2.4.tar.gz')


# # Imports

# In[ ]:


import numpy as np 
import pandas as pd
import scispacy
import spacy
import en_core_sci_lg
from spacy.matcher import PhraseMatcher
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from tqdm import tqdm


# # Load Metadata
# Loading the Metadata as we will do all the work on the abstract of the research papers only.

# In[ ]:


import numpy as np
import pandas as pd

root_path = '/kaggle/input/CORD-19-research-challenge/'
metadata_path = f'{root_path}/metadata.csv'
meta_df = pd.read_csv(metadata_path)
meta_df.head()


# # Extract relevant research papers only 
# Extracting research papers related to Covid-19 and its derivates as this is what we are interested in.

# In[ ]:


covid_research_papers = meta_df[meta_df['abstract'].astype(str).str.contains('COVID-19|SARS-CoV-2|2019-nCov|SARS Coronavirus 2|2019 Novel Coronavirus')]
covid_abstract = covid_research_papers.abstract
covid_abstract.shape


# # Load Sci Model and Define Tokenizer

# In[ ]:


nlp = en_core_sci_lg.load()

#Tokenizing and simple preprocessing of the documents to remove stop words, stemming and lemmatization of the words.

def spacy_tokenizer(sentence):
    return [word.lemma_ for word in nlp(sentence) if not (word.like_num or word.is_stop or word.is_punct or word.is_space or len(word)==1)]


# # Training the model on the research papers using Tf-Idf Vectorizer

# In[ ]:


vectorizer = TfidfVectorizer(tokenizer = spacy_tokenizer, min_df=2)
data_vectorized = vectorizer.fit_transform(tqdm(covid_abstract.values.astype('U')))
data_vectorized.shape


# # Graph TF-IDF 

# In[ ]:


# most frequent words
word_count = pd.DataFrame({'word': vectorizer.get_feature_names(), 'sum of tf-idf': np.asarray(data_vectorized.sum(axis=0))[0]})

word_count.sort_values('sum of tf-idf', ascending=False).set_index('word')[:20].sort_values('sum of tf-idf', ascending=True).plot(kind='barh')


# # Define Cosine Similirity to get top n documents

# In[ ]:


def compute_cosine_similarity(doc_features, corpus_features, top_n=10):
    # get document vectors
    doc_features = doc_features.toarray()[0]
    corpus_features = corpus_features.toarray()
    # compute similarites
    similarity = np.dot(doc_features, corpus_features.T)
    # get docs with highest similarity scores
    top_docs = similarity.argsort()[::-1][:top_n]
    top_docs_with_score = [(index, round(similarity[index], 3)) for index in top_docs]
    
    return top_docs_with_score


# # Define method to get top 10 documents for a search query and a method that answers any query

# In[ ]:


from IPython.display import display, HTML
import numpy as np
#Find the 10 most releveant papers to a given query and display them
def SearchDocuments(Query):
    query_docs_tfidf = vectorizer.transform(Query) #Vectorizing and calculating tf-idf for the query

    for index, doc in enumerate(Query):
        doc_tfidf = query_docs_tfidf[index]
        #Computing Cosine similarty between the query and the abstracts and get the 10 most relevant
        top_similar_docs = compute_cosine_similarity(doc_tfidf, data_vectorized, top_n=10)
        
        df = pd.DataFrame()
        Score=[]
        for doc_index, sim_score in top_similar_docs :
            #Getting the full data of the 10 most relevant papers and add them to the dataframe
            data =meta_df.loc[meta_df['cord_uid'] == covid_research_papers.cord_uid.values[doc_index]]
            Score.append(str(sim_score))
            df = df.append(data)

        df['Score']=Score
        # Display the relevant papers in a table
        DisplayTable(df)
        
def AnswerSearchQuery(Query):
    query_docs_tfidf = vectorizer.transform(Query) #Vectorizing and calculating tf-idf for the query

    for index, doc in enumerate(Query):
        doc_tfidf = query_docs_tfidf[index]
        #Computing Cosine similarty between the query and the abstracts and get the 10 most relevant
        top_similar_docs = compute_cosine_similarity(doc_tfidf, data_vectorized, top_n=1)
        result = covid_abstract.values[top_similar_docs[0][0]].split('Results: ')
        if(len(result)==1):
            print(covid_abstract.values[top_similar_docs[0][0]])
        else:
            print(result[1])


# # Define Methods to display the results in a table

# In[ ]:



#Displaying the dataframe in a table and styling
def DisplayTable(df):
    df = df.replace(np.nan, '', regex=True)
    df['Title'] = df['title'] + '#' + df['url']
    df =df[['Title','publish_time','abstract','Score']]
    dfStyler =df.style.format({'Title': make_clickable_both,'text-align': 'right'})
    dfStyler = dfStyler.set_properties(**{'text-align': 'left'})
    dfStyler=dfStyler.set_table_styles([dict(selector='th', props=[('text-align', 'center')])])
    display(HTML(dfStyler.render()))
#Making the title of the paper in the table as Hyperlink to get access to the full text paper    
def make_clickable_both(val): 
    
    name, url = val.split('#')
    if(url==''):
        return name
    return f'<a href="{url}">{name}</a>'


# # **Queries About Covid-19 Risk Factors**

# In[ ]:


SearchDocuments(['COVID-19 risk factors'])


# In[ ]:


SearchDocuments(['Data on potential risks factors'])


# In[ ]:


SearchDocuments(['Risk factors such as Smoking, pre-existing pulmonary disease'])


# In[ ]:


SearchDocuments(['Risk factors such as Co-infections (determine whether co-existing respiratory/viral infections make the virus more transmissible or virulent) and other co-morbidities'])


# In[ ]:


SearchDocuments(['Risk factors for Neonates and pregnant women'])


# In[ ]:


SearchDocuments(['Risk factors for Socio-economic and behavioral factors to understand the economic impact of the virus and whether there were differences'])


# In[ ]:


SearchDocuments(['Transmission dynamics of the virus, including the basic reproductive number, incubation period, serial interval, modes of transmission and environmental factors'])


# In[ ]:


SearchDocuments(['Severity of disease, including risk of fatality among symptomatic hospitalized patients, and high-risk patient groups'])


# In[ ]:


SearchDocuments(['Susceptibility of populations'])


# In[ ]:


SearchDocuments(['Public health mitigation measures that could be effective for control'])


# In[ ]:


SearchDocuments(['antiviral treatment'])


# In[ ]:


SearchDocuments(['risk factors such as age'])


# In[ ]:


SearchDocuments(['risk factors such as pollution'])


# In[ ]:


SearchDocuments(['risk factors such as population density'])


# In[ ]:


SearchDocuments(['risk factors such as humidity'])


# In[ ]:


SearchDocuments(['risk factors such as heart risks'])


# In[ ]:


SearchDocuments(['risk factors such as temperature'])


# # Answering queries
# We noticed that data preprocessing is done so well that our system returns answers to most of the queries. Below are examples for queries with the corresponding answer returned by our system.

# In[ ]:


AnswerSearchQuery(['Risk factors such as Smoking, pre-existing pulmonary disease'])


# In[ ]:


AnswerSearchQuery(['Transmission dynamics of the virus, including the basic reproductive number, incubation period, serial interval, modes of transmission and environmental factors'])


# In[ ]:


AnswerSearchQuery(['Risk factors for Neonates and pregnant women'])


# In[ ]:


AnswerSearchQuery(['COVID-19 risk factors'])


# # **User Dynamic Quer**y

# In[ ]:


print('Please Write down your own Query')


# In[ ]:


Query= input()
SearchDocuments([Query])

