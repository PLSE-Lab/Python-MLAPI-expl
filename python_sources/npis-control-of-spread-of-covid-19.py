#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
from nltk.tokenize import WhitespaceTokenizer
from nltk.stem import WordNetLemmatizer
import gensim
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
import string
import json
import re

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# - IDs: 4743095, 2077273, Marcus Dreyer, 4894652, 4833887
# - usernames: helenlord, katieymo, Marcus Dreyer, dincerti, shemrarizzo
# - Emails: lord.helen [at] gene [dot] com, katieymo [at] gmail [dot] com, dreyer [at] itprodqs-consultng [dot] com, devin.incerti [at] gmail [dot] com, shem.rizzo [at] gmail [dot] com
# 
# 
# ## Goal: to identify papers which discuss specific non-pharmaceutical interventions to decrease the spread of COVID-19
# 
# *Focus: Methods to control the spread in communities, barriers to compliance and how these vary among different populations*
# 
# 
# ## Methodology: 
# 1) Cleaned abstracts
# 
# 2) Use LDA on cleaned abstracts to identify papers most relevant to NPI topics
# 
# 3) Use keyword search to pull out NPI papers which focus on: methods to control the spread in communities, barriers to compliance and how these vary among different populations
# 
# 4) Pull out specific sentences and paragraphs from the identified papers with the keywords for quick identification
# 
# ## Notes:
# - The keyword search alone on the metadata is a useful methodology to identify relevant papers. It can be completed on either the title or the abstract. However, finding pertinent, specific keywords is very important for success
# - Cleaning the text and using topic modeling focuses on a smaller, more relevant subset of papers, allowing this methodology to scale. However, a some relevant papers may be eliminated by the topic modeling

# In[ ]:


m = pd.read_csv('/kaggle/input/CORD-19-research-challenge/metadata.csv')


# In[ ]:


m = m[(m['title'].notna() & m['abstract'].notna())]


# ## Clean Abstracts
# 
# - lowercase
# - remove punctuation
# - remove stopwords
# - lemmatize
# - bigrams/trigrams

# In[ ]:


w_tokenizer = WhitespaceTokenizer()
lemmatizer = WordNetLemmatizer()


# In[ ]:


def preprocess(sentence):
    sentence = sentence.lower()
    sentence_no_punctuation = sentence.translate(str.maketrans('', '', string.punctuation))
    lemmatized_list = [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(sentence_no_punctuation) 
                  if w not in stopwords.words('english')]
    return lemmatized_list


# In[ ]:


m['abstract_lemmatized']=m['abstract'].map(lambda s:preprocess(s)) 


# In[ ]:


data_words = list(m['abstract_lemmatized'])


# In[ ]:


bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
bigram_mod = gensim.models.phrases.Phraser(bigram)


# In[ ]:


trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  
trigram_mod = gensim.models.phrases.Phraser(trigram)


# In[ ]:


def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]


# In[ ]:


def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]


# In[ ]:


m['abstract_lemmatized_grams']= make_trigrams(m['abstract_lemmatized'])


# In[ ]:


def abstract_to_string(text):
    return ' '.join(word for word in text)


# In[ ]:


m['cleanAbstract'] = m['abstract_lemmatized_grams'].map(lambda s:abstract_to_string(s))


# ## LDA on Cleaned Abstracts

# In[ ]:


count_vectorizer = CountVectorizer(stop_words='english')


# In[ ]:


data_vectorized = count_vectorizer.fit_transform(m['cleanAbstract'])


# ## Compared sklean and gensim LDA models, completed GridSearch for n_components (options 5, 10, 15, 20, 25, 30, 35). The best model was sklearn and parameter was 5 - used below

# In[ ]:


number_topics = 5


# In[ ]:


lda = LDA(n_components=number_topics, n_jobs=-1)


# In[ ]:


lda.fit(data_vectorized)


# ## Print top words associated with Topics

# In[ ]:


# Helper function
def print_topics(model, count_vectorizer, n_top_words=10):
    words = count_vectorizer.get_feature_names()
    for topic_idx, topic in enumerate(model.components_):
        print("\nTopic #%d:" % topic_idx)
        print(" ".join([words[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))


# In[ ]:


# Print the topics found by the LDA model
print("Topics found via LDA:")
print_topics(lda, count_vectorizer)


# In[ ]:


topics = lda.transform(data_vectorized)


# In[ ]:


for idx in range(number_topics):
    col_name = 'Topic ' + str(idx)
    m[col_name] = topics[:, idx]


# ## Looking for Topic associated with NPI (non-pharm in abstract)

# In[ ]:


non_pharm = m[(m['abstract'].str.contains('non-pharm'))]


# In[ ]:


topic_cols = [x for x in m.columns if 'Topic ' in x]


# In[ ]:


non_pharm_topics = non_pharm[topic_cols].idxmax(axis=1)


# In[ ]:


def most_frequent(List): 
    return max(set(List), key = List.count)


# ## Take the topics that match most for NPI modeling papers, then find all papers with that as their top topic

# In[ ]:


Counter(non_pharm_topics)


# In[ ]:


top_topic = most_frequent(list(non_pharm_topics))

top_topic


# In[ ]:


m['Top_Topic'] = m[topic_cols].idxmax(axis=1)


# In[ ]:


m.groupby('Top_Topic').size()


# In[ ]:


top_topic_papers = m[m['Top_Topic'] == top_topic]


# ## Keywords
# 
# - Need a core covid keyword
# - And need a topic keyword

# In[ ]:


covid_keywords = ['corona', 'covid']


# In[ ]:


intervention_keywords = ['social distancing',
                        'contact tracing',
                        'case isolation',
                        'shelter-in-place',
                        'stay-at-home',
                        'movement restriction',
                        'event cancel',
                        'face mask',
                        'facial mask',
                        'travel ban',
                        'school closure']


# In[ ]:


def find_papers_w_keywords(topic_keywords, papers):
    for keyword in topic_keywords:
        num_papers_title = len(papers[(papers['title'].str.contains(keyword)) & 
                                        (papers['title'])])
        num_papers_abstract = len(papers[papers['abstract'].str.contains(keyword)])
        print ('Identified {} papers with "{}" in title, {} relevant papers with "{}" in abstract'                       .format(num_papers_title, keyword, num_papers_abstract, keyword)) 


# ## Identify core papers - about COVID-19 (keyword search and published date)

# In[ ]:


date_filter = '2019-12-01'


# In[ ]:


find_papers_w_keywords(covid_keywords, top_topic_papers)


# In[ ]:


top_topic_papers['core_abstract'] = top_topic_papers['abstract'].apply(lambda x: any([k in x for k in covid_keywords]))


# In[ ]:


covid_papers = top_topic_papers[(top_topic_papers['core_abstract'] == True) & 
                                (top_topic_papers['publish_time'] >= date_filter)]


# ## Intervention Papers

# In[ ]:


for keyword in intervention_keywords:
    covid_papers[keyword] = covid_papers['abstract'].str.contains(keyword)


# In[ ]:


covid_papers['# Keywords in Abstract'] = covid_papers[intervention_keywords].sum(axis=1)


# In[ ]:


find_papers_w_keywords(intervention_keywords, covid_papers)


# In[ ]:


intervention_papers = covid_papers[covid_papers['# Keywords in Abstract'] > 1]


# In[ ]:


len(intervention_papers)


# In[ ]:


intervention_papers.to_csv("intervention_papers_metadata.csv", index=False)


# ## Search Full Papers for relevant sentences and paragraphs

# In[ ]:


def find_keyword(keywords, text):
    """
    Iterates through a list of keywords and searches them in a string of text.

    inputs:
      keywords: list of keywords
      text: string of text

    output: number of times keywords are found in the text
    """
    find = []
    for keyword in keywords:
        find.extend(re.findall(keyword, text.lower()))
    return len(find)


# In[ ]:


def search_body_text(sha, folder1, folder2, keywords, sentence_only):
    """
    Searches a single full length text for sentences/paragraphs which contain a list of keywords.

    inputs:
      sha: sha file name
      folder1: text folder name
      folder2: pdf or pmc folder name
      keywords: list of keywords to search for
      sentence_only: whether or not to show sentence only or full paragraph
    
    output: list of sentences/paragraphs found containing keywords
    """

    #open text file
    with open('/kaggle/input/CORD-19-research-challenge/'+folder1+'/'+folder1+'/'+folder2+'/'+sha+'.json') as f:
        file = json.load(f)
    
    found = []
    for text_dict in file["body_text"]:
        
        #if show_sentence_only, then split the paragraph into sentences, then look for keywords
        if sentence_only:
            sentences = text_dict["text"].split(". ")
            for sentence in sentences:
                count = find_keyword(keywords, sentence)
                if count > 0:
                    found.append(sentence)
                    
        #otherwise, show the whole paragraph
        else:
            count = find_keyword(keywords, text_dict["text"])
            if count > 0:
                #print(text_dict["section"])
                found.append(text_dict["text"])
                
    return(found)


# In[ ]:


def automated_lit_search(metadata_subset, keywords, sentence_only=True):
    """
    Creates a table keyword findings.
    
    inputs:
      metadata_subset: subset of metadata file to search
      keywords: list of keywords to search
      sentence_only: whether or not to show sentence only or full paragraph
    
    output: dataframe table of results with columns containing index, title, and text snippet
    """
    results = []
    
    indices = metadata_subset[metadata_subset['has_pdf_parse'] == True].index
    indices_pmc = metadata_subset[metadata_subset['has_pmc_xml_parse'] == True].index
    indices.append(indices_pmc)
    
    for index in indices:
        
        #find text location
        sha = metadata_subset["sha"][index].split(';')[0]
        folder1 = metadata_subset["full_text_file"][index]
        if metadata_subset['has_pdf_parse'][index] == True:
            folder2 = 'pdf_json'
        elif metadata_subset['has_pmc_xml_parse'][index] == True:
            folder2 = 'pmc_json'
        
        #open text and search for keywords
        found = search_body_text(sha, folder1, folder2, keywords, sentence_only)
        if len(found) > 0:
            for f in found:
                results.append([index, metadata_subset["title"][index], f])
                
    results_df = pd.DataFrame(results, columns=["index","title","text"])
    return(results_df)


# In[ ]:


intervention_sentences = automated_lit_search(intervention_papers, intervention_keywords, True)
intervention_sentences.to_csv('intervention_sentences.csv', index=False)


# In[ ]:


intervention_paragraphs = automated_lit_search(intervention_papers, intervention_keywords, False)
intervention_paragraphs.to_csv('intervention_paragraphs.csv', index=False)


# In[ ]:


list(intervention_papers['title'])

