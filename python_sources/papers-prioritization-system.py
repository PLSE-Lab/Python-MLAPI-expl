#!/usr/bin/env python
# coding: utf-8

# # Prioritizing research papers based on abstracts
# ### The aim of this project is to make an easy way to prioritize and recommend papers to researchers in a fast way to save the researchers' time and make them able to direct all their efforts to innovate a new way to fight COVID-19. We will analyze the papers with highest priority according to our system.
# 
# #### We will start first by data exploration and cleaning then make a hypothesis based on our data exploration and domain knowledge, after that we will be implementing your solution using NLP and ML libraries like NLTK, TensorFlow and Scikit-Learn

# In[ ]:


get_ipython().system('pip install modin[ray]')
from gensim.models.word2vec import FAST_VERSION
print(FAST_VERSION)
import multiprocessing
import numpy as np
# import pandas as pd
import modin
import modin.pandas as pd
from matplotlib import pyplot as plt
import os
import string

import nltk
import gensim

from nltk.tokenize import word_tokenize, sent_tokenize
from gensim.parsing.preprocessing import remove_stopwords
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer

from gensim.test.utils import common_corpus, common_dictionary
from gensim.similarities import MatrixSimilarity

from gensim.test.utils import datapath, get_tmpfile
from gensim.similarities import Similarity


# There are a lot of files in the directory, json files for each paper, a description file and a metadata file for each paper. We will explore the metadata file to see whether we can find some useful features that serve our purpose or not.

# In[ ]:


meta = pd.read_csv("/kaggle/input/CORD-19-research-challenge/metadata.csv")
print("Cols names: {}".format(meta.columns))
meta.head(7)


# In[ ]:


plt.figure(figsize=(20,10))
meta.isna().sum().plot(kind='bar', stacked=True)


# It seems that the two features 'Microsoft Academic Paper ID', 'WHO #Covidence' have a very huge number of missing values as obvious from the histogram so we will remove them to regulate the scale of the histogram's frequencies

# In[ ]:


meta_dropped = meta.drop(['Microsoft Academic Paper ID', 'WHO #Covidence'], axis = 1)


# In[ ]:


plt.figure(figsize=(20,10))

meta_dropped.isna().sum().plot(kind='bar', stacked=True)


# From the above histogram we can see there is a small number of papers with missing urls and a considerable number with missing doi and abstracts. We are interested only in papers that have abstracts and either doi or url to be able to recommend them to the researcher. So let's explore some statistics about the papers with missing abstracts and remove them if possible.

# In[ ]:


miss = meta['abstract'].isna().sum()
print("The number of papers without abstracts is {:0.0f} which represents {:.2f}% of the total number of papers".format(miss, 100* (miss/meta.shape[0])))


# As we see from these previous results the percentage of missing abstracts is considerable but we can ignore it as it is one of the most important features in our approach. Now lets see the number of missing doi and the number of missing urls from the papers without missing abstracts.

# In[ ]:


abstracts_papers = meta[meta['abstract'].notna()]
print("The total number of papers is {:0.0f}".format(abstracts_papers.shape[0]))
missing_doi = abstracts_papers['doi'].isna().sum()
print("The number of papers without doi is {:0.0f}".format(missing_doi))
missing_url = abstracts_papers['url'].isna().sum()
print("The number of papers without url is {:0.0f}".format(missing_url))


# We will need to extract the year of publication to give high priority to the papers with earlier publication data as they are more likely to be cited in newer papers and also because the might not be sufficient for today's usage (ex: papers from 1955)

# In[ ]:


abstracts_papers = abstracts_papers[abstracts_papers['publish_time'].notna()]
abstracts_papers['year'] = pd.DatetimeIndex(abstracts_papers['publish_time']).year


# Now lets see if the papers with urls have doi or not

# In[ ]:


missing_url_data = abstracts_papers[abstracts_papers["url"].notna()]
print("The total number of papers with abstracts, urls, but missing doi = {:.0f}".format( missing_url_data.doi.isna().sum()))


# Based on the above data exploration, we will need to remove the data that doesn't have both url so that research scientists can find the paper recommended to them. We will only choose papers with abstracts and either doi or url as follows:

# In[ ]:


abstracts_papers = abstracts_papers[abstracts_papers["url"].notna()]


# ### Remove stopwords: 
# now we are going to remove the stop words which are common words as (the, a, an, etc.) we will also remove punctuation, we will then revert the words to its basis using the famous stemming algorithm (Porter algorithm) before using the TF-IDF techniques. We will also take care while removing punctuation as we don't want to lose the meaning of words like (non-medical) while using the dash but we will convert it to (nonmedical) instead to keep its meaning with respect to the search algorithm

# In[ ]:


porter = PorterStemmer()
lancaster=LancasterStemmer()

abstracts_only = abstracts_papers['abstract']
tokenized_abs = []

for abst in abstracts_only:
    tokens_without_stop_words = remove_stopwords(abst)
    tokens_cleaned = sent_tokenize(tokens_without_stop_words)
    words = [porter.stem(w.lower()) for text in tokens_cleaned for w in word_tokenize(text) if (w.translate(str.maketrans('', '', string.punctuation))).isalnum()]
    tokenized_abs.append(words)


# We will vectorize the corpus using TF-IDF, the TF-IDF technique to be computed to be used to compute the similarity between the query and the abstracts for prioritization in later steps.

# In[ ]:


dictionary = []
dictionary = gensim.corpora.Dictionary(tokenized_abs)
corpus = [dictionary.doc2bow(abstract) for abstract in tokenized_abs]
tf_idf = gensim.models.TfidfModel(corpus)


# Now, we will need to apply the same cleaning steps, that we applied before to the corpus, to the query itself so that we can get consistent results. We will do the following: Removing stop words, removing punctuation and stemming. then we will map the words to their integer ids using the dictionary of words computed before.

# In[ ]:


def query_tfidf(query):
    
    query_without_stop_words = remove_stopwords(query)
    tokens = sent_tokenize(query_without_stop_words)

    query_doc = [porter.stem(w.lower()) for text in tokens for w in word_tokenize(text) if (w.translate(str.maketrans('', '', string.punctuation))).isalnum()]

    # mapping from words into the integer ids
    query_doc_bow = dictionary.doc2bow(query_doc)
    query_doc_tf_idf = tf_idf[query_doc_bow]
    
    return query_doc_tf_idf


# Now we will use the cosine similarity algorithm from gensim, we tried to use text vectorizer instead of the TF-IDF but the TF-IDF gave better results. We will store the similarity in the same dataframe with the abstracts and sort from highest similarity to the lowest one.

# In[ ]:


def rankings(query):
    t = time.time()
    query_doc_tf_idf = query_tfidf(query)
    print(time.time()-t , 0)
    t = time.time()
    index_temp = get_tmpfile("index")
    print(time.time()-t , 1)
    t = time.time()
    len_dictionary = len(dictionary)
    print(time.time()-t , 2)
    t = time.time()
    index = Similarity(index_temp, tf_idf[corpus], num_features=len_dictionary)
    print(time.time()-t , 3)
    t = time.time()
    similarities = index[query_doc_tf_idf]
    print(time.time()-t , 4)

    # Storing similarity in the dataframe and sort from high to low simmilatiry
    t = time.time()
    abstracts_papers["similarity"] = similarities
    print(time.time()-t , 5)
    t = time.time()
#     abstracts_papers_sorted = abstracts_papers.sort_values(by ='similarity' , ascending=False)
    abstracts_papers_sorted = sorted(abstracts_papers, reverse=True)
    print(time.time()-t , 6)
    t = time.time()
    abstracts_papers_sorted.reset_index(inplace = True)
    print(time.time()-t , 7)
    t = time.time()
    top20 = abstracts_papers_sorted.head(20)
    print(time.time()-t , 8)
    t = time.time()
    norm_range = top20['year'].max() - top20['year'].min()
    print(time.time()-t , 9)
    t = time.time()
    top20["similarity"] -= (abs(top20['year'] - top20['year'].max()) / norm_range)*0.1
    print(time.time()-t , 10)
    t = time.time()
    top20 = top20.sort_values(by ='similarity' , ascending=False)
    print(time.time()-t , 11)
    t = time.time()
    top20.reset_index(inplace = True)
    print(time.time()-t , 12)
    
    return top20


# We will also reorder the top 20 papers by penalizing the older papers using the equation: similarity - 0.1*abs(x - newest_year_puplication) / (newest_year_puplication - oldest_year_puplication)

# In[ ]:


import time

# query = "COVID-19 (corona) non-pharmaceutical interventions, Methods to control the spread in communities, barriers to compliance and how these vary among different populations"
t = time.time()
top = rankings(input())
print(time.time()-t)

for abstract in range(10):
    print(top.abstract[abstract])
    print('\n>>>>>>>>>>>>>>>>>>>>>>\n')


# Now we will print the top 10 abstracts from the similarity algorithm

# In[ ]:


for paper in range(10):
    print(top.url[paper])


# # Analysis of the most related research papers
# ### Controlling the spread in communities:
# We will now discuss the methods to control the spread of the virus in the communities based on our top five papers with respect to the search query: ("COVID-19 (corona) non-pharmaceutical interventions, Methods to control the spread in communities, barriers to compliance and how these vary among different populations")

# Non-pharmaceutical intervention is often recommended to control the spread of pandemic[1] but it is debatable whether it is acceptable by the public or not and what barrier would we face when trying to apply concepts like social distancing, wearing masks or even convincing people to wash their hands more often to prevent the outbreak of a disease. We will discuss the details of the first results recommended by our system above.
# 
# It seems that there are many barriers that prevents some of the public to abide to some non-pharmaceutical intervention, for example: In the first paper, the study evaluates the acceptability of non-pharmaceutical interventions (as washing hands, social distancing,  as methods to fight respiratory diseases whether it was declared pandemic as H1N1 or non-pandemic. The study showed that a group of people were aware of the importance of non-pharmaceutical interventions to protect themselves and the society while others saw many barriers as stigma,social interaction or even economic feasibility. to make people adopt the non-pharmaceutical interventions habits we need to increase their awareness of the importance of those methods to protect them and the ones they care for and also to break the barriers concerning them about these methods.[2]
# 
# The above discussion and results show us the importance of evaluating some Non-pharmaceutical intervention methods and try to see if they could be effective in facing influenza outbreaks, this might convince more people to follow those methods, for example: The second paper discusses the non-medical intervention in the case of Influenza outbreak in one  of the schools. Many non-pharmaceutical intervention measures were tested including the following:
# 1. Closure of the school
# 2. Making a field hospital to isolate influenza patients from other patients in hospitals.
# 3. Public Health education campaign to increase the public awareness including instructions about:
#     ```css
#         1. Mask usage
#     2. Handshaking
#     3. Isolation of individuals 
#      ```
# 
# A high attack rate was found among the students maybe due to the close contact between the infected students and the rest of the students in the school although there is no mention of any obvious direct or indirect dependency between the attack rate and the close contact between the students. The study also stated that interventions could prevent the spread of the virus from the school to the outside world (other schools and places in the neighbourhood)[1]

# #### Citations:
# 
# [1] Teasdale E, Santer M, Geraghty AW, Little P, Yardley L. Public perceptions of non-pharmaceutical interventions for reducing transmission of respiratory infection: systematic review and synthesis of qualitative studies. BMC Public Health. 2014;14:589. Published 2014 Jun 11. doi:10.1186/1471-2458-14-589
# 
# [2] Sonthichai C, Iamsirithaworn S, Cummings D, et al. Effectiveness of Non-pharmaceutical Interventions in Controlling an Influenza A Outbreak in a School, Thailand, November 2007. Outbreak Surveill Investig Rep. 2011;4(2):611.
# 
