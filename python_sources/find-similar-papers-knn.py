#!/usr/bin/env python
# coding: utf-8

# # Which NIPS papers are similar? [a sklearn knn & tf-idf exercise]

# ## Goal: Find the papers that are similar based on abstract and full-text
# 
# ### Steps:
# 
# 1. Find the important keywords of each document using tf-idf
# 2. Apply knn_model on tf-idf to find similar papers
# 
# ### Cleaning: 
# 
# * Clean text from \n \x and things like that by 
#     1. Replace \n and \x0c with space
#     2. Apply unicode
#     3. Make everything lower case

# In[ ]:


import pandas as pd
import sklearn 
import numpy as np
import nltk
#nltk.download('punkt')
import re
import time
import codecs


# ### Let's import the data:

# In[ ]:


# import data using pandas and put into SFrames:
papers_data = pd.read_csv('../input/Papers.csv')
authors_data = pd.read_csv('../input/Authors.csv')
authorId_data = pd.read_csv('../input/PaperAuthors.csv')


# ### Define two functions for being able to go from index to id and visa-versa on papers_data: 
# 1. A function that takes paper_id and papers_data as input and gives its index
# 2. A function that takes index as input and gives its paper_id

# In[ ]:


def given_paperID_give_index(paper_id, paper_data):
    return paper_data[paper_data['Id']==paper_id].index[0]
#
def given_index_give_PaperID(index, paper_data):
    return paper_data.iloc[index]['Id']


# ### Let's look at second paper as an example before cleaning:

# In[ ]:


Ex_paper_id = 5941
Ex_paper_index = given_paperID_give_index(Ex_paper_id, papers_data)
papers_data.iloc[Ex_paper_index]['PaperText'][0:1000]


# ### Clean Abstract and PaperText:
# * Clean text from \n \x and things like that by 
#     1. Replace \n and \x0c with space
#     2. Apply unicode
#     3. Make everything lower case

# In[ ]:


def clean_text(text):
    list_of_cleaning_signs = ['\x0c', '\n']
    for sign in list_of_cleaning_signs:
        text = text.replace(sign, ' ')
    #text = unicode(text, errors='ignore')
    clean_text = re.sub('[^a-zA-Z]+', ' ', text)
    return clean_text.lower()


# In[ ]:


papers_data['PaperText_clean'] = papers_data['PaperText'].apply(lambda x: clean_text(x))
papers_data['Abstract_clean'] = papers_data['Abstract'].apply(lambda x: clean_text(x))


# ### Let's look at the example paper after cleaning:

# In[ ]:


papers_data.iloc[Ex_paper_index]['PaperText_clean'][0:1000]


# ### Build tf-idf matrix based on Abstract & PaperText:
# * Using Token and Stem [Thanks to the great post by Brandon Rose: http://brandonrose.org/clustering]

# In[ ]:


# here Brandon defines a tokenizer and stemmer which returns the set 
# of stems in the text that it is passed
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")
def tokenize_and_stem(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems


# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
# Producing tf_idf matrix separately based on Abstract
tfidf_vectorizer_Abstract = TfidfVectorizer(max_df=0.95, max_features=200000,
                                 min_df=0.05, stop_words='english',
                                 use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3))
get_ipython().run_line_magic('time', "tfidf_matrix_Abstract = tfidf_vectorizer_Abstract.fit_transform(papers_data['Abstract_clean'])")

# Producing tf_idf matrix separately based on PaperText
tfidf_vectorizer_PaperText = TfidfVectorizer(max_df=0.9, max_features=200000,
                                 min_df=0.1, stop_words='english',
                                 use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3))
get_ipython().run_line_magic('time', "tfidf_matrix_PaperText = tfidf_vectorizer_PaperText.fit_transform(papers_data['PaperText_clean'])")


# In[ ]:


terms_Abstract = tfidf_vectorizer_Abstract.get_feature_names()
terms_PaperText = tfidf_vectorizer_Abstract.get_feature_names()


# ### Let's create a function that takes a paper_id and tfidf_matrix and gives n-important keywords:
# * [Thanks to the great post by Thomas Buhrmann: https://buhrmann.github.io/tfidf-analysis.html]

# In[ ]:


def top_tfidf_feats(row, terms, top_n=25):
    topn_ids = np.argsort(row)[::-1][:top_n]
    top_feats = [(terms[i], row[i]) for i in topn_ids]
    df = pd.DataFrame(top_feats)
    df.columns = ['feature', 'tfidf']
    return df['feature']
def given_paperID_give_keywords(paper_data, tfidfMatrix, terms, paper_id, top_n=20):
    row_id = given_paperID_give_index(paper_id, paper_data)
    row = np.squeeze(tfidfMatrix[row_id].toarray())
    return top_tfidf_feats(row, terms, top_n)


# ### Let's check the top 10-keywords of the example paper based on Abstract:
# Note: The words are in stemmed form

# In[ ]:


paper_id_example = 5941
print ("Keywords based on Abstract:")
print (given_paperID_give_keywords(papers_data, tfidf_matrix_Abstract,
                                  terms_Abstract, paper_id_example, top_n = 10))


# ### Build NearestNeighbors models based on Abstract and PaperText:

# In[ ]:


from sklearn.neighbors import NearestNeighbors
# Based on Abstract
num_neighbors = 4
nbrs_Abstract = NearestNeighbors(n_neighbors=num_neighbors,
                                 algorithm='auto').fit(tfidf_matrix_Abstract)
distances_Abstract, indices_Abstract = nbrs_Abstract.kneighbors(tfidf_matrix_Abstract)
# Based on PaperText
nbrs_PaperText = NearestNeighbors(n_neighbors=num_neighbors,
                                  algorithm='auto').fit(tfidf_matrix_PaperText)
distances_PaperText, indices_PaperText = nbrs_PaperText.kneighbors(tfidf_matrix_PaperText)


# In[ ]:


print ("Nbrs of the example paper based on Abstract similarity: %r" % indices_Abstract[1])
print ("Nbrs of the example paper based on PaperText similarity: %r" % indices_PaperText[1])


# ### Let's check the abstract of the similar papers found for the example paper mentioned above:
# * a) Using the model based on Abstract
# * b) Using the model based on PaperText

# In[ ]:


Ex_paper_id = 5941
Ex_index = given_paperID_give_index(Ex_paper_id, papers_data)
print ("The Abstract of the example paper is:\n")
print (papers_data.iloc[indices_Abstract[Ex_index][0]]['Abstract'])
print ("The Abstract of the similar papers are:\n")
for i in range(1, len(indices_Abstract[Ex_index])):
    print ("Neighbor No. %r has following abstract: \n" % i)
    print (papers_data.iloc[indices_Abstract[Ex_index][i]]['Abstract'])
    print ("\n")


# In[ ]:


Ex_paper_id = 5941
Ex_index = given_paperID_give_index(Ex_paper_id, papers_data)
print ("The Abstract of the example paper is:\n")
print (papers_data.iloc[indices_PaperText[Ex_index][0]]['Abstract'])
print ("The Abstract of the similar papers are:\n")
for i in range(1, len(indices_PaperText[Ex_index])):
    print ("Neighbor No. %r has following abstract: \n" % i)
    print (papers_data.iloc[indices_PaperText[Ex_index][i]]['Abstract'])
    print ("\n")


# ### Some post-processing functions to help us read author's names and title of their papers:

# In[ ]:


def given_paperID_give_authours_id(paper_id, author_data, author_id_data):
    id_author_list = author_id_data[author_id_data['PaperId']==paper_id]['AuthorId']
    return id_author_list

def given_authorID_give_name(author_id, author_data):
    author_name = author_data[author_data['Id'] == author_id]['Name']
    return author_name

def given_similar_paperIDs_give_their_titles(sim_papers_list_index, paper_data):
    titles = []
    for index in sim_papers_list_index:
        titles.append(paper_data.iloc[index]['Title']+'.')
    return titles


# In[ ]:


Ex_paper_id = 5941
Ex_index = given_paperID_give_index(Ex_paper_id, papers_data)
print ("Title of similar papers to the example paper based on Abstract:\n\n")
for title in given_similar_paperIDs_give_their_titles(indices_Abstract[Ex_index], papers_data):
    print (title)


# In[ ]:


Ex_paper_id = 5941
Ex_index = given_paperID_give_index(Ex_paper_id, papers_data)
print ("Title of similar papers to the example paper based on Abstract:\n\n")
for title in given_similar_paperIDs_give_their_titles(indices_PaperText[Ex_index], papers_data):
    print (title)


# ### Questions & notes:
# 1. Are these papers really similar? i.e. Is there an automated way to evaluate?
#     * Maybe we can check if the similar papers reference the same papers? 
# 
# 
# 3. Which model is better? Abstract or PaperText? Which papers are recommended by both models? Are these more similar?
# 
# 4. Try different parameters in generating tf-idf and/or different algorithms in producing the knn model.
# 
# 

# ### References:
# 1. http://brandonrose.org/clustering
# 2. https://buhrmann.github.io/tfidf-analysis.html
# 3. "Machine Learning Foundations: A Case Study Approach" course on Coursera

# In[ ]:




