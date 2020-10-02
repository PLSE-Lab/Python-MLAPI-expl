#!/usr/bin/env python
# coding: utf-8

# ## What is Keyword Exctraction?
# 
# Keyword extraction is defined as the task that automatically identifies a set of the terms that best describe the subject of document. This is an important method in information retrieval (IR) systems: keywords simplify and speed up the search. Keyword extraction can be used to reduce the dimensionality of text for further text analysis (text classification ot topic modeling). [S.Art et al.](https://onlinelibrary.wiley.com/doi/abs/10.1002/smj.2699), for example, extracted keywords to measure patent similarity. Using keyword extraction, you can automatically index data, summarize a text, or generate tag clouds with the most representative keywords.
# 
# ## How to extract the keywords?
# 
# All keyword extraction algorithms include the following steps:
# 
# - *Candidate generation*. Detection of possible candidate keywords from the text.
# - *Property calculation*. Computation of properties and statistics required for ranking.
# - *Ranking*. Computation of a score for each candidate keyword and sorting in descending order of all candidates. The top n candidates are finally selected as the n keywords representing the text.
# 
# ## Automatic Keyword extraction algorithms
# 
# - Rapid Automatic Keyword Extraction (RAKE). Python implementations: [one](https://github.com/csurfer/rake-nltk), [two](https://github.com/zelandiya/RAKE-tutorial), [three](https://github.com/aneesha/RAKE)
# - TextRank. Python implementations [number one](https://pypi.org/project/summa/) and [number two](https://radimrehurek.com/gensim/summarization/keywords.html)
# - [Yet Another Keyword Extractor (Yake)](https://github.com/LIAAD/yake)
# 
# 
# ## If you want to know more...
# - [Slobodan Beliga.](https://pdfs.semanticscholar.org/bdbf/25f3dcf63d38cdb527a9ffca269fa0b8046b.pdf) Keyword extraction: a review of methods and approache
# - [Kamil Bennani-Smires et al.](https://arxiv.org/pdf/1801.04470.pdf) Simple Unsupervised Keyphrase Extraction using Sentence Embeddings
# - [YanYing et al.](https://www.sciencedirect.com/science/article/pii/S1877050917303629) A Graph-based Approach of Automatic Keyphrase Extraction
# - [Martin Dostal and Karel Jezek](http://ceur-ws.org/Vol-706/poster13.pdf) Automatic Keyphrase Extraction based on NLP and Statistical Methods

# In this kernel we will apply different keyword extraction approaches to the NIPS Paper dataset. Fisrt of all, let's load and prepare the data.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


# load the dataset
df = pd.read_csv('/kaggle/input/nips-papers/papers.csv')
df.head()


# In[ ]:


df.info()


# In[ ]:


print("{} abstracts are missing".format(df[df['abstract']=='Abstract Missing']['abstract'].count()))


# In[ ]:


import pprint
sample = 941
pprint.pprint("TITLE:{}".format(df['title'][sample]))
pprint.pprint("ABSTRACT:{}".format(df['abstract'][sample]))
pprint.pprint("FULL TEXT:{}".format(df['paper_text'][sample][:1000]))


# This dataset contains 7 columns: id, year, title, even_type, pdf_name, abstract and paper_text. We are mostly interested in the paper_text which include both title and abstract.

# ## Pre-processing

# In[ ]:


import re
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

stop_words = set(stopwords.words('english'))
##Creating a list of custom stopwords
new_words = ["fig","figure","image","sample","using", 
             "show", "result", "large", 
             "also", "one", "two", "three", 
             "four", "five", "seven","eight","nine"]
stop_words = list(stop_words.union(new_words))

def pre_process(text):
    
    # lowercase
    text=text.lower()
    
    #remove tags
    text=re.sub("&lt;/?.*?&gt;"," &lt;&gt; ",text)
    
    # remove special characters and digits
    text=re.sub("(\\d|\\W)+"," ",text)
    
    ##Convert to list from string
    text = text.split()
    
    # remove stopwords
    text = [word for word in text if word not in stop_words]

    # remove words less than three letters
    text = [word for word in text if len(word) >= 3]

    # lemmatize
    lmtzr = WordNetLemmatizer()
    text = [lmtzr.lemmatize(word) for word in text]
    
    return ' '.join(text)


# In[ ]:


get_ipython().run_cell_magic('time', '', "docs = df['paper_text'].apply(lambda x:pre_process(x))")


# In[ ]:


docs[1][0:103]


# ## 1.TF-IDF and Scikit-learn
# 
# Based on the tutorial of [Kavita Ganesan](https://github.com/kavgan/nlp-in-practice/blob/master/tf-idf/Keyword%20Extraction%20with%20TF-IDF%20and%20SKlearn.ipynb)
# 
# TF-IDF stands for Text Frequency Inverse Document Frequency. The importance of each word increases proportionally to the number of times a word appears in the document (Text Frequency - TF) but is offset by the frequency of the word in the corpus (Inverse Document Frequency - IDF). Using the tf-idf weighting scheme, the keywords are the words with the higherst TF-IDF score.
# 
# ### 1.1 CountVectorizer to create a vocabulary and generate word counts

# In[ ]:


get_ipython().run_cell_magic('time', '', 'from sklearn.feature_extraction.text import CountVectorizer\n#docs = docs.tolist()\n#create a vocabulary of words, \ncv=CountVectorizer(max_df=0.95,         # ignore words that appear in 95% of documents\n                   max_features=10000,  # the size of the vocabulary\n                   ngram_range=(1,3)    # vocabulary contains single words, bigrams, trigrams\n                  )\nword_count_vector=cv.fit_transform(docs)')


# ### 1.2 TfidfTransformer to Compute Inverse Document Frequency (IDF)

# In[ ]:


get_ipython().run_cell_magic('time', '', 'from sklearn.feature_extraction.text import TfidfTransformer\n\ntfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)\ntfidf_transformer.fit(word_count_vector)')


# Once we have our IDF computed, we are now ready to compute TF-IDF and extract the top keywords.

# In[ ]:


def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    """get the feature names and tf-idf score of top n items"""
    
    #use only topn items from vector
    sorted_items = sorted_items[:topn]

    score_vals = []
    feature_vals = []

    for idx, score in sorted_items:
        fname = feature_names[idx]
        
        #keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])

    #create a tuples of feature,score
    #results = zip(feature_vals,score_vals)
    results= {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]]=score_vals[idx]
    
    return results


# In[ ]:


# get feature names
feature_names=cv.get_feature_names()

def get_keywords(idx, docs):

    #generate tf-idf for the given document
    tf_idf_vector=tfidf_transformer.transform(cv.transform([docs[idx]]))

    #sort the tf-idf vectors by descending order of scores
    sorted_items=sort_coo(tf_idf_vector.tocoo())

    #extract only the top n; n here is 10
    keywords=extract_topn_from_vector(feature_names,sorted_items,10)
    
    return keywords

def print_results(idx,keywords, df):
    # now print the results
    print("\n=====Title=====")
    print(df['title'][idx])
    print("\n=====Abstract=====")
    print(df['abstract'][idx])
    print("\n===Keywords===")
    for k in keywords:
        print(k,keywords[k])


# In[ ]:


idx=941
keywords=get_keywords(idx, docs)
print_results(idx,keywords, df)


# I am not happy with result. I would like to add a filter that will remove similar keywords, or short keywords inside of complex ones. For instance, non-negative matrix factorization meets us 5 time: non negative matrix, negative matrix, nmf, matrix factorization, matrix. Adding a 4-grams does not change the situation. Similar keywords appears due to the fact that TF-IDF does not take into account the context, the keywords importance comes only from their frequencies relationship. Thus, TF-IDF is a quick, intuitive, but not the best way to extract keywords from the text. Let's look at other ways.

# ## 2. Gensim implementation of TextRank summarization algorithm
# 
# Gensim is a free Python library designed to automatically extract semantic topics from documents. The gensim implementation is based on the popular TextRank algorithm. 
# 
# [Documentation](https://radimrehurek.com/gensim/summarization/keywords.html)
# 
# [Tutorial](https://rare-technologies.com/text-summarization-with-gensim/)

# ### 2.1 Small text

# In[ ]:


import gensim
text = "Non-negative matrix factorization (NMF) has previously been shown to " + "be a useful decomposition for multivariate data. Two different multiplicative " + "algorithms for NMF are analyzed. They differ only slightly in the " + "multiplicative factor used in the update rules. One algorithm can be shown to " + "minimize the conventional least squares error while the other minimizes the  " + "generalized Kullback-Leibler divergence. The monotonic convergence of both  " + "algorithms can be proven using an auxiliary function analogous to that used " + "for proving convergence of the Expectation-Maximization algorithm. The algorithms  " + "can also be interpreted as diagonally rescaled gradient descent, where the  " + "rescaling factor is optimally chosen to ensure convergence."
gensim.summarization.keywords(text, 
         ratio=0.5,               # use 50% of original text
         words=None,              # Number of returned words
         split=True,              # Whether split keywords
         scores=False,            # Whether score of keyword
         pos_filter=('NN', 'JJ'), # Part of speech (nouns, adjectives etc.) filters
         lemmatize=True,         # If True - lemmatize words
         deacc=True)              # If True - remove accentuation


# In[ ]:


print("SUMMARY: ", gensim.summarization.summarize(text,
                                                  ratio = 0.5,
                                                  split = True))


# ### 2.2 Large text

# In[ ]:


def get_keywords_gensim(idx, docs):
    
    keywords=gensim.summarization.keywords(docs[idx], 
                                  ratio=None, 
                                  words=10,         
                                  split=True,             
                                  scores=False,           
                                  pos_filter=None, 
                                  lemmatize=True,         
                                  deacc=True)              
    
    return keywords

def print_results_gensim(idx,keywords, df):
    # now print the results
    print("\n=====Title=====")
    print(df['title'][idx])
    print("\n=====Abstract=====")
    print(df['abstract'][idx])
    print("\n===Keywords===")
    for k in keywords:
        print(k)


# In[ ]:


idx=941
keywords=get_keywords_gensim(idx, docs)
print_results_gensim(idx,keywords, df)


# The keywords highlight the main point , but still miss valuable information

# ## 3. Python implementation of the Rapid Automatic Keyword Extraction algorithm (RAKE) using NLTK
# 
# [Documentation](https://github.com/csurfer/rake-nltk)
# 
# ### Setup using pip

# In[ ]:


get_ipython().system('pip install rake-nltk')


# ### or directly from the repository

# In[ ]:


# !git clone https://github.com/csurfer/rake-nltk.git
# !python rake-nltk/setup.py install


# ### 3.1 Small text

# In[ ]:


text = "Non-negative matrix factorization (NMF) has previously been shown to " + "be a useful decomposition for multivariate data. Two different multiplicative " + "algorithms for NMF are analyzed. They differ only slightly in the " + "multiplicative factor used in the update rules. One algorithm can be shown to " + "minimize the conventional least squares error while the other minimizes the  " + "generalized Kullback-Leibler divergence. The monotonic convergence of both  " + "algorithms can be proven using an auxiliary function analogous to that used " + "for proving convergence of the Expectation-Maximization algorithm. The algorithms  " + "can also be interpreted as diagonally rescaled gradient descent, where the  " + "rescaling factor is optimally chosen to ensure convergence."


# In[ ]:


from rake_nltk import Rake
r = Rake()
r.extract_keywords_from_text(text)
r.get_ranked_phrases_with_scores()[:10]


# Wow! We see well interbretable machine learning terminology! But why diagonally rescaled gradient descent is more important than negative matrix factorization? 

# ### 3.2 Large Text

# In[ ]:


def get_keywords_rake(idx, docs, n=10):
    # Uses stopwords for english from NLTK, and all puntuation characters by default
    r = Rake()
    
    # Extraction given the text.
    r.extract_keywords_from_text(docs[idx][1000:2000])
    
    # To get keyword phrases ranked highest to lowest.
    keywords = r.get_ranked_phrases()[0:n]
    
    return keywords

def print_results(idx,keywords, df):
    # now print the results
    print("\n=====Title=====")
    print(df['title'][idx])
    print("\n=====Abstract=====")
    print(df['abstract'][idx])
    print("\n===Keywords===")
    for k in keywords:
        print(k)

idx=941
keywords = get_keywords_rake(idx, docs, n=10)
print_results(idx, keywords, df)


# Oups! Something goes wrong! Algorithm does not work for the preprocessed text without punctuations. Let's treat the raw text.

# In[ ]:


idx=941
keywords = get_keywords_rake(idx, df['paper_text'], n=10)
print_results(idx, keywords, df)


# Presented implementation works well on sentences, but it is not flexible enough for large text. However, those who are interested in RANK can expand the capabilities of this code to their needs. We will consider next options.

# ## 4. Yet Another Keyword Extractor (Yake)
# 
# [Documentation](https://github.com/LIAAD/yake)

# In[ ]:


get_ipython().system('pip install git+https://github.com/LIAAD/yake')


# In[ ]:


import yake

def get_keywords_yake(idx, docs):
    y = yake.KeywordExtractor(lan='en',          # language
                             n = 3,              # n-gram size
                             dedupLim = 0.9,     # deduplicationthresold
                             dedupFunc = 'seqm', #  deduplication algorithm
                             windowsSize = 1,
                             top = 10,           # number of keys
                             features=None)           
    
    keywords = y.extract_keywords(text)
    return keywords

idx=941
keywords = get_keywords_yake(idx, docs[idx])
print_results(idx, keywords, df)


# Key phrases are repeated, and the text needs pre-processing to remove stop words

# ## 5. Keyphrases extraction using pke
# 
# `pke` an open source python-based keyphrase extraction toolkit. It provides an end-to-end keyphrase extraction pipeline in which each component can be easily modified or extended to develop new models.
# 
# `pke` currently implements the following keyphrase extraction models:
# 
# * Unsupervised models
#   * Statistical models
#     * TfIdf [[documentation](https://boudinfl.github.io/pke/build/html/unsupervised.html#tfidf)]
#     * KPMiner [[documentation](https://boudinfl.github.io/pke/build/html/unsupervised.html#kpminer), [article by (El-Beltagy and Rafea, 2010)](http://www.aclweb.org/anthology/S10-1041.pdf)]
#     * YAKE [[documentation](https://boudinfl.github.io/pke/build/html/unsupervised.html#yake), [article by (Campos et al., 2020)](https://doi.org/10.1016/j.ins.2019.09.013)]
#   * Graph-based models
#     * TextRank [[documentation](https://boudinfl.github.io/pke/build/html/unsupervised.html#textrank), [article by (Mihalcea and Tarau, 2004)](http://www.aclweb.org/anthology/W04-3252.pdf)]
#     * SingleRank  [[documentation](https://boudinfl.github.io/pke/build/html/unsupervised.html#singlerank), [article by (Wan and Xiao, 2008)](http://www.aclweb.org/anthology/C08-1122.pdf)]
#     * TopicRank [[documentation](https://boudinfl.github.io/pke/build/html/unsupervised.html#topicrank), [article by (Bougouin et al., 2013)](http://aclweb.org/anthology/I13-1062.pdf)]
#     * TopicalPageRank [[documentation](https://boudinfl.github.io/pke/build/html/unsupervised.html#topicalpagerank), [article by (Sterckx et al., 2015)](http://users.intec.ugent.be/cdvelder/papers/2015/sterckx2015wwwb.pdf)]
#     * PositionRank [[documentation](https://boudinfl.github.io/pke/build/html/unsupervised.html#positionrank), [article by (Florescu and Caragea, 2017)](http://www.aclweb.org/anthology/P17-1102.pdf)]
#     * MultipartiteRank [[documentation](https://boudinfl.github.io/pke/build/html/unsupervised.html#multipartiterank), [article by (Boudin, 2018)](https://arxiv.org/abs/1803.08721)]
# * Supervised models
#   * Feature-based models
#     * Kea [[documentation](https://boudinfl.github.io/pke/build/html/supervised.html#kea), [article by (Witten et al., 2005)](https://www.cs.waikato.ac.nz/ml/publications/2005/chap_Witten-et-al_Windows.pdf)]
#     * WINGNUS [[documentation](https://boudinfl.github.io/pke/build/html/supervised.html#wingnus), [article by (Nguyen and Luong, 2010)](http://www.aclweb.org/anthology/S10-1035.pdf)]
# 

# In[ ]:


get_ipython().system('pip install git+https://github.com/boudinfl/pke.git')


# In[ ]:


import pke


# ### 5.1  SingleRank
# 
# This model is an extension of the TextRank model that uses the number of co-occurrences to weigh edges in the graph.

# In[ ]:


# define the set of valid Part-of-Speeches
pos = {'NOUN', 'PROPN', 'ADJ'}

# 1. create a SingleRank extractor.
extractor = pke.unsupervised.SingleRank()

# 2. load the content of the document.
extractor.load_document(input=text,
                        language='en',
                        normalization=None)

# 3. select the longest sequences of nouns and adjectives as candidates.
extractor.candidate_selection(pos=pos)

# 4. weight the candidates using the sum of their word's scores that are
#    computed using random walk. In the graph, nodes are words of
#    certain part-of-speech (nouns and adjectives) that are connected if
#    they occur in a window of 10 words.
extractor.candidate_weighting(window=10,
                              pos=pos)

# 5. get the 10-highest scored candidates as keyphrases
keyphrases = extractor.get_n_best(n=10)

idx = 941
# now print the results
print("\n=====Title=====")
print(df['title'][idx])
print("\n=====Abstract=====")
print(df['abstract'][idx])
print("\n===Keywords===")
for k in keyphrases:
    print(k[0])


# In[ ]:


Great job!


# ### 5.2 TopicRank

# In[ ]:


import string
from nltk.corpus import stopwords

# 1. create a TopicRank extractor.
extractor = pke.unsupervised.TopicRank()

# 2. load the content of the document.
extractor.load_document(input=text)

# 3. select the longest sequences of nouns and adjectives, that do
#    not contain punctuation marks or stopwords as candidates.
pos = {'NOUN', 'PROPN', 'ADJ'}
stoplist = list(string.punctuation)
stoplist += ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-']
stoplist += stopwords.words('english')
extractor.candidate_selection(pos=pos, stoplist=stoplist)

# 4. build topics by grouping candidates with HAC (average linkage,
#    threshold of 1/4 of shared stems). Weight the topics using random
#    walk, and select the first occuring candidate from each topic.
extractor.candidate_weighting(threshold=0.74, method='average')

# 5. get the 10-highest scored candidates as keyphrases
keyphrases = extractor.get_n_best(n=10)

idx = 941
# now print the results
print("\n=====Title=====")
print(df['title'][idx])
print("\n=====Abstract=====")
print(df['abstract'][idx])
print("\n===Keywords===")
for k in keyphrases:
    print(k[0])


# Great job too! I love implementation of graph-based models by `pke` library.

# # Conclusion
# 
# 
# #### In this kernel, we looked at different algorithms of keywords extraction. My personal top is:
#  1. SingleRank by pke
#  2. TopicRank by pke
#  3. TextRank by gensim 
#  
# I am new to NLP and it was a great journey for me: I learned a lot when I wrote this kernel. Hope, you will find it usefull too. Thanks for reading!

# In[ ]:




