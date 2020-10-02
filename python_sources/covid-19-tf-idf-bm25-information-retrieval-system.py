#!/usr/bin/env python
# coding: utf-8

# # COVID-19 TF-IDF and BM25 Based Information Retrieval System

# **If you find this helpful leave a like!**

# In this notebook we build a simple information retrieval system using **TF-IDF** to generate an array of **inverted indices**, and then using **BM25** (Best Match the 25th iteration) to search and index the articles. [More information about BM25](https://en.wikipedia.org/wiki/Okapi_BM25)

# ***

# # Description

# We break this task down into a few steps:
#  
#     1. Extract: here we extract the data from our dataset
#     
#     2. Organize: our organize step allows us to create a dictionary where we store our information of interest, then we generate a new dictionary where we store our inverted indices based off of keywords that we find for each document
#     
#     3. Retrieve: this step is where we run BM25 and generate our list of sorted results

# ***

# ### Libraries that we use 

# In[ ]:


#libraries for getting data and extracting
import os
import urllib.request
import tarfile
import json
import pandas as pd
import numpy as np
from tqdm import tqdm


#libraries for text preprocessing
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
nltk.download('wordnet') 
from nltk.stem.wordnet import WordNetLemmatizer

#libraries for keyword extraction with tf-idf
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from scipy.sparse import coo_matrix

#libraries for reading and writing files
import pickle

#libraries for BM25
get_ipython().system('pip install rank_bm25')
from rank_bm25 import BM25Okapi


# ***

# # EXTRACT

# #### getting data
# > input: None
#  
# > output: None
#  
# Downloads the jsonfiles from the collection and puts them all in a folder, called data. Within data, creates subfolders based on where each article is from.
# We will calling on these jsonfiles stored in data in the extract step to create our dataset.
# 

# In[ ]:


def getData():
    urls = ['https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/2020-03-27/comm_use_subset.tar.gz', 'https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/2020-03-27/noncomm_use_subset.tar.gz', 'https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/2020-03-27/custom_license.tar.gz', 'https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/2020-03-27/biorxiv_medrxiv.tar.gz']

    # Create data directory
    try:
        os.mkdir('./data')
        print('Directory created')
    except FileExistsError:
        print('Directory already exists')

    #Download all files
    for i in range(len(urls)):
        urllib.request.urlretrieve(urls[i], './data/file'+str(i)+'.tar.gz')
        print('Downloaded file '+str(i+1)+'/'+str(len(urls)))
        tar = tarfile.open('./data/file'+str(i)+'.tar.gz')
        tar.extractall('./data')
        tar.close()
        print('Extracted file '+str(i+1)+'/'+str(len(urls)))
        os.remove('./data/file'+str(i)+'.tar.gz')


# #### preprocessing
# >input: text 
#  
# >output: preprocessed text - take out stop words, punctuation, change all to lowercase, remove digits/special characters, stem, lemmatize
# 
# Necessary for extract step because we will be preprocessing all extracted text (title and abstract)

# In[ ]:


def preprocess(text):
    #define stopwords
    stop_words = set(stopwords.words("english"))
    #Remove punctuations
    text = re.sub('[^a-zA-Z]', ' ', text)
    #Convert to lowercase
    text = text.lower()
    #remove tags
    text=re.sub("&lt;/?.*?&gt;"," &lt;&gt; ",text)
    # remove special characters and digits
    text=re.sub("(\\d|\\W)+"," ",text)
    ##Convert to list from string
    text = text.split()
    ##Stemming
    ps=PorterStemmer()
    text = [ps.stem(word) for word in text if not word in stop_words]
    #Lemmatisation
    lem = WordNetLemmatizer()
    text = [lem.lemmatize(word) for word in text if not word in  stop_words] 
    text = " ".join(text) 
    
    return text


# #### extraction step
# >input: None
#  
# >output: a dataframe with all the extracted information 
#  
# For every article, we are collecting its: 
# 1. paper_id 
# 2. title and 
# 3. abstract 
# 
# We store these values in pandas dataframe, which we write to a pickle file.

# In[ ]:


def extract():
    #create our collection locally in the data folder
    
    #creating our initial datastructure
    x = {'paper_id':[], 'title':[], 'abstract': []}
    
    #Iterate through all files in the data directory
    for subdir, dirs, files in os.walk('./data'):
        for file in tqdm(files):
            with open(os.path.join(subdir, file)) as f:
                data = json.load(f)
                
               #Append paper ID to list
                x['paper_id'].append(data['paper_id'])
               #Append article title to list & preprocess the text
                x['title'].append((data['metadata']['title']))
                
                #Append abstract text content values only to abstract list & preprocess the text
                abstract = ""
                for paragraph in data['abstract']:
                    abstract += paragraph['text']
                    abstract += '\n'
                #if json file no abstract in file, set the body text as the abstract (happens rarely, but often enough that this edge case matters)
                if abstract == "": 
                    for paragraph in data['body_text']:
                        abstract += paragraph['text']
                        abstract += '\n'
                x['abstract'].append(preprocess(abstract))
                
    #Create Pandas dataframe & write to pickle file
    df = pd.DataFrame.from_dict(x, orient='index')
    df = df.transpose()
    pickle.dump( df, open( "full_data_processed_FINAL.p", "wb" ) )
    return df


# ***

# # ORGANIZE

# #### sort coordinate matrix
# >input: tf_idf coordinate representation
#  
# >output: tf_idf items sorted in descending order -- so things with highest scores at the top!
# 
# Function for sorting tf_idf in descending order
# 

# In[ ]:


def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)


# #### get top n words with highest tf-idf scores
# >input: 
#  
#         1. feature_names = vocabulary
#         2. sorted_items = tf-idf vectors sorted in descending order
#         3. topN = # of keywords you would like extract from text
#        
# >output: dictionary of topN # words with highest tf-idf scores in the text (key) and their corresponding tf-idf scores (value)
# 
# Gets keyword names and their tf-idf scores of topN items

# In[ ]:


def extract_topn_from_vector(feature_names, sorted_items, topN):
    #use only topn items from vector
    sorted_items = sorted_items[:topN]
 
    score_vals = []
    feature_vals = []
    # word index and corresponding tf-idf score
    for idx, score in sorted_items:
        #keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])
 
    #create a tuples of feature,score
    results= {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]]=score_vals[idx]
    return results


# #### getting abstract keywords
# > input: 
#  
#         1. entry = row in the article dataframe, which represents one article
#         2. cv = CountVectorizer, from sklearn.feature_extraction.text
#         3. X = vector that represents the CountVectorizer fit to the corpus
#         4. tfidf_transformer = object that holds our tf_idf data -- again, fit to our corpus
#         5. feature_names = vocabulary
#         6. topN = # of keywords we'd like to extract from the abstract
#         
# > output: the topN keywords from the abstract
# 
# Extracts the topN keywords from the abstract
# 

# In[ ]:


def getAbstractKeywords(entry, cv, X, tfidf_transformer, feature_names, topN):
    abstract = entry['abstract']
    
    #first check that abstract is full
    if type(abstract) == float:
        return []
 
    #generate tf-idf for the given document
    tf_idf_vector=tfidf_transformer.transform(cv.transform([abstract])) 
    #sort the tf-idf vectors by descending order of scores
    sorted_items=sort_coo(tf_idf_vector.tocoo())
    #extract only the topN # items
    keywords_dict=extract_topn_from_vector(feature_names,sorted_items,topN)
    #just want words themselves, so only need keys of the dictionary
    keywords = list(keywords_dict.keys()) 
     
    return keywords


# #### getting title keywords
# >input: entry = row in the article dataframe, which represents one article
# 
# >output: list of all the words in the title
# 
# Assumed that if a word is in the title of the article, it must be important to the article and treated like a keyword. 
# Thus, this method just extracts all the words from the (already processed) title.
# 

# In[ ]:


def getTitleKeywords(entry):
    title = entry['title']  
    title = preprocess(title)
    #first check that the title of that entry is full
    if type(title) == float:
        return []
    
    keywords_title = title.split(' ')
    return keywords_title


# #### getting final keywords
# >input: 
#  
#         1. entry = row in the article dataframe, which represents one article
#         2. cv = CountVectorizer, from sklearn.feature_extraction.text
#         3. X = vector that represents the CountVectorizer fit to the corpus
#         4. tfidf_transformer = object that holds our tf_idf data -- again, fit to our corpus
#         5. feature_names = vocabulary
#         6. topN = # of keywords we'd like to extract from the abstract
# > output: list of *all* keywords for an article -- extracted from both title and abstract!
# 
# Calls getTitleKeywords() and getAbstractKeywords() and concatenates the two lists, resulting in a final list of keywords

# In[ ]:


def getFinalKeywords(entry, cv, X, tfidf_trans, feature_names, topN):
    #get keywords from abstract and title
    fromAbstract = getAbstractKeywords(entry, cv, X, tfidf_trans, feature_names, topN)
    fromTitle = getTitleKeywords(entry)
    #concatenate two lists
    finalKeywords = fromAbstract + fromTitle
    #convert to set and then back to list to ensure there are no duplicates in list
    final_no_duplicates = list(set(finalKeywords))
    return final_no_duplicates


# #### getting corpus
# >input: dataframe that contains the 1) paper_id 2) abstract and 3) article for every article. All text is processed.
# 
# >output: a list that contains every abstracts in the in the article dataframe
# 
# Creating a corpus, which is a necessary input to our tf_idf step.

# In[ ]:


def getCorpus(articlesDf):
    #creating a new dataframe, abstractDf, of just the abstracts, so that we don't modify the original dataframe, articlesDf
    abstractDf = pd.DataFrame(columns = ['abstract'])
    #filling abstractDf with the abstract column from articlesDf
    abstractDf['abstract'] = articlesDf['abstract']
    #converting column of dataframe to a list
    corpus = abstractDf['abstract'].to_list()
    return corpus


# #### adding keywords
# 
# >input: 
# 
#         1. df = dataframe that contains the 1) paper_id 2) title and 3) abstract for every article. All text is processed 
#         2. topN = # of keywords we'd like to extract from the abstract
#         3. makeFile = boolean, whether you'd like this method to make a pickle file of the output dataframe
#         4. fileName = what the user would like to name the pickle file
#         
# > output: pandas dataframe that contains the 
# 
#         1. paper_id 
#         2. title 
#         3. abstract and 
#         4. keywords associated with every article

# In[ ]:


def addKeywords(df, topN, makeFile, fileName):
    #defining stopwords
    stop_words = set(stopwords.words("english"))

    #creating following variables that are needed for keyword extract from abstract, using tf-idf methodology,
    #all input in getFinalKewords method
    corpus = getCorpus(df)
    cv=CountVectorizer(max_df=0.8,stop_words=stop_words, max_features=1000, ngram_range=(1,1))    
    X=cv.fit_transform(corpus)
    tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
    tfidf_transformer.fit(X)
    feature_names=cv.get_feature_names()
    
    #adding keywords article to dataframe
    df = df.reindex(columns = ['paper_id', 'title', 'abstract','keywords'])                
    #getting keywords for each entry in article dataframe -- using apply to be more efficient
    df['keywords'] = df.apply(lambda row: getFinalKeywords(row, cv, X, tfidf_transformer, feature_names, topN), axis=1)

    #make pickle file depending on user input
    if makeFile == True:
        pickle.dump( df, open( fileName, "wb" ) )
    return df  


# #### creating inverted indices
# 
# >input: pandas dataframe that contains the 1) paper_id 2) title 3)abstract and 4) keywords associated with every article
# 
# >output: dictionary of inverted indices -- key = word that is a keyword; value = all articles' paper_id's that have word as a keyword
# 
# Creating an inverted indices dictionary. Will use this when deciding which subset of articles to run our ranking/retrieving algorithm on.
# It is important that our output is a dictionary because it has constant look up time.

# In[ ]:


def createInvertedIndices(df):
    numEntries = df.shape[0]
    invertInd = {}
    
    for i in range (numEntries):
        entry = df.iloc[i]
        paper_id = entry['paper_id']    
        keywords = entry['keywords']
        for k in keywords:
            if k not in invertInd:
                invertInd[k] = []
                invertInd[k].append(paper_id)
            else:
                invertInd[k].append(paper_id)
    return invertInd


# #### organize step

# In[ ]:


def organize():
    df_without_keywords = pickle.load(open("full_data_processed_FINAL.p", "rb"))
    df_with_keywords = addKeywords(df_without_keywords, 10, False, "full_data_withKeywords_FINAL.p")
    invertedIndices = createInvertedIndices(df_with_keywords)
    pickle.dump( invertedIndices, open( "invertedIndices_FINAL.p", "wb" ) )


# ***

# # Lets take a look at the processed data!

# In[ ]:


getData()
extract()

df_without_keywords = pickle.load(open("full_data_processed_FINAL.p", "rb"))
df_with_keywords = addKeywords(df_without_keywords, 10, False, "full_data_withKeywords_FINAL.p")
display(df_without_keywords)
display(df_with_keywords)


# ***

# # RETRIEVE

# #### getting our subset of articles
# >input: query
# 
# >output: a list of the potential articles that may be of interest, as they have some of the query terms as their keyword(s)
# 
# Doing this step so we don't have to run our ranking algorithm, BM25, over all ~30,000 articles. 
# Want to identify this subset in order to make our ranking (and therefore retrieving) more efficient!

# In[ ]:


def getPotentialArticleSubset(query):
    #load in inverted indices
    invertedIndices = pickle.load(open("invertedIndices_FINAL.p", "rb"))
    
    #preprocess query and split into individual terms
    query = preprocess(query)
    queryTerms = query.split(' ')
    
    potentialArticles = []
    #concatenate list of potential articles by looping through potential articles for each word in query
    for word in queryTerms:
        if word in invertedIndices: #so if someone types in nonsensical query term that's not in invertedIndices, still won't break!
            someArticles = invertedIndices[word]
            potentialArticles = potentialArticles + someArticles
            
    #convert to set then back to list so there are no repeat articles
    potentialArticles = list(set(potentialArticles))
    return potentialArticles


# #### bm25 method
# 
# >input: list of articles, dictionary with all of the documents, weight of the title, weight of the abstract, and the query
# 
# >output: list of ranked articles
# 
# This is the main information retrieval method implementing Okapi BM25

# In[ ]:


def bm25(articles, df_dic, title_w, abstract_w, query):
    corpus_title = []
    corpus_abstract = []
    
    for article in articles:
        arr = df_dic.get(article)
        #title
        if type(arr[0]) != float:
            preprocessedTitle = preprocess(arr[0])
            corpus_title.append(preprocessedTitle)
        else:
            corpus_title.append(" ")
        
        #abstract
        if type(arr[1]) != float:
            preprocessedAbst = preprocess(arr[1])
            corpus_abstract.append(preprocessedAbst)
        else:
            corpus_abstract.append(" ")
            
    query = preprocess(query)
    
    tokenized_query = query.split(" ")
    
    tokenized_corpus_title = [doc.split(" ") for doc in corpus_title]
    tokenized_corpus_abstract = [doc.split(" ") for doc in corpus_abstract]
    
    #running bm25 on titles
    bm25_title = BM25Okapi(tokenized_corpus_title)
    doc_scores_titles = bm25_title.get_scores(tokenized_query)
    #weighting array
    doc_scores_titles = np.array(doc_scores_titles)
    doc_scores_titles = doc_scores_titles**title_w
    
    #running bm25 on abstracts
    bm25_abstract = BM25Okapi(tokenized_corpus_abstract)
    doc_scores_abstracts = bm25_abstract.get_scores(tokenized_query)
    #weighting
    doc_scores_abstracts = np.array(doc_scores_abstracts)
    doc_scores_abstracts = doc_scores_abstracts ** abstract_w
    
    #summing up the two different scores
    doc_scores = np.add(doc_scores_abstracts,doc_scores_titles)
    
    #creating a dictionary with the scores
    score_dict = dict(zip(articles, doc_scores))
    
    #creating list of ranked documents high to low
    doc_ranking = sorted(score_dict, key=score_dict.get, reverse = True)
    
    #get top 100
    doc_ranking = doc_ranking[0:100]
    
    for i in range(len(doc_ranking)):
        dic_entry = df_dic.get(doc_ranking[i])
        doc_ranking[i] = dic_entry[0]
    
    return doc_ranking


# #### retrieval step

# In[ ]:


def retrieve(queries):
    #performing information retrieval
    df_without_keywords = pickle.load(open("full_data_processed_FINAL.p", "rb"))
    df_dic = df_without_keywords.set_index('paper_id').T.to_dict('list')
    results = []
    for q in queries:
        articles = getPotentialArticleSubset(q)
        result = bm25(articles,df_dic,1,2,q)
        results.append(result)

    #Output results
    for query in range(len(results)):
        for rank in range(len(results[query])):
            print(str(query+1)+'\t'+str(rank+1)+'\t'+str(results[query][rank]))
            


# ***

# # Putting it all together

# this will give us the top 100 articles for each query sorted by BM25 rank

# In[ ]:


getData()
extract()
organize()
q = ['coronavirus origin',
'coronavirus response to weather changes',
'coronavirus immunity']
retrieve(q)


# ***

# # References

# [bm25 documentation](https://pypi.org/project/rank-bm25/)
# 
# [sklearn TFidfTransformer documentation](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html)
