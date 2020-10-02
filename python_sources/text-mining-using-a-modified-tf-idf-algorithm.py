#!/usr/bin/env python
# coding: utf-8

# # Text mining research papers for answers using a modified TF-IDF algorithm

# This notebook attempts to find answers to the 10 tasks under the CORD-19 dataset challenge.
# The approach is as follows
# 
# * Extract data from json files. The title, abstract and body_text are extracted and converted to csv format (all_sources.csv)
# * Perform TF-IDF over documents in corpus to convert documents to document vectors.
# * TF-IDF is quite an elegant method but it suffers from quite a few limitations. So a modification to cosine similarity is explored during matching documents to search query. The exact details of this modification are explained in the notebook.
# * A search query is provided as input which seeks answers to the 10 tasks. Top N relevent documents (configurable) are retrieved according to the modified cosine similarity match. Top M relevent lines (configurable) per document are also retrieved. These are saved in output files.
# * Apart from outputting csv files TF-IDF models are also saved and are available as output data.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import json  # for reading json data files
import os
import pickle  # For saving data and models that are frequently used.
import sys
import time  # For timing execution.
from operator import itemgetter
import numpy as np
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import scipy
from gensim.parsing.preprocessing import STOPWORDS
from scipy.sparse import lil_matrix
from sklearn.feature_extraction.text import TfidfVectorizer  # For computing tfidf values of documents
import string
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # Read data and write to csv file 
# 
# Paper id, title, abstract and body is extracted from json files and written to csv file

# In[ ]:


# json files are processed and saved in csv format in this file. The fields considered are paperid,title,abstract,body_text
data_file_csv = "all_sources.csv"
output_data_directory = "/kaggle/working/"
# Mention directories where you put json data files
data_directories = "/kaggle/input/"
metadata_file = "/kaggle/input/CORD-19-research-challenge/metadata.csv"

# Return a dictionary 
def read_json(file_path):
    with open(file_path) as f:
        
        data = json.load(f)
        d = dict()
        try:
            d['paper_id'] = data['paper_id']
        except:
             d['paper_id'] = ""
        try:
            d['title'] = data['metadata']['title']
        except:
            d['title'] =""
        try:
            text = data['abstract']
        except:
            text = ""
            
        abstract_text = ""
        for content in text:
            abstract_text = abstract_text + content['text']
        d['abstract_text'] = abstract_text
        
        try:
            text = data['body_text']
        except:
            text = ""
            
        body_text = ""
        for content in text:
            body_text = body_text + content['text']
        d['body_text'] = body_text

        return d


# Function for pre processing json file content before writing to csv file.
#
def pre_process(dict_text):
    processed_text_list = []
    count = 0
    for text in dict_text.values():
        processed_text = text
        processed_text_list.append(processed_text)
        # processed_text_list contains paper_id, title, abstract and body_text
    return processed_text_list


def read_json_files_and_write_to_csv():
    f = open(output_data_directory + data_file_csv, mode="w")
    f.close()
    processed_row_list = []
    # Counter for number of files read
    number_of_files_read = 0
    # Write chunksize lines at a time to csv
    chunksize = 1000

    for dirname, _, filenames in os.walk(data_directories):
        for filename in filenames:
            if filename.endswith(".json"):
                file_path = os.path.join(dirname, filename)
                # Get parsed dictionary from json data file.
                dict_text = read_json(file_path)
                # Preprocess text and append to dataframe file by file
                processed_text_list_returned = pre_process(dict_text)
                dict_text = {"paper_id": processed_text_list_returned[0], "title": processed_text_list_returned[1],
                             "abstract": processed_text_list_returned[2], "body_text": processed_text_list_returned[3]}
                # Append dictionary to list
                processed_row_list.append(dict_text)
                number_of_files_read += 1

                # Write chunksize number of lines at a time to csv
                if number_of_files_read % chunksize == 0:
                    # Create dataframe out of list of dictionaries
                    df_csv = pd.DataFrame(processed_row_list)
                    # Dump dataframe to csv file. Write to csv one row at a time
                    df_csv.to_csv(output_data_directory + data_file_csv, mode='a', header=False, index=False)
                    processed_row_list = []
    # Write any remaining lines
    # Create dataframe out of list of dictionaries
    df_csv = pd.DataFrame(processed_row_list)
    # Dump dataframe to csv file. Write to csv one row at a time
    df_csv.to_csv(output_data_directory + data_file_csv, mode='a', header=False, index=False)

    print("Total number of files read = " + str(number_of_files_read))
    print("Saved in csv format to " + data_file_csv)
    
    
# Call read_json_files_and_write_to_csv() to convert json data to csv.
start_time = time.time()
print("Processing ...")
read_json_files_and_write_to_csv()
end_time = time.time()
print("Time usage " + str(end_time - start_time) + " seconds")


# # Perform TF-IDF
# * Data is retrieved from csv files and put in proper format for TFIDF Vectorizer.
# * TF-IDF of documents in corpus is computed. Document Vectors are made unit in length.
# * The TF-IDF models are saved for future use (fitted_vectorizer.pkl and tfidf_train_vectors.pkl).

# In[ ]:


# TF-IDF
# Retrieve data from csv file
def retrieve_data_from_csv_files_for_tfidf_input():
    chunksize = 1000
    chunklist = []
    for chunk in pd.read_csv(output_data_directory + data_file_csv, chunksize=chunksize, usecols=[1, 2, 3],
                             header=None):
        chunklist.append(chunk)
    df_concat_chunks = pd.concat(chunklist)
    tfidf_input = df_concat_chunks[1].map(str) + df_concat_chunks[2].map(str) + df_concat_chunks[3].map(str)
    tfidf_input = tfidf_input.values
    return tfidf_input


start_time = time.time()
print("Retrieving data from " + output_data_directory + data_file_csv + " for TF-IDF...")
tfidf_input = retrieve_data_from_csv_files_for_tfidf_input()
end_time = time.time()
print("Time usage " + str(end_time - start_time) + " seconds")

start_time = time.time()

print("Calculating TF-IDF for documents in corpus ...")

my_stop_words = STOPWORDS.union(
                {"preprint", "medrxiv", "copyright", "biorxiv", "reserved", "author", "permission", "text", "word",
                 "count", "rights", "count", "doi", "https", "funder", "peer", "reviewed", "org"})
tfidf_vectorizer = TfidfVectorizer(use_idf=True, decode_error='ignore',stop_words=my_stop_words,lowercase=True,
                                   norm='l2')

fitted_vectorizer = tfidf_vectorizer.fit(tfidf_input)

# Dump vectorizer
pickle.dump(fitted_vectorizer, open(output_data_directory+"fitted_vectorizer.pkl", "wb"))
tfidf_train_vectors = fitted_vectorizer.transform(tfidf_input)

print("Saving TF-IDF model to fitted_vectorizer.pkl and document vectors to tfidf_train_vectors.pkl...")
# Dump train vectors
pickle.dump(tfidf_train_vectors, open(output_data_directory+"tfidf_train_vectors.pkl", "wb"))

end_time = time.time()
print("Time usage " + str(end_time - start_time) + " seconds")


# # Load TF-IDF model

# In[ ]:


start_time = time.time()
print("Loading TF-IDF model ...")
fitted_vectorizer = pickle.load(open(output_data_directory+"fitted_vectorizer.pkl", 'rb'))
tfidf_train_vectors = pickle.load(open(output_data_directory+"tfidf_train_vectors.pkl", 'rb'))
print("Done ...")


# # Problems faced due to using cosine similarity measure
# 
# TF-IDF is usually used with cosine similarity to determine which documents match a search query the most. While trying to use this simplistic measure some problems were observed. These problems are best explained with help of an example.
# 
# Suppose we require the relevant documents to the search query "diabetes covid-19 risk factors". The word tokens are diabetes, covid, 19, risk and factors
# 
# The top 5 documents retrieved and the top 5 lines per document only using cosine similarity are displayed below. This illustrates that just cosine similarity is not good enough for our purpose.
# 
# 

# In[ ]:


# Load all_sources.csv into a dataframe
chunksize = 1000
chunklist = []
for chunk in pd.read_csv(output_data_directory + data_file_csv, chunksize=chunksize,
                         usecols=[0, 1, 2, 3], header=None):
    chunklist.append(chunk)
df_concat_chunks = pd.concat(chunklist)

search_query = "diabetes covid 19 risk factors"
# Vectorize query
tfidf_search_query_vectors = fitted_vectorizer.transform([search_query])
# Calculate cosine similarity between search query vector and document vectors
cosine_distance_between_search_query_and_doc_vectors = scipy.sparse.csr_matrix.dot(tfidf_train_vectors,
                                                       scipy.sparse.csr_matrix.transpose(tfidf_search_query_vectors))
top_n_docs = 5
top_n_lines_per_doc = 5
# Get top_n_docs which match the search query the most. The matrix cosine_distance_between_search_query_and_doc_vectors is negated to get document indices 
# in descending order of match
top_N_file_indices = np.squeeze(np.array((np.argsort(-cosine_distance_between_search_query_and_doc_vectors.todense().flatten(),axis=1)[0,:top_n_docs])))

print("Showing top matches for search query with just cosine similarity...")
doc_no = 0
for index in top_N_file_indices:
    # Show titles of top documents that match the search query the most
    doc_no+=1
    print("Document No: "+str(doc_no))
    print("Title : "+str(df_concat_chunks.iloc[index, 1]))
    print()
    all_lines_in_selected_doc = str(df_concat_chunks.iloc[index, 1]) +                                     str(df_concat_chunks.iloc[index, 2]) +                                     str(df_concat_chunks.iloc[index, 3])
    lines = all_lines_in_selected_doc.split(". ")
    # Vectorize lines in current file
    tfidf_line_vectors = fitted_vectorizer.transform(lines)
    # Calculate cosine distance between search query and line vectors
    cosine_distance_between_search_query_and_line_vectors = scipy.sparse.csr_matrix.dot(tfidf_line_vectors,
                                                            scipy.sparse.csr_matrix.transpose(
                                                            tfidf_search_query_vectors))
    # Rank lines according to cosine distance between lines of document and search query vectors
    top_N_line_indices = np.squeeze(np.array((np.argsort(
                            -cosine_distance_between_search_query_and_line_vectors.todense().flatten(), axis=1)[0, :top_n_lines_per_doc])))
    key_lines_in_doc = list(itemgetter(*top_N_line_indices)(lines))
    print("Lines :")
    for line in key_lines_in_doc:
          print(line)
    print("###########################################")
    print()


# Documents number 2 and 5 have little to no relevance to the search query.

# In[ ]:


dictionary_of_words_in_query_and_their_idf_values = dict()
idf_values = fitted_vectorizer.idf_
words_in_search_query = search_query.split()
for word in words_in_search_query:
    try:
        index_of_word_in_vocab_of_tfidf_vectorizer = fitted_vectorizer.vocabulary_[word]
        idf_of_word = idf_values[index_of_word_in_vocab_of_tfidf_vectorizer]
        dictionary_of_words_in_query_and_their_idf_values.update({word:idf_of_word})
    except:
        dictionary_of_words_in_query_and_their_idf_values = dictiionary_of_words_in_query_and_their_idf_values
print("IDF values for words in query ...")
print(dictionary_of_words_in_query_and_their_idf_values)


# The following IDF values for the words in the query was observed at the time of running this simulation.
# 
# {'diabetes': 3.453179026688757, 'covid': 3.9598320012919723, '19': 1.4025946028730936, 'risk': 1.8492836902078613, 'factors': 1.6826481323365186}
# 
# (These values may change as dataset changes but it should be realized that the values themselves are immaterial for the discussion here)
# 
# The cosine similarity calculates the following metric for a particular file i and the search query vector
# 
# COSINE_SIMILARITY(i) = TFIDFi(diabetes) x TFIDFq(diabetes) + TFIDFi(covid) x TFIDFq(covid) + TFIDFi(19) x TFIDFq(19) + TFIDFi(risk) x TFIDFq(risk) + TFIDFi(factors) x TFIDFq(factors)
# 
# where TFIDFi and TFIDFq represent TFIDF values for the word in a particular file i and search query respectively
# 
# Since covid has a high value of IDF it may produce a high value of cosine similarity for files with high values of term frequency for covid. Thus documents with only a high term frequency for covid might be ranked higher. This poses a problem for information retrieval relevant to all terms of the query.
# 
# The next cell plots the term frequencies of words present in query in top selected documents

# In[ ]:


dict_of_words_in_query_and_term_frequencies_in_a_document = dict()
doc_no = 0
for doc_index in top_N_file_indices:
    doc_no+=1
    for key in dictionary_of_words_in_query_and_their_idf_values:
        try:
            index_of_key_in_vocab_of_tfidf_vectorizer = fitted_vectorizer.vocabulary_[key]
            idf_of_key = dictionary_of_words_in_query_and_their_idf_values[key]
            tfidf_value_of_word_in_current_document = tfidf_train_vectors[doc_index,index_of_key_in_vocab_of_tfidf_vectorizer]
            term_frequency_of_word_in_current_document = tfidf_value_of_word_in_current_document/idf_of_key
            dict_of_words_in_query_and_term_frequencies_in_a_document.update({key:term_frequency_of_word_in_current_document})
        except:
            dict_of_words_in_query_and_term_frequencies_in_a_document = dict_of_words_in_query_and_term_frequencies_in_a_document
            
    print("Term frequencies for words in document: "+str(doc_no)+" (Showing term frequencies for words which are present in query)")
    print()
    print(dict_of_words_in_query_and_term_frequencies_in_a_document)
    k = dict_of_words_in_query_and_term_frequencies_in_a_document.keys()
    v = dict_of_words_in_query_and_term_frequencies_in_a_document.values()
    plt.bar(k,v)
    plt.title("Term frequencies for words in document: "+str(doc_no)+" (Showing term frequencies for words which are present in query)")
    plt.show()
    print()

        


# # Modification to cosine similarity measure
# 
# As speculated document number 2 and 5 have term frequency of diabetes = 0. Still those documents popped up in the top 5 results for search query "diabetes covid 19 risk factors". From the plotted bar graphs it is evident that we will get more relevant documents to our search query if documents contain as many words in the query as possible.
# 
# Rewriting the cosine similarity metric from before 
# 
# COSINE_SIMILARITY(i) = TFIDFi(diabetes) x TFIDFq(diabetes) + TFIDFi(covid) x TFIDFq(covid) + TFIDFi(19) x TFIDFq(19) + TFIDFi(risk) x TFIDFq(risk) + TFIDFi(factors) x TFIDFq(factors)  ---- (1)
# 
# where TFIDFi and TFIDFq represent TFIDF values for the word in a particular file i and search query respectively.
# COSINE_SIMILARITY(i) denotes how much file i matches with search query
# 
# We define component of cosine similarity as 
# 
# COMPONENT_OF_COSINE_SIMILARITY = TFIDFi(word) x TFIDFq(word)
# 
# So equation (1) has 5 components of cosine similarity. One for "diabetes" one for "covid" etc.
# 
# So we need documents where each component of cosine similarity contributes as much as possible to the final cosine similarity metric.
# 
# In short we would prefer documents where more number of cosine similarity components are non-zero
# 
# For e.g we would prefer a document where
# 
# TFIDFi(diabetes) x TFIDFq(diabetes) != 0 and TFIDFi(covid) x TFIDFq(covid) !=0 and TFIDFi(risk) x TFIDFq(risk) !=0
# 
# over a document where
# 
# TFIDFi(covid) x TFIDFq(covid) !=0 and TFIDFi(risk) x TFIDFq(risk) !=0
# 
# even though the latter may have a higher cosine similarity value.
# 
# A way to accomplish this is to calculate a metric similar to entropy over the components of cosine similarity
# 
# For e.g. 
# 
# COSINE_SIMILARITY(i) = 0.7 + 0 + 0 + 0 + 0 
# 
# After normalizing so that all components add to 1 we get 
# 
# NORMALIZED_COSINE_SIMILARITY(i) = 1 + 0 + 0 + 0 + 0
# 
# The components of normalized cosine similarity are (1,0,0,0,0) . If we consider this to be a probability distribution then information entropy of this distribution is 0.
# 
# Consider another cosine similarity measure
# 
# COSINE_SIMILARITY(i) = 0.3 + 0.2 + 0 + 0.2 + 0
# 
# NORMALIZED_COSINE_SIMILARITY(i) = 0.428 + 0.285 + 0 + 0.285 + 0
# 
# Components of normalized cosine similarity are (0.428,0.285,0,0.285,0). Information entropy of this distribution is about 1.56
# 
# So we would like to pick the latter document over the former even if they have the same cosine similarity measure of 0.7.
# 
# The idea is to multiply the information entropy with cosine similarity and use it as a final metric to rank results of a search query. Therefore if cosine similarity is high just because of one word (such as "covid" in the example above) the resulting metric will still be low. Thus this method will prefer documents which have more words of the search query.
# 
# Note that in practice we will use the following formula for calculating information entropy of components of normalized cosine similarity
# 
# ENTROPY_OF_NORMALIZED_COSINE_SIMILARITY_TERMS = SUM (Pi x log (1 + 1/Pi)) where Pi are components of normalized cosine similarity
# 
# This formula will prevent entropy from becoming 0 when search query is a single word.
# 
# The calculated entropy metric can be refined further. 
# If entropy of normalized cosine components of 2 documents is the same then the document which has more important words should be favoured. IDF is a measure of importance of words. A higher IDF means the word is rrer and hence more important in the context of the query.
# Thus entropy of normalized cosine components of documents can be weighed by the idf values of words in query.
# 
# For e.g. If a query has 3 words namely w1,w2 and w3 then 
# 
# SUM_OF_IDF_VALUES = IDF(w1) + IDF(w2) + IDF(w3)
# 
# If a document has w1 and w2 then WEIGHTED_ENTROPY_OF_NORMALIZED_COSINE_COMPONENTS = (IDF(w1) + IDF(w2)) * ENTROPY_OF_NORMALIZED_COSINE_SIMILARITY_TERMS/SUM_OF_IDF_VALUES
# This metric will improve the entropy measure of a document if more important words are present in document. 
# 
# The next few cells implement the explained functionality.

# In[ ]:


# Helper functions to preprocess query , get data from metedata.csv and write search results to output files.

def write_to_output_csv_file(data,output_filename):
    if output_filename !="":
        df = pd.DataFrame(data, columns=['search_query','paper_id', 'title','key_lines','doi'])
        df.to_csv(output_data_directory + output_filename,mode = 'a',header=True,index=False)

def get_metadata(paper_id):
    chunksize = 1000
    chunklist = []
    for chunk in pd.read_csv(metadata_file, chunksize=chunksize):
        chunklist.append(chunk)
    df_concat_chunks = pd.concat(chunklist)

    # filtering data
    matching_row = df_concat_chunks.loc[df_concat_chunks['sha'] == paper_id]
    if  matching_row.empty == False:

        title = matching_row.iloc[0]['title']
        doi = 'http://doi.org/'+str(matching_row.iloc[0]['doi'])
    else:
        title = None
        doi = None
    return title,doi

def preprocess_query(query):
    
    # Preprocess query
    # # Sanitize and preprocess queries
    my_stop_words = STOPWORDS
    # Replace punctuation with space
    words = query.split()
    # Replace punctuation with space
    punctuation_to_be_replaced_with_space = string.punctuation
    line1 = query.maketrans(punctuation_to_be_replaced_with_space,
                                     ' ' * len(punctuation_to_be_replaced_with_space))
    stripped = [w.translate(line1) for w in words]
    stripped = " ".join(stripped)
    search_queries_processed = stripped
    words = search_queries_processed.split()
    query_new = ''
    for word in words:
        if word not in my_stop_words:
            query_new = query_new + " " + word
    return query_new


# In[ ]:



# This function receives a query, top N number of documents , top M number of lines per document and output filename to write results of query
def search_corpus(query,top_n_docs,top_n_lines_per_doc,output_filename):
csv_data_search_results = []

query_new = preprocess_query(query)

dict_of_words_in_query_and_their_idf = dict()
words = query_new.split()

for word in words:
    word = word.lower()
    
    try:
        index_of_word_in_vectorizer_vocab = fitted_vectorizer.vocabulary_[word]
        # Store words in query and their IDFs
        dict_of_words_in_query_and_their_idf.update({word:idf_values[index_of_word_in_vectorizer_vocab]})
                                    
    except:
        dict_of_words_in_query_and_their_idf = dict_of_words_in_query_and_their_idf
        

# Vectorize query

tfidf_search_query_vectors = fitted_vectorizer.transform([query_new])

# Calculate cosine similarity components for each document in corpus. cosine_similarity_components will be a sparse matrix.
# For e.g. if tfidf_search_query_vectors is a sparse csr matrix and non zero elements are (0,2453) and (0,35654) then this matrix will be multiplied with all 
# tfidf_train_vectors (document vectors) . The broadcasted multiplication will be non zero for those documents where either 2453 or 35654 or both are nonzero.
    
cosine_similarity_components = scipy.sparse.csr_matrix.multiply(tfidf_train_vectors,tfidf_search_query_vectors)


# Make cosine similarity components add to one for all documents. Sum of numbers in axis 1 will be one.
normalized_cosine_similarity_components = scipy.sparse.csr_matrix(cosine_similarity_components). multiply(scipy.sparse.csr_matrix.power(scipy.sparse.csr_matrix(scipy.sparse.csr_matrix.sum(cosine_similarity_components,axis=1)),-1))

# idf_sparse_matrix is a csr matrix. It has shape equivalent to corpus of tfidf document vectors.
# For e.g suppose cosine_similarity_components  =              (0,20776) = 0.5
#                                                              (0,31212) = 0.3
#                                                                    .
#                                                                    .
#                                                              (29200,20776) = 0.4                                                       
# where axis 0 of csr matrix represents document number and axis 1 represents word index. idf_sparse_matrix will contain idf values of features in the same format.
# For e,g  idf_sparse_matrix                    =              (0,20776) = idf_of_20776
#                                                              (0,31212) = idf_of_31212 and so on.

idf_sparse_matrix = scipy.sparse.lil_matrix(tfidf_train_vectors.shape)

for row in cosine_similarity_components.nonzero()[0]:
    non_zero_columns = cosine_similarity_components[row].nonzero()[1]
    data = idf_values[non_zero_columns]
    data_index = 0
    for column in non_zero_columns:
        idf_sparse_matrix[row,column] = data[data_index]
        data_index+=1
        
idf_sparse_matrix = idf_sparse_matrix.tocsr()


# Compute the reciprocal of elements in normalized_cosine_similarity_components.
# For e.g. if normalized_cosine_similarity_components = (0,20776) = 0.625
#                                                       (0,31212) = 0.375
#                                                           .
#                                                           .
#                                                       (29200,20776) = 1

# then one_by_pi                                      = (0,20776) = 1/0.625
#                                                       (0,31212) = 1 / 0.375 and so on.
one_by_pi = scipy.sparse.csr_matrix(scipy.sparse.csr_matrix.power(normalized_cosine_similarity_components,-1))
# Take log1p of terms in one_by_pi
log_one_by_pi = scipy.sparse.csr_matrix(scipy.sparse.csr_matrix.log1p(one_by_pi))

# Calculate entropy (sum(Pi * lop1p(1/Pi))) of normalized cosine similarity components 
entropy_of_normalized_cosine_similarity_components = scipy.sparse.csr_matrix(scipy.sparse.csr_matrix.sum(
                                                     scipy.sparse.csr_matrix.multiply(normalized_cosine_similarity_components,log_one_by_pi),
                                                     axis=1))

# sum of idfs of words in query
sum_of_idfs_of_words_in_query = sum(list(dict_of_words_in_query_and_their_idf.values()))

weights_of_entropy_terms = scipy.sparse.csr_matrix.sum(idf_sparse_matrix,axis=1)/sum_of_idfs_of_words_in_query


cosine_distance_between_search_query_and_doc_vectors = scipy.sparse.csr_matrix.dot(tfidf_train_vectors,
                                                       scipy.sparse.csr_matrix.transpose(tfidf_search_query_vectors))

weighted_entropy_of_normalized_cosine_similarity_components = scipy.sparse.csr_matrix.multiply(
                                                              entropy_of_normalized_cosine_similarity_components,
                                                              weights_of_entropy_terms)
total_match_between_search_query_and_doc_vectors = scipy.sparse.csr_matrix.multiply(
                                                   cosine_distance_between_search_query_and_doc_vectors ,
                                                   weighted_entropy_of_normalized_cosine_similarity_components)

current_query_match = total_match_between_search_query_and_doc_vectors

top_N_file_indices = np.squeeze(np.array((np.argsort(-current_query_match.todense().flatten(),axis=1)[0,:top_n_docs])))

for index in top_N_file_indices:
    paper_id = df_concat_chunks.iloc[index, 0]
    title = df_concat_chunks.iloc[index, 1]
    

    all_lines_in_selected_doc = str(df_concat_chunks.iloc[index, 1]) +                                 str(df_concat_chunks.iloc[index, 2]) +                                 str(df_concat_chunks.iloc[index, 3])
    lines = all_lines_in_selected_doc.split(". ")

    # Match query with all lines of selected document
    tfidf_line_vectors = fitted_vectorizer.transform(lines)

    # Repeat the same algorithm as before for top M lines per document in top N documents
    
    cosine_similarity_components = scipy.sparse.csr_matrix.multiply(tfidf_search_query_vectors, tfidf_line_vectors)
    normalized_cosine_similarity_components = scipy.sparse.csr_matrix(cosine_similarity_components).                                               multiply(scipy.sparse.csr_matrix.power(
                                              scipy.sparse.csr_matrix(scipy.sparse.csr_matrix.sum(cosine_similarity_components, axis=1)), -1))

    one_by_pi = scipy.sparse.csr_matrix(scipy.sparse.csr_matrix.power(normalized_cosine_similarity_components, -1))
    log_one_by_pi = scipy.sparse.csr_matrix(scipy.sparse.csr_matrix.log1p(one_by_pi))
    entropy_of_normalized_cosine_similarity_components = scipy.sparse.csr_matrix(
                                                         scipy.sparse.csr_matrix.sum(
                                                         scipy.sparse.csr_matrix.multiply(
                                                         normalized_cosine_similarity_components, log_one_by_pi), axis=1))
    # Rank lines in a file according to match with search query
    cosine_distance_between_search_query_and_line_vectors = scipy.sparse.csr_matrix.dot(tfidf_line_vectors,
                                                            scipy.sparse.csr_matrix.transpose(
                                                            tfidf_search_query_vectors))

    total_match_between_search_query_and_line_vectors_in_curent_file = scipy.sparse.csr_matrix.multiply(
                                                    cosine_distance_between_search_query_and_line_vectors,
                                                    entropy_of_normalized_cosine_similarity_components)
    current_line_match = total_match_between_search_query_and_line_vectors_in_curent_file

    top_N_line_indices = np.squeeze(
                         np.array((np.argsort(
                        -current_line_match.todense().flatten(), axis=1)[0, :top_n_lines_per_doc])))
    key_lines = list(itemgetter(*top_N_line_indices)(lines))
   
    # Search metadata.csv to get doi,title,links to document
    title1, doi = get_metadata(paper_id)
    if title1 != "" and title1 != None:
        title = title1

    d_line = [query,paper_id, title,". \n".join(key_lines), doi]
    csv_data_search_results.append(d_line)
write_to_output_csv_file(csv_data_search_results, output_filename)
return top_N_file_indices


# # Testing after modification to cosine similarity

# In[ ]:


top_N_file_indices = search_corpus(query="diabetes covid 19 risk factors ", top_n_docs=5, top_n_lines_per_doc=5,output_filename="")

dict_of_words_in_query_and_term_frequencies_in_a_document = dict()
doc_no = 0
for doc_index in top_N_file_indices:
    doc_no+=1
    for key in dictionary_of_words_in_query_and_their_idf_values:
        try:
            index_of_key_in_vocab_of_tfidf_vectorizer = fitted_vectorizer.vocabulary_[key]
            idf_of_key = dictionary_of_words_in_query_and_their_idf_values[key]
            tfidf_value_of_word_in_current_document = tfidf_train_vectors[doc_index,index_of_key_in_vocab_of_tfidf_vectorizer]
            term_frequency_of_word_in_current_document = tfidf_value_of_word_in_current_document/idf_of_key
            dict_of_words_in_query_and_term_frequencies_in_a_document.update({key:term_frequency_of_word_in_current_document})
        except:
            dict_of_words_in_query_and_term_frequencies_in_a_document = dict_of_words_in_query_and_term_frequencies_in_a_document
            
    print("Term frequencies for words in document: "+str(doc_no)+" (Showing term frequencies for words which are present in query)")
    print()
    print(dict_of_words_in_query_and_term_frequencies_in_a_document)
    k = dict_of_words_in_query_and_term_frequencies_in_a_document.keys()
    v = dict_of_words_in_query_and_term_frequencies_in_a_document.values()
    plt.bar(k,v)
    plt.title("Term frequencies for words in document: "+str(doc_no)+" (Showing term frequencies for words which are present in query)")
    plt.show()
    print()


# As expected we see that all of the top 5 documents have a nonzero term frequency for important words in the search query (like diabetes,covid etc). Thus the modification gives somewhat more relevant results than cosine similarity.

# # Pros and Cons
# 
# Pros
# * TF-IDF with modified cosine similarity measure gives more relevant results to search queries as demonstrated than using simple cosine similarity.
# * TF-IDF is quite easy to understand and implement. It is also quite fast.
# 
# Cons
# * TF-IDF is a bag of words model and is unable to capture semantics of language.
# * The entropy based modification to cosine similarity is a heuristic measure based in intuition. Although it gives improved results there is no baseline to compare its performance

# # Queries for various tasks

# In[ ]:


# Task 1

questions = ["range of incubation periods of covid-19",
             "how incubation period of covid-19 varies with age",
             "how long individuals are contagious even after recovery from covid-19"]
output_file_name = "incubation_task1.csv"
for question in questions:
    search_corpus(query=question, top_n_docs=10, top_n_lines_per_doc=5,output_filename=output_file_name)
    
questions = ["asymptomatic shedding and transmission of covid-19 sars-cov-2",
             "asymptomatic shedding and transmission of covid-19 sars-cov-2 in children",
             "seasonality of transmission of covid-19",
             "role of the environment in transmission covid-19",
             "effectiveness of movement control strategies to prevent secondary transmission in health care and community settings covid-19"]
output_file_name = "transmission_task1.csv"
for question in questions:
    search_corpus(query=question, top_n_docs=10, top_n_lines_per_doc=5,output_filename=output_file_name)
    
questions = ["physical science of the coronavirus sars-cov-2 charge distribution",
             "physical science of the coronavirus sars-cov-2 adhesion to hydrophilic and hydrophobic surfaces",
             "persistence and stability of coronavirus sars-cov-2 in nasal discharge",
             "persistence and stability of coronavirus sars-cov-2 in sputum",
             "persistence and stability of coronavirus sars-cov-2 in urine",
             "persistence and stability of coronavirus sars-cov-2 in fecal matter",
             "persistence and stability of coronavirus sars-cov-2 in blood",
             "persistence of coronavirus sars-cov-2 on inanimate surfaces"]
output_file_name = "persistence_task1.csv"
for question in questions:
    search_corpus(query=question, top_n_docs=10, top_n_lines_per_doc=5,output_filename=output_file_name)

questions = ["natural history of the coronavirus sars-cov-2",
             "shedding of coronavirus sars-cov-2 from an infected person",
             "implementation of diagnostics and products to improve clinical processes for covid-19"]

output_file_name = "diagnostics_task1.csv"
for question in questions:
    search_corpus(query=question, top_n_docs=10, top_n_lines_per_doc=5,output_filename=output_file_name)

questions = ["disease models including animal models for infection covid-19",
             "disease models including animal models for transmission covid-19"]
output_file_name = "disease_models_task1.csv"
for question in questions:
    search_corpus(query=question, top_n_docs=10, top_n_lines_per_doc=5,output_filename=output_file_name)

questions = ["tools and studies to monitor phenotypic change sars-cov-2",
             "tools and studies to monitor potential adaptation sars-cov-2"]
output_file_name = "tools_task1.csv"
for question in questions:
    search_corpus(query=question, top_n_docs=10, top_n_lines_per_doc=5,output_filename=output_file_name)
    
questions = ["immune response and immunity against covid-19"]
output_file_name = "immunity_task1.csv"
for question in questions:
    search_corpus(query=question, top_n_docs=10, top_n_lines_per_doc=5,output_filename=output_file_name)
    
questions = ["effectiveness of personal protective equipment (PPE) covid-19",
             "usefulness of personal protective equipment (PPE) to reduce risk of transmission in health care and community settings"]
output_file_name = "ppe_task1.csv"
for question in questions:
    search_corpus(query=question, top_n_docs=10, top_n_lines_per_doc=5,output_filename=output_file_name)


# In[ ]:


#Task-2

questions = ["Smoking risk factors covid-19",
             "pre-existing pulmonary disease risk factors covid-19",
             "whether co-existing respiratory or viral infections make the sars-cov-2 virus more transmissible or virulent",
             "risk factors diabetes covid-19",
             "risk factors hypertension covid-19",
             "risk factors cardiac problems covid-19",
             "neonates and pregnant women risk factors covid-19",
             "socio-economic and behavioral risk factors covid-19"]
output_file_name = "risk_factors_task2.csv"
for question in questions:
    search_corpus(query=question, top_n_docs=10, top_n_lines_per_doc=5,output_filename=output_file_name)
    
questions = ["transmission dynamics of the sars-cov-2 the basic reproductive number",
             "incubation period of covid-19",
             "serial interval covid-19",
             "modes of transmission of sars-cov-2 covid-19",
             "environmental factors in transmission sars-cov-2 covid-19",]
output_file_name = "transmission_dynamics_task2.csv"
for question in questions:
    search_corpus(query=question, top_n_docs=10, top_n_lines_per_doc=5,output_filename=output_file_name)
    
questions = ["risk of fatality among symptomatic hospitalized patients covid-19",
             "high risk patient groups covid-19",
             "susceptibility of various populations covid-19"]
output_file_name = "severity_of_disease_task2.csv"
for question in questions:
    search_corpus(query=question, top_n_docs=10, top_n_lines_per_doc=5,output_filename=output_file_name)

questions = ["public health mitigation measures that could be effective for control covid-19 sars-cov-2"]

output_file_name = "mitigation_task2.csv"
for question in questions:
    search_corpus(query=question, top_n_docs=10, top_n_lines_per_doc=5,output_filename=output_file_name)


# In[ ]:


# Task-3

questions = ["real-time tracking of whole genomes sars-cov-2",
             "coordinating the rapid dissemination of information for the development of diagnostics and therapeutics and to track variations of sars-cov-2 over time",
             "geographic distribution and genomic differences of sars-cov-2",
             "number of strains of sars-cov-2 in circulation",
             "multi-lateral agreements such as the nagoya protocol sars-cov-2"]
output_file_name = "virus_genetics_task3.csv"
for question in questions:
    search_corpus(query=question, top_n_docs=10, top_n_lines_per_doc=5,output_filename=output_file_name)
    
questions = ["evidence that livestock could be infected with sars-cov-2 field surveillance genetic sequencing receptor binding",
             "whether livestock can serve as a reservoir of sars-cov-2",
             "evidence of whether farmers are infected with sars-cov-2 covid-19 and whether they could have played a role in the origin",
             "surveillance of mixed wildlife livestock farms for SARS-CoV-2 and other coronaviruses in southeast asia",
             "Experimental infections to test host range for sars-cov-2",
             "evidence of continued spill-over of sars-cov-2 from animals to humans",
             "socioeconomic and behavioral risk factors for spill-over of sars-cov-2 from animals to humans"
             ]
output_file_name = "livestock_task3.csv"
for question in questions:
    search_corpus(query=question, top_n_docs=10, top_n_lines_per_doc=5,output_filename=output_file_name)

questions = ["sustainable risk reduction strategies for covid-19 sars-cov-2"]
output_file_name = "sustainable_risk_reduction_task3.csv"
for question in questions:
    search_corpus(query=question, top_n_docs=10, top_n_lines_per_doc=5,output_filename=output_file_name)


# In[ ]:


# Task-4

questions = ["effectiveness of drugs being developed and tried to treat COVID-19 patients",
             "clinical and bench trials to investigate viral inhibitors against COVID-19 such as naproxen, clarithromycin, and minocycline",
             "methods evaluating potential complication of Antibody Dependent Enhancement (ADE) in vaccine recipients covid-19",
             "exploration of use of best animal models and their predictive value for a human vaccine covid-19",
             "capabilities to discover  therapeutics for covid-19",
             "clinical effectiveness studies to discover therapeutics, to include antiviral agents covid-19",
             "models  to aid decision makers in determining how to prioritize and distribute scarce, newly proven therapeutics against covid-19",
             "identifying approaches for expanding production capacity of vaccines and therapeutics against covid-19 to ensure equitable and timely distribution to populations in need",
             "efforts targeted at a universal coronavirus vaccine",
             "efforts to develop animal models and standardize challenge studies covid-19",
             "efforts to develop prophylaxis clinical studies and prioritize in healthcare workers covid-19",
             "approaches to evaluate risk for enhanced disease after vaccination covid-19",
             "assays to evaluate vaccine immune response covid-19",
             "process development for vaccines alongside suitable animal models covid-19"]
output_file_name = "vaccines_and_therapeutics_task4.csv"
for question in questions:
    search_corpus(query=question, top_n_docs=10, top_n_lines_per_doc=5,output_filename=output_file_name)
    


# In[ ]:


# Task-5

questions = ["resources to support skilled nursing facilities and long term care facilities",
             "mobilization of surge medical staff to address shortages in overwhelmed communities",
             "application of regulatory standards (EUA, CLIA) and ability to adapt care to crisis standards of care level",
             "approaches for encouraging and facilitating the production of elastomeric respirators",
             "best telemedicine practices, barriers and faciitators",
             "guidance on the simple things people can do at home to take care of sick people and manage covid-19",
             "use of AI or artificial intelligence in real-time health care delivery to evaluate interventions, risk factors, and outcomes",
             "innovative solutions and technologies in hospital flow and organization",
             "supply chain management to enhance capacity, efficiency, and outcomes"]
output_file_name = "medical_resources_task5.csv"

for question in questions:
    search_corpus(query=question, top_n_docs=10, top_n_lines_per_doc=5,output_filename=output_file_name)

questions = ["age adjusted mortality data for Acute Respiratory Distress Syndrome (ARDS) for viral etiologies",
             "extracorporeal membrane oxygenation (ECMO) outcomes data of COVID-19 patients",
             "outcomes data for COVID-19 after mechanical ventilation adjusted for age",
             "frequency, and course of extrapulmonary manifestations of COVID-19",
             "cardiomyopathy and cardiac arrest in covid-19",
             "oral medications that might potentially work covid-19",
             "determine adjunctive and supportive interventions that can improve the clinical outcomes of infected patients (steroids, high flow oxygen) covid-19"]
output_file_name = "clinical_charecterization_of_virus_task5.csv"
for question in questions:
    search_corpus(query=question, top_n_docs=10, top_n_lines_per_doc=5,output_filename=output_file_name)


# In[ ]:


# Task-6
questions = ["guidance on ways to scale up NPIs or non pharmaceutical interventions to handle covid-19 outbreak",
             "establish funding, infrastructure and authorities to give us time to enhance our health care delivery system capacity to respond to an increase in cases",
             "rapid assessment of the likely efficacy of school closures, travel bans, bans on mass gatherings of various sizes, and other social distancing approaches",
             "methods to control the spread of covid-19 in communities",
             "barriers to compliance and how these vary among different populations",
             "models of potential interventions to predict costs and benefits that take account of such factors as race, income, disability, age, geographic location, immigration status, housing status, employment status, and health insurance status",
             "Policy changes necessary to enable the compliance of individuals with limited resources and the underserved with NPIs or non pharmaceutical interventions",
             "why people fail to comply with public health advice, even if they want to do so (social or financial costs may be too high)",
             "economic impact of covid-19 outbreak or any pandemic",
             "identifying policy that lessen/mitigate risks to critical government services, food distribution and supplies, access to critical household supplies, and access to health diagnoses,treatment"]
output_file_name = "non_pharmaceutical_interventions_task6.csv"
for question in questions:
    search_corpus(query=question, top_n_docs=10, top_n_lines_per_doc=5,output_filename=output_file_name)


# In[ ]:


# Task-7
questions = ["immediate policy recommendations on mitigation measures covid-19",
             "denominators for testing covid-19",
             "mechanism for rapidly sharing testing data,including demographics for covid-19",
             "sampling methods to determine asymptomatic disease (use of serosurveys such as convalescent samples)",
             "sampling methods for early detection of disease (use of screening of neutralizing antibodies such as ELISAs)",
             "efforts to increase capacity on existing diagnostic platforms and tap into existing surveillance platforms",
             "leverage universities and private laboratories for testing purposes covid-19",
             "development of a point-of-care test (like a rapid influenza test) and rapid bed-side tests",
             "efforts to track the evolution of the virus (genetic drift or mutations)",
             "use of diagnostics such as host response markers (cytokines) to detect early disease or predict progression",
             "policies and protocols for screening and testing",
             "policies to mitigate the effects on supplies associated with mass testing",
             "barriers to developing and scaling up new diagnostic tests",
             "New platforms and technology (CRISPR) in covid-19",
             "rapid sequencing and bioinformatics to target regions of genome of sars-cov-2",
             "explore capabilities for distinguishing naturally-occurring pathogens from intentional"]
output_file_name = "diagnostics_and_surveilance_task7.csv"
for question in questions:
    search_corpus(query=question, top_n_docs=10, top_n_lines_per_doc=5,output_filename=output_file_name)


# In[ ]:


# Task-8
questions = ["existing ethical principles and standards to salient issues in COVID-19",
             "support sustained education, access, and capacity building in the area of ethics",
             "physical and psychological health of healthcare workers for Covid-19 patients",
             "underlying drivers of fear, anxiety and stigma that fuel misinformation and rumor, particularly through social media",
             "identification of the secondary impacts of prevention control measures covid-19"]
output_file_name = "ethical_considerations_task8.csv"
for question in questions:
    search_corpus(query=question, top_n_docs=10, top_n_lines_per_doc=5,output_filename=output_file_name)


# In[ ]:


# Task-9
questions = ["Methods for coordinating data-gathering with standardized nomenclature covid-19",
             "Understanding and mitigating barriers to information-sharing covid-19",
             "Integration of federal state local public health surveillance systems",
             "modes of communicating with target high-risk populations (elderly, health care workers)",
             "Risk communication and guidelines that are easy to understand and follow covid-19",
             "Misunderstanding around containment and mitigation covid-19",
             "Understanding coverage policies (barriers and opportunities) related to testing, treatment, and care covid-19",
             "Measures to reach marginalized and disadvantaged populations covid-19",
             "mitigating threats to incarcerated people from COVID-19"]
output_file_name = "info_sharing_and_intersectoral_collaboration_task9.csv"
for question in questions:
    search_corpus(query=question, top_n_docs=10, top_n_lines_per_doc=5,output_filename=output_file_name)

