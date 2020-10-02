#!/usr/bin/env python
# coding: utf-8

# # Task
# Given M documents compute the term-term relevance and for output return the term pairs and their similarity score descending
# 
# To calculate term-term relevance:
# 1.  Calculate tfidf of every term
# 2.  Compute and sort term-term relevance between a term and other terms
# 

# # Term Frequency Inverse Document Frequency
# ![Term Frequency functions](https://upload.wikimedia.org/wikipedia/commons/0/05/Plot_IDF_functions.png)
# TF-IDF is a numerial statistic which symbolizes how important a word is in a document from
# a collection of documents. In our context, we are using it first to determine each term's relevance within our collection of documents then using it again to determine similarity scores between term pairs

# In[ ]:


# Need to install pyspark bc it is not available
get_ipython().system('pip install pyspark')


# In[ ]:


# Libraries
import pandas as pd
import math
from pyspark import SparkContext, SparkConf
#import os
#print(os.listdir("../input"))


# In[ ]:


def perform_check(result,print_limit):
    """
    Takes result of Mapping to a RDD
    and prints a certain amount of lines
    """
    limit = print_limit
    count = 0
    for x in result.collect():
        count = count + 1
        print(x)
        if count == limit:
            break


# In[ ]:


def get_length(result):
    """
    Takes result of Mapping to a RDD
    and prints a certain amount of lines
    """
    count = 0
    for x in result.collect():
        count = count + 1
    print(count)
    return count


# In[ ]:


# A Quick Look At Our Data
source = pd.read_table("../input/project2_data.txt",header=None)
source[0] = source[0].str.split()
source["doc"] = source[0].apply(lambda x: x[0])
source["terms"] = source[0].apply(lambda x: x[1:])
source.head()


# In[ ]:


# Set app name and master for spark
appName = "TFIDF"
master = "local"
conf = SparkConf().setAppName(appName).setMaster(master)
sc = SparkContext(conf=conf)


# In[ ]:


# Convert our given text file for spark
data_file_path = "../input/project2_data.txt"
project2_data = sc.textFile(data_file_path)


# In[ ]:


key_value = project2_data.map(lambda x: x.split())


# In[ ]:


# Get the total number of documents in corpus
# Necessary for our TF-IDF calculation
TOTAL_DOCS = get_length(key_value)


# In[ ]:


perform_check(key_value,1)


# In[ ]:


def Filter_Terms(x):
    """
    Filters all extraneous terms within the list
    of the terms that we have to consider
    In this case we want to get the words
    with the following formats:
    (1) gene_word_gene
    (2) disease_word_disease
    """
    relevant_terms = []
    for word in x[1:]:
        if (word.startswith("gene_") and word.endswith("_gene")):
            relevant_terms.append((word,x[0]))
        if (word.startswith("disease_") and word.endswith("_disease")):
            relevant_terms.append((word,x[0]))
    return relevant_terms


# In[ ]:


word_count = key_value.flatMap(lambda x: Filter_Terms(x)).map(lambda x: (x, 1)).reduceByKey(lambda x, y: x + y)


# In[ ]:


perform_check(word_count,5)


# In[ ]:


def Get_All_Pairs(x):
    """
    Gathers all word doc pairs and
    resturctures so we have all words
    followed by their counts for each doc
    """
    docid_word_pair = x[0]
    word_count = x[1]
    unique_word = docid_word_pair[0]
    docid = docid_word_pair[1]
    return (docid, list((unique_word,word_count)))


# In[ ]:


doc_word_counts = word_count.map(lambda x: Get_All_Pairs(x)).cache().reduceByKey(lambda x, y: x + y)


# In[ ]:


perform_check(doc_word_counts,5)


# In[ ]:


def CreateTuple(x):
    """
    Gathers word and their counts
    from the list and puts
    them in tuples with the format:
    (word,word_count)
    This way a word and it's count
    is explicit
    """
    docid = x[0]
    converted_list = []
    tuple_list = x[1]
    for i in range(0, len(tuple_list), 2):
        converted_list.append((tuple_list[i], tuple_list[i+1]))
    return (docid, converted_list)


# In[ ]:


tuple_result = doc_word_counts.map(lambda x: CreateTuple(x))


# In[ ]:


perform_check(tuple_result,1)


# In[ ]:


def WordCountPerDoc(x):
    """
    Gets word and its document pair and reports 
    the occurences of the word in the document and the total
    number of words in the document in the following
    format:
    ((word,doc),(word occurences,total # of words in doc))
    """
    list_ = []
    docid = x[0]
    list_of_tuples = x[1]
    number_of_terms_in_doc = 0
    for each_tuple in list_of_tuples:
        number_of_terms_in_doc += each_tuple[1]
    for each in list_of_tuples:
        unique_word = each[0]
        word_occurences = each[1]
        list_.append(
            (
                (unique_word, docid),
                (word_occurences, number_of_terms_in_doc)
            )
        )
    return list_


# In[ ]:


word_count_per_doc= tuple_result.flatMap(lambda x: WordCountPerDoc(x))


# In[ ]:


perform_check(word_count_per_doc,5)


# In[ ]:


def All_Doc_Word_Count_Pairs(x):
    """
    Get all word counts and the total
    counts for each document within our
    database
    """
    word_and_doc = x[0]
    word = word_and_doc[0]
    docid = word_and_doc[1]
    word_count_and_total_word_in_doc = x[1]
    word_count = word_count_and_total_word_in_doc[0]
    total_word_count = word_count_and_total_word_in_doc[1]
    return (word, (docid, word_count, total_word_count)) 


# In[ ]:


word_per_doc = word_count_per_doc.map(lambda x: All_Doc_Word_Count_Pairs(x)).cache().reduceByKey(lambda x, y: x + y)


# In[ ]:


perform_check(word_per_doc,1)


# In[ ]:


def All_Word_Count_Pairs(x):
    """
    From list forms tuples
    of word and every document it appears
    and the total number words in that document
    """
    list_ = []
    word = x[0]
    tuple_list = x[1]
    for i in range(0,len(tuple_list),3):
        list_.append((tuple_list[i], tuple_list[i+1], tuple_list[i+2]))
    return (word, list_)


# In[ ]:


all_doc_word_counts= word_per_doc.map(lambda x: All_Word_Count_Pairs(x))


# In[ ]:


perform_check(all_doc_word_counts,1)


# In[ ]:


def CountDocsPerWord(x):
    """
    Determines the number of documents
    a term appears in
    """
    list_ = []
    docsPerWord = 0
    word = x[0]
    tuple_list = x[1]
    for each in tuple_list:
        docsPerWord += 1
    for each in tuple_list:
        docid = each[0]
        word_count = each[1]
        total_w_count = each[2]
        list_.append(
            (
                (word, docid),
                (word_count, total_w_count, docsPerWord)
            )
        )
    return list_


# In[ ]:


docs_per_word_result = all_doc_word_counts.flatMap(lambda x: CountDocsPerWord(x))


# In[ ]:


perform_check(docs_per_word_result,5)


# In[ ]:


def TFIDF(x,total_docs):
    """
    Calculates the term-frequency inverse document frequency
    for each term
    """
    term_name = x[0][0]
    second_tuple = x[1]
    term_word_count = second_tuple[0]
    all_word_count = second_tuple[1]
    docs_with_term = second_tuple[2]
    term_frequency = term_word_count / all_word_count
    inverse_doc_frequency = math.log(total_docs/docs_with_term)
    tfidf = term_frequency * inverse_doc_frequency
    return (term_name, tfidf)    


# In[ ]:


tfidf_result = docs_per_word_result.map(lambda x: TFIDF(x,TOTAL_DOCS)).groupByKey()


# In[ ]:


perform_check(tfidf_result,3)


# In[ ]:


# Gathers all calculated tfidf by
# term
tfidf = tfidf_result.cache().map(lambda x: (x[0], list(x[1])))


# In[ ]:


perform_check(tfidf,1)


# In[ ]:


def GetQueryVector(x, query_term):
    """
    Will return the vector for the query term
    """
    if x[0] == query_term:
        return True
    return False


# In[ ]:


query_term = 'gene_nmdars_gene'


# In[ ]:


query_vector = tfidf.filter(lambda x: GetQueryVector(x, query_term))


# In[ ]:


perform_check(query_vector,1)


# In[ ]:


def FilterOutQuery(x):
    """
    Filter out query terms
    """
    if x[0] == query_term:
        return False
    else:
        return True


# In[ ]:


cartesian_filter = tfidf.filter(lambda x: FilterOutQuery(x)).cache().cartesian(query_vector)


# In[ ]:


perform_check(cartesian_filter,1)


# In[ ]:


def SemanticSimilarity(x):
    """
    Calculates the Semantic Similarity
    for all term-term pairs
    """
    A_vector = x[0][1]
    B_vector = x[1][1]
    A_denominator = 0
    B_denominator = 0
    A_B_denominator = 0
    A_B_numerator = 0
    semantic_similarity = 0
    
    #calculates the denominator part for the A vector 
    for i in range(0, len(A_vector), 1):
        A_denominator += A_vector[i] * A_vector[i]

    A_denominator = math.sqrt(A_denominator)
    
    #calculates the denominator part for the B vector
    for i in range(0, len(B_vector), 1):
        B_denominator += B_vector[i] * B_vector[i]

    B_denominator = math.sqrt(B_denominator)
    
    #makes the vectors equal sized in order 
    #to allow multiplication of both vectors
    if len(B_vector) <= len(A_vector):
        difference = len(A_vector) - len(B_vector)
        for i in range(difference):
            B_vector.append(0)
    elif len(A_vector) <= len(B_vector):
        difference = len(B_vector) - len(A_vector)
        for i in range(difference):
            A_vector.append(0)

    #multiplies each element of A and B to find the numerator 
    #of the semantic similarity formula
    for i in range(len(A_vector)):
        A_B_numerator += A_vector[i] * B_vector[i]
    
    #calculates the denominator of the semantic similarity formula
    A_B_denominator = A_denominator * B_denominator
    
    #output is ((A-term, B-term), semantic similarity)
    return (x[1][0], x[0][0]), A_B_numerator/A_B_denominator


# In[ ]:


semantic_result = cartesian_filter.map(lambda x: SemanticSimilarity(x)).map(lambda x: (x[1], x[0])).sortByKey(False).map(lambda x: (x[1], x[0]))


# In[ ]:


perform_check(semantic_result,1)


# In[ ]:


final_output = semantic_result.collect()


# In[ ]:


len(final_output)


# In[ ]:


final_output[:5]


# In[ ]:


def Write_To_File(x):
    """
    Taking our final output
    and writes it file
    """
    list_ = x
    file = open("Final_Output.txt", "w")
    file.write('\n'.join('%s %s' % x for x in list_))


# In[ ]:


Write_To_File(final_output)


# Authors:
# # [Gael Blanchard](https://github.com/gaelblanchard)
# # [Lloyd Massiah](https://github.com/lazypassion)
