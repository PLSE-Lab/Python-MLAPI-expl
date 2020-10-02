import pandas as pd
import numpy as np
import os
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import string

# Description: py script takes abstracts that have been pre-processed from the JSON files in the challenge dataset. 
# It uses simple tfidf and cosine similarity to rank article abstracts according query terms. 
# Currently creates a list of lists of the top 100 paper IDs for each query. Also prints out ranked output of the IDs for each query term.


def getData():
    global df # globally define dataframe variable
    df = pd.read_csv('../input/titlesabstracts/titles_abstracts.csv') # read in df from csv file
    df = df.dropna(subset = ['abstract']) # drop all na values from the 'abstract' column (to improve in the future, this step could be avoided by taking care of it when creating the csv)

# Implementing tfidf with abstracts
def organize():
    global vectorizer # globally define the tfidf vectorizer from sklearn
    global X # globally define the fitted corpus tfidf object
    vectorizer = TfidfVectorizer(stop_words = 'english') # run the tfidf vectorizer with stop words, even though these were likely taken care of during pre-processing steps)
    X = vectorizer.fit_transform(df['abstract']) # fit on the abstracts of the corpus

# Querying
def retrieve(q):
    global results # globally define the list of lists of query results that will be printed out
    results = [] # instantiate empty list to hold the lists of results   
    for query in q: # for loop that goes through each query in q
        query_vec = vectorizer.transform([query]) # transform query according to the fitted vectorizer object from above
        query_results = cosine_similarity(X, query_vec).reshape((-1,)) # get similarity measure, which is cosine similarity in this case; reshape it for easier future use
        query_results = [df.iloc[i,1] for i in (-query_results).argsort()[:100]] # list comprehension that takes the indices of the first 100 papers that had highest cosine similarity with query and finds them in the paper dataframe
        results.append(query_results) # appends the 100 results list to the list of lists for all queries
        for i in range(len(query_results)):
            print(str(query)+'\t'+str(i+1)+'\t'+str(query_results[i]))

getData()
organize()
q = ['coronavirus origin','coronavirus response to weather changes','coronavirus immunity']
retrieve(q)
