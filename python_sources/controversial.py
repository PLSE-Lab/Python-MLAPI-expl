# Lets see if we can predict controversial posts
# Looks like we have a highly imbalanced problem - controversial posts are very rare
import sqlite3
import pandas as pd
import numpy as np
import scipy.sparse

import codecs
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

    
def predict(model, train_data, train_target, test_data, test_target):
    if(model == "NB"):
        classifier = MultinomialNB()
    elif(model == "SVM"):
        classifier = LinearSVC()
    elif(model == "majority"):
        classifier = DummyClassifier(strategy="most_frequent")
    classifier.fit(train_data, train_target)
    predict = classifier.predict(test_data)
    
    print(model)
    print("Accuracy: %f"%(accuracy_score(test_target,predict)))
    print("Precision: %f"%(precision_score(test_target,predict)))
    print("Recall: %f"%(recall_score(test_target,predict)))
    

def wordsInControversialPosts():
    #Which words appear most in controversial posts?
    controversialPosts = data.loc[data['controversiality'] == 1]
    cv = CountVectorizer(stop_words="english",min_df=2,ngram_range=(1,3))
    cv.fit_transform(controversialPosts['body'])
    return cv.vocabulary_

sql_conn = sqlite3.connect('../input/database.sqlite')
sql_cmd = "Select subreddit, body, controversiality From May2015 ORDER BY Random() LIMIT 500000" # 

data = pd.read_sql(sql_cmd, sql_conn)
utf8 = [codecs.encode(body,'utf-8') for body in data.body]
data.body = pd.Series(utf8)
#print(data.describe())

# Use words that appear in controversial posts because it's imbalanced
vocab = wordsInControversialPosts()
vectorizer = CountVectorizer(vocabulary=vocab,binary=True)
body_terms = vectorizer.fit_transform(data['body'])

# Add subreddit as a feature
dv = DictVectorizer()
subredditDict = data[['subreddit']].T.to_dict().values()
subredditFeatures = dv.fit_transform(subredditDict)
features = scipy.sparse.hstack([body_terms, subredditFeatures])

# LSA did not help, back to predicting majority class
#svd = TruncatedSVD(n_components=100)
#body_terms = svd.fit_transform(body_terms)

text_train, text_test, target_train, target_test = train_test_split(features, data['controversiality'])
    
predict("NB", text_train, target_train, text_test, target_test)
predict("SVM", text_train, target_train, text_test, target_test)
predict("majority", text_train, target_train, text_test, target_test)
