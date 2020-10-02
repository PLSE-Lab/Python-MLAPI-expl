#!/usr/bin/env python
# coding: utf-8

# # Sarcasm Detection Using XGBoost and Logistic Regression

# In[ ]:


########
#some eda from https://www.kaggle.com/danofer/loading-sarcasm-data
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from subprocess import check_output
from matplotlib import pyplot as plot
import os
import numpy as np
import xgboost
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import preprocessing, metrics, svm


# ##Data Exploration
# 
# Repeating a portion of the data exploration from YURY KASHNITSKY: https://www.kaggle.com/kashnitsky/a4-demo-sarcasm-detection-with-logit-solution

# In[ ]:


train = pd.read_csv("../input/train-balanced-sarcasm.csv")
print(train.shape)
print(train.columns)

#drop rows that have missing comments
train.dropna(subset=['comment'], inplace=True)

# Parse UNIX epoch timestamp as datetime: 
train.created_utc = pd.to_datetime(train.created_utc,infer_datetime_format=True) # Applies to original data , which had UNIX Epoch timestamp! 

train.describe()

########
train['label'].hist() #50 50 split

##see a sample of comments
train['comment'].sample(10)
train[train.label == 1]["comment"].sample(10).tolist()


# In[ ]:


#how many comments are in each subreddit?
train.groupby(["subreddit"]).count()["comment"].sort_values()

#learn more about the subreddits and the frequency of sarcastic labels
sub_df = train.groupby('subreddit')['label'].agg([np.size, np.mean, np.sum])
sub_df.sort_values(by='sum', ascending=False).head(10)
sub_df[sub_df['size'] > 1000].sort_values(by='mean', ascending=False).head(10)

#learn more about authors and the frequency of sarcastic labels
author_df = train.groupby('author')['label'].agg([np.size, np.mean, np.sum])
author_df.sort_values(by='sum', ascending=False).head(10)
author_df[author_df['size'] > 250].sort_values(by='mean', ascending=False).head(10)


# In[ ]:


#split the df into training and validation parts
train_texts, valid_texts, y_train, y_valid =         train_test_split(train['comment'], train['label'], random_state=17)
        
print(train_texts.shape, valid_texts.shape, y_train.shape, y_valid.shape)

'''
#take small sample for testing
train_texts_small = train_texts.sample(600, random_state=27)
y_train_small = y_train.sample(600, random_state=27)
valid_texts_small = valid_texts.sample(600, random_state=27)
y_valid_small = y_valid.sample(600, random_state=27)
'''


# ##Preprocessing with NLTK

# Porter Stemming was inspired by the following source: 
# https://medium.com/@chrisfotache/text-classification-in-python-pipelines-nlp-nltk-tf-idf-xgboost-and-more-b83451a327e0
# 
# The classifier function's source is the following: 
# https://www.analyticsvidhya.com/blog/2018/04/a-comprehensive-guide-to-understand-and-implement-text-classification-in-python/

# In[ ]:


get_ipython().run_cell_magic('time', '', '#Consider porter stemming\nimport nltk, re\ndef Tokenizer(str_input):\n    words = re.sub(r"[^A-Za-z0-9\\-]", " ", str_input).lower().split()\n    #words = re.search(r\'\\w{1,}\',str_input).lower().split()\n    porter_stemmer=nltk.PorterStemmer()\n    words = [porter_stemmer.stem(word) for word in words]\n    return words\n\n\ndef train_model(classifier, feature_vector_train, label, feature_vector_valid, label_valid,is_neural_net=False):\n    # fit the training dataset on the classifier\n    classifier.fit(feature_vector_train, label)\n    \n    # predict the labels on validation dataset\n    predictions = classifier.predict(feature_vector_valid)\n    \n    if is_neural_net:\n        predictions = predictions.argmax(axis=-1)\n    \n    return metrics.accuracy_score(predictions, label_valid)\n    return metrics.classification_report(predictions, label_valid)\n\n\n#count vectors\n\'\'\'\ncount_vect = CountVectorizer(analyzer=\'word\', token_pattern=r\'\\w{1,}\', ngram_range=(1, 3), max_features=50000, min_df=2, lowercase=True, max_df=0.9)\ncount_vect.fit(train[\'comment\'])\n\n#count vectors\n# transform the training and validation data using count vectorizer object\nxtrain_count =  count_vect.transform(train_texts)\nxvalid_count =  count_vect.transform(valid_texts)\n\'\'\'\n\n# word and n-gram level tf-idf\n#tfidf_vect = TfidfVectorizer(analyzer=\'word\', token_pattern=r\'\\w{1,}\', max_features=50000, ngram_range=(1,3), min_df=2, lowercase=True, max_df=0.9)\ntfidf_vect = TfidfVectorizer(analyzer=\'word\', tokenizer=Tokenizer, max_features=50000, ngram_range=(1,2), min_df=2, lowercase=True, max_df=0.95)\ntfidf_vect.fit(train[\'comment\'])\nxtrain_tfidf =  tfidf_vect.transform(train_texts)\nxvalid_tfidf =  tfidf_vect.transform(valid_texts)')


# ##XGBoost Classifier

# In[ ]:


get_ipython().run_cell_magic('time', '', '#try XGBoost on word- and ngram-level vectors\n#accuracy = train_model(xgboost.XGBClassifier(), xtrainSVD, y_train, xvalidSVD)\naccuracy = train_model(xgboost.XGBClassifier(n_estimators=400), xtrain_tfidf.tocsc(), y_train, xvalid_tfidf.tocsc(), y_valid)\nprint("Xgb, WordLevel TF-IDF: ", accuracy)\n\n#try XGBoost on count vectors\n#accuracy = train_model(xgboost.XGBClassifier(n_estimators=400), xtrain_count.tocsc(), y_train, xvalid_count.tocsc())\n#print("Xgb, Count Vectors: ", accuracy)')


# ##Logistic Regression Classifier

# In[ ]:


get_ipython().run_cell_magic('time', '', '#logistic regression\naccuracy = train_model(LogisticRegression(solver=\'lbfgs\', random_state=17, max_iter=1000), xtrain_tfidf, y_train, xvalid_tfidf, y_valid)\nprint("Logistic Regression: ", accuracy)\n')

