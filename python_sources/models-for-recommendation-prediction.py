#!/usr/bin/env python
# coding: utf-8

# # Analyzing Sentiment and Predicting Recommendation and Rating

# 1. [Looking at the Data](#looking-at-the-data)
# 2. [Text Preprocessing](#text-preprocessing)
# 3. [Creating a Model Pipeline](#creating-a- model-pipeline)
# 4. [Testing the Model](#testing-the-model)
# 5. [Tweaking the Model and Data](#tweaking-the-model-and-data)

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
from __future__ import unicode_literals
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import nltk
import re
import string

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.metrics import classification_report,confusion_matrix, roc_auc_score

from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# <a id="looking-at-the-data"></a>
# ## Looking at the Data
# ### Creating a Model to Predict Recommendation of Product
# #### For each review, the customer was asked if they would recommend the product. First, we will be taking only the text reviews to predict recommendation.
# 

# In[ ]:


file_path = "../input/Womens Clothing E-Commerce Reviews.csv"
df = pd.read_csv(file_path,index_col = 0 )
df.info()
df.groupby('Recommended IND').describe()


# <a id="text-preprocessing"></a>
# ## Text Preprocessing
# 
# the column 'Review Text' stores object values rather than strings.
# 
# There are also some null values to be taken care of.

# In[ ]:


# df.info()
# print(df.shape) # (23486, 10)
df = df.dropna(subset=['Review Text'])
df.isnull().sum() # no more null Review Text
rec = df.filter(['Review Text','Recommended IND'])
rec.columns = ['text','target']


# ### General Preprocessing Steps
# We will turn the words from the reviews into something that the computer can comprehend.
# Before we vectorize the words according to frequency, the words will require some preprocessing to increase the quality of words that will make up our vocabulary. 
# 
# The basic ones are:
# 1. Change all letters to **lowercase letter** so that Word, word, WORD are all recognized as the same entity.
# 2. Remove all **digits and punctuations** (for the sake of simplicity in this case)
# 3. Tokenize each word and **remove stop words**. Stop words are words that are the most commonly used, often providing no useful insight.
# 4. **Stemming each word**. There are many forms of one word (organize, organization, organizing) that all hold the same meaning. We want the machine to recognize these different forms as one word.
#     * There are two ways to achieve this: stemming and lemmatization.
#         * Stemming is faster, following an algorithm to chop off ends of words.
#             * **PorterStemmer(), LancasterStemmer(), SnowballStemmer(), **
#         * Lemmatization is slower but uses a vocabulary and morphologically analyzes the word to provide the base form (lemma) of the word. The only issue is that lemmatiazation also requires defined parts-of-speech.
#             * **use WordNetLemmatizer()**
# 5. Remove all words that consist of only one letter. They are likely not meaningful and therefore we will take them out.
# 
# I believe that punctuation and looking at capitalized words can be good indication for varying levels of emotions, both negative and positive.
# But for now, I will only be looking at words as indicators of positive or negative opinions, testing it by checking against whether the reviewer recommended the item or not.
# 
# For this, I will also not be spellchecking, which means that words like "sooo" are not going to be aligned with the correct spelling of "so." Spelling correction could help improve accuracy, so it may be considered in the future.

# In[ ]:


stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
# remove all tokens that are stop words or punctuations and perform lemmatization
def prep_clean(text):
    text = text.lower()
    text = re.sub(r'\d+','',text)
    tokens = word_tokenize(text)
    words = [token for token in tokens if not token in stop_words]
    words = [stemmer.stem(word) for word in words]
    words = [word for word in words if not word in string.punctuation]
    words = [word for word in words if len(word) > 1]
    return words


# We clean up the text, and now we should do something with these words.

# <a id = "creating-a-model-pipeline"></a>
# ## Creating a Model Pipeline
# 
# Now that we have a list of preprocessed tokens, let's look at different ways we can transform the words into something the model could use.
# 
# First, the most simple thing to do is to build vocabulary of words by word count.
# We can create a bag of words model using CountVectorizer

# Once we create our bow, we want to transform this count matrix into a normalized term-frequency representation.
# tf-idf is the term-ferquency times inverse document-frequency. It is a common term weighting scheme in information retrieval as well as document classification.
# 
# This will help scale down the impact of tokens that occur frequently in the corpus since features that occur n small fractions are more informative.

# We are only comparing the review text and the recommendation values
# 
# I want to explore several models to achieve this.
# 
# First, I will be looking at the Multinoial Naive Bayes model.
# By taking the word frequency for each review, the model will be able to learn which words are frequently associated with recommendation.
# 
# The model is naive because it takes an assumption that each word is independent of one another, therefore looking at individual words rather than the entire sentence.
# 
# I will explain some ways to improve the model along the way.

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(rec['text'], rec['target'], test_size=0.2)


# In[ ]:


# build the model and test its accuracy
def model(mod, name, X_train, X_test, y_train, y_test):
    mod.fit(X_train, y_train)
    print(name)
    acc = cross_val_score(mod, X_train, y_train, scoring = "accuracy", cv=5)
    predictions = cross_val_predict(mod, X_train, y_train, cv=5)
    print("Accuracy: ", round(acc.mean(),3))
    cm = confusion_matrix(predictions, y_train)
    print("Confusion Matrix: \n", cm)
    print("Classification Report: \n", classification_report(predictions, y_train))
    print("--------")
    print(predictions[:10])
    print(y_train[:10])
# the model and all the preprocessing steps
def pipeline(bow, tfidf, model):
    return Pipeline([('bow', bow),
               ('tfidf', tfidf),
               ('classifier', model),
              ])


# A confusion matrix is an easy way to visualize the performance of this binary classification model.
# 
# It is divided to show 
# 
# 
# $\begin{array}
# {rrr}
#  & predicted: no & predicted: yes \\
# actual: no  & correct & wrong \\
# actual: yes & wrong & correct \\
# \end{array}
# $

# In[ ]:


mnb = pipeline(CountVectorizer(analyzer=prep_clean, ngram_range=(1,2)), TfidfTransformer(), MultinomialNB())
mnb = model(mnb, "Multinomial Naive Bayes", X_train, X_test, y_train, y_test)

log = pipeline(CountVectorizer(analyzer=prep_clean, ngram_range=(1,2)), TfidfTransformer(), LogisticRegression(solver='lbfgs'))
log = model(log, "Logistic Regression", X_train, X_test, y_train, y_test)

svc = pipeline(CountVectorizer(analyzer=prep_clean, ngram_range=(1,2)), TfidfTransformer(), LinearSVC())
svc = model(svc, "Linear SVC", X_train, X_test, y_train, y_test)


# <a id="testing-the-model"></a>
# ## Testing the Model
# 
# #### Multimodal Naive-Bayes
# * Accuracy:  0.821
# * Confusion Matrix:
# $\left[\begin{array}
# {rrr}
# 45 & 1 \\
# 3240 & 14826 \\
# \end{array}\right]
# $
# 
# #### Logistic Regression
# * Accuracy:  0.884
# * Confusion Matrix:
# $\left[\begin{array}
# {rrr}
# 1602 & 413 \\
# 1683 & 14414 \\
# \end{array}\right]
# $
# 
# #### Linear SVC
# * Accuracy:  0.884
# * Confusion Matrix:
# $\left[\begin{array}
# {rrr}
# 1916 & 738 \\
# 1369 & 14089 \\
# \end{array}\right]
# $
# 
# 

# We can analyze each of these to understand what they mean. Although the overall accuracy of the Logistic Regression model and the Linear SVC model is very similar, the makeup of the error is different.
# 
# We can look at false negatives and false positives.
# 
# * predicted no, but was actually yes (FALSE NEGATIVE)
#     * log-reg:1683
#     * lin-svm:1369
# * predicted yes, but was actually no (FALSE POSITIVE)
#     * log-reg:413
#     * lin-svm:738
# 
# Adding a stemming step to the preprocessor did increase the overall accuracy of each of the models. However, it also increased the false negatives.
# (The code now shows the accuracy of the pipeline including the preprocessor with stemming)

# #### Tuning the model
# 
# Some of the reviews have titles, and the titles may be able to add more to the reviews.

# In[ ]:


rec["concat"] = df["Title"].fillna('') + " "+ df["Review Text"]
X_train, X_test, y_train, y_test = train_test_split(rec['concat'], rec['target'], test_size=0.2)
mnb = pipeline(CountVectorizer(analyzer=prep_clean, ngram_range=(1,2)), TfidfTransformer(), MultinomialNB())
mnb = model(mnb, "Multinomial Naive Bayes", X_train, X_test, y_train, y_test)

log = pipeline(CountVectorizer(analyzer=prep_clean, ngram_range=(1,2)), TfidfTransformer(), LogisticRegression(solver='lbfgs'))
log = model(log, "Logistic Regression", X_train, X_test, y_train, y_test)

svc = pipeline(CountVectorizer(analyzer=prep_clean, ngram_range=(1,2)), TfidfTransformer(), LinearSVC())
svc = model(svc, "Linear SVC", X_train, X_test, y_train, y_test)

# rfc = pipeline(CountVectorizer(analyzer=prep_clean, ngram_range=(1,2)), TfidfTransformer(), LinearSVC())
# rfc = model(rfc, "Random Forest Classifier", X_train, X_test, y_train, y_test)


# Adding the title data has increased the performance of all the models!
# This makes sense since many of the titles summarize the reviewer's experience, with the review containing the details of the sentiment they expressed in their title.
# The linear SVC now has an accuracy of 90%, with a 66% accuracy for not recommended.
# Since most of the data contains recommendation reviews, the focus of this is really to increase the accuracy of non-recommendation.
# There is also some discrepency in the text reviews and the recommendation in some cases, and even human guesses may be wrong for those.
# 
# <a id="tweaking-the-model-and-data"></a>
# ## Tweaking the Model and Data
# 
# #### Data:
# 1. Append the title and the review text together to create our X-value.
# 2. The recommendation (binary value, 1=recommended) is our y-value.
# 
# #### Preprocessing:
# 1. Tokenize the text and stem the words
# 2. Remove digits, stop words,  punctuations, and single-letter words
# 
# #### Modeling:
# 1. Split the dataset into training and testing set
# 2. Create a pipeline, consisting of:
#     * CountVectorizer(analyzer=prep_clean, ngram_range=(1,2))
#     * TfidfTransformer()
#     * LinearSVC()
# 3. Test the LinearSVC model's performance by
#     * Cross_Val_Score()
#     * Cross_Val_Predict()
#     * confusion_matrix()
#     * classification_report()
