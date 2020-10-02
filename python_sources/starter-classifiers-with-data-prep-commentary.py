#!/usr/bin/env python
# coding: utf-8

# **Grab the popcorn**
# 
# This kernel uses data from Movie Review Sentiment Analysis Playground Competition to classify sentiment. The training data set comes pre-labeled, with a sentiment score ranging from 0 (most negative) to 4 (most positive). The data has already been split into sentences, and further into phrases. This kernel is meant as a starter classification, with (hopefully) clear explanations of what each step is doing. 
# 
# Let's start by loading some libraries we'll use. 

# In[ ]:


import numpy as np 
import pandas as pd 

from nltk.tokenize import word_tokenize
from nltk import FreqDist
from nltk.stem import SnowballStemmer,WordNetLemmatizer
stemmer=SnowballStemmer('english')
lemma=WordNetLemmatizer()

from string import punctuation

import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import FeatureUnion

from xgboost import XGBClassifier


# Now let's read in the training data set, test data set, and sample submission... and take a look at our training data. We're using the *pandas* package to read in our data into data frames. 

# In[ ]:


train = pd.read_csv('../input/train.tsv', sep="\t")
test = pd.read_csv('../input/test.tsv', sep="\t")
sub = pd.read_csv('../input/sampleSubmission.csv', sep=",")


# In[ ]:


train.head()


# We can see in the train.head() output that there are multiple phrases identified per sentence. We can also see that some phrases are a subset of other phrases ("A series" vs. "series" vs. "A"), and some phrases are just one word. From user *artgor*'s [kernel on this data set](https://www.kaggle.com/artgor/movie-review-sentiment-analysis-eda-and-models), we learned average count of phrases per sentence in train is 18, and in test is 20; and the average word length of phrases in both data sets is 7. *Artgor* also noted, "Sometimes one word or even one punctuation mark influences the sentiment." 
# 
# **Data prep & feature engineering**
# 
# Let's do some clean-up first. Make things lower-case. Remove punctuation. Note, there could be value in making different data prep decisions; for example, keeping punctuation and treating it as its own words could be useful in predicting sentiment.

# In[ ]:


def clean_text(review_col):
    review_corpus=[]
    for i in range(0,len(review_col)):
        review=str(review_col[i])
        review=re.sub('[^a-zA-Z]',' ',review)
        #review=[stemmer.stem(w) for w in word_tokenize(str(review).lower())]
        review=[lemma.lemmatize(w) for w in word_tokenize(str(review).lower())]
        review=' '.join(review)
        review_corpus.append(review)
    return review_corpus

train['cleaned']=clean_text(train.Phrase.values)
train.head()

test['cleaned']=clean_text(test.Phrase.values)


# Now we'll need to convert the phrase texts to quantitative variables. We can do this by creating variables based on what words and phrases appear in the text. 
# 
# We can feed that that into *sklearn*'s TfidfVectorizer to come up with something that's a bit more nuanced than a normal document term matrix, which would be a binary 1/0 for whether each document contains each ngram (see [dtm Wikipedia explanation](https://en.wikipedia.org/wiki/Document-term_matrix) for more details). The tfidf is goes a bit further by assigning weights (rather than a binary 1/0) based on how important the word is to the document, with respect to the corpus. Downweighting more common words and upweighting less common words lets us come up with a final matrix that better differentiates betwen documents. 
# 
# TfidVectorizer determines these weights by:
# 1. Getting term frequency (tf): How many times does a word (or ngram) appear in a single document? For example, the word "the" is very common and will appear with high frequency in most documents -- and does not provide much information, so we will ultimately want to reduce the weight placed on "the." 
# 2. Getting inverse document frequency (idf): How many documents are in your corpus, divided by the number of documents containing the word (or ngram)? Then take the log of the result. If the word "the" appears in 100 documents in a corpus of 100 documents, its idf value will be log(100/100) = 0. If a word only appears in one of those documents, its idf value will be log(100/1) = 2.
# 3. To calculate the final weight for the word or ngram, multiply tf by idf. For example, if the word "the" appeared in 100 documents and had an idf value of 0, the tfidf value would be 100x0 = 0 -- so it will count for no weight at all because it appears in all documents and is not useful to differentiate between them. 

# In[ ]:


tfhash = [("tfidf", TfidfVectorizer(stop_words='english')),
       ("hashing", HashingVectorizer (stop_words='english'))]
train_vectorized = FeatureUnion(tfhash).fit_transform(train.cleaned)
test_vectorized = FeatureUnion(tfhash).transform(test.cleaned)


# **Modeling with "traditional" classifiers**
# 
# Now that we have our data quantified, let's try a classifier: xgboost. Let's just use default settings for now. We can get an idea of accuracy using cross-validation. 

# In[ ]:


# define outcome variable, y
y = train['Sentiment']

# train model
xgb = XGBClassifier()
xgb.fit(train_vectorized, y)
scores = cross_val_score(xgb, train_vectorized, y, scoring='accuracy', n_jobs=-1, cv=3)
print('Cross-validation mean accuracy {0:.2f}%, std {1:.2f}.'.format(np.mean(scores) * 100, np.std(scores) * 100))


# Whelp, that's not so hot. Let's try some good ol' logistic regression. Since logistic regressions have binary outcomes, we can run multiple logistic regressions, one for each outcome label, with *scipy* package's OneVsRestClassifier. We're doing this quick and dirty again, no model tuning or diagnostics yet. 

# In[ ]:


logreg = LogisticRegression()
ovr = OneVsRestClassifier(logreg)
ovr.fit(train_vectorized, y)
scores = cross_val_score(ovr, train_vectorized, y, scoring='accuracy', n_jobs=-1, cv=3)
print('Cross-validation mean accuracy {0:.2f}%, std {1:.2f}.'.format(np.mean(scores) * 100, np.std(scores) * 100))


# That's noticeably better. Still not very good. Let's try an SVM. 

# In[ ]:


svm = LinearSVC()
svm.fit(train_vectorized, y)
scores = cross_val_score(svm, train_vectorized, y, scoring='accuracy', n_jobs=-1, cv=3)
print('Cross-validation mean accuracy {0:.2f}%, std {1:.2f}.'.format(np.mean(scores) * 100, np.std(scores) * 100))


# That's in the same ballpark as the logistic regressions. 
# 
# How about an ensemble of the models?

# In[ ]:


#estimators = [ ('xgb',xgb) , ('ovr', ovr), ('svm',svm) ]
estimators = [ ('ovr', ovr), ('svm',svm) ]
clf = VotingClassifier(estimators , voting='soft')
clf.fit(train_vectorized,y)
#scores = cross_val_score(clf, train_vectorized, y, scoring='accuracy', n_jobs=-1, cv=3)
#print('Cross-validation mean accuracy {0:.2f}%, std {1:.2f}.'.format(np.mean(scores) * 100, np.std(scores) * 100))
#scores = clf.predict(train_vectorized)
#print(classification_report(scores, y))
#print(accuracy_score(scores, y))


# Now let's do an example deep learning model.
# 
# **Stay tuned**
# 
# Hoping to add more here, including model tuning and/or deep learning models. 

# In[ ]:


#from sklearn.model_selection import train_test_split
#seed = 1234
#X_train, X_val, Y_train, Y_val = train_test_split(train_vectorized, y, test_size=0.25, random_state=seed)


# **Scoring**

# In[ ]:


sub['Sentiment'] = clf.predict(test_vectorized) 
sub.to_csv("clf.csv", index=False)

