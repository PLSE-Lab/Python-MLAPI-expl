#!/usr/bin/env python
# coding: utf-8

# <img src="https://cdn.steemitimages.com/DQmQZCf7ME7Haj3X3MzXtG8R8JtGmTpuh5NXDSd3wKueva7/rottentomatoes.png" />

# # Approaching (Almost) Any NLP Problem for Classification on Kaggle
# 
# In this post I'll talk about approaching natural language processing problems on Kaggle. As an example, we will use the data from this competition. I have create a very basic all classification model first and then improve algorithm parameter. 
# 
# ### Cover all Classification Algorithm 
# * XGBClassifier
# * LogisticRegression
# * MultinomialNB
# * AdaBoostClassifier
# * KNeighborsClassifier
# * LinearSVC
# * GradientBoostingClassifier
# * ExtraTreesClassifier
# * DecisionTreeClassifier
# 
# Note : You can also use other classification algorithm **which is load in Library (Classification Model Packages for Sentiment)**
# 
# **Important Note : ** * you must use del and gc function because of same time kernal ram and memory is full so you have to delete object and clear the ram*
# 
# ### Blending Technique Apply on Best Score Algorithm
# * XGBClassifier 
# * LogisticRegression 
# * MultinomialNB 
# * KNeighborsClassifier 
# * LinearSVC 
# * ExtraTreesClassifier
# * DecisionTreeClassifier

# ## Road Map We must have to follow 
# * NLP Library for Preprocessing and Cleaning
# * Load all Classification Model Packages for Sentiment
# * Load Data Set
# * Analyse the Data and Take same few insights
# * PreProcessing Function
# * Split the Data Train and Validation
# * Generate Feature using TfidfVectorizer
# * Load the all classification model
#         * XGBClassifier
#         * LogisticRegression
#         * MultinomialNB
#         * AdaBoostClassifier
#         * KNeighborsClassifier
#         * LinearSVC
#         * GradientBoostingClassifier
#         * ExtraTreesClassifier
#         * DecisionTreeClassifier
# 
# ## Hello Kagglers you have fork this notebook and  improve your score using parameter tunning technique
# * Gride Search
# * Random Search
# * Bayes Search so on

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from wordcloud import WordCloud
from collections import Counter

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import re,gc
from string import punctuation
print(os.listdir("../input"))
# Any results you write to the current directory are saved as output.


# ### NLP Library for Preprocessing and Cleaning

# In[ ]:


import nltk
from nltk.tokenize import TweetTokenizer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import FreqDist
from nltk.corpus import stopwords
from nltk.util import ngrams
from nltk.stem import SnowballStemmer,WordNetLemmatizer
stemmer=SnowballStemmer('english')
lemma=WordNetLemmatizer()


# ### Load all Classification Model Packages for Sentiment

# In[ ]:


import datetime
import lightgbm as lgb
from scipy import stats
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier,RadiusNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC,SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier,ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.multiclass import OneVsRestClassifier
pd.set_option('max_colwidth',400)


# 
# **<p>Rotten Tomatoes is an American review-aggregation website for film and television. The company was launched in August 1998 by three undergraduate students at the University of California, Berkeley: Senh Duong, Patrick Y. Lee and Stephen Wang. The name "Rotten Tomatoes" derives from the practice of audiences throwing rotten tomatoes when disapproving of a poor stage performance</p>**
# 
# #### Evalution
# 
# <p>Submissions are evaluated on classification accuracy (the percent of labels that are predicted correctly) for every parsed phrase. The sentiment labels are:</p>
# 
# * 0 - negative
# * 1 - somewhat negative
# * 2 - neutral
# * 3 - somewhat positive
# * 4 - positive

# ### Load Data Set

# In[ ]:


train = pd.read_csv('../input/train.tsv', sep="\t")
test = pd.read_csv('../input/test.tsv', sep="\t")
sub = pd.read_csv('../input/sampleSubmission.csv', sep=",")


# In[ ]:


train.head(10)


# ### Analyse the Data and Take same few insights

# In[ ]:


print('Average count of phrases per sentence in train is {0:.0f}.'.format(train.groupby('SentenceId')['Phrase'].count().mean()))
print('Average count of phrases per sentence in test is {0:.0f}.'.format(test.groupby('SentenceId')['Phrase'].count().mean()))


# In[ ]:


print('Number of phrases in train: {}. Number of sentences in train: {}.'.format(train.shape[0], len(train.SentenceId.unique())))
print('Number of phrases in test: {}. Number of sentences in test: {}.'.format(test.shape[0], len(test.SentenceId.unique())))


# In[ ]:


print('Average word length of phrases in train is {0:.0f}.'.format(np.mean(train['Phrase'].apply(lambda x: len(x.split())))))
print('Average word length of phrases in test is {0:.0f}.'.format(np.mean(test['Phrase'].apply(lambda x: len(x.split())))))


# In[ ]:


text = ' '.join(train.loc[train.Sentiment == 4, 'Phrase'].values)
text_trigrams = [i for i in ngrams(text.split(), 3)]


# In[ ]:


text = ' '.join(train.loc[train.Sentiment == 4, 'Phrase'].values)
text = [i for i in text.split() if i not in stopwords.words('english')]
text_trigrams = [i for i in ngrams(text, 3)]
Counter(text_trigrams).most_common(30)


# ### PreProcessing Function

# In[ ]:


def clean_review(review_col):
    review_corpus=[]
    for i in range(0,len(review_col)):
        review=str(review_col[i])
        review=re.sub('[^a-zA-Z]',' ',review)
        #review=[stemmer.stem(w) for w in word_tokenize(str(review).lower())]
        review=[lemma.lemmatize(w) for w in word_tokenize(str(review).lower())]
        review=' '.join(review)
        review_corpus.append(review)
    return review_corpus


# In[ ]:


train['csen']=clean_review(train.Phrase.values)
test['csen']=clean_review(test.Phrase.values)


# In[ ]:


y = train['Sentiment']


# ### Split the Data Train and Validation

# In[ ]:


xtrain, xvalid, ytrain, yvalid = train_test_split(train.csen.values, y, 
                                                  stratify=y, 
                                                  random_state=42, 
                                                  test_size=0.1, shuffle=True)


# ### Generate Feature using TfidfVectorizer

# In[ ]:


vectorizer = TfidfVectorizer(min_df=3,  max_features=None, 
            strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
            ngram_range=(1, 2), use_idf=1,smooth_idf=1,sublinear_tf=1,
            stop_words = 'english')
full_text = list(train['csen'].values)
vectorizer.fit(full_text)


# In[ ]:


xtrain_tfv =  vectorizer.transform(xtrain)
xvalid_tfv = vectorizer.transform(xvalid)
xtest_tfv = vectorizer.transform(test['csen'].values)


# ### LogisticRegression

# In[ ]:


clf = LogisticRegression(C=1.0)
clf.fit(xtrain_tfv, ytrain)
predictions = clf.predict(xvalid_tfv)
print("accuracy_score",accuracy_score(yvalid, predictions))


# In[ ]:


lr = clf.predict(xtest_tfv)
del clf,predictions
gc.collect()


# ### XGBClassifier

# In[ ]:


clf = xgb.XGBClassifier(max_depth=7, n_estimators=200, colsample_bytree=0.8, 
                        subsample=0.8, nthread=10, learning_rate=0.1)
clf.fit(xtrain_tfv, ytrain)
predictions = clf.predict(xvalid_tfv)
print("accuracy_score",accuracy_score(yvalid, predictions))


# In[ ]:


xgb = clf.predict(xtest_tfv)
del clf,predictions
gc.collect()


# ### MultinomialNB

# In[ ]:


clf = MultinomialNB()
clf.fit(xtrain_tfv, ytrain)
predictions = clf.predict(xvalid_tfv)
print("accuracy_score",accuracy_score(yvalid, predictions))


# In[ ]:


mnb = clf.predict(xtest_tfv)
del clf,predictions
gc.collect()


# ### AdaBoostClassifier

# In[ ]:


clf = AdaBoostClassifier()
clf.fit(xtrain_tfv, ytrain)
predictions = clf.predict(xvalid_tfv)
print("accuracy_score",accuracy_score(yvalid, predictions))


# In[ ]:


adboost = clf.predict(xtest_tfv)
del clf,predictions
gc.collect()


# ### KNeighborsClassifier

# In[ ]:


clf = KNeighborsClassifier()
clf.fit(xtrain_tfv, ytrain)
predictions = clf.predict(xvalid_tfv)
print("accuracy_score",accuracy_score(yvalid, predictions))


# In[ ]:


knc = clf.predict(xtest_tfv)
del clf,predictions
gc.collect()


# ### LinearSVC

# In[ ]:


clf = LinearSVC()
clf.fit(xtrain_tfv, ytrain)
predictions = clf.predict(xvalid_tfv)
print("accuracy_score",accuracy_score(yvalid, predictions))


# In[ ]:


lsvc = clf.predict(xtest_tfv)
del clf,predictions
gc.collect()


# ### GradientBoostingClassifier

# In[ ]:


clf = GradientBoostingClassifier()
clf.fit(xtrain_tfv, ytrain)
predictions = clf.predict(xvalid_tfv)
print("accuracy_score",accuracy_score(yvalid, predictions))


# In[ ]:


gbc = clf.predict(xtest_tfv)
del clf,predictions
gc.collect()


# ### ExtraTreesClassifier 

# In[ ]:


clf = ExtraTreesClassifier()
clf.fit(xtrain_tfv, ytrain)
predictions = clf.predict(xvalid_tfv)
print("accuracy_score",accuracy_score(yvalid, predictions))


# In[ ]:


etc = clf.predict(xtest_tfv)
del clf,predictions
gc.collect()


# ### DecisionTreeClassifier

# In[ ]:


clf = DecisionTreeClassifier()
clf.fit(xtrain_tfv, ytrain)
predictions = clf.predict(xvalid_tfv)
print("accuracy_score",accuracy_score(yvalid, predictions))


# In[ ]:


dtc = clf.predict(xtest_tfv)
del clf,predictions
gc.collect()


# In[ ]:


sub['Sentiment'] = pd.DataFrame(lr)
sub.to_csv('lr.csv',index=False)
sub['Sentiment'] = pd.DataFrame(xgb)
sub.to_csv('xgb.csv',index=False)
sub['Sentiment'] = pd.DataFrame(mnb)
sub.to_csv('mnb.csv',index=False)
sub['Sentiment'] = pd.DataFrame(lsvc)
sub.to_csv('lsvc.csv',index=False)
sub['Sentiment'] = pd.DataFrame(etc)
sub.to_csv('etc.csv',index=False)
sub['Sentiment'] = pd.DataFrame(knc)
sub.to_csv('knc.csv',index=False)
sub['Sentiment'] = pd.DataFrame(dtc)
sub.to_csv('dtc.csv',index=False)


# ### Blending Technique Apply on Best Score
# * XGBClassifier 
# * LogisticRegression 
# * MultinomialNB 
# * KNeighborsClassifier 
# * LinearSVC 
# * ExtraTreesClassifier
# * DecisionTreeClassifier

# In[ ]:


df = pd.DataFrame(lr,columns=['lr'])


# In[ ]:


df['xgb'] = xgb
df['mnb'] = mnb
df['lsvc'] = lsvc
df['etc'] = etc
df['knc'] = knc
df['dtc'] = dtc
df.head(15)


# In[ ]:


sub['Sentiment'] = df.mode(axis=1)
sub['Sentiment'] = sub.Sentiment.astype(int)


# In[ ]:


sub.to_csv('submission.csv',index=False)

