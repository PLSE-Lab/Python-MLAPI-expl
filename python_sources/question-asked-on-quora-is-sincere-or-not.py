#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# all import statements
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
#from wordcloud import WordCloud as wc   # not needed
from nltk.corpus import stopwords
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
from pandas import get_dummies
import matplotlib as mpl
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib
import warnings
import sklearn
import string
import scipy
import numpy
import nltk
import json
import sys
import csv
import os


# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


train = pd.read_csv('/kaggle/input/quora-insincere-questions-classification/train.csv')
test = pd.read_csv('/kaggle/input/quora-insincere-questions-classification/test.csv')
train.head()


# # 2  We would first do EDA ( Exploratory Data Analysis ) over Quora Data set :
# 

# In[ ]:


print(train.info())
print(test.info())


# In[ ]:


# shape for train and test
print('Shape of train:',train.shape)
print('Shape of test:',test.shape)


# Data Cleaning
# 
# When dealing with real-world data, dirty data is the norm rather than the exception.
# 
# The primary goal of data cleaning is to detect and remove errors and anomalies to increase the value of data in analytics and decision making. While it has been the focus of many researchers for several years, individual problems have been addressed separately. These include missing value imputation, outliers detection, transformations, integrity constraints violations detection and repair, consistent query answering, deduplication, and many other related problems such as profiling and constraints mining

# In[ ]:


# How many NA elements in every column!!
# Good news, it is Zero!
# To check out how many null info are on the dataset, we can use isnull().sum().
# recall from info() -> we found that it has zero Nulls. 

train.isnull().sum()

# data is infact clean and ready for use.


# In[ ]:


# in case , their were NA or None values in any row then we would drop the row.

# remove rows that have NA's
print('Before Droping',train.shape)
train = train.dropna()
print('After Droping',train.shape)


# In[ ]:


# Number of words in the text

train["num_words"] = train["question_text"].apply(lambda x: len(str(x).split()))
test["num_words"] = test["question_text"].apply(lambda x: len(str(x).split()))
print('maximum of num_words in train',train["num_words"].max())
print('min of num_words in train',train["num_words"].min())
print("maximum of  num_words in test",test["num_words"].max())
print('min of num_words in train',test["num_words"].min())


# In[ ]:


# Number of unique words in the text
train["num_unique_words"] = train["question_text"].apply(lambda x: len(set(str(x).split())))
test["num_unique_words"] = test["question_text"].apply(lambda x: len(set(str(x).split())))

print('maximum of num_unique_words in train',train["num_unique_words"].max())

print("maximum of num_unique_words in test",test["num_unique_words"].max())


# In[ ]:


# Number of stopwords in the text

#from nltk.corpus import stopwords
eng_stopwords = set(stopwords.words("english"))

train["num_stopwords"] = train["question_text"].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))
test["num_stopwords"] = test["question_text"].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))

print('maximum of num_stopwords in train',train["num_stopwords"].max())
print("maximum of num_stopwords in test",test["num_stopwords"].max())


# In[ ]:


# Number of punctuations in the text

train["num_punctuations"] =train['question_text'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )
test["num_punctuations"] =test['question_text'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )
print('maximum of num_punctuations in train',train["num_punctuations"].max())
print("maximum of num_punctuations in test",test["num_punctuations"].max())


# In[ ]:


# lets figure out how many unique target values exist.
# like we expect : 0 -> sincere qns and 1 -> un-sincere qns

# You see number of unique item for Target with command below:
train_target = train['target'].values

np.unique(train_target)


# YES, quora problem is a binary classification!
# 
# The data is absolutely clean. But is it balanced ?
# 
# I mean do we have equal no. of sincere and un-sincere questions in the dataset.
# 
# lets find out :

# In[ ]:


#train.where(train ['target']==1).count()
train[train.target==1].count()


# Such a dataset is imbalanced.
# 
# Imbalanced dataset is relevant primarily in the context of supervised machine learning involving two or more classes.
# 
# Imbalance means that the number of data points available for different classes is different: If there are two classes, then balanced data would mean 50% points for each of the class. For most machine learning techniques, little imbalance is not a problem. So, if there are 60% points for one class and 40% for the other class, it should not cause any significant performance degradation. Only when the class imbalance is high, e.g. 90% points for one class and 10% for the other, standard optimization criteria or performance measures may not be as effective and would need modification.
# 
# imbalacedQuoraData
# 
# A typical example of imbalanced data is encountered in e-mail classification problem where emails are classified into ham or spam. The number of spam emails is usually lower than the number of relevant (ham) emails. So, using the original distribution of two classes leads to imbalanced dataset.
# 
# Using accuracy as a performace measure for highly imbalanced datasets is not a good idea. For example, if 90% points belong to the true class in a binary classification problem, a default prediction of true for all data points leads to a classifier which is 90% accurate, even though the classifier has not learnt anything about the classification problem at hand!

# In[ ]:


# visualising the imbalance in data set

ax=sns.countplot(x='target',hue="target", data=train  ,linewidth=5,edgecolor=sns.color_palette("dark", 3))
plt.title('Is data set imbalance?');


# # 3  Data Preprocessing

# 3  Data Preprocessing
# This basically involves transforming raw data into an understandable format for NLP models.
# 
# Remember, our feature "Question_Text" is Text or String Object and No ML algo say KNN or Bayes classification would accept Text. Hence Pre-Processing is mandatory in this case.
# 
# Below, I have recalled the two most important techniques that are also performed besides other easy to understand steps in data pre-processing:
# 
# Tokenization: This is a process of breaking a stream of text up into words, phrases, symbols, or other meaningful elements called tokens. The list of tokens becomes input for further processing. NLTK Library has word_tokenize and sent_tokenize to easily break a stream of text into a list of words or sentences, respectively.
# Word Stemming/Lemmatization: The aim of both processes is the same, reducing the inflectional forms of each word into a common base or root. Lemmatization is closely related to stemming. The difference is that a stemmer operates on a single word without knowledge of the context, and therefore cannot discriminate between words which have different meanings depending on part of speech. However, stemmers are typically easier to implement and run faster, and the reduced accuracy may not matter for some applications.
# Quora_data_preprocessing
# 
# We would atleast the following (Text) Pre-Processing steps
# 
# 1.Change all the text to lower case
# 
# 2.Word Tokenization
# 
# 3.Remove Stop words
# 
# 4.Remove Non-alpha text
# 
# 5.Word Lemmatization
# 
# 5.Converting the text data into Numeric vectors( called Vectorization )

# In[ ]:


# step 1: Change all the text to lower case. 

# This is required as python interprets 'quora' and 'QUORA' differently

train['question_text'] = [entry.lower() for entry in train['question_text']]

test['question_text'] = [entry.lower() for entry in test['question_text']]

train.head()


# In[ ]:


# more imports for NLP
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score


# In[ ]:


# taking backup
trainbackup=train
testbackup=test
trainbackup.shape


# In[ ]:


# keeping only 2000 questions for analysis
train= train.head(2000)
test= test.head(2000)


# In[ ]:


# step 2 : Tokenization : In this each entry in the corpus will be broken 
#                         into set of words


train['question_text']= [word_tokenize(entry) for entry in train['question_text']]

test['question_text']= [word_tokenize(entry) for entry in test['question_text']]
train.head()


# In[ ]:


# Set random seed
# This is used to reproduce the same result every time 
# if the script is kept consistent otherwise each run 
# will produce different results. The seed can be set to any number.
np.random.seed(500)


# In[ ]:


## for train data

# step 3, 4 and 5
# Remove Stop words and Numeric data 
# and perfom Word Stemming/Lemmenting.

# WordNetLemmatizer requires Pos tags to understand if the word is noun or verb
# or adjective etc. By default it is set to Noun
tag_map = defaultdict(lambda : wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV
# the tag_map would map any tag to 'N' (Noun) except
# Adjective to J, Verb -> v, Adverb -> R
# that means if you get a Pronoun then it would still be mapped to Noun


for index,entry in enumerate(train['question_text']):
    # Declaring Empty List to store the words that follow the rules for this step
    Final_words = []
    
    # Initializing WordNetLemmatizer()
    word_Lemmatized = WordNetLemmatizer()
    #print(help(pos_tag(entry)))
    # pos_tag function below will provide the 'tag' 
    # i.e if the word is Noun(N) or Verb(V) or something else.
    for word, tag in pos_tag(entry):
    
        # Below condition is to check for Stop words and consider only 
        # alphabets
        if word not in stopwords.words('english') and word.isalpha():
            word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
            
            Final_words.append(word_Final)
    #print(Final_words)        
    # The final processed set of words for each iteration will be stored 
    # in 'question_text_final'
    train.loc[index,'question_text_final'] = str(Final_words)
    
   


# In[ ]:


## for test data

# step 3, 4 and 5
# Remove Stop words and Numeric data 
# and perfom Word Stemming/Lemmenting.

# WordNetLemmatizer requires Pos tags to understand if the word is noun or verb
# or adjective etc. By default it is set to Noun
tag_map = defaultdict(lambda : wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV
# the tag_map would map any tag to 'N' (Noun) except
# Adjective to J, Verb -> v, Adverb -> R
# that means if you get a Pronoun then it would still be mapped to Noun


for index,entry in enumerate(test['question_text']):
    # Declaring Empty List to store the words that follow the rules for this step
    Final_words_test = []
    
    # Initializing WordNetLemmatizer()
    word_Lemmatized = WordNetLemmatizer()
    
    # pos_tag function below will provide the 'tag' 
    # i.e if the word is Noun(N) or Verb(V) or something else.
    for word, tag in pos_tag(entry):
        # Below condition is to check for Stop words and consider only 
        # alphabets
        if word not in stopwords.words('english') and word.isalpha():
            word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
            Final_words_test.append(word_Final)
            
    # The final processed set of words for each iteration will be stored 
    # in 'question_text_final'
    test.loc[index,'question_text_final'] = str(Final_words_test)
    


# In[ ]:


test.head()


# In[ ]:


Tfidf_vect = TfidfVectorizer()
Tfidf_vect.fit(train['question_text_final'])

Train_X_Tfidf = Tfidf_vect.transform(train['question_text_final'])

Test_X_Tfidf = Tfidf_vect.transform(test['question_text_final'])


# In[ ]:


#print(Train_X_Tfidf)
print(Test_X_Tfidf[:4])


# In[ ]:


# You can use the below syntax to see the vocabulary that 
# it has learned from the corpus
print(Tfidf_vect.vocabulary_)


# # Data Pre-processing is over !!

# # 5  Use ML Algorithms to Predict the outcome

# In[ ]:




# fit the training dataset on the NB classifier
Naive = naive_bayes.MultinomialNB()

Naive.fit(Train_X_Tfidf,train['target'])

# predict the labels on validation dataset
predictions_NB = Naive.predict(Test_X_Tfidf)

# Use accuracy_score function to get the accuracy
#print("Naive Bayes Accuracy Score -> ",accuracy_score(predictions_NB, train['target'])*100)
print(predictions_NB)

accuracy_score(predictions_NB,train.target)*100


# In[ ]:


# Classifier - Algorithm - SVM
# fit the training dataset on the classifier
SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')

SVM.fit(Train_X_Tfidf,train['target'])

# predict the labels on validation dataset
predictions_SVM = SVM.predict(Test_X_Tfidf)

# Use accuracy_score function to get the accuracy
# print("SVM Accuracy Score -> ",accuracy_score(predictions_SVM, train['target'])*100)
print(predictions_SVM)
predictions_SVM[0]


accuracy_score(predictions_SVM,train.target)*100


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




