#!/usr/bin/env python
# coding: utf-8

# # Clickbait detector using Naive Bayes Classifier
# 
# This kernel focuses on classifying News headlines into clickbaits and non-clickbaits.
# 
# The clickbaits are labelled as **1** and non-clickbaits as **0**.
# The headlines are collected from different news sites.
# 
# The dataset consists of 32000 headlines of which 50% are clickbaits and the other 50% are non-clickbait.
# 
# I have used a *Multinomial Naive Bayes* classification algorithm for text classification of the given dataset. 

# # Importing different tools and libraries
# 
# The main libraries used are *Numpy*, *Pandas*, *NLTK*(Natural language toolkit) and *Scikit-learn*.

# In[ ]:


import numpy as np 
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.probability import FreqDist
import string as s
import re

import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score

from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# # Loading the Dataset

# In[ ]:


cb_data= pd.read_csv('/kaggle/input/clickbait-dataset/clickbait_data.csv')
cb_data.head()


# # Splitting into Train and Test sets
# 
# The dataset is splitted into training and testing sets. The percentage of training data is 75% and testing data is 25%.

# In[ ]:


x=cb_data.headline
y=cb_data.clickbait
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.25,random_state=2)


# # Analyzing Train and Test Data

# In[ ]:


print("No. of elements in training set")
print(train_x.size)
print("No. of elements in testing set")
print(test_x.size)


# In[ ]:


train_x.head(10)


# In[ ]:


train_y.head(10)


# In[ ]:


test_x.head()


# In[ ]:


test_y.head()


# # Tokenization of Data
# 
# The data is tokenized i.e. split into tokens which are the smallest or minimal meaningful units. The data is split into words.

# In[ ]:


def tokenization(text):
    lst=text.split()
    return lst
train_x=train_x.apply(tokenization)
test_x=test_x.apply(tokenization)


# # Converting to lowercase
# 
# The data is converted into lowercase to avoid ambiguity between same words in different cases like 'NLP', 'nlp' or 'Nlp'. 

# In[ ]:


def lowercasing(lst):
    new_lst=[]
    for i in lst:
        i=i.lower()
        new_lst.append(i)
    return new_lst
train_x=train_x.apply(lowercasing)
test_x=test_x.apply(lowercasing)  


# # Removing punctuation
# 
# The punctuations are removed to increase the efficiency of the model. They are irrelevant because they provide no added information.

# In[ ]:


def remove_punctuations(lst):
    new_lst=[]
    for i in lst:
        for j in s.punctuation:
            i=i.replace(j,'')
        new_lst.append(i)
    return new_lst
train_x=train_x.apply(remove_punctuations)
test_x=test_x.apply(remove_punctuations)  


# # Removing Numbers

# In[ ]:


def remove_numbers(lst):
    nodig_lst=[]
    new_lst=[]
    for i in lst:
        for j in s.digits:    
            i=i.replace(j,'')
        nodig_lst.append(i)
    for i in nodig_lst:
        if i!='':
            new_lst.append(i)
    return new_lst
train_x=train_x.apply(remove_numbers)
test_x=test_x.apply(remove_numbers)


# # Removing Stopwords

# In[ ]:


print("All stopwords of English language ")
", ".join(stopwords.words('english'))


# In[ ]:


def remove_stopwords(lst):
    stop=stopwords.words('english')
    new_lst=[]
    for i in lst:
        if i not in stop:
            new_lst.append(i)
    return new_lst

train_x=train_x.apply(remove_stopwords)
test_x=test_x.apply(remove_stopwords)  


# # Removing extra spaces

# In[ ]:


def remove_spaces(lst):
    new_lst=[]
    for i in lst:
        i=i.strip()
        new_lst.append(i)
    return new_lst
train_x=train_x.apply(remove_spaces)
test_x=test_x.apply(remove_spaces)


# # Analyzing data after preprocessing
# 
# After preprocessing the data i.e. after removing punctuation, stopwords, spaces and numbers.

# In[ ]:


train_x.head()


# In[ ]:


test_x.head()


# # Lemmatization
# 
# Lemmatization in linguistics is the process of grouping together the inflected forms of a word so they can be analysed as a single item, identified by the word's lemma, or dictionary form. It involves the morphological analysis of words.
# 
# In lemmatization we find the root word or base form of the word rather than just clipping some characters from the end e.g. *is, are, am* are all converted to its base form *be* in Lemmatization
# 
# Here lemmatization is done using NLTK library.

# In[ ]:


lemmatizer=nltk.stem.WordNetLemmatizer()
def lemmatzation(lst):
    new_lst=[]
    for i in lst:
        i=lemmatizer.lemmatize(i)
        new_lst.append(i)
    return new_lst
train_x=train_x.apply(lemmatzation)
test_x=test_x.apply(lemmatzation)


# In[ ]:


train_x=train_x.apply(lambda x: ''.join(i+' ' for i in x))
test_x=test_x.apply(lambda x: ''.join(i+' ' for i in x))


# In[ ]:


freq_dist={}
for i in train_x.head(20):
    x=i.split()
    for j in x:
        if j not in freq_dist.keys():
            freq_dist[j]=1
        else:
            freq_dist[j]+=1
freq_dist


# # TF-IDF (Term frequency-Inverse Data Frequency)
# 
# This method is used to convert the text into features.

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf=TfidfVectorizer()
train_1=tfidf.fit_transform(train_x)
test_1=tfidf.transform(test_x)


# In[ ]:


print("Number of features extracted")
print(len(tfidf.get_feature_names()))
print()
print("The 100 features extracted from TF-IDF ")
print(tfidf.get_feature_names()[:100])


# In[ ]:


print("Shape of train set",train_1.shape)
print("Shape of test set",test_1.shape)


# In[ ]:


train_arr=train_1.toarray()
test_arr=test_1.toarray()


# # Define Naive Bayes Classifier

# In[ ]:


NB_MN=MultinomialNB()


# # Training the model

# In[ ]:


NB_MN.fit(train_arr,train_y)
pred=NB_MN.predict(test_arr)
print('first 20 actual labels: ',test_y.tolist()[:20])
print('first 20 predicted labels: ',pred.tolist()[:20])


# # Evaluation of Result
# 
# The Accuracy and F1 score of the model are printed to evaluate the model for text classification.

# In[ ]:


from sklearn.metrics import f1_score,accuracy_score
print("F1 score of the model")
print(f1_score(test_y,pred))
print("Accuracy of the model")
print(accuracy_score(test_y,pred))
print("Accuracy of the model in percentage")
print(accuracy_score(test_y,pred)*100,"%")


# In[ ]:


from sklearn.metrics import confusion_matrix
print("Confusion Matrix")
print(confusion_matrix(test_y,pred))

from sklearn.metrics import classification_report
print("Classification Report")
print(classification_report(test_y,pred))


# ## A second model for TF-IDF with different n-grams and fixed feature size

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf=TfidfVectorizer(ngram_range=(1,3),max_features=6500)
train_2=tfidf.fit_transform(train_x)
test_2=tfidf.transform(test_x)

NB_MN.fit(train_2.toarray(),train_y)
pred2=NB_MN.predict(test_2.toarray())

print("Accuracy of the model in percentage")
print(accuracy_score(test_y,pred2)*100,"%")

from sklearn.metrics import confusion_matrix
print("Confusion Matrix")
print(confusion_matrix(test_y,pred2))

from sklearn.metrics import classification_report
print("Classification Report")
print(classification_report(test_y,pred2))

