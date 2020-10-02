#!/usr/bin/env python
# coding: utf-8

# ## Goal: To classify future SMS messages as either spam or ham with a Naive Bayes model.
# 
# Steps:
# 
# 1.  Convert the words ham and spam to a binary indicator variable(0/1)
# 
# 2.  Convert the txt to a sparse matrix of TFIDF vectors
# 
# 3.  Fit a Naive Bayes Classifier
# 
# 4.  Measure your success using roc_auc_score
# 
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import naive_bayes
from sklearn.metrics import roc_auc_score


# In[ ]:


#nltk.download()


# In[ ]:


df= pd.read_csv("../input/sms_spam.csv")


# In[ ]:


df.head()


# #### Train the classifier if it is spam or ham based on the text

# #### Convert the spam and ham to 1 and 0 values respectively for probability testing

# In[ ]:


df.type.replace('spam', 1, inplace=True)


# In[ ]:


df.type.replace('ham', 0, inplace=True)


# In[ ]:


df.head()


# In[ ]:


df.shape


# In[ ]:


##Our dependent variable will be 'spam' or 'ham' 
y = df.type


# In[ ]:


df.text


# In[ ]:


#TFIDF Vectorizer
stopset = set(stopwords.words('english'))
vectorizer = TfidfVectorizer(use_idf=True, lowercase=True, strip_accents='ascii', stop_words=stopset)


# In[ ]:


#Convert df.txt from text to features
X = vectorizer.fit_transform(df.text)


# In[ ]:


X.shape


# In[ ]:


X.data


# ### TF(t) = (Number of times term t appears in a document) / (Total number of terms in the document)
# 
# ### IDF(t) = log_e(Total number of documents / Number of documents with term t in it).
# 
# ## tf-idf score=TF(t)*IDF(t)

# In[ ]:


df.text[0]


# In[ ]:


## Spliting the SMS to separate the text into individual words
splt_txt1=df.text[0].split()
print(splt_txt1)


# In[ ]:


## Count the number of words in the first SMS
len(splt_txt1)


# ### It means in the first SMS there are 20(len(splt_txt1)) words & out of which only 14 elements have been taken, that;s why we'll get only 14 tf-idf values for the first the SMS.Likewise elements or words of all other SMSes are taken into consideration

# In[ ]:


X[0]


# ## 0 is the first SMS,3536,4316 etc are the positions of the elements or the words & 0.15,0.34,0.27 are the tf_idf value of the words . Like wise we can find the next SMSes & the tf-idf value of the words of the SMSes

# In[ ]:


print(X[0])


# In[ ]:


vectorizer.get_feature_names()[8585]## 4316 is the position of the word jurong


# ## Second SMS

# In[ ]:


## Spliting the SMS to separate the text into individual words
splt_txt2=df.text[1].split()
print(splt_txt2)


# In[ ]:


len(splt_txt2)


# In[ ]:


X[1]## Second SMS


# In[ ]:


print (X[1])


# In[ ]:


## Finding the most frequent word appearing in the second SMS
max(splt_txt2)


# ### From the above in the 2nd SMS there are 6 words  & out of which only 5 elements have been taken, that's why
# ### we'll get only 5 tf-idf values for the 2nd the SMS.Likewise elements or words of all other SMSes are taken into consideration

# In[ ]:


## Last word in the vocabulary
max(vectorizer.get_feature_names())


# In[ ]:


len(vectorizer.get_feature_names())


# In[ ]:


print (y.shape)
print (X.shape)


# In[ ]:


##Split the test and train
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)


# In[ ]:


X_train


# In[ ]:


##Train Naive Bayes Classifier
## Fast (One pass)
## Not affected by sparse data, so most of the 8605 words dont occur in a single observation
clf = naive_bayes.MultinomialNB()
model=clf.fit(X_train, y_train)


# In[ ]:


clf.feature_log_prob_


# In[ ]:


clf.coef_


# In[ ]:


predicted_class=model.predict(X_test)
print(predicted_class)


# In[ ]:





# ### First 3 SMSes are correctly assigned to Ham(0) based on the tf-idf scores of the words given in the SMSes

# In[ ]:


print(y_test)


# In[ ]:


df.loc[[19]]


# In[ ]:


predicted_class[19]## This SMS(SMS no. 19) has been classified as Ham but Actually it's SPAM


# ### Find the probability of assigning a SMS to a specific class

# In[ ]:


prd=model.predict_proba(X_test)


# In[ ]:


prd


# In[ ]:





# In[ ]:


clf.predict_proba(X_test)[:,0]


# In[ ]:


roc_auc_score(y_train, clf.predict_proba(X_train)[:,1])


# In[ ]:


##Check model's accuracy
roc_auc_score(y_test, clf.predict_proba(X_test)[:,1])


# ### With the model, the success rate is ~98.60%

# Identify the words which are the most important in classifying a message as spam

# In[ ]:


clf.coef_


# In[ ]:


def get_most_important_features(vectorizer, model, n=5):
    index_to_word = {v:k for k,v in vectorizer.vocabulary_.items()}
    
    # loop for each class
    classes ={}
    for class_index in range(model.coef_.shape[0]):
        word_importances = [(el, index_to_word[i]) for i,el in enumerate(model.coef_[class_index])]
        sorted_coeff = sorted(word_importances, key = lambda x : x[0], reverse=True)
        tops = sorted(sorted_coeff[:n], key = lambda x : x[0])
        bottom = sorted_coeff[-n:]
        classes[class_index] = {
            'tops':tops,
            'bottom':bottom
        }
    return classes

importance = get_most_important_features(vectorizer, clf, 20)


# In[ ]:


print (importance)


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
def plot_important_words(top_scores, top_words, bottom_scores, bottom_words, name):
    y_pos = np.arange(len(top_words))
    top_pairs = [(a,b) for a,b in zip(top_words, top_scores)]
    top_pairs = sorted(top_pairs, key=lambda x: x[1])
    
    bottom_pairs = [(a,b) for a,b in zip(bottom_words, bottom_scores)]
    bottom_pairs = sorted(bottom_pairs, key=lambda x: x[1], reverse=True)
    
    top_words = [a[0] for a in top_pairs]
    top_scores = [a[1] for a in top_pairs]
    
    bottom_words = [a[0] for a in bottom_pairs]
    bottom_scores = [a[1] for a in bottom_pairs]
    fig = plt.figure(figsize=(10, 10))  

    plt.subplot(121)
    plt.barh(y_pos,bottom_scores, align='center', alpha=0.5)
    plt.title('Ham', fontsize=20)
    plt.yticks(y_pos, bottom_words, fontsize=14)
    plt.suptitle('Key words', fontsize=16)
    plt.xlabel('Importance', fontsize=20)
    
    plt.subplot(122)
    plt.barh(y_pos,top_scores, align='center', alpha=0.5)
    plt.title('Spam', fontsize=20)
    plt.yticks(y_pos, top_words, fontsize=14)
    plt.suptitle(name, fontsize=16)
    plt.xlabel('Importance', fontsize=20)
    
    plt.subplots_adjust(wspace=0.8)
    plt.show()
top_scores = [a[0] for a in importance[0]['tops']]
top_words = [a[1] for a in importance[0]['tops']]
bottom_scores = [a[0] for a in importance[0]['bottom']]
bottom_words = [a[1] for a in importance[0]['bottom']]

plot_important_words(top_scores, top_words, bottom_scores, bottom_words, "Most important words for relevance")


# # END**
