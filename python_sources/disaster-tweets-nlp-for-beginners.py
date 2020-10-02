#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


train = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')
test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')


# # Removing Stopwords(EXAMPLE): 

# In[ ]:


train['text'].iloc[2]


# In[ ]:


from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# In[ ]:


stop_words = set(stopwords.words('english'))


# In[ ]:


word_tokens = word_tokenize(train['text'].iloc[2])
print(word_tokens)


# **Pretty cool, isn't it? You can separate sentences using word_tokenize!**

# In[ ]:


filtered_tweet = [w for w in word_tokens if not w in stop_words]
print(filtered_tweet)


# **There you go! We remove unnecessary words just like that.**

# # Converting words to numbers(EXAMPLE):

# **To fit a model to anything, be it words or images, we gotta convert them to numbers, to work with them!**

# In[ ]:


from sklearn.feature_extraction import text


# In[ ]:


count_vectorizer = text.CountVectorizer()

train_vectors = count_vectorizer.fit_transform(train['text'])
test_vectors = count_vectorizer.transform(test['text'])

train_vectors


# **If you use fit_transform for the test_vectors here, it returns the number of unique words in the test set, which is less than the train set. We want to use the unique words from the train set. So, we use transform to just convert the words to numbers, without returning the unique words.**

# In[ ]:


print(train_vectors[0].todense().shape)
print(train_vectors[0].todense())


# **Boy, there are 21,637 unique tokens in the tweets!**

# # Implementing the above techniques:

# In[ ]:


def remove_stopwords(df):
    
    for i in range(len(df)):
        
        word_tokens = word_tokenize(df['text'].loc[i])
        filtered_set = []
        
        for w in word_tokens:
            if w not in stop_words:
                filtered_set.append(w)
                
        filtered_sentence = ' '.join(filtered_set)
        df['text'].iloc[i] = filtered_sentence


# In[ ]:


remove_stopwords(train)
remove_stopwords(test)


# In[ ]:


count_vectorizer = text.CountVectorizer()

train_vectors = count_vectorizer.fit_transform(train['text'])
test_vectors = count_vectorizer.transform(test['text'])

print(train_vectors[0].shape)


# **A small difference of 21637-21617 = 20 words, but each small change counts.**

# In[ ]:


from sklearn.model_selection import cross_val_score
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB


# In[ ]:


clf = RidgeClassifier()
logreg = LogisticRegression()
sgd = SGDClassifier()
svc = LinearSVC()
mnb = MultinomialNB()


# In[ ]:


clf_scores = cross_val_score(clf, train_vectors, train['target'], cv=10, scoring='f1')
logreg_scores = cross_val_score(logreg, train_vectors, train['target'], cv=10, scoring='f1')
sgd_scores = cross_val_score(sgd, train_vectors, train['target'], cv=10, scoring='f1')
svc_scores = cross_val_score(svc, train_vectors, train['target'], cv=10, scoring='f1')
mnb_scores = cross_val_score(mnb, train_vectors, train['target'], cv=10, scoring='f1')


# In[ ]:


print("Ridge Classifier: ", np.mean(clf_scores))
print("Logistic Regression: ", np.mean(logreg_scores))
print("Stochastic Gradient Descent Classifier: ", np.mean(sgd_scores))
print("Support Vector Classifier: ", np.mean(svc_scores))
print("Multinomial Naive Bayes: ", np.mean(mnb_scores))


# **Clearly, the best model to use here is the Multinomial Naive Bayes classifier.**

# In[ ]:


mnb.fit(train_vectors, train['target'])


# In[ ]:


preds = mnb.predict(test_vectors)
result = pd.DataFrame({'id':test['id'], 'target':preds})


# In[ ]:


result.to_csv("submission.csv", index=False)

