#!/usr/bin/env python
# coding: utf-8

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


train = pd.read_csv('/kaggle/input//quora-insincere-questions-classification/train.csv')
train.shape


# In[ ]:


train.head()


# In[ ]:


docs = train['question_text']


# In[ ]:


train['target'].value_counts()/train.shape[0]*100


# In[ ]:


import nltk
stemmer = nltk.stem.PorterStemmer()


# In[ ]:


stemmer.stem('organization')


# # test cleaning process for text classification 
# - convert to lower case
# - regular expression to remove non alphabets 
# - apply stemming to get root form of a word 
# - apply stop word removel 

# In[ ]:


docs.head()


# In[ ]:


docs_clean = docs.str.lower().str.replace('[^a-z ]','')
docs_clean.head()


# In[ ]:


def clean_fun(data):
        docs_clean = data.str.lower().str.replace('[^a-z ]','')
        return docs_clean


# In[ ]:


stopwords = nltk.corpus.stopwords.words('english')
stopwords.extend([])
def clean_sentence(doc):
    words = nltk.word_tokenize(doc)
    words_clean = [stemmer.stem(word) for word in words if word not in stopwords]
    return ' '.join(words_clean)
docs_clean = docs_clean.apply(clean_sentence)
    


# In[ ]:


docs_clean.head()


# In[ ]:


#DTM
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.model_selection import train_test_split


# In[ ]:


x_train,x_validate,y_train,y_validate = train_test_split(docs_clean,train['target'],test_size= 0.3,random_state = 1)
x_train.shape,x_validate.shape,y_train.shape,y_validate.shape


# In[ ]:


vectorizer = CountVectorizer(min_df=10).fit(x_train)
train_dtm = vectorizer.transform(x_train)
validate_dtm = vectorizer.transform(x_validate)


# In[ ]:


from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB().fit(train_dtm,y_train)
validate_pred = model.predict(validate_dtm)

from sklearn.metrics import f1_score
f1_score(y_validate,validate_pred)


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3).fit(train_dtm,y_train)
validate_pred = knn.predict(validate_dtm)
f1_score(y_validate,validate_pred)


# In[ ]:


#SVM
from sklearn.svm import SVC
model_svm = SVC(kernel='linear').fit(train_dtm,y_train)
validate_pred_svm=model_svm.predict(validate_dtm)
f1_score(y_validate,validate_pred_svm)


# In[ ]:


test_data = pd.read_csv('/kaggle/input/quora-insincere-questions-classification/test.csv')
test_data.head()


# In[ ]:


test_doc = test_data['question_text']
test_doc.head(2)


# **Uisng Keras**

# In[ ]:


from keras.models import Sequential
 from keras import layers

 input_dim = X_train.shape[1]  # Number of features

 model = Sequential()
 model.add(layers.Dense(10, input_dim=input_dim, activation='relu'))
>>> model.add(layers.Dense(1, activation='sigmoid'))


# In[ ]:


model.compile(loss='binary_crossentropy', 
               optimizer='adam', 
               metrics=['accuracy'])
 model.summary()


# **Word to Vec**

# In[ ]:


import gensim 
path = '/kaggle/input/quora-insincere-questions-classification/embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin'
embendings = gensim.models.KeyedVectors.load_word2vec_format(path,binary= True)


# In[ ]:


embendings.most_similar(['sad'],topn=10) # using cosine similarity 


# In[ ]:


embendings.most_similar(positive=['king','woman'],negative=['man'],topn=1)


# In[ ]:


imdb = pd.read_csv('https://github.com/skathirmani/datasets/blob/master/imdb_sentiment.csv')

