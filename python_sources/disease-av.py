#!/usr/bin/env python
# coding: utf-8

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


from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer


# In[ ]:


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
submit = pd.read_csv("../input/sample_submission.csv")


# In[ ]:


train.shape


# In[ ]:


test.shape


# In[ ]:


submit.shape


# In[ ]:


train.isnull().sum()


# In[ ]:


train.Word[train["Word"] == '.']


# In[ ]:


train.pop("id")
train.head()


# In[ ]:





# In[ ]:


from nltk.probability import FreqDist
fdist = FreqDist(train["Word"])
fdist


# In[ ]:





# In[ ]:





# In[ ]:


'''import nltk
nltk.pos_tag(train["Word"][0])'''


# In[ ]:


train.dropna(inplace=True)


# In[ ]:





# In[ ]:


'''from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
token = RegexpTokenizer(r'[a-zA-Z0-9]+')
cv = CountVectorizer(lowercase=True,stop_words='english',ngram_range = (1,1),tokenizer = token.tokenize)
text_counts= cv.transform(train["Word"])'''


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


train.tag.unique()


# In[ ]:


train.tag[train.tag == 'O']=1
train.tag[train.tag == 'B-indications']=2
train.tag[train.tag == 'I-indications']=3


# In[ ]:





# In[ ]:


y = train["tag"]
y=y.astype('int')


# In[ ]:


test.fillna("None", inplace=True)


# In[ ]:


test.shape


# In[ ]:


test.head()


# In[ ]:


'''cv = CountVectorizer()
text_counts= cv.fit_transform(train["Word"])
text_counts'''


# In[ ]:


le = LabelEncoder()
train["Word"] = le.fit_transform(train["Word"])


# In[ ]:



test["Word"] = le.fit_transform(test["Word"])


# In[ ]:


x = train.drop(labels = "tag", axis = 1)


# In[ ]:


test.pop("id")


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)


# In[ ]:


from sklearn.metrics import f1_score
clf = MultinomialNB().fit(X_train, y_train)
predicted= clf.predict(X_test)
print("MultinomialNB Accuracy:",metrics.f1_score(y_test, predicted, average = 'weighted'))


# In[ ]:





# In[ ]:


'''cv = CountVectorizer()
text_counts1= cv.fit_transform(test["Word"])
'''


# In[ ]:





# In[ ]:


pred= clf.predict(test)


# In[ ]:


submit.shape


# In[ ]:


len(pred)
2994463
2994370


# In[ ]:


submit["tag"] = pred 


# In[ ]:


submit.head()


# In[ ]:





# In[ ]:


submit.tag[submit.tag == 1]='O'


# In[ ]:


submit.head()


# In[ ]:


submit.to_csv("submit.csv", index=False)


# In[ ]:




