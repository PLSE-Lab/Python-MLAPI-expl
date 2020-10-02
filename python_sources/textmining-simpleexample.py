#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import nltk as nlp
from nltk.corpus import stopwords
import re
from sklearn.naive_bayes import GaussianNB

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv(r"../input/gender-classifier-DFE-791531.csv", encoding="latin1")


# In[ ]:


df.describe()


# In[ ]:


df.head()


# In[ ]:


df.info()


# In[ ]:


df = pd.concat([df.gender, df.description], axis=1)


# In[ ]:


df.head()


# In[ ]:


df.dropna(axis = 0, inplace = True)


# In[ ]:


df.info()


# In[ ]:


df.gender = [ 1 if each == "female" else 0 for each in df.gender]


# In[ ]:


df.head()


# In[ ]:


description_list = []
for description in df.description:
    description = re.sub("[^a-zA-Z]", " ", description)
    description = description.lower()
    description = nlp.word_tokenize(description)
    description = [word for word in description if not word in set(stopwords.words("english"))]
    lemma = nlp.WordNetLemmatizer()
    description = [ lemma.lemmatize(word) for word in description]
    description = " ".join(description)
    description_list.append(description)


# In[ ]:


max_features = 500
count_vectorizer = CountVectorizer(max_features = max_features, stop_words = "english")
sparce_matrix = count_vectorizer.fit_transform(description_list).toarray()


# In[ ]:


print("most using {} words: {} ".format(max_features, count_vectorizer.get_feature_names()))


# In[ ]:


x = sparce_matrix
y = df.iloc[:,0].values


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size =0.1, random_state=42)


# In[ ]:


model = GaussianNB()
model.fit(x_train,y_train)


# In[ ]:


y_pred = model.predict(x_test)


# In[ ]:


print("accuracy : ", model.score(y_pred.reshape(-1,1),y_test))

