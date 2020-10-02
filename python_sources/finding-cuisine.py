#!/usr/bin/env python
# coding: utf-8

# Finding cuisine from the ingredients of recipes

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import nltk
import re
from nltk.stem import WordNetLemmatizer

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


train = pd.read_json('../input/train.json')
train.head()


# In[ ]:


train['ingredient_list'] = [','.join(z).strip() for z in train['ingredients']]


# In[ ]:


train.head()


# In[ ]:


test = pd.read_json('../input/test.json')
test.head()


# In[ ]:


ingredients = train['ingredient_list']
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(ingredients).todense()


# In[ ]:


cuisines = train['cuisine']
cuisines.head()


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(tfidf_matrix, cuisines)


# In[ ]:


test['ingredient_list'] = [','.join(z).strip() for z in test['ingredients']]
test.head()
test_ingredients = test['ingredient_list']
test_tfidf_matrix = vectorizer.transform(test_ingredients)
test_cuisines = clf.predict(test_tfidf_matrix)


# In[ ]:


test['cuisine'] = test_cuisines


# In[ ]:


test.head()


# In[ ]:


test[['id' , 'cuisine' ]].to_csv("submission.csv", index=False)


# In[ ]:


test[['id','cuisine']].head()


# In[ ]:




