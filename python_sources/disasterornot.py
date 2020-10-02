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


train=pd.read_csv('../input/nlp-getting-started/train.csv')
test= pd.read_csv('../input/nlp-getting-started/test.csv')
train.head()


# In[ ]:


train.shape


# In[ ]:


train.isnull().sum()


# In[ ]:


not_disaster = train[train['target']==0]
not_disaster.head()


# In[ ]:


disaster = train [train['target']==1]
disaster.head()


# In[ ]:


no_of_disaster = len(disaster)
no_of_disaster
no_of_not_disaster = len(not_disaster)
targets = train['target'].value_counts()


# In[ ]:


import seaborn as sns
sns.barplot(x=targets.index,y=targets)


# In[ ]:


train['text'] = train['text'].apply(lambda x: x.lower())
test['text'] = test['text'].apply(lambda x: x.lower())


# In[ ]:


import re
import string
def clean_text(text):
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text


# In[ ]:


train['text'] = train['text'].apply(lambda x: clean_text(x))
test['text'] = test['text'].apply(lambda x: clean_text(x))


# In[ ]:


train.head()


# In[ ]:


import nltk
nltk.download('stopwords')


# In[ ]:


from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# In[ ]:


stopwords = stopwords.words('english')


# In[ ]:


def remove_stopwords(text):
    text = [item for item in text.split() if item not in stopwords]
    return ' '.join(text)

train['updated_text'] = train['text'].apply(remove_stopwords)
test['updated_text'] = test['text'].apply(remove_stopwords)

train.head()


# In[ ]:


from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer("english")

def stemming(text):
    text = [stemmer.stem(word) for word in text.split()]
    return ' '.join(text)

train['stemmed_text'] = train['updated_text'].apply(stemming)
test['stemmed_text'] = test['updated_text'].apply(stemming)
train.head()


# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(analyzer='word',binary=True)
cv.fit(train['stemmed_text'])

trained = cv.fit_transform(train['stemmed_text'])
tested = cv.transform(test['stemmed_text'])


# In[ ]:


y=train['target']


# In[ ]:


from sklearn.svm import SVC
from sklearn import model_selection

model1 = SVC()
scores = model_selection.cross_val_score(model1,trained,y,cv=3,scoring='f1')


# In[ ]:


scores.mean()


# In[ ]:


from sklearn.naive_bayes import MultinomialNB
model2 = MultinomialNB(alpha=1)
scores = model_selection.cross_val_score(model2, trained, y, cv=3, scoring="f1")
scores.mean()


# In[ ]:


from sklearn.linear_model import LogisticRegression
model3 = LogisticRegression()
scores = model_selection.cross_val_score(model3,trained,y,cv=3,scoring='f1')
scores.mean()


# In[ ]:


model2.fit(trained,y)


# In[ ]:


submission = pd.read_csv('../input/nlp-getting-started/sample_submission.csv')
submission['target'] = model2.predict(tested)


# In[ ]:


submission.head()


# In[ ]:


submission.to_csv("submission.csv", index=False)

