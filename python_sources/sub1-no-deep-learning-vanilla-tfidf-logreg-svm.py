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


## Using simple TFIDF features and SVM 
## Observation -- Although accuracy comes out to be high, that is not a good measure-the F1 score is low


# In[ ]:


import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer


# In[ ]:


train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')


# In[ ]:


train_df.tail(30)
insincere_q = train_df[train_df['target']==1]
insincere_q.shape


# In[ ]:


# function to clean data
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
from nltk.stem import WordNetLemmatizer

lemm_ = WordNetLemmatizer()
st = PorterStemmer()
stops = set(stopwords.words("english"))
def cleanData(text, lowercase = True, remove_stops = True, stemming = False, lemma = True):
    #txt = str(text)
    #print(text)
    #txt = text.encode('utf-8').strip()
    txt = str(text)
    txt = re.sub(r'[^a-zA-Z. ]+|(?<=\\d)\\s*(?=\\d)|(?<=\\D)\\s*(?=\\d)|(?<=\\d)\\s*(?=\\D)',r'',txt)
    txt = re.sub(r'\n',r' ',txt)
    
    #converting to lower case
    if lowercase:
        txt = " ".join([w.lower() for w in txt.split()])
    
    # removing stop words
    if remove_stops:
        txt = " ".join([w for w in txt.split() if w not in stops])
    
    # stemming
    if stemming:
        txt = " ".join([st.stem(w) for w in txt.split()])
        
    if lemma:
        txt = " ".join([lemm_.lemmatize(w) for w in txt.split()])

    return txt


# In[ ]:


train_df['clean_question_text'] = train_df['question_text'].map(lambda x: cleanData(x))


# In[ ]:


test_df['clean_question_text'] = test_df['question_text'].map(lambda x: cleanData(x))


# In[ ]:


max_features = 50000  ##More than this would filter in noise also
tfidf_vectorizer = TfidfVectorizer(ngram_range =(2,4) , max_df=0.90, min_df=5, max_features=max_features) ##4828 features found
#tfidf_feature_names = tfidf_vectorizer.get_feature_names()


# In[ ]:


X = tfidf_vectorizer.fit_transform(train_df['clean_question_text'])


# In[ ]:


X_te = tfidf_vectorizer.transform(test_df['clean_question_text'])


# In[ ]:


tfidf_feature_names = tfidf_vectorizer.get_feature_names()


# In[ ]:


y = train_df["target"]


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3,random_state=42)


# In[ ]:


# Classification and prediction
clf = LogisticRegression(C=10, penalty='l1')
clf.fit(X_train, y_train)


# In[ ]:


clf.score(X_val, y_val)


# In[ ]:


p_test = clf.predict_proba(X_te)[:, 0]
y_te = (p_test > 0.5).astype(np.int)


# In[ ]:


from sklearn.svm import LinearSVC
svm_model = LinearSVC(C=0.5).fit(X_train, y_train)


# In[ ]:


score = svm_model.score(X_train, y_train)
print('score', score)


# In[ ]:


#pred_test_y = (pred_test_y > best_thresh).astype(int)
pred_test_y = svm_model.predict(X_te)

out_df = pd.DataFrame({"qid":test_df["qid"].values})
out_df['prediction'] = pred_test_y
out_df.to_csv("submission.csv", index=False)


# In[ ]:


out_df.tail()


# In[ ]:




