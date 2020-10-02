#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install pytorch-transformers > null')


# In[ ]:


import string
import numpy as np
import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer
import tqdm
import spacy

import torch
from pytorch_transformers import *

class StockNewsDataset:
    def __init__(self, file_path='Combined_News_DJIA.csv'):
        data = pd.read_csv(file_path)
        self.train = data[data['Date'] < '2014-01-01']
        self.val = data[(data['Date'] >= '2014-01-01') & (data['Date'] < '2015-01-01')]
        self.test = data[data['Date'] >= '2015-01-01']

    def get_splits(self):
        return self.train, self.val, self.test

    def get_headlines(self, df):
        headlines = []
        for _, row in df.iterrows():
            headlines.append(self._cleaning(' '.join(str(x) for x in row[2:27])))
            #headlines.append(self._cleaning(' '.join(str(x) for x in row[2:3])))
        return headlines
        
    def _cleaning(self, text):
        table = str.maketrans({key: ' ' for key in string.punctuation})
        text = text.replace("b\'", " ")
        text = text.replace("b\"", " ")
        text = text.translate(table)
        return text

    def get_tfidfs(self):
        train_headlines = self.get_headlines(self.train)
        val_headlines = self.get_headlines(self.val)
        test_headlines = self.get_headlines(self.test)

        vectorizer = TfidfVectorizer(min_df=0.03, max_df=0.97, max_features=10000, ngram_range=(2, 2))

        train_tfidf = vectorizer.fit_transform(train_headlines)
        val_tfidf = vectorizer.transform(val_headlines)
        test_tfidf = vectorizer.transform(test_headlines)

        return (train_tfidf, self.train['Label']),             (val_tfidf, self.val['Label']),             (test_tfidf, self.test['Label']) 
    def get_vectors(self):
        nlp = spacy.load('en_core_web_lg')

        train_vec = np.array([nlp(headline).vector for headline in self.get_headlines(self.train)])
        val_vec = np.array([nlp(headline).vector for headline in self.get_headlines(self.val)])
        test_vec = np.array([nlp(headline).vector for headline in self.get_headlines(self.test)])

        return (train_vec, self.train['Label']), (val_vec, self.val['Label']), (test_vec, self.test['Label'])

    def _bert_vector(self, df):
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel(BertConfig.from_pretrained('bert-base-uncased'))
        output = []
        for _, row in tqdm.tqdm(df.iterrows()):
            vec = np.zeros((768,))
            for headline in row[2:12]:
                input_ids = tokenizer.encode('[CLS] '+self._cleaning(str(headline))+' [SEP]')[:512]
                vec += model(torch.tensor([input_ids]))[1].detach().numpy().squeeze()
            output.append(vec/10.)
        return np.array(output)

    def get_bert_vectors(self):
        train_vec = self._bert_vector(self.train)
        val_vec = self._bert_vector(self.val)
        test_vec = self._bert_vector(self.test)
        return (train_vec, self.train['Label']), (val_vec, self.val['Label']), (test_vec, self.test['Label'])


# In[ ]:


import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from utils import *


# In[ ]:


dataset = StockNewsDataset('../input/Combined_News_DJIA.csv')
(train_x, train_y), (val_x, val_y), (test_x, test_y) = dataset.get_tfidfs()
print(train_x.shape, val_x.shape, test_x.shape)


# In[ ]:


model = LogisticRegression()
model = model.fit(train_x, train_y)
val_pred = model.predict(val_x)
test_pred = model.predict(test_x)
val_acc = accuracy_score(val_y, val_pred)
test_acc = accuracy_score(test_y, test_pred)
print(val_acc, test_acc)


# In[ ]:


(train_x, train_y), (val_x, val_y), (test_x, test_y) = dataset.get_vectors()
print(train_x.shape, val_x.shape, test_x.shape)


# In[ ]:


model = LogisticRegression()
model = model.fit(train_x, train_y)
val_pred = model.predict(val_x)
test_pred = model.predict(test_x)
val_acc = accuracy_score(val_y, val_pred)
test_acc = accuracy_score(test_y, test_pred)
print(val_acc, test_acc)


# In[ ]:


(train_x, train_y), (val_x, val_y), (test_x, test_y) = dataset.get_bert_vectors()
print(train_x.shape, val_x.shape, test_x.shape)


# In[ ]:


model = LogisticRegression(penalty='l2')
model = model.fit(train_x, train_y)
val_pred = model.predict(val_x)
test_pred = model.predict(test_x)
val_acc = accuracy_score(val_y, val_pred)
test_acc = accuracy_score(test_y, test_pred)
print(val_acc, test_acc)


# In[ ]:




