#!/usr/bin/env python
# coding: utf-8

# # I. Importing required libraries and data

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


import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.wrappers.scikit_learn import KerasClassifier
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer
from sklearn.metrics import roc_auc_score
import scikitplot as skplt
import matplotlib.pyplot as plt


# In[ ]:


data_train = pd.read_csv('/kaggle/input/quora-insincere-questions-classification/train.csv')
data_val = pd.read_csv('/kaggle/input/quora-insincere-questions-classification/test.csv')
print(data_train.shape)
print(data_train.isnull().sum())
print(data_val.isnull().sum())
print(data_train['target'].value_counts())
data_train.head()


# # II. Lemmatization

# In[ ]:


def lemmatize(data):
    lemmatizer = WordNetLemmatizer()
    lem_data = []
    for text in data:
        lem_text = ''
        for word in text.split():
            word = word.lower()
            lem_word = lemmatizer.lemmatize(word)
            lem_word = lemmatizer.lemmatize(lem_word, pos='v')
            lem_text = lem_text + ' ' + lem_word
        lem_data.append(lem_text)
    return lem_data
    


# In[ ]:


X = data_train['question_text']
y = data_train['target']
ques_id = data_val['qid']
X_val = data_val['question_text']
X_lem = lemmatize(X.tolist())
X_val = lemmatize(X_val.tolist())
X_lem[:10]


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X_lem, y, random_state=0)
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)
word2idx = tokenizer.word_index
vocab_size = len(word2idx)
max_len = 100
train_seq = tokenizer.texts_to_sequences(X_train)
train_pad = pad_sequences(train_seq, maxlen=max_len)
test_seq = tokenizer.texts_to_sequences(X_test)
test_pad = pad_sequences(test_seq, maxlen=max_len)
val_seq = tokenizer.texts_to_sequences(X_val)
val_pad = pad_sequences(val_seq, maxlen=max_len)


# # III. Vectorization using Tfidf

# In[ ]:


tfidf = TfidfVectorizer(ngram_range=(1,3), min_df=5, max_df=0.9)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)
X_val_tfidf = tfidf.transform(X_val)


# # IV. Training the Classifier using RandomizedSearchCV

# In[ ]:


clf = LogisticRegression(max_iter=1000)
params = {
    'C':[0.001,0.01,0.1,1,10],
}
scorer=make_scorer(f1_score)
grid = RandomizedSearchCV(clf, params, scoring=scorer)
cv_results = grid.fit(X_train_tfidf, y_train)


# In[ ]:


cv_results.best_params_


# In[ ]:


cv_results.cv_results_


# # V. Model prediction and evaluation

# **Here, we use the ROC curve as a metric for evaluating our classification model. Since the dataset is imbalanced with high number of '0' labels as compared '1' labels, accuracy_score is an inappropriate metric as it tends to be biased towards the majority class. Hence, area under the ROC curve is used instead, whose value is between 0.5 and 1, and the higher, the better.**

# In[ ]:


best_clf = cv_results.best_estimator_
best_clf.fit(X_train_tfidf, y_train)
y_pred = best_clf.predict(X_test_tfidf)
print(y_pred)
y_val = best_clf.predict(X_val_tfidf)
y_val


# In[ ]:


y_prob = best_clf.predict_proba(X_test_tfidf)
pos_class_prob = [prob[1] for prob in y_prob]
print(roc_auc_score(y_test, pos_class_prob))


# In[ ]:


#Plotting ROC curve for test data.
skplt.metrics.plot_roc_curve(y_test, y_prob)
plt.show()


# In[ ]:


#Submitting predictions.
df = pd.DataFrame()
df['qid'] = ques_id
df['prediction'] = y_val
df.to_csv('submission.csv',index=False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




