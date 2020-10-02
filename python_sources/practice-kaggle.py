#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.svm import SVC, NuSVC, LinearSVC


# In[ ]:


raw_test = pd.read_csv('../input/test.csv')
raw_train = pd.read_csv('../input/train.csv')

train_text = raw_train['question_text']
train_target = raw_train['target']


# In[ ]:


from sklearn.model_selection import train_test_split


x_train, x_test, y_train, y_test = train_test_split(train_text, train_target, test_size = 0.2)


# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(x_train)


# In[ ]:


from sklearn.feature_extraction.text import TfidfTransformer

#tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
#X_train_tf = tf_transformer.transform(X_train_counts)

tfidf_transformer = TfidfTransformer()
#X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)


# In[ ]:


from sklearn.naive_bayes import MultinomialNB


# In[ ]:


#clf = MultinomialNB().fit(X_train_tfidf, x_test)


# In[ ]:


#clf


# In[ ]:


from sklearn.pipeline import Pipeline
text_clf = Pipeline([
    ('vect', CountVectorizer(min_df = 10, max_df = 0.9, stop_words = 'english')),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB())
])


# In[ ]:


text_clf.fit(x_train,y_train)


# In[ ]:


text_clf.score(x_test,y_test)


# In[ ]:


predict_test = raw_test['question_text']


# In[ ]:


predicted = text_clf.predict(predict_test)


# In[ ]:


my_df=pd.DataFrame(predicted)


# In[ ]:


submission_pd = pd.concat([raw_test["qid"],pd.DataFrame(predicted)],axis=1)


# In[ ]:


submission_pd.columns = ['qid','prediction']


# In[ ]:


submission_pd.to_csv("submission.csv",index=False)


# In[ ]:




