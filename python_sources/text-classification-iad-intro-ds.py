#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import gc


# In[ ]:


Train = pd.read_csv('../input/texts-classification-iad-hse-intro-2020/train.csv')


# In[ ]:


Test = pd.read_csv('../input/texts-classification-iad-hse-intro-2020/test.csv')


# In[ ]:


Train.head()


# In[ ]:


Test.head()


# In[ ]:


Train.isnull().sum()


# In[ ]:


Test.isnull().sum()


# In[ ]:


Train.fillna('', inplace=True)


# In[ ]:


Test.fillna('', inplace=True)


# In[ ]:


Train['title&description'] = Train['title'].str[:] + ' ' + Train['description'].str[:]


# In[ ]:


Test['title&description'] = Test['title'].str[:] + ' ' + Test['description'].str[:]


# In[ ]:


Train.drop(columns=['title', 'description'], inplace=True)
Test.drop(columns=['title', 'description'], inplace=True)


# In[ ]:


gc.collect()


# In[ ]:


import nltk

nltk.download('stopwords')

from nltk.corpus import stopwords

ru_stopwords = list(stopwords.words("russian"))


# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[ ]:


tf_idf = TfidfVectorizer(stop_words=ru_stopwords)
tf_idf.fit(Train['title&description'])


# In[ ]:


from sklearn.svm import LinearSVC


# In[ ]:


Train_tf_idf = tf_idf.transform(Train['title&description'])
Test_tf_idf = tf_idf.transform(Test['title&description'])


# In[ ]:


clf = LinearSVC()
clf.fit(Train_tf_idf, Train['Category'])


# In[ ]:


del Train, Train_tf_idf
gc.collect()


# In[ ]:


Answer = pd.DataFrame(columns=['Id', 'Category'])
Answer['Id'] = Test['itemid']


# In[ ]:


Answer['Category'] = clf.predict(Test_tf_idf)


# In[ ]:


Answer.to_csv('my_submission_tfidfvect_nltk_stopwords_lsvc.csv', index=None)

