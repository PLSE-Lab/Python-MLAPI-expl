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


true = pd.read_csv('/kaggle/input/fake-and-real-news-dataset/True.csv')
fake = pd.read_csv('/kaggle/input/fake-and-real-news-dataset/Fake.csv')
true


# In[ ]:


fake


# In[ ]:


fake['label'] = '0'
true['label'] = '1'


# In[ ]:


fake


# In[ ]:


true


# In[ ]:


news = pd.concat([true, fake], ignore_index=True)


# In[ ]:


news.subject.unique()


# In[ ]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(news['text'], news['label'], test_size=0.2, random_state=7)


# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)


# In[ ]:


tfidf_train=tfidf_vectorizer.fit_transform(x_train) 
tfidf_test=tfidf_vectorizer.transform(x_test)


# In[ ]:


from sklearn.linear_model import PassiveAggressiveClassifier
pac=PassiveAggressiveClassifier(max_iter=80)


# In[ ]:


pac.fit(tfidf_train,y_train)


# In[ ]:


from sklearn.metrics import accuracy_score, confusion_matrix
y_pred=pac.predict(tfidf_test)
score=accuracy_score(y_test,y_pred)
print(f'Accuracy: {round(score*100,2)}%')


# In[ ]:


confusion_matrix(y_test,y_pred, labels=['0','1'])

