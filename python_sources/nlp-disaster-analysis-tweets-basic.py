#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import re


# loading data*

# In[ ]:


df_train = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')
df_test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')


# In[ ]:


df_train.head()


# In[ ]:


target = df_train['target']


# In[ ]:


df_train.drop('target', axis=1, inplace=True)


# In[ ]:


join_df = [df_train, df_test]
both_df = pd.concat(join_df, sort=True, keys=['x','y'])


# In[ ]:


both_df.head()


# In[ ]:


no_na_df = pd.DataFrame()


# In[ ]:


no_na_df['text'] =  both_df['text'] + both_df['keyword'].apply(lambda x: ' '+str(x)) + both_df['location'].apply(lambda x: ' '+str(x))


# In[ ]:


no_na_df['text'] = no_na_df['text'].replace(r'http\S+', '', regex=True).replace(r'www\S+', '', regex=True)


# In[ ]:


no_na_df


# In[ ]:


from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
stop = stopwords.words('english')


# In[ ]:


from nltk.stem import PorterStemmer
pst = PorterStemmer()


# In[ ]:


no_na_df['text'] = no_na_df['text'].apply(lambda x: " ".join(re.split("[^a-zA-Z]*", x)) if x else '')


# In[ ]:


no_na_df['text'] = no_na_df['text'].apply(lambda x: x.lower())


# In[ ]:


no_na_df


# In[ ]:


no_na_df['text'] = no_na_df['text'].apply(lambda x: ' '.join([lemmatizer.lemmatize(i) for i in x.split() if i not in stop]))


# In[ ]:


no_na_df['text'] = no_na_df['text'].apply(lambda x: ' '.join([pst.stem(i) for i in x.split() if i not in stop]))


# In[ ]:


corpus = np.array(no_na_df['text'])


# In[ ]:


corpus1 = np.array(no_na_df.loc['y']['text'])


# In[ ]:


len(no_na_df.loc['x']['text'])


# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1000000)
X = cv.fit_transform(corpus).toarray()


# In[ ]:


y = cv.transform(corpus1).toarray()


# new_dict = dict()
# for i in unique_keywords:
#     full_data_to_train_list = []
#     for j in full_data_to_train:
#         full_data_to_train_list.append(1 if i in j else 0)
#     new_dict[i] = full_data_to_train_list

# In[ ]:


len(X)


# In[ ]:


df_submit = pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')
df_submit.head()


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X[:7613], target.values, test_size=0.8, random_state=0)


# In[ ]:


from sklearn.naive_bayes import MultinomialNB


# In[ ]:


from sklearn.linear_model import RidgeClassifier


# In[ ]:


from sklearn.tree import DecisionTreeClassifier


# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


gnb = RandomForestClassifier(max_depth=10, random_state=0)
classifier = gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)


# In[ ]:


print(sum([1 for i in range(len(y_pred)) if y_pred[i]!=y_test[i]]),len(y_pred))


# In[ ]:


gnb = DecisionTreeClassifier(random_state=0)
classifier = gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)


# In[ ]:


print(sum([1 for i in range(len(y_pred)) if y_pred[i]!=y_test[i]]),len(y_pred))


# In[ ]:


gnb = RidgeClassifier()
classifier = gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)


# In[ ]:


print(sum([1 for i in range(len(y_pred)) if y_pred[i]!=y_test[i]]),len(y_pred))


# In[ ]:


gnb = GaussianNB()
classifier = gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)


# In[ ]:


print(sum([1 for i in range(len(y_pred)) if y_pred[i]!=y_test[i]]),len(y_pred))


# In[ ]:


gnb = MultinomialNB()
classifier = gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)


# In[ ]:


print(sum([1 for i in range(len(y_pred)) if y_pred[i]!=y_test[i]]),len(y_pred))


# In[ ]:


gnb = MultinomialNB()
classifier = gnb.fit(X[:7613], target.values)
y_pred = gnb.predict(X[7613:])


# In[ ]:


df_submit['target'] = y_pred


# In[ ]:


df_submit.to_csv('sample_submission.csv', index=False)

