#!/usr/bin/env python
# coding: utf-8

# #### Bagi yang bingung cara mengolah dataset tersebut, kernel ini bisa dijadikan panduan. Masih banyak hal yang bisa teman-teman lakukan di luar kernel, misalnya:
# - menghapus stopwords
# - menangani kata-kata yang tidak baku
# - melakukan stemming
# - melakukan word embedding
# - menggunakan model lainnya
# - dan lainnya

# ## Import Library

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, classification_report, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

get_ipython().run_line_magic('matplotlib', 'inline')


# ## Load dataset

# In[ ]:


df = pd.read_csv('../input/train-data-2.csv')


# In[ ]:


df.head()


# In[ ]:


df = df.drop(['id'], axis=1) # drop id karena tidak penting


# In[ ]:


# pada kasus ini hanya menggunakan kolom review_sangat_singkat, header_review tidak digunakan.
X = df['review_sangat_singkat']
y = df['rating']


# ### Proporsi kelas

# In[ ]:


plt.title('Proporsi tiap kelas')
sns.countplot(y)


# dari plot tersebut, sebaran kelas data tidak merata, review 4/5 sangat banyak dibandingkan yang lain. Banyak hal yang bisa teman-teman lakukan untuk menangani data yang seperti ini. Penting untuk menganalisis nilai recall untuk rating minoritas.

# ## Split data

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)


# ## Pipeline
# - Feature extraction
#     - CountVectorizer
#     - TfIdf
# - Modeling

# In[ ]:


cvec = CountVectorizer(min_df=25, ngram_range=(1,2))
tfidf_trans = TfidfTransformer()
model = LogisticRegression()

text_clf = Pipeline([
    ('vect', cvec),
    ('tfidf', tfidf_trans),
    ('clf', model),
])

text_clf.fit(X_train, y_train)


# ## Train score

# In[ ]:


y_train_pred = text_clf.predict(X_train)


# In[ ]:


print(classification_report(y_train, y_train_pred))
print('accuracy', accuracy_score(y_train, y_train_pred))
print('rmse', np.sqrt(mean_squared_error(y_train, y_train_pred)))


# ## Test score

# In[ ]:


y_test_pred = text_clf.predict(X_test)


# In[ ]:


print(classification_report(y_test, y_test_pred))
print('accuracy', accuracy_score(y_test, y_test_pred))
print('rmse', np.sqrt(mean_squared_error(y_test, y_test_pred)))


# hasil masil bertendensi besar untuk memprediksi ke rating 1 dan 5.
# 
# Apakah model logistic regression tepat untuk kita gunakan untuk kasus ini?
# 
# Banyak hal lain yang bisa teman-teman lakukan untuk meningkatkan performa model

# ## Full Train

# In[ ]:


X_full = pd.concat([X_train, X_test])
y_full = pd.concat([y_train, y_test])


# In[ ]:


text_clf.fit(X_full, y_full)


# In[ ]:


y_full_pred = text_clf.predict(X_full)


# In[ ]:


print(classification_report(y_full, y_full_pred))
print('accuracy', accuracy_score(y_full, y_full_pred))
print('rmse', np.sqrt(mean_squared_error(y_full, y_full_pred)))


# ## Submit to Kaggle

# In[ ]:


# test_data = pd.read_csv('../input/test-data-2.csv')


# In[ ]:


# test_data.head()


# In[ ]:


# test_data_pred = text_clf.predict(test_data['review_sangat_singkat'])


# In[ ]:


# test_data_pred


# In[ ]:


# submission = pd.read_csv('../input/sample-submission-2.csv')


# In[ ]:


# submission['rating'] = test_data_pred


# In[ ]:


# submission.to_csv('submission-2.csv', index=False)


# In[ ]:




