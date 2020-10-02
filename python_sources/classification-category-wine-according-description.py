#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import eli5
import pandas as pd
import numpy as np

from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

import matplotlib.pyplot as plt


# In[ ]:


df_1 = pd.read_csv('../input/winemag-data_first150k.csv')
df_2 = pd.read_csv('../input/winemag-data-130k-v2.csv')
wine_df = pd.concat([df_1, df_2], ignore_index=True)
wine_df.shape


# In[ ]:


category = []
for point in wine_df['points']:
    if point >= 80 and point <= 89: category.append(0)
    if point >= 90 and point <= 100: category.append(1)
wine_df['category'] = pd.Series(category)


# In[ ]:


wine_df = wine_df.drop_duplicates('description')
wine_df.shape


# In[ ]:


p = np.bincount(wine_df['category'])
print(p)
print(max(p) / sum(p)) # acc if random predict 


# In[ ]:


train_text, test_text, ytrain, ytest = train_test_split(
    wine_df['description'], wine_df['category'], random_state=42)


# In[ ]:


get_ipython().run_cell_magic('time', '', "word_vectorizer = TfidfVectorizer(\n    sublinear_tf=True,\n    strip_accents='unicode',\n    analyzer='word',\n    token_pattern=r'\\w{1,}',\n    ngram_range=(1, 8))\nword_vectorizer.fit(train_text)")


# In[ ]:


from sklearn.linear_model import SGDClassifier
sgd_cls = SGDClassifier(max_iter=2)
sgd_cls.fit(word_vectorizer.transform(train_text), ytrain)


# In[ ]:


eli5.show_weights(sgd_cls, vec=word_vectorizer)


# In[ ]:


# show analysis description wine having 80-82 point. class - 0
eli5.show_prediction(
    sgd_cls, 
    wine_df['description'][wine_df['points'] <= 82].values[0], 
    vec=word_vectorizer)


# In[ ]:


# show analysis description wine having 98-100 point. class - 1
eli5.show_prediction(
    sgd_cls, 
    wine_df['description'][wine_df['points'] >= 98].values[0], 
    vec=word_vectorizer)


# In[ ]:


get_ipython().run_cell_magic('time', '', "char_vectorizer = TfidfVectorizer(\n    sublinear_tf=True,\n    strip_accents='unicode',\n    analyzer='char',\n    ngram_range=(1, 5))\nchar_vectorizer.fit(train_text)")


# In[ ]:


get_ipython().run_cell_magic('time', '', 'X = hstack([word_vectorizer.transform(train_text), char_vectorizer.transform(train_text)])')


# In[ ]:


from sklearn.linear_model import SGDClassifier
sgd_cls = SGDClassifier(max_iter=2)
sgd_cls.fit(X, ytrain)


# In[ ]:


predict = sgd_cls.predict(
    hstack([word_vectorizer.transform(test_text), char_vectorizer.transform(test_text)]))
acc = np.mean(ytest == np.around(predict))
print('accuracy: {0:.3}'.format(acc))


# In[ ]:




