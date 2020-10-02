#!/usr/bin/env python
# coding: utf-8

# Base line Code Using Tfidf-bigram/trigram

# In[43]:


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from lightgbm import LGBMRegressor
from scipy import sparse
from category_encoders.hashing import HashingEncoder
import os
from scipy.sparse import hstack, csr_matrix
import tqdm
print(os.listdir("../input"))


# In[44]:


get_ipython().run_cell_magic('time', '', "\ntrain = pd.read_csv('../input/train.csv',nrows =10000)\ntest = pd.read_csv('../input/test.csv',nrows=10000)")


# In[45]:


cat_feats = ['region', 'city', 'parent_category_name', 'category_name', 'param_1', 'param_2', 'param_3', 'user_type', 'image_top_1']
text_feats = ['title', 'description']
num_feats = ['price', 'item_seq_number']
allcols = cat_feats + text_feats + num_feats


# In[46]:


merged = pd.concat((train[allcols], test[allcols]), axis=0)
merged['price'] = merged['price'].apply(np.log1p)


# In[47]:


merged.head()


# In[48]:


merged.isnull().sum()


# In[49]:


word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    stop_words = None,
    encoding='KOI8-R',
    analyzer='word',
    token_pattern=r'\w{1,}',
    ngram_range=(1,1),
    dtype=np.float32,
    max_features=9000
)
# Character Stemmer
char_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    encoding='KOI8-R',
    analyzer='char',
    ngram_range=(1, 4),
    dtype=np.float32,
    max_features=5000
)


# In[50]:


get_ipython().run_cell_magic('time', '', "tfidf_matrices_1 = []\nfor feat in text_feats:\n    tfidf_matrices_1.append(word_vectorizer.fit_transform(merged[feat].fillna('').values))")


# In[51]:


get_ipython().run_cell_magic('time', '', "tfidf_matrices_2 = []\nfor feat in text_feats:\n    tfidf_matrices_2.append(char_vectorizer.fit_transform(merged[feat].fillna('').values))")


# In[52]:


get_ipython().run_cell_magic('time', '', "tfidf_matrices = sparse.hstack(tfidf_matrices_1,format='csr')")


# In[53]:


get_ipython().run_cell_magic('time', '', 'he = HashingEncoder()\ncat_df = he.fit_transform(merged[cat_feats].values)')


# In[54]:


full_matrix = sparse.hstack([cat_df.values, tfidf_matrices, merged[num_feats].fillna(-1).values], format='csr')


# In[56]:


get_ipython().run_cell_magic('time', '', "model = LGBMRegressor(max_depth=4, learning_rate=0.3, n_estimators=500)\nres = cross_val_score(model, full_matrix[:train.shape[0]], train['deal_probability'].values, cv=5, scoring='neg_mean_squared_error')\nres = [np.sqrt(-r) for r in res]\nprint(np.mean(res), np.std(res))")


# In[57]:


model.fit(full_matrix[:train.shape[0]], train['deal_probability'].values)
preds = model.predict(full_matrix[train.shape[0]:])


# In[58]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[59]:


plt.figure(figsize=(10, 7))
plt.hist(preds, bins=50);


# In[60]:


#sub = pd.read_csv('../input/sample_submission.csv')
#sub['deal_probability'] = preds
#sub['deal_probability'].clip(0.0, 1.0, inplace=True)
#sub.to_csv('../input/first_attempt.csv', index=False)
#sub.head()


# In[ ]:




