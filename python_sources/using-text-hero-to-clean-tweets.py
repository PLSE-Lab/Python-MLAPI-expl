#!/usr/bin/env python
# coding: utf-8

# # Text Hero Sample
# - https://github.com/jbesomi/texthero

# In[ ]:


get_ipython().system('pip install texthero')


# In[ ]:


import texthero as hero
import pandas as pd 


# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


df_train = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')
df_train.head()


# In[ ]:


df_train['tfidf'] = (
    df_train['text']
    .pipe(hero.clean)
    .pipe(hero.tfidf)
)


# In[ ]:


df_train['kmeans_labels'] = (
    df_train['tfidf']
    .pipe(hero.kmeans, n_clusters=5)
    .astype(str)
)


# In[ ]:


df_train.head()


# In[ ]:


df_train['pca'] = df_train['tfidf'].pipe(hero.pca)


# In[ ]:


hero.scatterplot(df_train, 'pca', color='target', title="Disastor Tweets by Target")

