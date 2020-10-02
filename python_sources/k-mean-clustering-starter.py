#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


train = pd.read_csv('../input/training_text', sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])
test = pd.read_csv('../input/test_text', sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])


# In[ ]:


train.head()


# In[ ]:


len(train['Text'][0].split())


# In[ ]:


train['len_text'] = train['Text'].apply(lambda x: len(x.split()))


# In[ ]:


import seaborn as sns
sns.distplot(train['len_text'])


# In[ ]:


import nltk
from nltk.corpus import stopwords
stopwords = set(stopwords.words('english'))


# In[ ]:


train['Text'] = train['Text'].apply(lambda x : " ".join([word for word in x.lower().split() if word not in stopwords]))


# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
vect = TfidfVectorizer()
X = vect.fit_transform(train['Text'])
#X = vect.fit_transform(train['Text'][1:100])


# In[ ]:


get_ipython().run_line_magic('time', '')
from sklearn.cluster import KMeans
km = KMeans(n_clusters=10)
km.fit(X)


# In[ ]:


# https://stackoverflow.com/questions/27889873/clustering-text-documents-using-scikit-learn-kmeans-in-python
print("Top terms per cluster:")
order_centroids = km.cluster_centers_.argsort()[:, ::-1]
terms = vect.get_feature_names()
for i in range(10):
    print ("Cluster %d:" % i)
    for ind in order_centroids[i, :20]:
        print(' %s' % terms[ind])
    print()


# In[ ]:


train_var = pd.read_csv('../input/training_variants')
train_df = pd.merge(train, train_var, on='ID')


# In[ ]:


train_df[km.labels_ == 0].groupby('Class').size()


# In[ ]:


train_df[km.labels_ == 1].groupby('Class').size()


# In[ ]:


train_df[km.labels_ == 2].groupby('Class').size()


# In[ ]:


train_df[km.labels_ == 3].groupby('Class').size()


# In[ ]:


train_df[km.labels_ == 4].groupby('Class').size()


# In[ ]:


train_df[km.labels_ == 5].groupby('Class').size()


# In[ ]:


train_df[km.labels_ == 6].groupby('Class').size()


# In[ ]:


train_df[km.labels_ == 7].groupby('Class').size()


# In[ ]:


train_df[km.labels_ == 8].groupby('Class').size()


# In[ ]:


train_df[km.labels_ == 9].groupby('Class').size()


# So we can see that K-mean is able to cluster some classes e.g. check classes corresponding to labels 6 and 7 above while cluster for label 9 is not so good. I suspect that if we remove noisy and common occuring words, clustering could improve further

# In[ ]:




