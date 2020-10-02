#!/usr/bin/env python
# coding: utf-8

# In[4]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[5]:


train_data = pd.read_csv("../input/train.csv")


# **Statewise statistics of training data. Here, it tells us the count of projects submitted from each state.**

# In[6]:


train_data.groupby("school_state").describe()


# **Applying Tf-idf feature exratcion technique for the project_essay_1 and project_essay_2 attributes**

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
term_vectors = TfidfVectorizer(min_df=80).fit(train_data["project_essay_1"])

print("no.of Features: {0}".format(len(term_vectors.get_feature_names())))

#sorted_coef = tf_vectors.coef_[0].argsort()
features = term_vectors.get_feature_names()
tfidf = term_vectors.transform(train_data["project_essay_1"])
sorted_arguments = tfidf.max(0).toarray()[0].argsort()

feature_names = np.array(features)
print("features with samll tdidf:{0}".format(feature_names[sorted_arguments[0:10]]))
print("features with large tdidf:{0}".format(feature_names[sorted_arguments[-11:-1]]))
#print(tf_vectors._sort_features())

