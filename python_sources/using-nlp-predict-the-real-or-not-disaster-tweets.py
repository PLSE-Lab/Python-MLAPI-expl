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


import numpy as np 
import pandas as pd
from sklearn import feature_extraction, linear_model, model_selection, preprocessing


# Reading files of train and test

# In[ ]:


train=filenames[0]
test=filenames[1]
print(train)
print(test)


# In[ ]:


train_pd=pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')
test_pd=pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')
print(train_pd)


# In[ ]:


print(train_pd.head())


# In[ ]:


train_pd[train_pd["target"] == 1]["text"].values[3]


# In[ ]:


train_pd[train_pd["target"] == 1]["text"].values[4]


# In[ ]:


count_vectorizer = feature_extraction.text.CountVectorizer()
print(count_vectorizer)


# In[ ]:


## let's get counts for the first 5 tweets in the data
count_words = count_vectorizer.fit_transform(train_pd["text"][0:10])


# In[ ]:


print(count_words[0].todense().shape)


# In[ ]:


train_vectors = count_vectorizer.fit_transform(train_pd["text"])
test_vectors = count_vectorizer.transform(test_pd["text"])


# In[ ]:


model_linear = linear_model.RidgeClassifier()


# In[ ]:


predict = model_selection.cross_val_score(model_linear, train_vectors, train_pd["target"], cv=3, scoring="f1")


# In[ ]:


predict


# In[ ]:


model_linear.fit(train_vectors, train_pd["target"])


# In[ ]:


sample_submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")


# In[ ]:


sample_submission.head()


# In[ ]:


sample_submission["target"] = model_linear.predict(test_vectors)


# In[ ]:


sample_submission.head()


# **save the file**

# In[ ]:


sample_submission.to_csv("submission_competition.csv", index=False)

