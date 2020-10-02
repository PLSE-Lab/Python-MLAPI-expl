#!/usr/bin/env python
# coding: utf-8

# ## Does it work on gujarati datasets?

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


train=pd.read_csv('/kaggle/input/gujarati-news-dataset/train.csv')
test=pd.read_csv('/kaggle/input/gujarati-news-dataset/valid.csv')


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


X=train['headline']
y=train['label']


# In[ ]:


train['label'].value_counts()


# In[ ]:


from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

text_clf = Pipeline([('tfidf', TfidfVectorizer()),
                     ('clf', LinearSVC()),
])


# In[ ]:


text_clf.fit(X, y)  


# In[ ]:


test_X=test['headline']
test_y=test['label']


# In[ ]:


predictions = text_clf.predict(test_X)


# In[ ]:


predictions


# ## The final check..

# In[ ]:


from sklearn import metrics

cm = metrics.confusion_matrix(test_y, predictions)
plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Confusion Matrix'
plt.title(all_sample_title, size = 15);


# Yes, it does work..
