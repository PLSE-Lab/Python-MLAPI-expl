#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
from collections import Counter
import matplotlib.pyplot as plt
#import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns # for making plots with seaborn
color = sns.color_palette()
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from sklearn.pipeline import FeatureUnion


# In[ ]:


train = pd.read_csv('../input/train.tsv',  sep="\t")
test = pd.read_csv('../input/test.tsv',  sep="\t")
sub = pd.read_csv('../input/../input/sampleSubmission.csv')


# In[ ]:


print('size of train data',train.shape)
print('size of test data',test.shape)


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


sub.head()


# In[ ]:


tfhash = [("tfidf", TfidfVectorizer(stop_words='english')),
       ("hashing", HashingVectorizer (stop_words='english'))]
X_train = FeatureUnion(tfhash).fit_transform(train.Phrase)
X_test = FeatureUnion(tfhash).transform(test.Phrase)
y = train.Sentiment
sub['Sentiment'] = LinearSVC(dual=False).fit(X_train,y).predict(X_test) 
sub.to_csv("svc.csv", index=False)


# **To Do :**
#   * Some plots
#   * LB Accuracy improvement

# # More To Come Stayed Tuned !!
