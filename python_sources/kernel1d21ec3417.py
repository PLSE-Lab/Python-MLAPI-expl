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


import nltk
import string
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from nltk.stem.porter import PorterStemmer
import re
import string
from bs4 import BeautifulSoup
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import pickle

from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score


# In[ ]:


df = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/train.csv.zip')
test_data = pd.read_csv("../input/jigsaw-toxic-comment-classification-challenge/test.csv.zip")


# In[ ]:


df.head(10)


# In[ ]:


df.shape


# In[ ]:


df.describe()


# In[ ]:


rslt_df = df[(df['toxic'] == 0) & (df['severe_toxic'] == 0) & (df['obscene'] == 0) & (df['threat'] == 0) & (df['insult'] == 0) & (df['identity_hate'] == 0)]
rslt_df2 = df[(df['toxic'] == 1) & (df['severe_toxic'] == 0) & (df['obscene'] == 0) & (df['threat'] == 0) & (df['insult'] == 0) & (df['identity_hate'] == 0)]
new1 = rslt_df[['id', 'comment_text', 'toxic']].iloc[:23891].copy() 
new2 = rslt_df2[['id', 'comment_text', 'toxic']].iloc[:946].copy()
new = pd.concat([new1, new2], ignore_index=True)


# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_df=0.95, min_df=5)
Xv = vectorizer.fit(new['comment_text'])
import pickle


# In[ ]:



import numpy as np
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(new["comment_text"], new['toxic'], test_size=0.33)


# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
#vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_df=0.95, min_df=5)
X1 = vectorizer.transform(X_train)
X_test1= vectorizer.transform(X_test)


# In[ ]:



from sklearn.tree import DecisionTreeClassifier


# In[ ]:


dt = DecisionTreeClassifier(max_depth=4)
dt= dt.fit(X1,y_train)


# In[ ]:


y_pred = dt.predict(X_test1)


# In[ ]:


#evaluating algo

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test,y_pred))


# In[ ]:


y_pred[:34]


# In[ ]:


from sklearn import tree
from IPython.display import SVG
from graphviz import Source
from IPython.display import display
graph = Source(
    tree.export_graphviz(
        dt,
        out_file=None,
        feature_names=vectorizer.get_feature_names(),
        class_names=['1' , '0'],
        filled = True)
)
display(SVG(graph.pipe(format='svg')))


# In[ ]:


pip install lime


# In[ ]:


import lime
from lime import lime_text
from sklearn.pipeline import make_pipeline
c = make_pipeline(vectorizer, dt)


# In[ ]:





# In[ ]:





# In[ ]:


pip install treeinterpreter


# In[ ]:


import treeinterpreter
from treeinterpreter import treeinterpreter as ti


# In[ ]:


predictions, biases, contributions = ti.predict( dt, vectorizer)
predictions[4], biases[4]


# In[ ]:




