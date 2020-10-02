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


# ## Step 1: load packages
# 

# In[ ]:


# Load packages 
import timeit
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
import time
import importlib

from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
logging.basicConfig(format='[%(asctime)s %(levelname)8s] %(message)s', level=logging.INFO, datefmt='%m-%d %H:%M:%S')

import tensorflow_hub as hub 
import tensorflow as tf 
from tqdm.notebook import tqdm 


# ## Step 2: Transfrom
# A [Universal Sentence Encoders](https://tfhub.dev/google/universal-sentence-encoder-large/5) encodes text into high dimensional vectors that can be used for text classification, semantic similarity, clustering and other natural language tasks.
# 
# There is a [Universal Sentence Encoders family](https://tfhub.dev/google/collections/universal-sentence-encoder/1), we choose the large one. 

# In[ ]:





def transfrom(text_train, text_test):
    large_use = 'https://tfhub.dev/google/universal-sentence-encoder-large/5'
    embed = hub.load(large_use)

    vector_train = [tf.reshape(embed([line]), [-1]).numpy() for line in tqdm(text_train)]
    vector_test = [tf.reshape(embed([line]), [-1]).numpy() for line in tqdm(text_test)]

    return vector_train, vector_test
    


# Now transform texts into vectors

# In[ ]:


train = pd.read_csv('../input/nlp-getting-started/train.csv')
test = pd.read_csv('../input/nlp-getting-started/test.csv')
vector_train, vector_test = transfrom(train.text, test.text)


# ## Step 3: build model and train

# In[ ]:


model = svm.SVC()
X_train, X_val, y_train, y_val = train_test_split(vector_train, train.target, test_size=0.2, random_state=2020)
model.fit(X_train, y_train)

preds = model.predict(X_val)
print('Accuracy score', accuracy_score(y_val, preds))
print('f1_score', f1_score(y_val, preds))


# ## Step 4: predict and export results

# In[ ]:


final_preds = model.predict(vector_test)
sub = pd.read_csv(f"../input/nlp-getting-started/sample_submission.csv")
sub['target'] = final_preds
sub.to_csv('submission.csv', index=False)


# This is a very simple kernel showing how to use SVM to predict disaster tweets. Potential improvements include: 
# 1. `GridSearchCV` to find better parameters for the model
# 2. considering using **keywords** and **location** information
# 3. use `predict_proba` instead of `predict` and find the best threshold to maximum `f1_score`
# 4. data cleaning, e.g., remove urls, punctuations, etc. 
# 
