#!/usr/bin/env python
# coding: utf-8

# > This kernel is based on the work of https://www.kaggle.com/thebrownviking20/bert-multiclass-classification

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import sys
import zipfile
import datetime
print(os.listdir("../input"))
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/"))
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
df = train

df.columns = ['ArticleId', 'Text', 'Drug', 'Category']
test.columns = ['ArticleId', 'Text', 'Drug']
df = df.drop('ArticleId', axis=1)
test = test.drop('ArticleId', axis=1)
df.drop('Drug', axis=1, inplace=True)
test.drop('Drug', axis=1, inplace=True)
df.head()

# Any results you write to the current directory are saved as output.
#     print(os.listdir(f"../input/{d}"))
# print(os.listdir("../input/googles-bert-model/chinese_l-12_h-768_a-12/chinese_L-12_H-768_A-12"))
# Any results you write to the current directory are saved as output.


# # 1. Kernel Overview

# In[ ]:


get_ipython().system('pip install git+https://github.com/charles9n/bert-sklearn.git')


# In[ ]:


from bert_sklearn import *


# In[ ]:


# !wget https://storage.googleapis.com/bert_models/2018_10_18/cased_L-12_H-768_A-12.zip
# !wget https://raw.githubusercontent.com/google-research/bert/master/modeling.py 
# !wget https://raw.githubusercontent.com/google-research/bert/master/optimization.py 
# !wget https://raw.githubusercontent.com/google-research/bert/master/run_classifier.py 
# !wget https://raw.githubusercontent.com/google-research/bert/master/tokenization.py 


# # 2. Data Exploration

# ### Step 2.1 Load Dataset

# In[ ]:


import pandas as pd

data=df


# In[ ]:


df.columns = ['text', 'category']


# In[ ]:


data.head()


# In[ ]:


data['category'].value_counts()


# # 3. Implementation

# ### Step 2.2 Map Textual labels to numeric using Label Encoder

# In[ ]:


from sklearn.preprocessing import LabelEncoder
df2 = pd.DataFrame()
df2["text"] = data["text"]
df2["label"] = (data["category"])


# In[ ]:


df2.head()


# ### Step 2.3 Divide dataset to test and train dataset

# In[ ]:


test.columns = ['text']
test.head()


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df2["text"].values, df2["label"].values, test_size=0.0001, random_state=42)
X_test = test['text'].values


# ### Step 2.4 Setting up BERT Configurations

# In[ ]:


import bert_sklearn
# bert_sklearn.model.model.BertModel.


# In[ ]:


model = BertClassifier()
# try different options...
model.bert_model = 'biobert-v1.0-pmc-base-cased'
# model.bert_model = 'biobert-v1.1-pubmed-base-cased'
model.num_mlp_layers = 1
model.max_seq_length = 300
model.epochs = 8
model.learning_rate = 1e-5
model.gradient_accumulation_steps = 4
model.validation_fraction = 0
# finetune


# In[ ]:


model.fit(X_train, y_train)


# In[ ]:


y_pred = model.predict(X_test)
y_pred


# In[ ]:


y_pred.mean()


# In[ ]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

print("Accuracy of BERT is:",f1_score(y_test,y_pred, average='macro'))


# # 4. Results

# In[ ]:





# In[ ]:





# In[ ]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

print("Accuracy of BERT is:",f1_score(y_test,preds, average='macro'))


# In[ ]:


# from sklearn.metrics import classification_report

# print(classification_report(y_test,preds))


# >** Past Work mentioned on this dataset at max achieved 95.22 accuracies. BERT base model for the same, without any preprocessing and achieved 97.75 accuracies.**

# In[ ]:


len(preds)


# In[ ]:


sam = pd.read_csv('../input/sample.csv')
sam.head()


# In[ ]:


sam['sentiment'] = y_pred


# In[ ]:


sam.head()


# In[ ]:


sam.to_csv('sub.csv', index=False)
from IPython.display import FileLinks
FileLinks('.')


# In[ ]:





# In[ ]:





# # 5. Future Improvements on this kernel:

# * Explore preprocessing steps on data.
# * Explore other models as baseline.
# * Make this notebook more informative and illustrative.
# * Explaination on Bert Model.
# * More time on data exploration
# and many more...

# # 6. References

# In[ ]:


get_ipython().system('wget https://github.com/charles9n/bert-sklearn/master/BertClassifier')


# In[ ]:


get_ipython().system('wget https://raw.githubusercontent.com/charles9n/bert-sklearn/')


# In[ ]:


import os
import math
import random
import csv
import sys

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
import statistics as stats

from bert_sklearn import BertClassifier
from bert_sklearn import BertRegressor
from bert_sklearn import BertTokenClassifier
from bert_sklearn import load_model


# In[ ]:


model.


# In[ ]:


X_train


# In[ ]:


model = BertClassifier()         # text/text pair classification
model.fit(X_train, y_train)


# https://www.kaggle.com/thebrownviking20/bert-multiclass-classification
