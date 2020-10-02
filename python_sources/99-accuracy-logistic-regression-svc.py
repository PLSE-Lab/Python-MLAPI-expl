#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer,StandardScaler,MinMaxScaler
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud,STOPWORDS
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize,sent_tokenize
from bs4 import BeautifulSoup
import re,string,unicodedata
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.stem import LancasterStemmer,WordNetLemmatizer
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.model_selection import train_test_split
from string import punctuation
from nltk import pos_tag
from nltk.corpus import wordnet


# In[ ]:


df = pd.read_csv("../input/creditcardfraud/creditcard.csv")


# In[ ]:


df.head()


# In[ ]:


df.isna().sum() # Checking for any Nan values in Dataset


# In[ ]:


del df['Time']


# In[ ]:


df.Class.value_counts()


# In[ ]:


scaler = MinMaxScaler()
df['Amount'] = scaler.fit_transform(df[['Amount']])


# In[ ]:


df.head()


# In[ ]:


x_values = []
for i in df.values:
    x_values.append(i[:-1])


# In[ ]:


x_values[0]


# In[ ]:


x_train,x_test,y_train,y_test = train_test_split(x_values,df.Class)


# In[ ]:


print(len(x_train) , len(y_train))
print(len(x_test) , len(y_test))


# In[ ]:


clf = LogisticRegression(penalty = 'l2', max_iter = 500)
clf.fit(x_train,y_train)


# In[ ]:


pred_lr = clf.predict(x_test)


# In[ ]:


accuracy_score(pred_lr,y_test)


# In[ ]:


lr_report = classification_report(y_test,pred_lr,target_names=['0','1'])
print(lr_report)


# In[ ]:


cm_lr = confusion_matrix(y_test,pred_lr)
cm_lr


# In[ ]:


cm_lr = pd.DataFrame(cm_lr, index=[0,1], columns=[0,1])
cm_lr.index.name = 'Actual'
cm_lr.columns.name = 'Predicted'


# In[ ]:


plt.figure(figsize = (10,10))
sns.heatmap(cm_lr,cmap= "Blues",annot = True, fmt='')


# In[ ]:


svc = SVC()
svc.fit(x_train,y_train)


# In[ ]:


pred_svc = svc.predict(x_test)


# In[ ]:


accuracy_score(pred_svc,y_test)


# In[ ]:


cm_svc = confusion_matrix(y_test,pred_svc)
cm_svc


# In[ ]:


cm_svc = pd.DataFrame(cm_svc, index=[0,1], columns=[0,1])
cm_svc.index.name = 'Actual'
cm_svc.columns.name = 'Predicted'


# In[ ]:


plt.figure(figsize = (10,10))
sns.heatmap(cm_svc,cmap= "Blues",annot = True, fmt='')


# In[ ]:




