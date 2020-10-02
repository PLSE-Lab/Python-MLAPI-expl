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
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df = pd.read_csv('/kaggle/input/text-classificationheathcare/TextClassification_Data.csv', encoding='latin')


# In[ ]:


df.head()


# In[ ]:


df = df[['SUMMARY', 'categories']]


# In[ ]:


df.head()


# In[ ]:


df['categories'].value_counts()


# In[ ]:


all_mind = {'PRESCRIPTION': 'PRESCRIPTION', 'APPOINTMENTS': 'APPOINTMENTS', 'MISCELLANEOUS': 'MISCELLANEOUS', 'mISCELLANEOUS': 'MISCELLANEOUS', 'JUNK': 'MISCELLANEOUS', 'ASK_A_DOCTOR':'ASK_A_DOCTOR', 'asK_A_DOCTOR': 'ASK_A_DOCTOR', 'LAB':'LAB' }


# In[ ]:


df['categories'] = [all_mind[x] for x in df['categories']]


# In[ ]:


df.head()


# In[ ]:


df['categories'].value_counts()


# In[ ]:


df.isnull().sum()


# In[ ]:


df.dropna(inplace=True)


# In[ ]:


df.isnull().sum()


# In[ ]:


plt.figure(figsize=(20,12))
sns.countplot(x = 'categories', data =df)


# In[ ]:


df.head()


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X = df['SUMMARY']
y = df['categories']


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[ ]:


from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC


# In[ ]:


text_cla = Pipeline([('tfid', TfidfVectorizer()), ('clas', LinearSVC())])


# In[ ]:


text_cla.fit(X_train, y_train)


# In[ ]:


prediction = text_cla.predict(X_test)


# In[ ]:


from sklearn import metrics


# In[ ]:


print(metrics.classification_report(y_test, prediction))


# In[ ]:


print(metrics.accuracy_score(y_test, prediction))


# ****Test Data****

# In[ ]:


text_cla.predict(['please call doctor'])


# In[ ]:


text_cla.predict(['lab report'])

