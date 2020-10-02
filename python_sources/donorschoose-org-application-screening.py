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

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import nltk
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV


# In[ ]:


rawdf = pd.read_csv('../input/train.csv')


# In[ ]:


rawdf.head()


# In[ ]:


dftest = pd.read_csv('../input/test.csv')


# In[ ]:


rawdf['text'] = rawdf['project_title'].astype(str) + ' ' +rawdf['project_essay_1'].astype(str) + ' ' + rawdf['project_essay_2'].astype(str) + ' ' + rawdf['project_essay_3'].astype(str) + ' ' + rawdf['project_essay_4'].astype(str) +rawdf['project_resource_summary'].astype(str) 


# In[ ]:


dftest['text'] = dftest['project_title'].astype(str) + ' ' +dftest['project_essay_1'].astype(str) + ' ' + dftest['project_essay_2'].astype(str) + ' ' + dftest['project_essay_3'].astype(str) + ' ' + dftest['project_essay_4'].astype(str) +dftest['project_resource_summary'].astype(str) 


# In[ ]:


rawdf.drop(['teacher_id', 'teacher_prefix', 'school_state',
       'project_submitted_datetime', 'project_grade_category',
       'project_subject_categories', 'project_subject_subcategories',
       'project_title', 'project_essay_1', 'project_essay_2',
       'project_essay_3', 'project_essay_4', 'project_resource_summary',
       'teacher_number_of_previously_posted_projects'], axis=1, inplace=True)


# In[ ]:


dftest.drop(['teacher_id', 'teacher_prefix', 'school_state',
       'project_submitted_datetime', 'project_grade_category',
       'project_subject_categories', 'project_subject_subcategories',
       'project_title', 'project_essay_1', 'project_essay_2',
       'project_essay_3', 'project_essay_4', 'project_resource_summary',
       'teacher_number_of_previously_posted_projects'], axis=1, inplace=True)


# In[ ]:


data_final_vars=rawdf.columns.values.tolist()
x=[i for i in data_final_vars if i not in ['id','teacher_id','project_is_approved']]
y = ['project_is_approved']


# In[ ]:


X = rawdf[x]
Y = rawdf[y]


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0,stratify=Y)


# In[ ]:


text_clf = Pipeline([('vect', CountVectorizer(stop_words='english')), ('tfidf', TfidfTransformer()),('clf-svm', SGDClassifier(loss="modified_huber"))])


# In[ ]:


X_train = X_train.values.flatten()
y_train = y_train.values.flatten()


# In[ ]:


text_clf = text_clf.fit(X_train, y_train) #SGDClassifier


# In[ ]:


X_test = X_test.values.flatten()
y_test = y_test.values.flatten()


# In[ ]:


predicted = text_clf.predict(X_test)


# In[ ]:


np.mean(predicted == y_test)


# In[ ]:


X_final = dftest[x]


# In[ ]:


predicted_prob = text_clf.predict_proba(X_final)[:,1]


# In[ ]:


test  = pd.DataFrame(columns = ['id','project_is_approved'])
test['id'] = dftest['id']
test['project_is_approved'] = pd.Series(predicted_prob.tolist())


# In[ ]:


test.to_csv('./submission.csv',index = False)

