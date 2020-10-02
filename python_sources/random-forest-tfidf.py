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

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import numpy as np
import pandas as pd

from sklearn import pipeline,ensemble,preprocessing,feature_extraction,cross_validation,metrics


# In[ ]:


train=pd.read_json('../input/train.json')


# In[ ]:


train.head()


# In[ ]:


train.ingredients=train.ingredients.apply(' '.join)


# In[ ]:


train.head()


# In[ ]:


clf=pipeline.Pipeline([
        ('tfidf_vectorizer', feature_extraction.text.TfidfVectorizer(lowercase=True)),
        ('rf_classifier', ensemble.RandomForestClassifier(n_estimators=500,verbose=1,n_jobs=-1))
    ])


# In[ ]:


# step 1: testing
X_train,X_test,y_train,y_test=cross_validation.train_test_split(train.ingredients,train.cuisine, test_size=0.2)


# In[ ]:


clf.fit(X_train,y_train)


# In[ ]:


y_pred=clf.predict(X_test)


# In[ ]:


metrics.confusion_matrix(y_test,y_pred)


# In[ ]:


metrics.accuracy_score(y_test,y_pred)


# In[ ]:


# step 2: real training
test=pd.read_json('../input/test.json')


# In[ ]:


test.ingredients=test.ingredients.apply(' '.join)


# In[ ]:


test.head()


# In[ ]:


clf.fit(train.ingredients,train.cuisine)


# In[ ]:


pred=clf.predict(test.ingredients)


# In[ ]:


df=pd.DataFrame({'id':test.id,'cuisine':pred})


# In[ ]:


df.to_csv('rf_tfidf.csv', columns=['id','cuisine'],index=False)


# In[ ]:


# LB score ~ 0.753

