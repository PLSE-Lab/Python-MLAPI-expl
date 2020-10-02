#!/usr/bin/env python
# coding: utf-8

# # 99% train AND 98% test accuracy

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


text=pd.read_csv("../input/spam-text-message-classification/SPAM text message 20170820 - Data.csv")


# In[ ]:


text.head()


# In[ ]:


text.info()


# In[ ]:


text.describe()


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


text.groupby(["Category"]).count().plot(kind="bar",color='blue')


# In[ ]:


text.groupby(["Category"]).describe()


# In[ ]:


text.drop_duplicates(inplace=True)


# In[ ]:


text.groupby(["Category"]).describe()


# In[ ]:


import string
import nltk
from nltk.corpus import stopwords


# In[ ]:


stopword=stopwords.words('english')


# In[ ]:


def text_preprocessing(texts):
    tex=texts.strip()
    texts_word=[word for word in tex.split() if "@" not in word]
    tex=" ".join(texts_word)
    texts_word=[word for word in tex.split() if "#" not in word]
    tex=" ".join(texts_word)
    texts_word=[word for word in tex.split() if "www." not in word]
    tex=" ".join(texts_word)
    texts_word=[word for word in tex.split() if "http" not in word]
    tex=" ".join(texts_word)
    texts_word=[word for word in tex if word not in string.punctuation]
    tex="".join(texts_word)
    texts_word=[word for word in tex.split() if word not in stopword]
    tex=" ".join(texts_word)
    texts_word=[word for word in tex.split() if word.isalpha()]
    tex=" ".join(texts_word)
    texts_word=[word.lower() for word in tex.split()]
    tex=" ".join(texts_word)
    tex=tex.strip()
    return tex.split()


# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer


# In[ ]:


text["Message"]=text["Message"].astype('str')


# In[ ]:


cv=CountVectorizer(analyzer=text_preprocessing).fit(text.Message)


# In[ ]:


cv_trans=cv.transform(text.Message)


# In[ ]:


print(cv_trans[2])


# In[ ]:


cv.get_feature_names()[900]


# In[ ]:


type(cv_trans)


# In[ ]:


tfidf=TfidfTransformer().fit(cv_trans)
tfidf_trans=tfidf.transform(cv_trans)


# In[ ]:


print(tfidf_trans[0])


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score


# In[ ]:


X_train,X_test,y_train,y_test=train_test_split(tfidf_trans,text.Category,test_size=0.15,random_state=42)


# In[ ]:


get_ipython().run_line_magic('pinfo', 'SVC')


# In[ ]:


svc=SVC(gamma=0.5,tol=0.51,decision_function_shape='ovo')


# In[ ]:


svc.fit(X_train,y_train)


# In[ ]:


print("train accuracy: {0} \ntest accuracy: {1}".format(accuracy_score(y_train,svc.predict(X_train)),
                                                       accuracy_score(y_test,svc.predict(X_test))))


# In[ ]:


classification_report(y_test,svc.predict(X_test))


# In[ ]:


confusion_matrix(y_test,svc.predict(X_test))


# In[ ]:


svc=SVC(gamma=0.5,tol=0.51,decision_function_shape='ovr',random_state=42)
svc.fit(X_train,y_train)
print("train accuracy: {0} \ntest accuracy: {1}".format(accuracy_score(y_train,svc.predict(X_train)),
                                                       accuracy_score(y_test,svc.predict(X_test))))
print("******************************************************")
print(classification_report(y_test,svc.predict(X_test)))
print("******************************************************")
print(confusion_matrix(y_test,svc.predict(X_test)))


# In[ ]:


cv_trans.shape


# In[ ]:




