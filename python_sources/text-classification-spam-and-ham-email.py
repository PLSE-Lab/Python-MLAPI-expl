#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib as plt


# In[ ]:


df =pd.read_csv('/kaggle/input/spam.csv',encoding= 'latin-1')


# In[ ]:


df.head()


# In[ ]:


df.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1, inplace= True)


# In[ ]:


df


# In[ ]:


data=df.rename(columns={'v1':'class','v2':'text'})


# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split


# In[ ]:


x= data['text']
y=data['class']
x_train,x_test, y_train,y_test= train_test_split(x,y,test_size=0.2)


# In[ ]:


v=CountVectorizer()
v.fit(x_train)
vec_x_train= v.transform(x_train).toarray()
vec_x_test= v.transform(x_test).toarray()


# In[ ]:


v.vocabulary


# In[ ]:


from sklearn.naive_bayes import MultinomialNB,GaussianNB, BernoulliNB


# In[ ]:


m= GaussianNB()
m.fit(vec_x_train,y_train)
print(m.score(vec_x_test,y_test))


# In[ ]:


sample = input('ask a question:')
vec = v.transform([sample]).toarray()
m.predict(vec)


# In[ ]:




