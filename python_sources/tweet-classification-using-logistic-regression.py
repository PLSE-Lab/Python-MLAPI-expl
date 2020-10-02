#!/usr/bin/env python
# coding: utf-8

# In[ ]:



import numpy as np 
import pandas as pd 
from sklearn import feature_extraction
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# In[ ]:


train_data=pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')
test_data=pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')
train_data[train_data["target"]==1]['text'].values[2]


# In[ ]:


count_vectorizer= feature_extraction.text.CountVectorizer()

text= train_data['text'].values
target= train_data['target'].values

train_vectors=count_vectorizer.fit_transform(text)
#Vectorizing text
test_vectors=count_vectorizer.transform(test_data['text'])


# In[ ]:


classifier=LogisticRegression()
classifier.fit(train_vectors,target)


# In[ ]:


for value in range(0,144):
    print(test_data['text'][value])
    print("Predicted: ", classifier.predict(test_vectors[value]))
    


# 

# In[ ]:


submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")
submission["target"] = classifier.predict(test_vectors)


# In[ ]:


submission.to_csv("submission.csv", index=False)


# In[ ]:




