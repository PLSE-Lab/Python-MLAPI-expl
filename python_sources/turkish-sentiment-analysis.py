#!/usr/bin/env python
# coding: utf-8

# ## Introduction
# * Predict comments is positive or negative  with Logistic Regression algorithm
# * Calculate reliability -- Accuracy

# In[ ]:


import numpy as np 
import pandas as pd 
import nltk
from nltk.corpus import stopwords
import string


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


data = pd.read_csv("/kaggle/input/product-comments-dataset/data.csv",sep=",",encoding ='utf-8')


# In[ ]:


data['Duygu'].value_counts()


# In[ ]:


# Get equal data from both statement for better results

data_n=data[data.Duygu==0]
data_negative=data_n.iloc[:230]

data_p=data[data.Duygu==1]
data_positive=data_p.iloc[:230]


# In[ ]:


# Concat datasets
dataset=pd.concat([data_positive,data_negative])


# In[ ]:


# split dataset
x = dataset['Yorum'].copy()
y = dataset['Duygu'].values.reshape(-1,1)


# In[ ]:


WPT = nltk.WordPunctTokenizer()
stop_word_list = nltk.corpus.stopwords.words('turkish')
print(stop_word_list)


# In[ ]:


# function for remove stopwords and punctuations
def text_preprocess(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = [word for word in text.split() if word.lower() not in stop_word_list]
    return " ".join(text)

x = x.apply(text_preprocess)


# In[ ]:


# train test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state = 42)


# In[ ]:


# frequency of words appearing in a document is converted to a matrix
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer(encoding ='utf-8').fit(x_train) # fit and transform
x_train_vectorized = vect.transform(x_train)


# In[ ]:


# import LogisticRegression
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()


# In[ ]:


# fitting model
lr.fit(x_train_vectorized, y_train)


# In[ ]:


# prediction
predictions = lr.predict(vect.transform(x_test))


# In[ ]:


# accuracy 
from sklearn.metrics import roc_auc_score
print('AUC: ', roc_auc_score(y_test, predictions))


# ## Conclusion
# * accuracy:  0.8390070921985816 
