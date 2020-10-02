#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


data=pd.read_csv('/kaggle/input/stockmarket-sentiment-dataset/stock_data.csv')
data.head()


# In[ ]:


data.info()


# In[ ]:


data.describe()


# **Let's check if we are having missing data ??**

# In[ ]:


sns.heatmap(data.isnull(),cmap='Blues')


# Well no missing data that's great   :)

# In[ ]:


data['Sentiment'].value_counts()


# In[ ]:


sns.countplot(x=data['Sentiment'])


# Getting a view on our sentiment data 

# In[ ]:


data.groupby('Sentiment').describe()


# In[ ]:


data['Length']=data['Text'].apply(lambda x:len(x))


# In[ ]:


data['Length'].plot.hist(bins=200)


# Above curve shows how our length varies in our text data

# In[ ]:


data['Length'].describe()


# Now comparing how length affect sentiment

# In[ ]:


plt.figure(figsize=(12,5))
data.hist(column='Length',by='Sentiment',bins=150)


# **Now let's focus on our data prediction**

# In[ ]:


import string
from nltk.corpus import stopwords


# **Setting up tool for cleaning the data in required form**

# In[ ]:


def clean(text):
    a=[f for f in text if f not in string.punctuation]
    a=''.join(a)
    b=[w for w in a.split() if w.lower() not in stopwords.words('english')]
    return b


# In[ ]:


check=data['Text'].head(1).apply(clean)


# In[ ]:


print(check[0])


# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score


# In[ ]:


words=CountVectorizer(analyzer=clean).fit(data['Text']) # Cleaning all our data set from punctuations and stopwords


# In[ ]:


print(len(words.vocabulary_))


# These are the no of different no of words present in our data 

# In[ ]:


sample=data['Text'][1]
sample


# In[ ]:


trans=words.transform([sample])
print(trans)


# L.H.S show's the position of a particular word and R.H.S tell the count of that word

# In[ ]:


print(trans.shape) 


# We have 1 documnet Vs 11425 Diffrent Words in total 

# Printing the particular word with the help of index :)

# In[ ]:


words.get_feature_names()[363]


# In[ ]:


allmessgae=words.transform(data['Text'])


# In[ ]:


print(allmessgae.shape)


# Above shape telling us that we have in total 5791 documnets vs 13456 different words among them

# In[ ]:


allmessgae.nnz # No of non - zero's value


# In[ ]:


sparsity = (100.0 * allmessgae.nnz / (allmessgae.shape[0] * allmessgae.shape[1]))
print('sparsity: {}'.format(sparsity))


# No of non-zero value by total values 

# In[ ]:


tf=TfidfTransformer()


# In[ ]:


tf.fit(allmessgae)


# In[ ]:


tfidf=tf.transform(trans)
print(tfidf) 


# These values depict that how much a word is important in that particular document 

# In[ ]:


tf.idf_[words.vocabulary_['return']] # Checking the IDF value of particular word how imp a term is in whole dataset 


# In[ ]:


final_transfrom=tf.transform(allmessgae)


# In[ ]:


modelfitting=MultinomialNB().fit(final_transfrom,data['Sentiment'])


# In[ ]:


result=modelfitting.predict(final_transfrom)


# In[ ]:


print(result)


# **Another method instead of doing all above stuff**

# In[ ]:


pipe=Pipeline([
 ('cv',CountVectorizer(analyzer=clean)),
 ('tfidf',TfidfTransformer()),
 ('Classifier',MultinomialNB())
])


# In[ ]:


x_train,x_test,y_train,y_test=train_test_split(data['Text'],data['Sentiment'],test_size=0.3,random_state=101)


# In[ ]:


pipe.fit(x_train,y_train)


# In[ ]:


pipe_predict=pipe.predict(x_test)


# In[ ]:


print(classification_report(pipe_predict,y_test))


# In[ ]:


print(confusion_matrix(pipe_predict,y_test))


# In[ ]:


print(accuracy_score(pipe_predict,y_test))

