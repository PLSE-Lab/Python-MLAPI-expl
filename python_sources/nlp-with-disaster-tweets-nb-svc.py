#!/usr/bin/env python
# coding: utf-8

# In[ ]:


## Importing required libraries for importing and visulation
import numpy as np # linear algebra
import pandas as pd # data processing
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


## Importing Training & Testing Data
test = pd.read_csv('../input/nlp-getting-started/test.csv')
train = pd.read_csv('../input/nlp-getting-started/train.csv')


# In[ ]:


# Checking the length of Training and Testing Data
print((len(train),len(test)))


# In[ ]:


## Preview of Training Data
train.describe()


# In[ ]:


## Preview of Testing Data
test.describe()


# In[ ]:


## Saving the Test id into new variable
test_id = test['id']


# In[ ]:


## Dropping all the unnecessary column
test.drop(['id','keyword','location'],axis = 1, inplace=True)


# In[ ]:


## Dropping all the unnecessary column which also contains null values
train.drop(['id','keyword','location'],axis = 1, inplace=True)


# In[ ]:


## Adding new column to differenciate between Training & Testing data
test['istest'] = 1
train['istest'] = 0
test['target'] = "Null"


# In[ ]:


## combining the data to one 
dataset = pd.concat([train,test], join = 'inner')


# In[ ]:


## Checking the top 50 observation
dataset.head(50)


# In[ ]:


dataset.isna().sum()


# In[ ]:


len(dataset)


# In[ ]:


import re
import string


# In[ ]:


def remove_url(text):
   url=re.compile(r'https?://\S+|www\.\S+')
   return url.sub(r"", text)

def remove_html(text):
   html=re.compile(r'<.*?>')
   return html.sub(r"", text)


# In[ ]:


# removing html and https content
dataset['text'] = dataset['text'].apply(lambda x: remove_html(x)) 


# In[ ]:


# provided space between the words and special characters.
dataset['text'] = dataset['text'].str.replace(r'[^\w]+', ' ')  
#Removed punctuations as well removed from the data     


# In[ ]:


# removed numbers from the dataset
dataset['text'] = dataset['text'].str.replace(r'[\d]+', ' ') 


# In[ ]:


import nltk
stopwords = nltk.corpus.stopwords.words('english')


# In[ ]:


from nltk.stem import WordNetLemmatizer, PorterStemmer
wn=nltk.WordNetLemmatizer()


# In[ ]:


dir(stopwords)


# In[ ]:


def process(text):
    text=[wn.lemmatize(word.lower()) for word in text.split() if word.lower() not in stopwords]
    return " ".join(text)


# In[ ]:


dataset['text'] = dataset['text'].apply(lambda x: process(x))


# In[ ]:


len(dataset)


# In[ ]:


dataset.head()


# In[ ]:


# Checking for imbalance in the data
sns.countplot(x= "target",data = dataset )


# In[ ]:


# without tfidf
from sklearn.feature_extraction.text import CountVectorizer


# In[ ]:


# Convert a collection of text documents to a matrix of token counts
cv = CountVectorizer()


# In[ ]:


x = cv.fit_transform(dataset['text'])


# In[ ]:


cv.vocabulary_


# In[ ]:


x_train = x[:7613,:]


# In[ ]:


x_test = x[7613:]


# In[ ]:


y_train = train.target[:7613]


# In[ ]:


# Building multinomial naive bayes model
from sklearn.naive_bayes import MultinomialNB as MB


# In[ ]:


from collections import Counter


# In[ ]:


clf = MB() 
classifier = clf.fit(x_train, y_train)


# In[ ]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
pred = classifier.predict(x_train)
print(classification_report(y_train,pred))


# In[ ]:


print('accuracy:', accuracy_score(y_train,pred)) # accuracy of 92%


# In[ ]:


target = classifier.predict(x_test)


# In[ ]:


print(len(target),len(test_id))


# In[ ]:


kaggle_submission = pd.DataFrame(test_id,columns = ['id'])


# In[ ]:


kaggle_submission["target"] = target


# In[ ]:


kaggle_submission


# In[ ]:


# saving the dataframe 
kaggle_submission.to_csv('kaggle_submission.csv', index=False)

