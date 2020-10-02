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


# Importing Necessary libraries
import re
import nltk
import spacy


# In[ ]:


# Loading Data
train = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')
test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')


# In[ ]:


train.head()


# In[ ]:


train.isnull().sum()


# Here we will remove keyword and location variables. They donot have much of a value here. In addition to that, there are many NaN values. So it is better to remove those variables. 

# In[ ]:


train=train.dropna()
target = train['target']


# In[ ]:


train = train['keyword'] + ' ' + train['location'] + " " + train['text'] 
test = test['keyword'] + " " + test['location'] + " " + test['text'] 
train.head()


# Now we have everything required for us to get started. Here id is the reference like an index. 'text' is our asset. We will be working on that. 'target' is the target variable. 

# In[ ]:


import string


# In[ ]:


train = train.str.lower()


# In[ ]:


train.head()


# In the earlier preview, you can find a mixture of upper and lower case letters. Now, you can see that whole of the text is in lower case. This forms the first step of text preprocessing. Let us now split our dataset for the target and text variables. 

# In[ ]:





# In[ ]:


target.head()


# In[ ]:


text=train


# In[ ]:


text.head()


# The above data has punctuation with it and they do not have any semantic meaning in our data. So we will remoce it. The following is a better way of removing it. 

# In[ ]:


def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))

text_clean = text.apply(lambda text: remove_punctuation(text))


# In[ ]:


text_clean.head()


# You can find that the functuations are removed. Now we will remove the so called stopwords. They are highly repetitive words in the text but do not posses a greater value for their presence. So we will remove it. 

# In[ ]:


from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))


# In[ ]:


def stopwords_(text):
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])

text_clean = text_clean.apply(lambda text: stopwords_(text))


# In[ ]:


text_clean.head()


# You can find ityourself right. The use of removing these words. Want to know what are those words. Take a look at it. 

# In[ ]:


", ".join(stopwords.words('english'))


# Yeah. you have now completed the first phase of the text preprocessing. Now let us proceed to the next one. 
# 
# Lemmatization is the process of reducing the words to their roots. Let us take a look at an example for better understanding. 

# In[ ]:


from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
def lemmatizer_(text):
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])


# In[ ]:


lemmatizer.lemmatize("wrote", "v")


# In[ ]:


lemmatizer.lemmatize("written", "v")


# Do I need to explain further. Hahaha. Not at all necesary. It is self explanatory. But if you have any doubts donot hesitate to comment in the comment section. 
# 
# Let us apply this to our text. 

# In[ ]:


text_clean = text_clean.apply(lambda text: lemmatizer_(text))

train = text_clean


# In[ ]:


# train = train.drop(['id'],axis=1)
train.head()


# All of these can also be done by in-built packages. But it is a good parctice in the beginning to understand our data better. 
# 
# Now for the fun part, we will look at the most used words in the cleaned text. We will use wordcloud library for that. 

# In[ ]:


from wordcloud import WordCloud
import matplotlib.pyplot as plt
fig, (ax1) = plt.subplots(1, figsize=[12, 12])
wordcloud = WordCloud( background_color='white',
                        width=600,
                        height=400).generate(" ".join(text_clean))
ax1.imshow(wordcloud)
ax1.axis('off')
ax1.set_title('Frequent Words',fontsize=16);


# In[ ]:


from sklearn.linear_model import LogisticRegression

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
print(train.shape)
train_vectors = vectorizer.fit_transform(train)
test_vectors = vectorizer.transform(test)
print(train_vectors.shape)
clf = LogisticRegression(random_state=0).fit(train_vectors, target)

print(clf.predict(test_vectors))

print(clf.score(train_vectors,target))


# In[ ]:


df=clf.predict(test_vectors)
result=pd.DataFrame(df)
x=pd.concat([test['id'], result],axis=1)
x.to_csv('mycsvfile.csv',index=False)


# ![](http://)![](http://)If you find anything unnecessary or to be removed, you can do so by appending it to the stopwords or remove it manually.
# 
# Sometime later, I will share a kernel on how to solve this problem in an efficient manner. 

# Hope you find this useful. If you like this kernel, please consider upvoting it. 
