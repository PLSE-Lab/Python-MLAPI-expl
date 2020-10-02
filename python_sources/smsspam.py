#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Import Dependencies
import os
print(os.listdir("../input"))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import string
from nltk.corpus import stopwords
from nltk import PorterStemmer as Stemmer


# In[ ]:


#Read the csv file
df = pd.DataFrame(pd.read_csv('../input/spam.csv', encoding='latin-1')[['v1', 'v2']])
df.columns=['Label','Message']
df.head()


# In[ ]:


df.groupby('Label').describe()


# In[ ]:


sns.countplot(data=df, x='Label')


# **Cleaning the Data**
# * Remove all the punctuations.
# * Remove all the stopwords.
# * Apply stemming.

# In[ ]:


def process(text):
    # lowercase it
    text = text.lower()
    # remove punctuation
    text = ''.join([t for t in text if t not in string.punctuation])
    # remove stopwords
    text = [t for t in text.split() if t not in stopwords.words('english')]
    # stemming
    st = Stemmer()
    text = [st.stem(t) for t in text]
    # return token list
    return text


# In[ ]:


#Testing the code
process('It\s has been a long day running.')


# In[ ]:


# Test with our dataset
df['Message'][:20].apply(process)


# **Convert each message to vectors that machine learning models can understand.**

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[ ]:


tfidfv = TfidfVectorizer(analyzer=process)
data = tfidfv.fit_transform(df['Message'])


# In[ ]:


mess = df.iloc[2]['Message']
print(mess)


# In[ ]:


print(tfidfv.transform([mess]))


# In[ ]:


j = tfidfv.transform([mess]).toarray()[0]
print('index\tidf\ttfidf\tterm')
for i in range(len(j)):
    if j[i] != 0:
        print(i, format(tfidfv.idf_[i], '.4f'), format(j[i], '.4f'), tfidfv.get_feature_names()[i],sep='\t')


# **Having messages in form of vectors, we are ready to train our classifier.
# We will use Naive Bayes which is well known classifier while working with text data.
# Before that we will use pipeline feature of sklearn to create a pipeline of TfidfVectorizer followed by Classifier.**

# In[ ]:


from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
spam_filter = Pipeline([
    ('vectorizer', TfidfVectorizer(analyzer=process)), # messages to weighted TFIDF score
    ('classifier', MultinomialNB())                    # train on TFIDF vectors with Naive Bayes
])


# In[ ]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(df['Message'], df['Label'], test_size=0.20, random_state = 21)


# In[ ]:


spam_filter.fit(x_train, y_train)


# In[ ]:


predictions = spam_filter.predict(x_test)


# In[ ]:


count = 0
for i in range(len(y_test)):
    if y_test.iloc[i] != predictions[i]:
        count += 1
print('Total number of test cases', len(y_test))
print('Number of wrong of predictions', count)


# In[ ]:


x_test[y_test != predictions]


# In[ ]:


from sklearn.metrics import classification_report
print(classification_report(predictions, y_test))


# In[ ]:


def detect_spam(s):
    return spam_filter.predict([s])[0]
detect_spam('Your cash-balance is currently 500 pounds - to maximize your cash-in now, send COLLECT to 83600.')

