#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:





# In[ ]:


import pandas as pd
data = pd.read_csv("../input/onion-or-not/OnionOrNot.csv")


# In[ ]:


data.head()


# In[ ]:


data['text'].iloc[5]


# In[ ]:


data['label'].value_counts(normalize=True) * 100


# In[ ]:


data['length'] = data['text'].apply(len)


# In[ ]:


data.info()


# In[ ]:


yes = data[data['label'] == 1]
no = data[data['label'] == 0]


# In[ ]:


fig, ax = plt.subplots(1, 1, figsize = (15,7))

sns.distplot(yes['length'], kde = False, color = 'gold')
sns.distplot(no['length'], kde = False, color = 'purple')


# In[ ]:


data.hist(column='length', by='label', bins=50,figsize=(12,4))


# In[ ]:


from nltk.corpus import stopwords
import string


# In[ ]:


def text_process(data):
    """
    Takes in a string of text, then performs the following:
    1. Remove all punctuation
    2. Remove all stopwords
    3. Returns a list of the cleaned text
    """
    # Check characters to see if they are in punctuation
    nopunc = [char for char in data if char not in string.punctuation]

    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)
    
    # Now just remove any stopwords
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]


# In[ ]:


clean = data['text'].apply(text_process)


# In[ ]:


clean


# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer


# In[ ]:


# Might take awhile...
bow_transformer = CountVectorizer(analyzer=text_process).fit(data['text'])

# Print total number of vocab words
print(len(bow_transformer.vocabulary_))


# In[ ]:


text4 = data['text'][4]


# In[ ]:


bow4 =bow_transformer.transform([text4])
print(bow4)


# In[ ]:


bow4.shape


# In[ ]:


messages_bow = bow_transformer.transform(data['text'])


# In[ ]:


messages_bow.shape


# In[ ]:


from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transformer = TfidfTransformer().fit(messages_bow)
tfidf4 = tfidf_transformer.transform(bow4)
print(tfidf4)


# In[ ]:


messages_tfidf = tfidf_transformer.transform(messages_bow)
print(messages_tfidf.shape)


# In[ ]:


from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(messages_tfidf, data['label'])


# In[ ]:


print('predicted:', spam_detect_model.predict(tfidf4)[0])
print('expected:', data.label[3])


# In[ ]:


all_predictions = spam_detect_model.predict(messages_tfidf)
print(all_predictions)


# In[ ]:


from sklearn.metrics import classification_report
print (classification_report(data['label'], all_predictions))


# In[ ]:


from sklearn.model_selection import train_test_split

msg_train, msg_test, label_train, label_test = train_test_split(data['text'], data['label'], test_size=0.2)

print(len(msg_train), len(msg_test), len(msg_train) + len(msg_test))


# In[ ]:


from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer=text_process)),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
])


# In[ ]:


pipeline.fit(msg_train,label_train)


# In[ ]:


predictions = pipeline.predict(msg_test)


# In[ ]:


print(classification_report(predictions,label_test))


# In[ ]:




