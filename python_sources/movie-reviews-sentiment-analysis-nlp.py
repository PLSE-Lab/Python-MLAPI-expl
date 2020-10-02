#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import warnings
warnings.simplefilter('ignore')


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


data = pd.read_csv('/kaggle/input/sentiment-analysis-on-movie-reviews/train.tsv.zip',sep='\t')


# In[ ]:


data.head(6)


# In[ ]:


data.info()


# In[ ]:


data.Sentiment.value_counts()


# In[ ]:


Sentiment_count=data.groupby('Sentiment').count()
plt.bar(Sentiment_count.index.values, Sentiment_count['Phrase'])
plt.xlabel('Review Sentiments')
plt.ylabel('Number of Review')
plt.show()


# In[ ]:


print(data.iloc[0]['Phrase'],'Sentiment - ',data.iloc[0]['Sentiment'])


# In[ ]:


print(data.iloc[1]['Phrase'],'Sentiment - ',data.iloc[1]['Sentiment'])


# In[ ]:


print(data.iloc[32]['Phrase'],'Sentiment - ',data.iloc[32]['Sentiment'])
print('\n')
print(data.iloc[33]['Phrase'],'Sentiment - ',data.iloc[33]['Sentiment'])


# In[ ]:


import string
string.punctuation


# In[ ]:


from nltk.stem import WordNetLemmatizer
lm = WordNetLemmatizer()

def own_analyser(phrase):
    phrase = phrase.split()
    for i in range(0,len(phrase)):
        k = phrase.pop(0)
        if k not in string.punctuation:
                phrase.append(lm.lemmatize(k).lower())    
    return phrase


# In[ ]:


data.columns


# In[ ]:


X = data['Phrase']
y = data['Sentiment']


# In[ ]:


from sklearn.model_selection import train_test_split
phrase_train,phrase_test,sentiment_train,sentiment_test = train_test_split(X,y,test_size=0.3)


# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB


# In[ ]:


from sklearn.pipeline import Pipeline

pipeline = Pipeline([('BOW',CountVectorizer(analyzer=own_analyser)),
                    ('tfidf',TfidfTransformer()),
                    ('classifier',MultinomialNB())])


# In[ ]:


pipeline.fit(phrase_train,sentiment_train)


# In[ ]:


predictions = pipeline.predict(phrase_test)


# In[ ]:


from sklearn.metrics import classification_report


# In[ ]:


print(classification_report(sentiment_test,predictions))


# In[ ]:


test_data = pd.read_csv('/kaggle/input/sentiment-analysis-on-movie-reviews/test.tsv.zip',sep='\t')


# In[ ]:


test_data.head()


# In[ ]:


test_data.info()


# In[ ]:


test_predictions = pipeline.predict(test_data['Phrase'])


# In[ ]:


phrase_id = test_data['PhraseId'].values


# In[ ]:


test_predictions.shape


# In[ ]:


final_answer = pd.DataFrame({'PhraseId':phrase_id,'Sentiment':test_predictions})


# In[ ]:


final_answer.head(10)


# In[ ]:


filename = 'Sentiment_analysis_output.csv'

final_answer.to_csv(filename,index=False)

print('Saved file: ' + filename)


# In[ ]:




