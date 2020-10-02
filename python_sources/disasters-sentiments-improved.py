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


import matplotlib.pyplot as plt


# In[ ]:


import pandas as pd
sample_submission = pd.read_csv("../input/nlp-getting-started/sample_submission.csv")
testData = pd.read_csv("../input/nlp-getting-started/test.csv")
trainData = pd.read_csv("../input/nlp-getting-started/train.csv")


# In[ ]:


trainData.drop(['keyword','location'],axis=1,inplace=True)
testData.drop(['keyword','location'],axis=1,inplace=True)


# In[ ]:


print(trainData.head())
print(testData.head())


# In[ ]:


import string


# In[ ]:


trainData['text']=trainData['text'].str.lower()
testData['text']=testData['text'].str.lower()


# In[ ]:


trainData.head()


# In[ ]:


textData=trainData['text']
text1Data=testData['text']
textData.head()


# In[ ]:


def remove_punctuation(text):
    return text.translate(str.maketrans('','',string.punctuation))
text_clean=textData.apply(lambda text:remove_punctuation(text))
text_clean1=text1Data.apply(lambda text1:remove_punctuation(text1))


# In[ ]:


text_clean.head()


# In[ ]:


from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))


# In[ ]:


def stopwords_(text):
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])
text_clean = text_clean.apply(lambda text: stopwords_(text))
text_clean1 = text_clean1.apply(lambda text1: stopwords_(text1))


# In[ ]:


text_clean.head()


# In[ ]:


text_clean.head()


# In[ ]:


text_clean1.head()


# In[ ]:


from nltk.stem import WordNetLemmatizer
lemmatizer=WordNetLemmatizer()
def lemma(text):
    return ' '.join([lemmatizer.lemmatize(word) for word in text.split()])


# In[ ]:


import nltk
from nltk.stem import WordNetLemmatizer   
lemmatizer = WordNetLemmatizer() 
text_clean=text_clean.apply(lambda text: lemma(text))
text_clean1=text_clean1.apply(lambda text1: lemma(text1))


# In[ ]:


def remove_URL(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'',text)


# In[ ]:


import re
text_clean=text_clean.apply(lambda x : remove_URL(x))
text_clean1=text_clean1.apply(lambda x : remove_URL(x))


# In[ ]:


def remove_html(text):
    html=re.compile(r'<.*?>')
    return html.sub(r'',text)


# In[ ]:


text_clean=text_clean.apply(lambda x : remove_html(x))
text_clean1=text_clean1.apply(lambda x : remove_html(x))


# In[ ]:


from wordcloud import WordCloud


# In[ ]:


all_words = ' '.join([text for text in text_clean])
from wordcloud import WordCloud
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words)

plt.figure(figsize=(16, 10))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()


# In[ ]:


df = pd.DataFrame({"text": text_clean})
df.head()


# In[ ]:


df.head()


# In[ ]:


trainData.update(df)


# In[ ]:


df1 = pd.DataFrame({"text": text_clean1})
df1.head()


# In[ ]:


testData.update(df1)


# In[ ]:


testData.head()


# In[ ]:


testData_text=testData.drop('id',axis=1)


# In[ ]:


testData_text.head()


# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer()


# In[ ]:


X_all = pd.concat([trainData["text"],testData["text"]])

tfidf = TfidfVectorizer(stop_words = 'english')
tfidf.fit(X_all)

X = tfidf.transform(trainData["text"])
X_test = tfidf.transform(testData["text"])
del X_all


# In[ ]:


x=X
y=trainData.iloc[:,-1]


# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


logit = LogisticRegression()


# In[ ]:


logit = LogisticRegression(penalty='l2',solver='saga',l1_ratio=0.2)


# In[ ]:


logit.fit(x,y)


# In[ ]:


x1=X_test


# In[ ]:


y1=logit.predict(x1)


# In[ ]:


prediction = pd.DataFrame(y1, columns=['target'])


# In[ ]:


prediction.head()


# In[ ]:


testData.head()


# In[ ]:


testData['id'].count()


# In[ ]:


Final_result = pd.concat([testData, prediction],axis=1)
Final_result


# In[ ]:


Final_result.drop(['text'],axis=1,inplace=True)


# In[ ]:


Final_result['target'].value_counts()


# In[ ]:


Final_result.head()


# In[ ]:


Final_result.to_csv('final_outputnew3.csv',index=False)


# 
