#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


import pandas as pd
sample_submission = pd.read_csv("../input/nlp-getting-started/sample_submission.csv")
test = pd.read_csv("../input/nlp-getting-started/test.csv")
train = pd.read_csv("../input/nlp-getting-started/train.csv")


# In[ ]:


train.drop(['keyword','location'],axis=1,inplace=True)
test.drop(['keyword','location'],axis=1,inplace=True)


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


import string


# In[ ]:


train['text']=train['text'].str.lower()
test['text']=test['text'].str.lower()


# In[ ]:


text=train['text']
text1=test['text']


# In[ ]:


def remove_punctuation(text):
    return text.translate(str.maketrans('','',string.punctuation))
text_clean=text.apply(lambda text:remove_punctuation(text))
text_clean1=text1.apply(lambda text1:remove_punctuation(text1))


# In[ ]:


text_clean.head()


# In[ ]:


text_clean1.head()


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


text_clean.head()


# In[ ]:


text_clean1.head()


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


train.update(df)


# In[ ]:


train.head()


# In[ ]:


df1 = pd.DataFrame({"text": text_clean1})
df1.head()


# In[ ]:


test.update(df1)


# In[ ]:


test.drop('id',axis=1,inplace=True)


# In[ ]:


test.head()


# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[ ]:


tfidf = TfidfVectorizer()


# In[ ]:


X_all = pd.concat([train["text"],test["text"]])

tfidf = TfidfVectorizer(stop_words = 'english')
tfidf.fit(X_all)

X = tfidf.transform(train["text"])
X_test = tfidf.transform(test["text"])
del X_all


# In[ ]:


x=X
y=train.iloc[:,-1]


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


sample_submission.head()


# In[ ]:


from sklearn.metrics import accuracy_score


# In[ ]:


score =accuracy_score(sample_submission['target'],y1)


# In[ ]:


score*100


# In[ ]:


from sklearn.linear_model import SGDClassifier


# In[ ]:


clf=SGDClassifier(loss='modified_huber',verbose=1)


# In[ ]:


clf.fit(x,y)


# In[ ]:


y2=clf.predict(x1)


# In[ ]:


from sklearn.metrics import accuracy_score


# In[ ]:


score1 =accuracy_score(sample_submission['target'],y1)


# In[ ]:


score1


# In[ ]:


from sklearn.model_selection import GridSearchCV
from sklearn import svm


# In[ ]:


parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10, 100, 1000], 'gamma' : [0.001,0.0001]}


# In[ ]:


clf = GridSearchCV(estimator=svm.SVC(), param_grid=parameters, n_jobs=-1)


# In[ ]:


clf.fit(x,y)


# In[ ]:


y3=clf.predict(x1)


# In[ ]:


score2 =accuracy_score(sample_submission['target'],y3)


# In[ ]:


score2


# In[ ]:


prediction = pd.DataFrame(y3, columns=['y3']).to_csv('prediction.csv')


# In[ ]:




