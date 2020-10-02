#!/usr/bin/env python
# coding: utf-8

# # Problem Statement

# You work in an event management company. On Mother's Day, your company has organized an event where they want to cast positive Mother's Day related tweets in a presentation. Data engineers have already collected the data related to Mother's Day that must be categorized into positive, negative, and neutral tweets.
# 
# You are appointed as a Machine Learning Engineer for this project. Your task is to build a model that helps the company classify these sentiments of the tweets into positive, negative, and neutral.

# Github Repo: https://github.com/bilalProgTech/online-data-science-ml-challenges/tree/master/HackerEarth-Mothers-Day-ML-Challenge

# In[ ]:


import re
import nltk
import string
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


test = pd.read_csv('/kaggle/input/test.csv')
train = pd.read_csv('/kaggle/input/train.csv')
test1 = pd.read_csv('/kaggle/input/test.csv')


# In[ ]:


train.head()


# In[ ]:


train = train[['original_text', 'sentiment_class']]
test = test[['original_text']]


# In[ ]:


train['sentiment_class'].value_counts()


# In[ ]:


str_len_test = test[test['original_text'].str.len() < 700]['original_text'].str.len()
str_len_train = train[train['original_text'].str.len() < 700]['original_text'].str.len()

plt.hist(str_len_train, bins=100, label="Train Text")
plt.hist(str_len_test, bins=100, label="Test Text")
plt.legend()
plt.show()


# In[ ]:


def clean_summary(text):
    text = re.sub("\'", "", text)
    text = re.sub("[^a-zA-Z#]"," ",text)
    text = re.sub("http\S+|www.\S+","", text)
    text = re.sub("igshid","", text)
    text = re.sub("instagram","", text)
    text = re.sub("twitter","", text)
    text = re.sub("mothersday","", text)
    text = re.sub("happymothersday","", text)
    text = re.sub("motheringsubday","", text)
    text = re.sub("happy","", text)
    text = ' '.join(text.split())
    text = text.lower()
    text = ' '.join([w for w in text.split() if len(w)>3])
    words = text.split()
    text = " ".join(sorted(set(words), key=words.index))
    return text


# In[ ]:


train['clean_text'] = train['original_text'].apply(lambda x: clean_summary(x))
train.head(2)


# In[ ]:


from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
def remove_stopwords(text):
    no_stopword_text = [w for w in text.split() if not w in stop_words]
    return ' '.join(no_stopword_text)

nltk.download('words')
words = set(nltk.corpus.words.words())

def remove_nonenglish(text):
    return " ".join(w for w in nltk.wordpunct_tokenize(text) if w.lower() in words or not w.isalpha())
    
train['clean_text'] = train['clean_text'].apply(lambda x: remove_stopwords(x))

train.head()


# In[ ]:


test['clean_text'] = test['original_text'].apply(lambda x: clean_summary(x))
test['clean_text'] = test['clean_text'].apply(lambda x: remove_stopwords(x))
test.head()


# In[ ]:


combine = train.append(test)


# In[ ]:


from wordcloud import WordCloud


# In[ ]:


all_words = ' '.join([text for text in combine['clean_text']])
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()


# In[ ]:


not_sarcastic =' '.join([text for text in combine['clean_text'][combine['sentiment_class'] == 0]])
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(not_sarcastic)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()


# In[ ]:


not_sarcastic =' '.join([text for text in combine['clean_text'][combine['sentiment_class'] == 1]])
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(not_sarcastic)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()


# In[ ]:


not_sarcastic =' '.join([text for text in combine['clean_text'][combine['sentiment_class'] == -1]])
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(not_sarcastic)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()


# In[ ]:


def hashtag_extract(x):
    hashtags = []
    for i in x:
        ht = re.findall(r"#(\w+)", i)
        hashtags.append(ht)
    return hashtags


# In[ ]:


HT_positive = hashtag_extract(combine['clean_text'][combine['sentiment_class'] == 0])
HT_negative = hashtag_extract(combine['clean_text'][combine['sentiment_class'] == 1])
HT_neutral = hashtag_extract(combine['clean_text'][combine['sentiment_class'] == -1])


# In[ ]:


HT_positive = sum(HT_positive,[])
HT_negative = sum(HT_negative,[])
HT_neutral = sum(HT_neutral,[])


# In[ ]:


a = nltk.FreqDist(HT_positive)
d = pd.DataFrame({'Hashtag': list(a.keys()),'Count': list(a.values())})
d = d.nlargest(columns='Count', n=10)
plt.figure(figsize=(16,5))
ax = sns.barplot(data=d, x= "Hashtag", y = "Count")
ax.set(ylabel = 'Count')
plt.show()


# In[ ]:


a = nltk.FreqDist(HT_negative)
d = pd.DataFrame({'Hashtag': list(a.keys()),'Count': list(a.values())})
d = d.nlargest(columns='Count', n=10)
plt.figure(figsize=(16,5))
ax = sns.barplot(data=d, x= "Hashtag", y = "Count")
ax.set(ylabel = 'Count')
plt.show()


# In[ ]:


a = nltk.FreqDist(HT_neutral)
d = pd.DataFrame({'Hashtag': list(a.keys()),'Count': list(a.values())})
d = d.nlargest(columns='Count', n=10)
plt.figure(figsize=(16,5))
ax = sns.barplot(data=d, x= "Hashtag", y = "Count")
ax.set(ylabel = 'Count')
plt.show()


# In[ ]:


def clean_summary_modelling(text):
    text = re.sub("\'", "", text)
    text = re.sub("[^a-zA-Z]"," ",text)
    text = re.sub("http\S+|www.\S+","", text)
    text = re.sub("igshid","", text)
    text = re.sub("instagram","", text)
    text = re.sub("twitter","", text)
    text = re.sub("mothersday","", text)
    text = re.sub("happymothersday","", text)
    text = re.sub("motheringsubday","", text)
    text = ' '.join(text.split())
    text = text.lower()
    text = ' '.join([w for w in text.split() if len(w)>3])
    words = text.split()
    text = " ".join(sorted(set(words), key=words.index))
    return text


# In[ ]:


train['clean_text'] = train['original_text'].apply(lambda x: clean_summary_modelling(x))
train['clean_text'] = train['clean_text'].apply(lambda x: remove_stopwords(x))
train['clean_text'] = train['clean_text'].apply(lambda x: remove_nonenglish(x))

train.head()


# In[ ]:


test['clean_text'] = test['original_text'].apply(lambda x: clean_summary_modelling(x))
test['clean_text'] = test['clean_text'].apply(lambda x: remove_stopwords(x))
test['clean_text'] = test['clean_text'].apply(lambda x: remove_nonenglish(x))
test.head()


# In[ ]:


X = train.drop(['sentiment_class','original_text'], axis=1)
y = train['sentiment_class']

X_test = test.drop(['original_text'], axis=1)
X.shape, y.shape, X_test.shape


# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# In[ ]:


xtrain, xval, ytrain, yval = train_test_split(X, y, test_size=0.2, random_state=10)


# In[ ]:


tfidf_vectorizer = TfidfVectorizer(max_df=0.46, min_df=1, max_features=3000)
x_train = tfidf_vectorizer.fit_transform(xtrain['clean_text'])
x_val = tfidf_vectorizer.transform(xval['clean_text'])


# In[ ]:


classifier = LogisticRegression()

classifier.fit(x_train, ytrain)


# In[ ]:


from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
pred = classifier.predict(x_val)
100 * f1_score(yval, pred, average='weighted'), accuracy_score(yval, pred)


# In[ ]:


confusion_matrix(yval, pred)


# In[ ]:


test_x = tfidf_vectorizer.transform(X_test['clean_text'])
pred_test = classifier.predict(test_x)


# In[ ]:


submission = pd.DataFrame()
submission['id'] = test1['id']
submission['sentiment_class'] = pred_test
submission['sentiment_class'].value_counts()


# In[ ]:


submission.to_csv('Submission.csv', index=False)

