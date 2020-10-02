#!/usr/bin/env python
# coding: utf-8

# ### Loading Required Libs

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt 
import seaborn as sns
import spacy
from spacy.matcher import Matcher


# ### Reading Input data

# In[ ]:


train = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")
test = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")
submission =  pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")


# # Sample Training Data

# In[ ]:


train.head()


# In[ ]:


train[~train.keyword.isnull()].tail()


# In[ ]:


sns.countplot(data = train, x = "target")


# In[ ]:





# ### Number of unique Keywords in the dataset
# 
# Hoping for better predictability for these observations

# In[ ]:


train.keyword.nunique()


# # Word Cloud of Keywords in the train dataset (based on Frequency)
# 
# ### Disaster Tweets (Target == 1)

# In[ ]:


text1 = dict(train[train.target==1].keyword.value_counts())
wordcloud = WordCloud(width=800, height=400,background_color="white").generate_from_frequencies(text1)
plt.figure(figsize=[14,8])
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# ### Disaster Tweets (Target == 0)

# # Sample Testing Data

# In[ ]:


text2 = dict(train[train.target==0].keyword.value_counts())
wordcloud = WordCloud(width=800, height=400,background_color="white").generate_from_frequencies(text2)
plt.figure(figsize=[14,8])
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[ ]:


test.head()


# # Word Cloud of Keywords in the test dataset (based on Frequency)
# 

# In[ ]:


text3 = dict(test.keyword.value_counts())
wordcloud = WordCloud(width=800, height=400,background_color="white").generate_from_frequencies(text3)
plt.figure(figsize=[14,8])
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# ### Keywords present in Train also in Test Data

# In[ ]:


len(set(test.keyword.unique()).intersection(train.keyword.unique()))


# In[ ]:


"outbreak" in test.keyword.unique()


# In[ ]:


"explode" in test.keyword.unique()


# ### Length

# In[ ]:


df_train = train


# In[ ]:


df_train["len"] = df_train.text.str.len()


# In[ ]:


print("max tweet length of no disaster {0}" .format(max(df_train[df_train.target == 0].len)))
print("max tweet length of disaster {0}" .format(max(df_train[df_train.target == 1].len)))


# In[ ]:


print("min tweet length of no disaster {0}" .format(min(df_train[df_train.target == 0].len)))
print("min tweet length of disaster {0}" .format(min(df_train[df_train.target == 1].len)))


# In[ ]:


print(df_train[df_train.target == 0]["len"].mean())
print(df_train[df_train.target == 1]["len"].mean())


# # Improvements

# ### Extract Hashtags

# In[ ]:


nlp = spacy.load('en')
matcher = Matcher(nlp.vocab)
matcher.add('HASHTAG', None, [{'ORTH': '#'}, {'IS_ASCII': True}])


# The below function extracts hashtag - strips of `#` and converts the hashtag to lower case for normaliztion

# In[ ]:


def extract_hashtags(text):
    doc = nlp(text)
    matches = matcher(doc)
    hashtags = []
    for match_id, start, end in matches:
        hashtags.append(doc[start+1:end].text.lower()) 
    return hashtags


# ### This is just for tests 

# In[ ]:


#hashtags = extract_hashtags(train.text[4])
#hashtags


# In[ ]:


train['hashtags'] = train.text.apply(extract_hashtags)


# In[ ]:


hash_df_0 = dict(pd.Series(np.concatenate(train[train.target==0].hashtags.reset_index(drop = True))).value_counts())


# In[ ]:



wordcloud = WordCloud(width=800, height=400,background_color="white").generate_from_frequencies(hash_df_0)
plt.figure(figsize=[14,8])
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[ ]:


hash_df_1 = dict(pd.Series(np.concatenate(train[train.target==1].hashtags.reset_index(drop = True))).value_counts())


# In[ ]:



wordcloud = WordCloud(width=800, height=400,background_color="white").generate_from_frequencies(hash_df_1)
plt.figure(figsize=[14,8])
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# ### Model Baseline 

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn import feature_extraction, linear_model, model_selection, preprocessing
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


# In[ ]:


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf


# In[ ]:


# For CV 

X = train.text

y = train.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=122)


# ### Tokenizer

# In[ ]:


count_vectorizer = feature_extraction.text.CountVectorizer()


# In[ ]:


train_vectors = count_vectorizer.fit_transform(X_train)
test_vectors = count_vectorizer.transform(X_test)


# In[ ]:


train_vectors.shape
test_vectors.shape


# ### TF_IDF

# In[ ]:


tfidf_transformer = TfidfTransformer()


# In[ ]:


train_tfidf = tfidf_transformer.fit_transform(train_vectors)
test_tfidf = tfidf_transformer.fit_transform(test_vectors)


# ### Naive Bayes

# In[ ]:


model =  MultinomialNB().fit(train_tfidf, y_train)


# In[ ]:


y_pred = model.predict(test_tfidf)


# In[ ]:


#model = RandomForestClassifier(n_estimators=1000, random_state=0).fit(train_tfidf, train.target)
#predicted = model.predict(test_tfidf)


# In[ ]:


f1_score(y_test, y_pred, average='micro')


# In[ ]:


confusion_matrix(y_test, y_pred)


# ### SVM

# In[ ]:


model_svm = SVC(gamma='scale').fit(train_tfidf, y_train)


# In[ ]:


y_pred = model_svm.predict(test_tfidf)


# In[ ]:


f1_score(y_test, y_pred, average='micro')


# In[ ]:


confusion_matrix(y_test, y_pred)


# In[ ]:


test_vectors_og = count_vectorizer.transform(test.text)
test_tfidf_og = tfidf_transformer.fit_transform(test_vectors_og)


# In[ ]:



predicted = model.predict(test_tfidf_og)


# In[ ]:


submission.target = predicted


# In[ ]:


submission.head()


# In[ ]:


submission.to_csv("submission.csv", index=False)


# In[ ]:




