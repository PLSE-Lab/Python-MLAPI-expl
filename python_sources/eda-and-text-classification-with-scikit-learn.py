#!/usr/bin/env python
# coding: utf-8

# # Quora Insincere Questions Classification:
# 

# ## Detect toxic content to improve online conversations
# 

# ## Problem:
# * **Handle toxic and disivie content / miseleading content**

# * **To do that we have to develop a model that identify and flag insincere questions**

# ## Evaluation:

# * **For each qid in the testset, predict the corresponding questions_text:**
# 
#  * **Is insincere => 1 (Target)**
# 
#  * **Is sincere => 0**
# 
# * **Submissions are evaluated on F1 Score between the predicted and the observed targets**

# ## Data Fields:

# * **qid : unique question identifer**
# 
# * **question_text : Quora question text**
# 
# * **Target : a question labeled << insincere >> has a value of 1, otherwise 0**

# In[ ]:


import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from wordcloud import WordCloud


import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.metrics import classification_report

import warnings
warnings.filterwarnings("ignore")


# In[ ]:


X_train_filepath = os.path.join('..', 'input', 'train.csv')
X_test_filepath = os.path.join('..', 'input', 'test.csv')
sample_filepath = os.path.join('..', 'input', 'sample_submission.csv')
X_train_filepath, X_test_filepath, sample_filepath


# In[ ]:


df_train = pd.read_csv(X_train_filepath, encoding='ISO-8859-1')
df_train.head()


# In[ ]:


df_test = pd.read_csv(X_test_filepath, encoding='ISO-8859-1')
df_test.head()


# In[ ]:


df_train.shape, df_test.shape


# In[ ]:


df_sample = pd.read_csv(sample_filepath, encoding='ISO-8859-1')
df_sample.head()


# In[ ]:


df_train.info()


# In[ ]:


df_train["target"].value_counts()


# ## Exploratory Data Analysis:

# In[ ]:


insincere = df_train[df_train["target"] == 1]
sincere = df_train[df_train["target"] == 0]


# In[ ]:


sincere.head()


# In[ ]:


sincere.info()


# In[ ]:


insincere.head()


# In[ ]:


insincere.info()


# ## Distribution sincere/insincere plots:

# In[ ]:


ax, fig = plt.subplots(figsize=(10, 7))
question_class = df_train["target"].value_counts()
question_class.plot(kind= 'bar', color= ["blue", "orange"])
plt.title('Bar chart')
plt.show()


# In[ ]:


print(df_train['target'].value_counts())
print(sum(df_train['target'] == 1) / sum(df_train['target'] == 0) * 100, "percent of questions are insincere.")
print(100 - sum(df_train['target'] == 1) / sum(df_train['target'] == 0) * 100, "percent of questions are sincere")


# **We have a Unbalenced Data**

# ## Insincere Word cloud:

# In[ ]:


stop_words = stopwords.words("english")


# In[ ]:


insincere_words = ''

for question in insincere.question_text:
    text = question.lower()
    tokens = word_tokenize(text)
    for words in tokens:
        insincere_words = insincere_words + words + ' '


# In[ ]:


# Generate a word cloud image
insincere_wordcloud = WordCloud(width=600, height=400).generate(insincere_words)
#Insincere Word cloud
plt.figure( figsize=(10,8), facecolor='k')
plt.imshow(insincere_wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()


# ## Length distribution in sincere and insincere questions:

# In[ ]:


insincere["questions_length"] = insincere.question_text.apply(lambda x: len(x))
sincere["questions_length"] = sincere.question_text.apply(lambda x: len(x))


# In[ ]:


insincere["questions_length"].mean()


# In[ ]:


sincere["questions_length"].mean()


# In[ ]:


sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.distplot(insincere.questions_length, hist=True, label="insincere")
sns.distplot(sincere.questions_length, hist=True, label="sincere");


# ## Number of words distribution in sincere and insincere questions:

# In[ ]:


insincere['number_words'] = insincere.question_text.apply(lambda x: len(x.split()))
sincere['number_words'] = sincere.question_text.apply(lambda x: len(x.split()))


# In[ ]:


insincere['number_words'].mean()


# In[ ]:


sincere['number_words'].mean()


# In[ ]:


sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.distplot(insincere.number_words, hist=True, label="insincere")
sns.distplot(sincere.number_words, hist=True, label="sincere");


# ## TfidfVectorizer :

# In[ ]:


vectorizer = TfidfVectorizer(min_df=5, ngram_range=(1,3),
                        strip_accents='unicode',lowercase =True, 
                        stop_words = 'english',tokenizer=word_tokenize)


# In[ ]:


train_vectorized = vectorizer.fit_transform(df_train.question_text.values)


# In[ ]:


test_vectorized = vectorizer.fit_transform(df_test.question_text.values)


# In[ ]:


X_train, X_val, y_train, y_val = train_test_split(train_vectorized, df_train.target.values, test_size=0.1, stratify = df_train.target.values)


# ## Classification with Logistic Regression :

# In[ ]:


lr = LogisticRegression(C=10, class_weight={0:0.07 , 1:1})


# In[ ]:


lr.fit(X_train, y_train)


# In[ ]:


y_pred_train1 = lr.predict(X_train)


# In[ ]:


print(f1_score(y_train, y_pred_train1))


# In[ ]:


y_pred_val1 = lr.predict(X_val)


# In[ ]:


print(f1_score(y_val, y_pred_val1))


# In[ ]:


cm1 = confusion_matrix(y_val, y_pred_val1)
cm1


# In[ ]:


sns.heatmap(cm1, cmap="Blues", annot=True, square=True, fmt=".0f");


# In[ ]:


print(classification_report(y_val, y_pred_val1))


# ## Classification with MultinomialNB:

# In[ ]:


mnb = MultinomialNB(alpha=0.1)


# In[ ]:


mnb.fit(X_train, y_train)


# In[ ]:


y_pred_train2 = mnb.predict(X_train)


# In[ ]:


print(f1_score(y_train, y_pred_train2))


# In[ ]:


y_pred_val2 = mnb.predict(X_val)


# In[ ]:


print(f1_score(y_val, y_pred_val2))


# In[ ]:


cm2 = confusion_matrix(y_val, y_pred_val2)
cm2


# In[ ]:


sns.heatmap(cm2, cmap="Blues", annot=True, square=True, fmt=".0f");


# In[ ]:


print(classification_report(y_val, y_pred_val2))


# ## Classification with Linear SVC:

# In[ ]:


from sklearn.svm import LinearSVC

svc = LinearSVC(C=5, class_weight={0:0.07 , 1:1})
svc.fit(X_train, y_train)


# In[ ]:


y_pred_train3 = svc.predict(X_train)


# In[ ]:


print(f1_score(y_train, y_pred_train3))


# In[ ]:


y_pred_val3 = svc.predict(X_val)


# In[ ]:


print(f1_score(y_val, y_pred_val3))


# In[ ]:


cm3 = confusion_matrix(y_val, y_pred_val3)
cm3


# In[ ]:


sns.heatmap(cm3, cmap="Blues", annot=True, square=True, fmt=".0f");


# In[ ]:


print(classification_report(y_val, y_pred_val3))

