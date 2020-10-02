#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# # 1. EDA

# In[ ]:


import os
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


from IPython.display import Markdown, display
def printmd(string):
    display(Markdown(string))
#printmd('**bold**')


# In[ ]:


data_path = "../input/train.csv"


# In[ ]:


data_raw = pd.read_csv(data_path)
#data_raw = data_raw.loc[np.random.choice(data_raw.index, size=2000)]
data_raw.shape


# In[ ]:



print("Number of rows in data =",data_raw.shape[0])
print("Number of columns in data =",data_raw.shape[1])
print("\n")
printmd("**Sample data:**")
data_raw.head()


# ## Checking for missing values

# In[ ]:


missing_values_check = data_raw.isnull().sum()
print(missing_values_check)


# ## Calculating number of comments under each label

# In[ ]:


# Comments with no label are considered to be clean comments.
# Creating seperate column in dataframe to identify clean comments.

# We use axis=1 to count row-wise and axis=0 to count column wise

rowSums = data_raw.iloc[:,2:].sum(axis=1)
clean_comments_count = (rowSums==0).sum(axis=0)

print("Total number of comments = ",len(data_raw))
print("Number of clean comments = ",clean_comments_count)
print("Number of comments with labels =",(len(data_raw)-clean_comments_count))


# In[ ]:


categories = list(data_raw.columns.values)
categories = categories[2:]
print(categories)


# In[ ]:


# Calculating number of comments in each category

counts = []
for category in categories:
    counts.append((category, data_raw[category].sum()))
df_stats = pd.DataFrame(counts, columns=['category', 'number of comments'])
df_stats


# In[ ]:


sns.set(font_scale = 2)
plt.figure(figsize=(15,8))

ax= sns.barplot(categories, data_raw.iloc[:,2:].sum().values)

plt.title("Comments in each category", fontsize=24)
plt.ylabel('Number of comments', fontsize=18)
plt.xlabel('Comment Type ', fontsize=18)

#adding the text labels
rects = ax.patches
labels = data_raw.iloc[:,2:].sum().values
for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom', fontsize=18)

plt.show()


# ## Calculating number of comments having multiple labels

# In[ ]:


rowSums = data_raw.iloc[:,2:].sum(axis=1)
multiLabel_counts = rowSums.value_counts()
multiLabel_counts = multiLabel_counts.iloc[1:]

sns.set(font_scale = 2)
plt.figure(figsize=(15,8))

ax = sns.barplot(multiLabel_counts.index, multiLabel_counts.values)

plt.title("Comments having multiple labels ")
plt.ylabel('Number of comments', fontsize=18)
plt.xlabel('Number of labels', fontsize=18)

#adding the text labels
rects = ax.patches
labels = multiLabel_counts.values
for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom')

plt.show()


# In[ ]:





# # 2. Data Pre-Processing

# In[ ]:


data = data_raw
data = data_raw.loc[np.random.choice(data_raw.index, size=2000)]
data.shape


# In[ ]:


import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import re

import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")


# ##  Cleaning Data

# In[ ]:


def cleanHtml(sentence):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, ' ', str(sentence))
    return cleantext


def cleanPunc(sentence): #function to clean the word of any punctuation or special characters
    cleaned = re.sub(r'[?|!|\'|"|#]',r'',sentence)
    cleaned = re.sub(r'[.|,|)|(|\|/]',r' ',cleaned)
    cleaned = cleaned.strip()
    cleaned = cleaned.replace("\n"," ")
    return cleaned


def keepAlpha(sentence):
    alpha_sent = ""
    for word in sentence.split():
        alpha_word = re.sub('[^a-z A-Z]+', ' ', word)
        alpha_sent += alpha_word
        alpha_sent += " "
    alpha_sent = alpha_sent.strip()
    return alpha_sent


# In[ ]:


data['comment_text'] = data['comment_text'].str.lower()
data['comment_text'] = data['comment_text'].apply(cleanHtml)
data['comment_text'] = data['comment_text'].apply(cleanPunc)
data['comment_text'] = data['comment_text'].apply(keepAlpha)
data.head()


# ##  Removing Stop Words

# In[ ]:


stop_words = set(stopwords.words('english'))
stop_words.update(['zero','one','two','three','four','five','six','seven','eight','nine','ten','may','also','across','among','beside','however','yet','within'])
re_stop_words = re.compile(r"\b(" + "|".join(stop_words) + ")\\W", re.I)
def removeStopWords(sentence):
    global re_stop_words
    return re_stop_words.sub(" ", sentence)

data['comment_text'] = data['comment_text'].apply(removeStopWords)
data.head()


# ## Stemming

# In[ ]:


stemmer = SnowballStemmer("english")
def stemming(sentence):
    stemSentence = ""
    for word in sentence.split():
        stem = stemmer.stem(word)
        stemSentence += stem
        stemSentence += " "
    stemSentence = stemSentence.strip()
    return stemSentence

data['comment_text'] = data['comment_text'].apply(stemming)
data.head()


# ## Train-Test Split

# In[ ]:


from sklearn.model_selection import train_test_split

train, test = train_test_split(data, random_state=42, test_size=0.30, shuffle=True)

print(train.shape)
print(test.shape)


# In[ ]:


train_text = train['comment_text']
test_text = test['comment_text']


# ## TF-IDF

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(strip_accents='unicode', analyzer='word', ngram_range=(1,3), norm='l2')
vectorizer.fit(train_text)
vectorizer.fit(test_text)


# In[ ]:


x_train = vectorizer.transform(train_text)
y_train = train.drop(labels = ['id','comment_text'], axis=1)

x_test = vectorizer.transform(test_text)
y_test = test.drop(labels = ['id','comment_text'], axis=1)


#  # 3. Multi-Label Classification
# ##  Multiple Binary Classifications - (One Vs Rest Classifier)

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier


# In[ ]:


get_ipython().run_cell_magic('time', '', '\n# Using pipeline for applying logistic regression and one vs rest classifier\nLogReg_pipeline = Pipeline([\n                (\'clf\', OneVsRestClassifier(LogisticRegression(solver=\'sag\'), n_jobs=-1)),\n            ])\n\nfor category in categories:\n    printmd(\'**Processing {} comments...**\'.format(category))\n    \n    # Training logistic regression model on train data\n    LogReg_pipeline.fit(x_train, train[category])\n    \n    # calculating test accuracy\n    prediction = LogReg_pipeline.predict(x_test)\n    print(\'Test accuracy is {}\'.format(accuracy_score(test[category], prediction)))\n    print("\\n")')


# ## Multiple Binary Classifications - (Binary Relevance)

# In[ ]:


get_ipython().run_cell_magic('time', '', '\n# using binary relevance\nfrom skmultilearn.problem_transform import BinaryRelevance\nfrom sklearn.naive_bayes import GaussianNB\n\n# initialize binary relevance multi-label classifier\n# with a gaussian naive bayes base classifier\nclassifier = BinaryRelevance(GaussianNB())\n\n# train\nclassifier.fit(x_train, y_train)\n\n# predict\npredictions = classifier.predict(x_test)\n\n# accuracy\nprint("Accuracy = ",accuracy_score(y_test,predictions))\nprint("\\n")')


# ## Classifier Chains

# In[ ]:


# using classifier chains
from skmultilearn.problem_transform import ClassifierChain
from sklearn.linear_model import LogisticRegression


# In[ ]:


get_ipython().run_cell_magic('time', '', '\n# initialize classifier chains multi-label classifier\nclassifier = ClassifierChain(LogisticRegression())\n\n# Training logistic regression model on train data\nclassifier.fit(x_train, y_train)\n\n# predict\npredictions = classifier.predict(x_test)\n\n# accuracy\nprint("Accuracy = ",accuracy_score(y_test,predictions))\nprint("\\n")')


# In[ ]:





# In[ ]:




