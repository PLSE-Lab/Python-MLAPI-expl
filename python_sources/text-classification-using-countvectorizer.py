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


df = pd.read_csv("/kaggle/input/dataisbeautiful/r_dataisbeautiful_posts.csv")
df.head()


# In[ ]:


df.dtypes


# In[ ]:


df.isnull().sum()


# In[ ]:


import seaborn as sns


# In[ ]:


sns.heatmap(df.isnull())


# In[ ]:


df = df.drop(columns=['author_flair_text','total_awards_received','removed_by','awarders','full_link','created_utc'],axis=1)
df.head(2)


# In[ ]:


df['over_18'] = df['over_18'].replace(True,1)
df['over_18'] = df['over_18'].replace(False,0)
df['title'] = df['title'].fillna(' ')


# In[ ]:


df.head().style.background_gradient(cmap='Purples')


# In[ ]:


df['text'] = df['title']+" "+ df['author']

df.head().style.background_gradient(cmap='Purples')


# In[ ]:


df.drop(columns=['title','author'],axis=1,inplace=True)


# In[ ]:


df['over_18'].value_counts()


# In[ ]:


sns.heatmap(df.isnull())


# In[ ]:


df.info()


# In[ ]:


df.shape


# In[ ]:


import nltk
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer,HashingVectorizer,TfidfTransformer
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix,accuracy_score
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
import matplotlib.pyplot as plt


# In[ ]:


df.head()


# In[ ]:


df['text'] = df['text'].astype(str)


# In[ ]:


porter = PorterStemmer()
corpus = []
for i in range(0,len(df)):
    review = re.sub('[^a-zA-Z]',' ',df['text'][i])
    review = review.lower()
    review = review.split()
    review = [porter.stem(word) for word in review if not word in stopwords.words('english')]
    review = " ".join(review)
    corpus.append(review)
    


# In[ ]:


# CountVectorizer
cv = CountVectorizer(max_features=4000,ngram_range=(1,4))
X = cv.fit_transform(corpus).toarray()
cv.get_feature_names()[:20]


# In[ ]:


cv.get_params()


# In[ ]:


y = df['over_18']


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# # # Multinomial NaiveBayes

# In[ ]:


# Multinomial NaiveBayes
from sklearn.naive_bayes import MultinomialNB
mnb = MultinomialNB()
mnb.fit(X_train,y_train)
y_pred = mnb.predict(X_test)
cm = confusion_matrix(y_test,y_pred)
print("naive bayes accuracy is ",round(accuracy_score(y_test,y_pred),2)*100,"%")
print("\n")
print("confusion matrix for naive bayes ")
plot_confusion_matrix(cm,class_names=['FAKE','REAL'],cmap='Purples',show_normed=True,colorbar=True,figsize=(6,6))
plt.show()


# # PassiveAgressiveClassifier

# In[ ]:


# PassiveAgressiveClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
pac = PassiveAggressiveClassifier()
pac.fit(X_train,y_train)
y_pred = pac.predict(X_test)
cm = confusion_matrix(y_test,y_pred)
print("Passive Aggressive Classifier accuracy is ",round(accuracy_score(y_test,y_pred),2)*100,"%")
print("\n")
print("confusion matrix for Passive Aggressive Classifier ")
plot_confusion_matrix(cm,class_names=['fake','Real'],cmap='Purples',show_normed=True,colorbar=True,figsize=(6,6))
plt.show()


# # Furthur modifications will be done including visualisations and other vectorizers

# 

# 

# In[ ]:




