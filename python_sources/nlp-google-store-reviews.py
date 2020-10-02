#!/usr/bin/env python
# coding: utf-8

# Natural Language Processing is the sub-field of AI that is focused on enabling computers to understand and process human languages.
# 
# **In this kernel I will show you the computers can understand the human language or not?** We will also classify the types of reviews which are Positive, Negative and Neutral.
# 
# Here on the photo you can check the schema that we will do step by step.
# 
# ref: Adam Geitgey on www.medium.com

# ![](https://i.ibb.co/bstrG04/1-z-HLs87sp8-R61eh-Uo-Xep-WHA.png)

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# We are using google reviews and their Sentiment types which are Positive-Negative etc..

# In[ ]:


data=pd.read_csv("../input/googleplaystore_user_reviews.csv",encoding="latin1")


# In[ ]:


data.head()


# **Selecting reviews and types of reviews.**

# In[ ]:


df=pd.concat([data.Translated_Review,data.Sentiment],axis=1)
df.dropna(axis=0,inplace=True)
df.head(10)


# In[ ]:


df.Sentiment.value_counts()


# Converting review types to int form in order to use classification methods.
# 
# **0= Positive, 1=Negative, 2= Neutral**

# In[ ]:


df.Sentiment=[0 if i=="Positive" else 1 if i== "Negative" else 2 for i in df.Sentiment]
df.head(10)


# **Here we will remove characters which are not letters.**         ":) # $ @ ()!-/*"   like that!
# 
# Also converting them lower case.

# In[ ]:


#Data cleaning
import re
first_text=df.Translated_Review[0]
text=re.sub("[^a-zA-Z]"," ",first_text) #changing characters with space
text=text.lower()


# In[ ]:


print(df.Translated_Review[0]) #lets review of changings
print(text)


# Here we can see basic text cleaning. Keep going deeper..

# In[ ]:


#stopwords (irrelavent words)
import nltk
#nltk.download("stopwords")
#nltk.download("punkt")
from nltk.corpus import stopwords
text=nltk.word_tokenize(text) #separate all words


# **With "tokenize" wi seperated all words.**
# 
# Also with stopwords we can remove irrelavent words. But I will do this step later.

# In[ ]:


text


# **What is lemmatization??? With lemmatization we can convert words to their root format.** 
# 
# For instance books--->book

# In[ ]:


#lemmatization books----> book
import nltk as nlp
lemma=nlp.WordNetLemmatizer()
text=[lemma.lemmatize(i) for i in text]
text=" ".join(text)
text


# Yes! Now our sentence is really simple. Keep going to apply to all dataset.

# In[ ]:


text_list=[]
for i in df.Translated_Review:
    text=re.sub("[^a-zA-Z]"," ",i)
    text=text.lower()
    text=nltk.word_tokenize(text)
    lemma=nlp.WordNetLemmatizer()
    text=[lemma.lemmatize(word) for word in text]
    text=" ".join(text)
    text_list.append(text)


# In[ ]:


text_list[:10]


# Here with "bag of words" we are removing irrelavent words and creating matrix form in order to make them in order. Also after matrix form we will have our sentences with numbers. This means that **now our computer can understand human language!**
# 
# 
# Let's understand the process easier with next images.

# ![](https://i.ibb.co/42Gs1GH/atap-0401.png)
# 

# ![](https://i.ibb.co/NrbjNxW/1-e-Ueduf-Al7-s-I-QWSEIst-Zg.png)

# In[ ]:


#bag of words
from sklearn.feature_extraction.text import CountVectorizer
max_features=200000
cou_vec=CountVectorizer(max_features=max_features,stop_words="english")
sparce_matrix=cou_vec.fit_transform(text_list).toarray()
all_words=cou_vec.get_feature_names()
print("Most used words: ",all_words[50:100])


# **We converted our textes to array form, also removed irrelavent words.**
# 
# Before classification let's make it more colorful :) Here I visualized most used words:

# In[ ]:


from wordcloud import WordCloud
plt.subplots(figsize=(12,12))
wordcloud=WordCloud(background_color="white",width=1024,height=768).generate(" ".join(all_words[100:]))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()


# In[ ]:


#classification
y=df.iloc[:,1].values
x=sparce_matrix
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


# In[ ]:


#Random Forest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators = 10, random_state=42)
rf.fit(x_train,y_train)
print("accuracy: ",rf.score(x_test,y_test))


# In[ ]:


#confussion matrix
y_pred=rf.predict(x_test)
y_true=y_test
from sklearn.metrics import confusion_matrix
import seaborn as sns
names=["Positive","Negative","Neutral"]
cm=confusion_matrix(y_true,y_pred)
f,ax=plt.subplots(figsize=(5,5))
sns.heatmap(cm,annot=True,linewidth=.5,linecolor="r",fmt=".0f",ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
ax.set_xticklabels(names)
ax.set_yticklabels(names)
plt.show()


# Wowww %87 accuracy! The result is really good, let's try logistic regression to improve accuracy.

# In[ ]:


#logistic Regression
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(x_train,y_train)
print("lr accuracy: ",lr.score(x_test,y_test))


# In[ ]:


#confussion matrix
y_pred=lr.predict(x_test)
y_true=y_test
from sklearn.metrics import confusion_matrix
import seaborn as sns
names=["Positive","Negative","Neutral"]
cm=confusion_matrix(y_true,y_pred)
f,ax=plt.subplots(figsize=(5,5))
sns.heatmap(cm,annot=True,linewidth=.5,linecolor="r",fmt=".0f",ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
ax.set_xticklabels(names)
ax.set_yticklabels(names)
plt.show()


# * In Conclusion
# We used NLP tools in order to train our dataset. After training we classified reviews with our model. At the final we have accuracy more than %90.
# 
# * If you have any question or any suggestions feel free to write me.
