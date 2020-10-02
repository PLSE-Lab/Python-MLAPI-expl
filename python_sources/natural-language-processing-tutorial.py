#!/usr/bin/env python
# coding: utf-8

# ## What is Natural Language Processing (NPL) ?
# Natural language processing (NLP) is a subfield of computer science, information engineering, and artificial intelligence concerned with the interactions between computers and human (natural) languages, in particular how to program computers to process and analyze large amounts of natural language data.
# 
# * In this kernel we will learn how we classify types of review. 
# * Our steps are:
#         ** Read Data
#         ** PreProcess Data
#         ** Stopwords
#         ** Lemmatazation
# 

# In[ ]:


## Import Libraries
import numpy as np 
import pandas as pd 
import os

# Read Data
data=pd.read_csv("../input/googleplaystore_user_reviews.csv",encoding="latin1")


# In[ ]:


data.head()    # Show information about our data. 


# Translated_Review &  Sentiment columns are the columns which we are interested. So lets take them.

# In[ ]:


data=pd.concat([data.Translated_Review,data.Sentiment],axis=1)
data.dropna(axis=0,inplace=True)  # For drop nan values. It makes confuse for our model.
data.tail()


# Here is our data. We need to classify reviews according to Sentiment. So Translated_Review is our x column and Sentiment is our y column which we will predict.
# * Now Lets learn our sentiments.

# In[ ]:


data.Sentiment.unique() 


# We have 3 values:
# * Posivitive (0)
# * Negative  (1)
# * Neutral     (2)
#  
# We can accept it like this. Lets convert these to our values.

# In[ ]:


data.Sentiment=[0 if i=="Positive" else 1 if i== "Negative" else 2 for i in data.Sentiment]

#visualization
import matplotlib.pyplot as plt
import seaborn as sns
sns.countplot(data.Sentiment)
plt.title("Count of Sentiments")


# In[ ]:


data.Sentiment.value_counts()


# ## Here our Preprocessing Side
# 
# ** So we need to preprocess our data which means clean data for model.
# 
# ** First we need to remove characters which are not letters.

# ## *For giving example I will work on only one sample. Then works for all

# In[ ]:


import re ## Regular expression for deleting characters which are not letters.
first_sample = data.Translated_Review[9] 
sample = re.sub("[^a-zA-Z]"," ",first_sample)
sample = sample.lower()
print("[{}] convert to \n[{}]".format(data.Translated_Review[9],sample))


# *See we removed  ' ! ' from sentence.

# ## *Now StopWords Turn

# In[ ]:


## import libraries

import nltk  ## Natural Language Tool Kit
from nltk.corpus import stopwords 

sample=nltk.word_tokenize(sample)
print(sample)


# ### *Tokenize provides us split the sentence.

# ### *Drop Unnecessary Words

# In[ ]:


sample = [word for word in sample if not word in set(stopwords.words("english"))]
print(sample)   ## drop unnecesarry words like it, I, you.


# # - What is Lemmatazation
# **  With lemmatization we can convert words to stem. For example; Liked and Like. 
# ** Is it important ?
#     - Yes, because in your 	perspective liked and like seem like same thing but for machine they both are different. We need to make easier it for machine.

# In[ ]:


lemma = nltk.WordNetLemmatizer()  ##We have already imported nltk.
sample = [ lemma.lemmatize(word) for word in sample]
sample = " ".join(sample)
## for this example there is no paragoge I cant show you but if there is -ed or -s or something like these,
## lemmatizer will drop them and returns stem of word


# # Lets apply it for all !

# In[ ]:


text_list=[]
for i in data.Translated_Review:
    text=re.sub("[^a-zA-Z]"," ",i)
    text=text.lower()
    text=nltk.word_tokenize(text)
    lemma=nltk.WordNetLemmatizer()
    text=[lemma.lemmatize(word) for word in text]
    text=" ".join(text)
    text_list.append(text)


# In[ ]:


text_list[:5]


# # We have bag words, clean and relevant words.

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
max_features=1000
cou_vec=CountVectorizer(max_features=max_features) # stop_words="english" you can add but we have already applied it.
sparce_matrix=cou_vec.fit_transform(text_list).toarray()
all_words=cou_vec.get_feature_names()
print("Most used 50 words: ",all_words[0:50])


# # Our data is ready for models. Its time to choose best one !

# # Lets try with Naive Bayes :

# ** Split data to train and test

# In[ ]:


y = data.iloc[:,1].values
x= sparce_matrix

from sklearn.model_selection import train_test_split

xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.2,random_state=1)


# In[ ]:


from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(xtrain,ytrain)
print("acc : ", nb.score(xtest,ytest))


# 59%... Not a good score lets try other models

# In[ ]:


y_pred=nb.predict(xtest)
from sklearn.metrics import confusion_matrix
import seaborn as sns
names=["Positive","Negative","Neutral"]
cm=confusion_matrix(ytest,y_pred)
f,ax=plt.subplots(figsize=(5,5))
sns.heatmap(cm,annot=True,linewidth=.5,linecolor="r",fmt=".0f",ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
ax.set_xticklabels(names)
ax.set_yticklabels(names)
plt.show()


# ### This table is confusion matrix and show us predict of models. Y coordinates is y_true, x coordinates is y_pred. 
# #### In this table if we look neutral, our model predict 38 positive and 58 negative instead of neutral

# # RandomForest Classifier

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
rf = RandomForestClassifier(n_estimators = 10, random_state=42)
rf.fit(xtrain,ytrain)
print("acc: ",rf.score(xtest,ytest))


# In[ ]:


y_pred=rf.predict(xtest)
from sklearn.metrics import confusion_matrix
import seaborn as sns
names=["Positive","Negative","Neutral"]
cm=confusion_matrix(ytest,y_pred)
f,ax=plt.subplots(figsize=(5,5))
sns.heatmap(cm,annot=True,linewidth=.5,linecolor="r",fmt=".0f",ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
ax.set_xticklabels(names)
ax.set_yticklabels(names)
plt.show()


# ## 89% is good score, but lets try 
# 
# ## -Logistic Regression

# In[ ]:


from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(xtrain,ytrain)
print("Logistic Regression accuracy: ",lr.score(xtest,ytest))


# # 90%, A little bit better.

# In[ ]:


y_pred=lr.predict(xtest)
from sklearn.metrics import confusion_matrix
import seaborn as sns
names=["Positive","Negative","Neutral"]
cm=confusion_matrix(ytest,y_pred)
f,ax=plt.subplots(figsize=(5,5))
sns.heatmap(cm,annot=True,linewidth=.5,linecolor="r",fmt=".0f",ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
ax.set_xticklabels(names)
ax.set_yticklabels(names)
plt.show()


# # Conclusion
# 
# ### We used npl on google play store apps data set. We follow basic npl processes which are PreProcess Data, remove Stopwords, Lemmatazation, create bag of words and finally use our model.
# 
# ### **We split our data test and train for model.
# 
# ### ** We used random forest, naive bayes and finally logistic regression.
# 
# ###  ** According to score table Logistic Regression gives us best accurancy like 90%.
# 
# ###  ** If you have any question please feel free to ask. And if you like it please upvote to motivate :)

# In[ ]:




