#!/usr/bin/env python
# coding: utf-8

# * [What you will learn? ](#section1)
# * [Importing ](#section2)
# * [Performing EDA ](#section3)
# * [Which type of tweets are more & how much? ](#section4)
# * [What pattern keywords help us to find? ](#section5)
# * [Cleaning Tweets ](#section6)
# * [Finding most used words in disaster & non-disaster tweets using Wordcloud ](#section7)
# * [Splitting the dataset into train & test set ](#section8)
# * [Model Selection ](#section9)
# * [Hyperparameter-tuning ](#section10)
# * [Fitting tuned model ](#section11)
# * [Prediction ](#section12)
# * [Complete Evaluation ](#section13)

# # What you will learn? <a id='section1'></a>

# This notebook will help you to learn:
# 
# <li>How to use Pipeline in ML specially in NLP to make your work too easy and simple</li>
# <li>Basic EDA performed while analysing the text</li>
# <li>How to remove unwanted words, symbols and other things from text using regular expression</li>
# <li>How to use Wordcloud to make our search for most used words easy while performing analysing.</li>
# <li>Hyperparameter tuning with Pipeline</li>
# <li>How to check accuracy of different Machine Learning Algoritms (models) simultaneously and that too fast with Pipeline.</li>

# Learning from others work is the best tool for mastering anything.
# <h3>Your learning tour begins!</h3>

# # Importing <a id='section2'></a>

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import nltk
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import seaborn as sns
import regex as re


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV


from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # Loading Dataset <a id='section3'></a>

# In[ ]:


df = pd.read_csv('../input/nlp-getting-started/train.csv')


# In[ ]:


df.head()


# # Performing EDA

# # Which type of tweets are more & how much? <a id='section4'></a>

# In[ ]:


print(df.isnull().sum())
sns.countplot(df['target'])


# Their are most Non-Disastrous tweets and thats intuitive. Disasters are rarely prevailed. If not that it's a point to be considered.

# # What pattern keywords help us to find? <a id='section5'></a>

# In[ ]:


key1 = df[df['target'] == 1]['keyword'].value_counts()
f, ax = plt.subplots(figsize=(6, 55))
sns.barplot(key1.values,key1.index,
            label="Keyword", color="b")
sns.despine(left=True, bottom=True)


# This shows the number of times different keywords are used to label the disaster-tweet. Keywords like outbreak, suicude_bomb are used more number of times. Which shows that such disaster events may had happen more than other.

# In[ ]:


df[df['target'] == 1]['keyword'].isnull().sum()


# In[ ]:


key2 = df[df['target'] == 0]['keyword'].value_counts()
f, ax = plt.subplots(figsize=(6, 55))
sns.barplot(key2.values,key2.index,
            label="Keyword", color="g")
sns.despine(left=True, bottom=True)


# Finding most used keywords to label non-disaster tweets.

# # Cleaning tweets <a id='section6'></a>

# In[ ]:


df['text'] = df['text'].replace('http://\w+.\w+/\w+',' ',regex = True)

df['text'] = df['text'].replace('https://\w+.\w+/\w+',' ',regex = True)

df['text'] = df['text'].replace('http://\w*.?\w*/?\w+',' ',regex = True)

df['text'] = df['text'].replace('@\w+',' ',regex = True)

df['text'] = df['text'].replace('\d+',' ',regex = True)

df['text'] = df['text'].replace('[!@#$%^&*()_+\-":;.,/?\=}{}"<>~`]',' ',regex = True)

df['text'] = df['text'].replace(r'\s{2:}',' ',regex = True)


# We have removed urls, symbols, numbers, tags and other unwanted things from raw tweets as they provide no value to us. 

# In[ ]:


df['text'].head()


# # Finding most used words in disaster & non-disaster tweets using Wordcloud <a id='section7'></a>

# In[ ]:


target1 = df[df['target'] == 1]
target0 = df[df['target'] == 0]


# In[ ]:


string = ' '.join(list(target1['text'].values))
stopwords = set(STOPWORDS)
wordcloud = WordCloud(background_color="white",max_words =200,stopwords = stopwords,random_state=42,max_font_size = 100)
wordcloud.generate(string)
plt.figure(figsize=(24,29))
plt.imshow(wordcloud)
plt.axis('off')
plt.tight_layout()


# This shows that most of the disasters may have happened because of fire, storm, suicide, car accidents, flood and much more as this words are highly used in many tweets. And one more pattern it brings is most of may have happened in California.

# In[ ]:


string = ' '.join(list(target0['text'].values))
stopwords = set(STOPWORDS)
wordcloud = WordCloud(background_color="white",max_words =200,stopwords = stopwords,random_state=42)
wordcloud.generate(string)
plt.figure(figsize=(24,16))
plt.imshow(wordcloud)
plt.axis('off')
plt.tight_layout()


# # Splitting the dataset into train & test set. <a id='section8'></a>

# In[ ]:


df_main = df[['text','target']] 
X = df_main['text']
y = df_main['target']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.15,random_state = 42)


# # Model Selection <a id='section9'></a>

# In[ ]:


classifiers = [
    KNeighborsClassifier(3),
    LinearSVC(),
    SVC(kernel="rbf"),
    NuSVC(probability=True),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    MultinomialNB()
    ]


# In[ ]:


for classifier in classifiers:
    text_clf = Pipeline([('tfidf',TfidfVectorizer()),('clf',classifier)])
    text_clf.fit(X_train,y_train)
    predictions = text_clf.predict(X_test)
    print(classifier)
    print(accuracy_score(y_test,predictions))


# Highest Accuracy is given by LinearSVC algorithm without any hyperparameter tuning. So we will move forward with it. Now let's tune it's hyperparameter and make our accuracy better.

# # Hyperparameter-tuning <a id='section10'></a>

# In[ ]:


parameters = {
    'tfidf__stop_words':('english',None),
    'tfidf__max_df':(0.5,0.75,1),
    'tfidf__max_features': (None, 5000, 10000, 50000),
    'clf__max_iter':(800,1000,2000,4000),
    'clf__C':(0.1,0.5,1,2)
    
}


# In[ ]:


pipeline = Pipeline([('tfidf',TfidfVectorizer()),('clf',LinearSVC())])


# Defining the pipeline. First the data will get pass into tfidvectorizer whose output will be fitted in the model.

# In[ ]:


grid = GridSearchCV(pipeline,parameters,n_jobs=-1, verbose=1)
grid.fit(X_train, y_train)


# We tested multiple options on the selected model. Now it's time to see the best hyperparameters that fit our model for highest accuracy.

# In[ ]:


grid.best_estimator_.get_params()


# From the parameters we provided the best selected are as under: <br><br>
# <h6>For tfidfvectorizer</h6>
# <li>stop_words=None</li>
# <li>max_df=0.5</li>
# <li>max_features = 10000</li>
# 
# <h6>For our Model LinearSVC</h6>
# <li>max_iter=800</li>
# <li>C=0.1</li>
# 
# 

# **NOTE: You can try more by tuning other parameters. I have just tuned few to show you how to do hyperparameter tuning for increasing the accuracy.**

# # Fitting in the tuned model <a id='section11'></a>

# In[ ]:


text_clf = Pipeline([('tfidf',TfidfVectorizer(stop_words=None,max_df=0.5,max_features = 10000)),('clf',LinearSVC(max_iter=800,C=0.1))])


# In[ ]:


text_clf.fit(X_train,y_train)


# # Prediction <a id='section12'></a>

# In[ ]:


predictions = text_clf.predict(X_test)


# # Complete Evaluation <a id='section13'></a>

# In[ ]:


print("Confusion matrix\n",confusion_matrix(y_test,predictions))


# In[ ]:


print("Classification Report\n",classification_report(y_test,predictions))


# In[ ]:


print("Accuracy Score\n",accuracy_score(y_test,predictions))


# NOTE: Check what's the accuracy of the same model without tuning hyperparameters when we selected our model. This shows that tuning hyperparameters is must.

# ***Congratulations! You learned alot many things to make your project accurate.***

# # Thanks Giving <a id='section13'></a>

# Thanks for learning from my notebook. Much Appreciated!

# <h1 style="color:red">Don't forget to upvote my notebook if you found it useful</h1>
