#!/usr/bin/env python
# coding: utf-8

# # <center><span style = 'color:red'>Sentiment Analysis IMDB 50k Movie Dataset</span></center>

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        filepath = os.path.join(dirname, filename) 
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# * # <center><span style = 'color:red'>Sentiment Analysis IMDB 50k Movie Dataset</span></center>

# ### Since the competition is over, I am creating this notebook to help the begineers to learn from this solution and also the experts can provide me suggestions to further improve my solution. Let's start the project by exploring the data. 
# 
# I have stored the filename and path in 'filepath'. Let's load the data as a pandas dataframe and play with it. 

# In[ ]:


df = pd.read_csv(filepath)
df.sentiment.value_counts()


# In[ ]:


df.info()


# Our dataset contains 50k rows indexed 0 to 49,999 and two columns. One column contains the review and other contains sentiment. In total there are 25k +ive and 25k -ive sentiments. Let's check if the reviews are randomly distributed or are in some order. 

# In[ ]:


df.head()


# Using df.head(), df.tail() we can observe that reviews are randomly distributed thus we don't need to redistribute the data to randomize it. We also observe that data contains (review column): lower case and upper case letters, some html tags within "< > ", punctuations, apostrophies e.t.c. It would be prudent to clean the data before proceeding but just to show you the difference we will do this project with and without data pre-processing.   

# ## TF-IDF 
# 
# Since we have reviews as text and we want to run a mathematical model we need a method to convert the text to numbers. We will use TF-IDF method for this, inorder to know the details about TF-IDF you may read about it __[here](https://towardsdatascience.com/sentiment-analysis-introduction-to-naive-bayes-algorithm-96831d77ac91)__. Note that it contains 50k rows so its a mid-sized database so we expect the TF-IDF based model to work better than deep learning models.  
# 
# We will use pre-built TF-IDF vectorizer from sklearn library. Let us create a document term matrix (DTM) using TF-IDF. 

# In[ ]:


import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(min_df=2, max_df=0.5, ngram_range=(1,2))
text_count_matrix = tfidf.fit_transform(df.review)


# In[ ]:


#splitting the complete dataset in test and training dataset:
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(text_count_matrix, df.sentiment, test_size=0.30, random_state=2)


# In[ ]:


#converting the sentiments (positive and negatives) to 1 and 0. 
y_train = (y_train.replace({'positive': 1, 'negative': 0})).values
y_test = (y_test.replace({'positive': 1, 'negative': 0})).values


# In[ ]:


# let's use Naive Bayes classifier and fit our model:
from sklearn.naive_bayes import MultinomialNB 
MNB = MultinomialNB()
MNB.fit(x_train, y_train)
#4. Evaluating the model
from sklearn import metrics
accuracy_score = metrics.accuracy_score(MNB.predict(x_test), y_test)
print("accuracy_score without data pre-processing = " + str('{:04.2f}'.format(accuracy_score*100))+" %")


# ## With Data Pre-Processing
# 
# I am using some steps of data pre-processing, for ex: I am not using lemmatization you can use it and see how the result varies. You may add some more steps to it for example if there is any happy smiley replace it with a positive word like good and a sad smiley with negative word like bad. 

# In[ ]:


#let's investigate what kind of special characters and language is used by the reviewers to review the content. 
#we can observe some html tags
#use of parenthesis
#punctuation (apostrophy, '' e.t.c)

import re
import nltk 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
stop_words = set(stopwords.words('english'))
lemmatizer = nltk.stem.WordNetLemmatizer()
#print(df.review[4])

processed_review = []
single_review = "string to iniialize <br /> my email id is charilie@waoow.com. You can also reach to me at charlie's "
reviews = df.review
for review in range(0,50000):
    single_review = df.loc[review,'review']
    
    #start processing the single_review 
    
    #removing html tags:
    single_review = re.sub('<.*?>',' ',single_review)
    #removing special characters (punctuation) '@,!' e.t.c.
    single_review = re.sub('\W',' ',single_review)
    #removing single characters
    single_review = re.sub('\s+[a-zA-Z]\s+',' ', single_review)
    #substituting multiple spaces with single space
    single_review = re.sub('\s+',' ', single_review)
   
    #removing stop words
    #word_tokens = []
    word_tokens = word_tokenize(single_review)
    #lemmatization
    #lemmatized_sentence = " ".join(lemmatizer.lemmatize(token) for token in word_tokens if token not in stop_words)
    filtered_sentence = []
    #filtered_sentence.append([w for w in word_tokens if w not in stop_words])
    filtered_sentence2 = " ".join([w for w in word_tokens if w not in stop_words])
    
    
    #compile all the sentences to make a complete dictionary of processed reviews
    processed_review.append(filtered_sentence2)
    
print(processed_review[10])
#print(filtered_sentence2)


# In[ ]:


text_count_matrix2 = tfidf.fit_transform(processed_review)


# In[ ]:


X_train, X_test, Y_train, Y_test = train_test_split(text_count_matrix2, df.sentiment, test_size=0.30, random_state=2)


# In[ ]:


Y_train = (Y_train.replace({'positive': 1, 'negative': 0})).values
Y_test = (Y_test.replace({'positive': 1, 'negative': 0})).values


# In[ ]:


MNB.fit(X_train, Y_train)
#4. Evaluating the model
accuracy_score = metrics.accuracy_score(MNB.predict(X_test), Y_test)
print(str('{:04.2f}'.format(accuracy_score*100))+" %")


# In[ ]:


from sklearn.metrics import classification_report, confusion_matrix

print("Classification Report: \n", classification_report(Y_test, MNB.predict(X_test),target_names=['Negative','Positive']))
print("Confusion Matrix: \n", confusion_matrix(Y_test, MNB.predict(X_test)))


# ##  Using Linear-SVC with pre-processing and SGDC

# In[ ]:


from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC


# In[ ]:


LSVC = LinearSVC()
LSVC.fit(X_train, Y_train)
accuracy_score = metrics.accuracy_score(LSVC.predict(X_test), Y_test)
print("Linear SVC accuracy = " + str('{:04.2f}'.format(accuracy_score*100))+" %")
print("Classification Report: \n", classification_report(Y_test, LSVC.predict(X_test),target_names=['Negative','Positive']))
print("Confusion Matrix: \n", confusion_matrix(Y_test, LSVC.predict(X_test)))


# In[ ]:


SGDC = SGDClassifier()
SGDC.fit(X_train, Y_train)
predict = SGDC.predict(X_test)
accuracy_score = metrics.accuracy_score(predict, Y_test)
print("Stocastic Gradient Classifier accuracy = " + str('{:04.2f}'.format(accuracy_score*100))+" %")
print("Classification Report: \n", classification_report(Y_test, predict,target_names=['Negative','Positive']))
print("Confusion Matrix: \n", confusion_matrix(Y_test, predict))


# In[ ]:


LR = LogisticRegression()
LR.fit(X_train, Y_train)
predict = LR.predict(X_test)
accuracy_score = metrics.accuracy_score(predict, Y_test)
print("LR = " + str('{:04.2f}'.format(accuracy_score*100))+" %")
print("Classification Report: \n", classification_report(Y_test, predict,target_names=['Negative','Positive']))
print("Confusion Matrix: \n", confusion_matrix(Y_test, predict))


# ## Comparing Different Models
# 
# Here we have seen that LSVM with limited data pre-processing brings the best result to us with the accuracy of over 91.10% . This is a significant improvement over NB model. 
# 
# I am open for your suggestions and feedback. Kindly let me know if you want any elaborate explanantion on any of the steps or have ideas to further improve the model.   

# In[ ]:




