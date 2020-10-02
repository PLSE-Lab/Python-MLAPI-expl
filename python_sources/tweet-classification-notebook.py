#!/usr/bin/env python
# coding: utf-8

# # Who's Tweeting? Trump vs Trudeau 
# ## is a project I've seen on Datacamp. It asks you to classify a given tweet of either Donald Trump or Justin Trudeau.
# The dataset consists of three columns, ID, Tweet itself and the authors (being either Donald Trump or Justin Trudeau). I have used support vector classifier and logistic regressor in this code, and also compared two word vectorizers; count vectorizer and TF-IDF vectorizer.

# **Importing the libraries**

# In[ ]:


import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression


# **Importing the data, and I've also removed the label row that was given in the dataset**

# In[ ]:


tweetsdf = pd.read_table('../input/tweets-of-trump-and-trudeau/tweets.csv', sep=',', names=('ID', 'Author', 'tweet'))
tweetsdf=tweetsdf.iloc[1:]
tweetsdf.head()


# **We will predict the author from the tweet column, splitting the data as training and test**

# In[ ]:


y=tweetsdf['Author']
x=tweetsdf['tweet']
x_train, x_test, y_train, y_test =train_test_split(x,y,test_size=0.33, random_state=50)
print(x_train)


# **TF-IDF vectorizer, vectorizes the words by dividing the frequency of that specific word by how many times that word appears in how many documents, it yields a matrix with values between 0 and 1 so it gives better precision than the count vectorizer** The columns of matrix are the words and the rows are the documents. 
# It removes English stopwords, and n-gram determines the number of words taken in a phrase, and max and min df values get rid of words either used too much or too rare.

# In[ ]:


tvec= TfidfVectorizer(stop_words="english", ngram_range=(1,2), max_df=0.9, min_df=0.05)


# Splitting the data for the comparison of vectorizers.

# In[ ]:


t_train=tvec.fit_transform(x_train)
t_test=tvec.fit_transform(x_test)


# **Count vectorizer basically counts the words that appear and returns a matrix with columns being the words and rows being tweets.** The elements of matrix are integers. Applying the same procedure with TF-IDF. 

# In[ ]:


cvec = CountVectorizer(stop_words="english",ngram_range=(1,2), max_df=0.9, min_df=0.05)
c_train=cvec.fit_transform(x_train)
c_test=cvec.fit_transform(x_test)


# **Classification with SVC with RBF kernel on the TF-IDF data**

# In[ ]:


svclassifier = SVC(kernel='rbf')
svclassifier.fit(t_train, y_train)
t_predsvc = svclassifier.predict(t_test)


# **Classification with SVC with RBF kernel on Count Vectorizer data**

# In[ ]:


svclassifier = SVC(kernel='rbf')
svclassifier.fit(c_train, y_train)
c_predsvc = svclassifier.predict(c_test)


# **Calculation of accuracies of both vectorizers with SVC**

# In[ ]:


countsvcacc = accuracy_score(c_predsvc,y_test)
print(confusion_matrix(y_test,c_predsvc))
print(classification_report(y_test,c_predsvc))

tfidfsvmacc = accuracy_score(t_predsvc,y_test)
print(confusion_matrix(y_test,t_predsvc))
print(classification_report(y_test,t_predsvc))


# **Classification with logistic regressor on the TF-IDF data**

# In[ ]:


logclassifier=LogisticRegression(random_state=0, solver='lbfgs') 
logclassifier.fit(t_train, y_train) 
t_predlog = logclassifier.predict(t_test)


# **Classification with logistic regressor on the Count Vectorizer data**

# In[ ]:


logclassifier=LogisticRegression(random_state=0, solver='lbfgs')
logclassifier.fit(c_train, y_train)
c_predlog = logclassifier.predict(c_test)


# **Calculation of accuracies of both vectorizers with Logistic Regression**

# In[ ]:


countlogacc = accuracy_score(c_predlog,y_test)
print(confusion_matrix(y_test,c_predlog))
print(classification_report(y_test,c_predlog))

countlogacc = accuracy_score(t_predlog,y_test)
print(confusion_matrix(y_test,t_predlog))
print(classification_report(y_test,t_predlog))


# **Confusion matrices for both vectorizers**

# In[ ]:


tlog_confmatrix = confusion_matrix(t_predlog,y_test)
clog_confmatrix = confusion_matrix(c_predlog,y_test)

tsvc_confmatrix = confusion_matrix(t_predsvc,y_test)
csvc_confmatrix = confusion_matrix(c_predsvc,y_test)
print(tlog_confmatrix)
print(clog_confmatrix)
print(tsvc_confmatrix)
print(csvc_confmatrix)

