#!/usr/bin/env python
# coding: utf-8

# **Description:**
# Dataset contains headlines and categories for 400k news items. 
# 
# Columns are:
# ID : the numeric ID of the article
# TITLE : the headline of the article
# URL : the URL of the article
# PUBLISHER : the publisher of the article
# CATEGORY : the category of the news item; one of: b : business, t : science and technology, e : entertainment, m : health
# STORY : alphanumeric ID of thet news story that the article discusses
# HOSTNAME : hostname where the article was posted
# TIMESTAMP : approximate timestamp of the article's publication, given in Unix time (seconds since midnight on Jan 1, 1970)
# 

# I have used different classifiers which predicts the category of news articles based on headline of news article. For classifiers I've considered only one feature which is headline of the news article.

# In[2]:


import pandas as pd
from sklearn.metrics import confusion_matrix
mydataset = pd.read_csv('../input/news.csv') 
X = mydataset.iloc[:,1]#taking all rows and title column from dataset
y = mydataset.iloc[:,4]#taking all rows and category column from dataset


# **Splitting the data based on training and test set**

# In[4]:


from sklearn.model_selection import train_test_split
X_train, X_test,y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 0)


# **Converting text data into numeric as algorithm expects numerical feature vector of a fixed size rather than row text documents with variable size**

# In[5]:


from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer()
vect.fit(X_train)
X_train_dtm = vect.transform(X_train)
X_train_dtm = vect.fit_transform(X_train)
X_test_dtm = vect.transform(X_test)


# **Training the model with Naive Bayes**

# In[7]:


from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
nb.fit(X_train_dtm, y_train)


# **predicting our test data with Naive Bayes**

# In[8]:


y_pred_class  = nb.predict(X_test_dtm)

from sklearn import metrics
print('Accuracy Precentage with Naive Bayes',metrics.accuracy_score(y_test,y_pred_class)*100)
print("Confusion Matrix of Naive Bayes ",metrics.confusion_matrix(y_test,y_pred_class))


# **Training the model with Decision Tree Classifier**

# In[9]:


from sklearn.tree import DecisionTreeClassifier 
DTC = DecisionTreeClassifier()
DTC.fit(X_train_dtm, y_train)
y1_pred_class = DTC.predict(X_test_dtm)
print('Accuracy Precentage with Decision tree classifier',metrics.accuracy_score(y_test,y1_pred_class)*100)
print("Confusion Matrix of Decision Tree Classifier ",metrics.confusion_matrix(y_test,y1_pred_class))


# **Using KNN Classifier**

# In[10]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(X_train_dtm, y_train)
y3_pred_class = knn.predict(X_test_dtm)
print('Accuracy Precentage with KNN',metrics.accuracy_score(y_test,y3_pred_class)*100)
print("Confusion Matrix of KNN ",metrics.confusion_matrix(y_test,y3_pred_class))

