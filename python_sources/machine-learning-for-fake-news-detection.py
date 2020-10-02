#!/usr/bin/env python
# coding: utf-8

# Hello there, we are going to use **Machine Learning to detect fake news**.So lets start.
# here we are using following algorithms.
# 
# 
# **1.Naivy bayes
# 2.logistic regression
# 3.Decision Tree
# 4.Random Forest
# 5.KNN
# 6.SVM(Support vector machine)**
# 
# 
# Note:-we can code one aspect in different ways. So my way of coding may be different from yours. but untimately output will be same.
# 
# Lets start with importing basic pakages

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Importing datasets

# In[ ]:



fake = pd.read_csv('../input/fake-and-real-news-dataset/Fake.csv', delimiter = ',')
true = pd.read_csv('../input/fake-and-real-news-dataset/True.csv', delimiter = ',')


# Now We are going to combine these two dataset to one dataset to simplify processing.
# also to combine we need to add an extra column as sentiment to differtiate news as 1=true_news 0=fake_news

# In[ ]:


fake['sentiment']= 0
true['sentiment']= 1

dataset =pd.DataFrame()
dataset = true.append(fake)


# Column **'Date' and 'Subject'** are important to Descriptive analysis but here for prediction they are less important so i am going to drop these columns.
# Also Created array of **'title' column as input_array** for preprocessing.

# In[ ]:


column = ['date','subject']
dataset = dataset.drop(columns=column)
input_array=np.array(dataset['title'])


# Next part is most important, That is **claning the text and forming curpus**.
# such as removing symbols, stpwords,using porterstremer etc.
# I have selected 40000 title as input. you can chose these number depending on your system performance but not less than 10000.

# In[ ]:


import re
import nltk
# ltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 40000):
    review = re.sub('[^a-zA-Z]', ' ', input_array[i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)


# Now, We crate bag of world model. if you dont know just google it.
# Also we are going to initialize  x and y that is x=independent_variable(title) y=dependent_variable(sentiment 0 or 1).
# I am going to select max features as 5000. this figure also depends on your preference.

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 5000)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[0:40000, 2].values


# Splitting the dataset into the Training set and Test set

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


# Fitting Naive Bayes to the Training set.
# Canculate accuracy form confusion matrix that is 88.58%.

# In[ ]:


# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

print(cm)


# Fitting Logistic Regression to the Training set.
# Finding accuracy by confusion matrix that is 94.92%

# In[ ]:


from sklearn.linear_model import LogisticRegression
classifier1 = LogisticRegression(random_state = 0)
classifier1.fit(X_train, y_train)

# Predicting the Test set results
y_predL = classifier1.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm1 = confusion_matrix(y_test, y_predL)

print(cm1)


# Fitting Decision Tree Classification to the Training set
# Finding accuracy by confusion matrix that is 89.70%
# 

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
classifier2 = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier2.fit(X_train, y_train)

# Predicting the Test set results
y_predD = classifier2.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm2 = confusion_matrix(y_test, y_predD)

print(cm2)


# Fitting Random Forest Classification to the Training set.
# Finding accuracy by confusion matrix that is 92.37%
# 

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
classifier3 = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier3.fit(X_train, y_train)

# Predicting the Test set results
y_predR = classifier3.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm3 = confusion_matrix(y_test, y_predR)


print(cm3)


# Fitting SVM to the Training set.
# Finding accuracy by confusion matrix that is 94.92%

# In[ ]:


from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm5 = confusion_matrix(y_test, y_pred)

print(cm5)


# Fitting K-NN to the Training set.
# here , i have given 5 as n_neighbors value for simple processing time. if  you change value accuracy also changes.
# Finding accuracy by confusion matrix that is 94.92%

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
classifier4 = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier4.fit(X_train, y_train)

# Predicting the Test set results
y_predK = classifier4.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm4 = confusion_matrix(y_test, y_predK)


# Note:- Decision Tree and random forest take time so you have to be patient while running code.
# Thank you.
