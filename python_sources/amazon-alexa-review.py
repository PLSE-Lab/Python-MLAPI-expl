#!/usr/bin/env python
# coding: utf-8

# # The objective is to discover insights into consumer reviews and perform sentiment analysis on the data.
# 
# ## Decision Tree and Random Forest
# 
# Note: There a many other ways to do the analysis. This is one of the way to do the same.

# # Step 1: Importing the Data

# In[ ]:


import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


print(os.listdir("../input/amazon-alexa-reviews"))


# In[ ]:


my_alexa = pd.read_csv('../input/amazon-alexa-reviews/amazon_alexa.tsv', sep = '\t')


# In[ ]:


my_alexa.head()


# In[ ]:


my_alexa.keys()


# In[ ]:


my_alexa['verified_reviews']


# In[ ]:


my_alexa['variation'].unique()


# # Step 2: Visualizing the data

# In[ ]:


my_alexa['feedback'].unique() 
# 1 is Positive Feedback and 0 is Negative Feedback


# In[ ]:


positive_feedback = my_alexa[my_alexa['feedback'] == 1]


# In[ ]:


positive_feedback.shape # Count of positive feedback


# In[ ]:


negative_feedback = my_alexa[my_alexa['feedback'] == 0]


# In[ ]:


negative_feedback.shape # Count of negative feedback


# In[ ]:


sns.countplot(my_alexa['feedback'],label='count')


# In[ ]:


sns.countplot(my_alexa['rating'],label='count')


# In[ ]:


my_alexa['rating'].hist(bins=5)


# In[ ]:


plt.figure(figsize=(40,15))
sns.barplot(x='variation',y='rating',data=my_alexa, palette='deep')


# # Step 3: Data Cleaning / Engineering

# In[ ]:


# 'feedback' is what we are predicting and 'variation' and 'verified_reviews' are used for analysis.
# So, dropping the 'date' and 'rating' fields

my_alexa = my_alexa.drop(['date','rating'],axis=1)


# In[ ]:


my_alexa.keys() # After 'date' and 'rating' dropped


# In[ ]:


# Encoding the 'variation' to avoid the dummy trap
variation_dummy = pd.get_dummies(my_alexa['variation'], drop_first = True)


# In[ ]:


variation_dummy


# In[ ]:


my_alexa.drop(['variation'],axis=1,inplace=True)


# In[ ]:


my_alexa.keys()


# In[ ]:


# Merging the dataframes
my_alexa = pd.concat([my_alexa,variation_dummy],axis = 1)


# In[ ]:


my_alexa.keys()


# In[ ]:


# Vectorizing the 'verified_reviews' for analysis
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
alexa_cv = vectorizer.fit_transform(my_alexa['verified_reviews'])


# In[ ]:


alexa_cv.shape


# In[ ]:


print(vectorizer.get_feature_names())


# In[ ]:


print(alexa_cv.toarray())


# In[ ]:


my_alexa.drop(['verified_reviews'],axis=1,inplace = True)


# In[ ]:


my_alexa.keys()


# In[ ]:


encoded_reviews = pd.DataFrame(alexa_cv.toarray())


# In[ ]:


my_alexa = pd.concat([my_alexa,encoded_reviews],axis = 1)


# In[ ]:


my_alexa


# In[ ]:


X = my_alexa.drop(['feedback'],axis=1)


# In[ ]:


X.shape


# In[ ]:


y = my_alexa['feedback']


# In[ ]:


y.shape


# # Step 4: Model the Training

# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)
# try setting the test_size to 0.3 and 0.4


# In[ ]:


X_train.shape


# In[ ]:


X_test.shape


# In[ ]:


y_train.shape


# In[ ]:


y_test.shape


# In[ ]:


# RandomForest Classifer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
randomforest_classifier = RandomForestClassifier(n_estimators=200,criterion = 'entropy')
randomforest_classifier.fit(X_train,y_train)
# Try the n_estimators with 50,100,150,200,250,300


# # Step 5: Evaluating the Model

# In[ ]:


y_predict_train = randomforest_classifier.predict(X_train)


# In[ ]:


y_predict_train


# In[ ]:


cm = confusion_matrix(y_train,y_predict_train)


# In[ ]:


sns.heatmap(cm,annot = True)


# In[ ]:


print(classification_report(y_train,y_predict_train))


# In[ ]:


y_predict = randomforest_classifier.predict(X_test)


# In[ ]:


cm = confusion_matrix(y_test,y_predict)


# In[ ]:


sns.heatmap(cm,annot=True)


# In[ ]:


print(classification_report(y_test,y_predict))


# # Step 6: Improving the Model

# In[ ]:


my_alexa = pd.read_csv('../input/amazon-alexa-reviews/amazon_alexa.tsv', sep = '\t')


# In[ ]:


my_alexa = pd.concat([my_alexa,pd.DataFrame(alexa_cv.toarray())],axis = 1)


# In[ ]:


my_alexa.shape


# In[ ]:


# Adding the length fo the 'verified_review' as a last column in my_alexa dataframe
my_alexa['length'] = my_alexa['verified_reviews'].apply(len)


# In[ ]:


my_alexa


# In[ ]:


X = my_alexa.drop(['rating','date','variation','verified_reviews','feedback'],axis=1)


# In[ ]:


X


# In[ ]:


y = my_alexa['feedback']


# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)


# In[ ]:


from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
randomforest_classifier = RandomForestClassifier(n_estimators=300,criterion = 'entropy') # Earlier we had n_estimator as 200
randomforest_classifier.fit(X_train,y_train)

y_predict = randomforest_classifier.predict(X_test)
cm = confusion_matrix(y_test,y_predict)
sns.heatmap(cm,annot=True)


# In[ ]:


print(classification_report(y_test,y_predict))


# # End of Classification
