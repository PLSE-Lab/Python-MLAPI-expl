#!/usr/bin/env python
# coding: utf-8

# # Spooky Author Identification - 1

# Simple implementation of Scikit-Learn Multinomial Naive Bayes classifier to identify the "Spooky Author". No Natural Language Processing techniques (e.g. NLTK) are implemented in this example. 

# ### Importing necessary libraries

# In[ ]:


import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics


# ### Data Preprocessing

# In[ ]:


# Read train csv file into Pandas data frame
book = pd.read_csv('train.csv', encoding = 'utf-8')


# In[ ]:


# Read test csv file into Pandas data frame
test_df = pd.read_csv('test.csv', encoding = 'utf-8')


# In[ ]:


# First 5 rows of test data frame
print test_df.head()


# In[ ]:


# Shape of test data frame
print test_df.shape


# In[ ]:


# First 5 rows of train data frame
print book.head()


# In[ ]:


# Shape of the train data frame
print book.shape


# In[ ]:


# Examine the class distribution
book.author.value_counts()


# In[ ]:


# Convert author (target variables) into a numerical variable
book['author_num'] = book.author.map({'EAP':0, 'MWS':1, 'HPL':2})


# In[ ]:


print book.head()


# In[ ]:


# Define X (frature variables) and y (target variable)
X = book.text
y = book.author_num
print X.shape
print y.shape


# ### Split dataset into test and train for model evaluation

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 1)
print X_train.shape
print X_test.shape
print y_train.shape
print y_test.shape


# ### Vectorizing our Dataset

# In[ ]:


# instantiate the count vectorizer
vect = CountVectorizer()


# In[ ]:


# Learn the training data vocabulary
vect.fit(X_train)


# In[ ]:


# Use it to create Document Term Matrix
X_train_dtm = vect.transform(X_train)


# In[ ]:


# Examine the document term matrix
X_train_dtm


# In[ ]:


# Tranform testing data into document-term matrix 
X_test_dtm = vect.transform(X_test)
X_test_dtm


# ### Building and Evaluating a Model

# In[ ]:


# Instantiate Multinomial Naive Bayes
nb = MultinomialNB()


# In[ ]:


# Train the model using X_train_dtm
nb.fit(X_train_dtm, y_train)


# In[ ]:


# Make class predictions for X_test_dtm
y_pred_class = nb.predict(X_test_dtm)


# In[ ]:


# Calculate the accuracy of class predictions
metrics.accuracy_score(y_test, y_pred_class)


# ### Train final model with full train data - Preprocessing

# In[ ]:


# instantiate the count vectorizer for final model
vect_final = CountVectorizer()


# In[ ]:


# Learn entire training data vocabulary
vect_final.fit(X)


# In[ ]:


# Use vect_final to create Document Term Matrix
X_dtm = vect_final.transform(X)


# In[ ]:


# Examine the document term matrix
X_dtm


# In[ ]:


# Define X (frature variables) for test data
X_test_1 = test_df.text


# In[ ]:


# Tranform test data into document-term matrix 
X_test_dtm_1 = vect_final.transform(X_test_1)
X_test_dtm_1


# ### Final Model

# In[ ]:


# Instantiate Multinomial Naive Bayes
nb_final = MultinomialNB()


# In[ ]:


# Train the final model using X_dtm
nb_final.fit(X_dtm, y)


# In[ ]:


# Make class predictions for X_test_dtm_1
y_pred_class_final = nb_final.predict(X_test_dtm_1)


# In[ ]:


print len(y_pred_class_final)


# ### Predicted probabilities of different classes 

# In[ ]:


# storing the predicted probabilities for class 0
pred_prob_class0 = nb_final.predict_proba(X_test_dtm_1)[:,0]


# In[ ]:


# storing the predicted probabilities for class 1
pred_prob_class1 = nb_final.predict_proba(X_test_dtm_1)[:,1]


# In[ ]:


# storing the predicted probabilities for class 2
pred_prob_class2 = nb_final.predict_proba(X_test_dtm_1)[:,2]


# ### Prepare submission dataframe

# In[ ]:


test_df['EAP'] = pred_prob_class0


# In[ ]:


test_df['HPL'] = pred_prob_class2


# In[ ]:


test_df['MWS'] = pred_prob_class1


# In[ ]:


sub_df = test_df


# In[ ]:


del sub_df['text']


# In[ ]:


print sub_df.head()
print sub_df.shape


# In[ ]:


sub_df.to_csv('submission_file_1.csv', encoding = 'utf-8', index = False)

