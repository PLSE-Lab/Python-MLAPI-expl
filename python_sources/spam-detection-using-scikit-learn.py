#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# # Read csv file using panda 

# In[ ]:


data = pd.read_csv("../input/emails.csv", encoding= "latin-1")


# There are 2 column in csv dataset. One is Text data and other on is spam or not spam.  We want to see number of class in spam cloumn and number of element each one contain.

# In[ ]:


data.spam.value_counts()


# In[ ]:


from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y = train_test_split(data["text"],data["spam"], test_size=0.2, random_state=10)


# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
help(CountVectorizer)


# [stop_words](http://https://github.com/scikit-learn/scikit-learn/blob/bac89c2/sklearn/feature_extraction/text.py#L645) contains some frequent work in english language.

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer(stop_words="english")
vect.fit(train_X) # Find some word that cause most of spam email
print(vect.get_feature_names()[0:20])
print(vect.get_feature_names()[-20:])


# Extract token counts out of raw text documents using the vocabulary
#     fitted with fit or the one provided to the constructor. [Details..](http://
# http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html#sklearn.feature_extraction.text.CountVectorizer.fit_transform)

# In[ ]:


X_train_df = vect.transform(train_X)
X_test_df = vect.transform(test_X)
type(X_test_df)


# In[ ]:


from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
model = MultinomialNB(alpha=1.8)
model.fit(X_train_df,train_y)
pred = model.predict(X_test_df)
accuracy_score(test_y, pred)


# In[ ]:


print(classification_report(test_y, pred , target_names = ["Not Spam", "Spam"]))


# In[ ]:


confusion_matrix(test_y,pred)


# # Select a Non Spam email 

# In[ ]:


print(data["text"][1472])
pred = model.predict(vect.transform(data["text"]))
print("Pred : ",pred[1472])
print("Main : ",data["spam"][1472])


# # Select a spam Email

# In[ ]:


print(data["text"][10])
pred = model.predict(vect.transform(data["text"]))
print("Pred : ",pred[10])
print("Main : ",data["spam"][10])


# In[ ]:


dir(vect.transform)

