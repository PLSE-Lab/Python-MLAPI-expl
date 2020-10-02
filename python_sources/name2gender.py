#!/usr/bin/env python
# coding: utf-8

# # <font color=blue|red|green|pink|yellow>GURU</font>
# [www.aiguru.az](http://aiguru.az)
# ## Determine Gender By Name
# ### Author   : Ramin Hashimzade
# ### Location : Azerbaijan, Baku

# ### Datasset Description
# Data Generated for test purpose, for confidential reason I can not share real Data. For training you have to have much more data.
# 
# *   FIRST_NAME
# *   LAST_NAME
# *   SEX (M/F)

# ### Import Library

# In[ ]:


import pandas as pd
# load data
#dataset = pd.read_csv("../input/gender_testdata.csv")
dataset = pd.read_excel("../input/gender/gender.xlsx")


# ### Concatenate First Name & Last Name

# In[ ]:


dataset["NAME"] = dataset["FIRST_NAME"] + " " + dataset["LAST_NAME"]
dataset.groupby('SEX')['FIRST_NAME'].count()


# In[ ]:


dataset.head()


# Convert a collection of text documents to a matrix of token counts
# 
# This implementation produces a sparse representation of the counts using scipy.sparse.csr_matrix.
# 
# If you do not provide an a-priori dictionary and you do not use an analyzer that does some kind of feature selection then the number of features will be equal to the vocabulary size found by analyzing the data.

# In[ ]:


# Dropping last 20K rows as we have limitation in memory in this kernel. In real case this line code have to be commented
dataset.drop(dataset.tail(20000).index,inplace=True)


# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(dataset["NAME"].values.astype('U')).toarray()
y = dataset.iloc[:, 2].values


# Splitting  To Train/Test Data

# In[ ]:


# split to train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 1)


# In[ ]:


print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)


# # **Training Model - Naive Bayes**

# In[ ]:


from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)


# In[ ]:


y_pred= classifier.predict(X_test)


# In[ ]:


########## Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
print('Accurancy: {:.0f}%'.format(classifier.score(X_test, y_test)*100))

