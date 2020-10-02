#!/usr/bin/env python
# coding: utf-8

# ### Text Classification: Naive Bayes Algorithm

# In[ ]:


import os
import pandas as pd
import numpy as np
from subprocess import check_output
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


# In[ ]:


train = pd.read_csv("../input/train.csv")
test=pd.read_csv("../input/test.csv")


# train data: Check the top 5 entries. How the data looks like

# In[ ]:


train.head()


# train data: Check the bottom 5 entries. How the data looks like

# In[ ]:


train.tail()


# Test data: Check the top 5 entries. How the data looks like

# In[ ]:


test.head()


# test data: Check the bottom 5 entries. How the data looks like

# In[ ]:


test.head()


# Check the train data shape

# In[ ]:


print('Train data shape: ',train.shape)


# Check the test data shape

# In[ ]:


print('Test data shape: ',test.shape)


# Check is there any blank values in train data

# In[ ]:


train.isnull().sum()


# Check the information about the train data

# In[ ]:


print(train.info())


# Till now we understood the shape and size of the train and test data i.e. dimensions

# Now by just looking at the top 4 entries, it must be clear that does id have any role in determining who the author is? 
# No. Id is just used as a identifier for text NOT for author. So one thing is clear over here, that id is useless and 
# would not help in any way to our model to learn.
# 
# OR
# 
# Check out the unique count of 'id' i.e. 19579 also the length of dataframe is 19579. It clears that it is not useful for 
# classification purpose so remove it.

# In[ ]:


print("Length of unique id's in train: ",len(np.unique(train['id'])))
print("Length of train dataframe is: ",len(train))
id = test['id'].copy()
train = train.drop('id', axis = 1)


# In[ ]:


train['author'] = train.author.map({'EAP':0,'HPL':1,'MWS':2})
train.head()


# Now lets split the 'train' data into x & y

# In[ ]:


x = train['text'].copy()
y = train['author'].copy()

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3, random_state=42)

print(x.head())
print(y.head())


# Now, till here we are ready with nthe test & train data. The real real fun game begins here. Let's 'Vectorize' the data rows

# In[ ]:


# Example
# In short it returns the count of each word in row under consideration.

text = ["My name is Bhagwat Chate Bhagwat Chate"]
toy  = CountVectorizer(lowercase=False, token_pattern = r'\w+|\,')
toy.fit_transform(text)

print (toy.vocabulary_)
matrix = toy.transform(text)

print(matrix[0,0])
print(matrix[0,1])
print(matrix[0,2])
print(matrix[0,3])
print(matrix[0,4])


# In[ ]:


vect = CountVectorizer(lowercase=False, token_pattern=r'\w+|\,')

x_v = vect.fit_transform(x)
x_train_v = vect.transform(x_train)
x_test_v = vect.transform(x_test)

print (x_train_v.shape)


# Let's fit the model

# In[ ]:


model = MultinomialNB()
model.fit(x_train_v, y_train)


# The Accuracy of the model is

# In[ ]:


print('Naive Bayes accuracy: ',round(model.score(x_test_v, y_test)*100,2),'%')


# For training purpose what we did is we split the train dataset into 'training' & 'testing'. We have seperate dataset for \
# 'testing' lets work with the 'test' dataset ptovided in this question.

# In[ ]:


x_test=vect.transform(test["text"])


# Now we have successfully vectorized the data given by kaggle Now we fit the whole training data without any split \
# into our Naive Bayes Model Next we give it the testing vectorized data to predict the probabilities

# In[ ]:


model = MultinomialNB()
model.fit(x_v, y)

predicted_result = model.predict_proba(x_test)

predicted_result.shape


# We see that we got a result with 8392 rows presenting each text and 3 columns each column representing probability of\
# each author.

# In[ ]:


result=pd.DataFrame()

result["id"]  = test['id']
result["EAP"] = predicted_result[:,0]
result["HPL"] = predicted_result[:,1]
result["MWS"] = predicted_result[:,2]

result.head()


# Let's save the result into Excel file name 'Result.csv'

# In[ ]:


result.to_csv("Result.csv", index=False)

