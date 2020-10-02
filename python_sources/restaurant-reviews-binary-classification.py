#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np 
import re
import nltk
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
import scikitplot as skplt
from sklearn.linear_model import LogisticRegression


# In[ ]:


df=pd.read_csv('../input/restaurant-reviews/Restaurant_Reviews.csv')
df.head()


# ## Cleaning Data

# ## Stop Words
# Words such as ('the' , 'is' , 'are' , 'at' , 'a' , 'an' , 'on' ) doesn't convey any meaningful information regarding reviews.
# So it's better to remove all these words from reviews.It will also reduce the size of our sparse Matrix.

# In[ ]:


from nltk.corpus import stopwords

# xyz is a list consisting of English Stopwords
xyz=stopwords.words('english') 
xyz.remove('not')


# ## Stemming
# Same words might be present in different forms in reviews.
# Consider this example:
# 1. I Loved the food.
# 2. I love the food.
# 
# Meaning of both the senteance is same.So it's better to stem the words rather than filling the sparse matrix with every form of the word. 

# In[ ]:


from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()


# ## Removing Punctuation 
# 
# We can remove punctuation with the help of re librarry

# ## Applying Stemming, Removing Stopwords and Punctuations from Reviews

# In[ ]:


corpus=[]

for i in range(len(df)):
    review=re.sub('[^a-zA-Z]',' ',df.iloc[i,0]) # Removing punctuations
    review=review.lower() # Converting to lower case.
    review=review.split() # List of words in a review.
    ps=PorterStemmer() # Stemming Words
    review=[ps.stem(word) for word in review if not word in set(xyz)] #Stemming words those are not in list xyz i.e list of stopping words
    review=' '.join(review)
    corpus.append(review)
    


# ## Original Reviews vs Transformed Ones.
# Let's See how the reviews look after cleaning process.

# In[ ]:


original=list(df.Review)
original[:10]


# In[ ]:


corpus[0:10]


# ## Creating Bag of Words
# 
# How many words should be have in our bag ? Let's see total number of different words in corpus list.

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
no_of_words=cv.fit_transform(corpus).toarray()
len(no_of_words[0])


# So we have 1566 different words in corpus list.But some of them might occur only once such as name of secific dish etc.
# So we can take 1500 words for creating our bag of words.

# In[ ]:


cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = df.iloc[:, 1].values


# ## Splitting Training and Test Set

# In[ ]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)


# ## Naive Bayes Classifier

# In[ ]:


classifier = GaussianNB()
classifier.fit(X_train, y_train)
print('Test Score : {} %'.format(classifier.score(X_test,y_test)*100))


# ## Cross Val Score of Naive Bayes

# In[ ]:


cross_val_score(classifier,X_train,y_train,cv=10).mean()*100


# In[ ]:


skplt.metrics.plot_confusion_matrix(y_test,classifier.predict(X_test),figsize=(8,8))


# ## Logistic Regression

# In[ ]:


logistic_classifier=LogisticRegression()
logistic_classifier.fit(X_train,y_train)
print('Test Score : {} %'.format(logistic_classifier.score(X_test,y_test)*100))


# ## Cross Val Score for Logistic Regression

# In[ ]:


cross_val_score(logistic_classifier,X_train,y_train,cv=10).mean()*100


# In[ ]:


skplt.metrics.plot_confusion_matrix(y_test,logistic_classifier.predict(X_test),figsize=(8,8))

