#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[4]:


data = pd.read_csv("../input/spam.csv",encoding='latin-1')


# In[5]:


data.head()


# In[6]:


#Remove unknown columns
data = data.drop(["Unnamed: 2","Unnamed: 3","Unnamed: 4"],axis=1)
data.head()


# In[7]:


#Rename the columns
data = data.rename(columns={"v1":"label","v2":"text"})


# In[8]:


data.info()


# In[9]:


data.label.value_counts()


# In[10]:


#Cleaning the texts
import re
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = [] #Initialize the empty list
for i in range(0, 5572):
    text = re.sub('[^a-zA-Z]', ' ', data['text'][i]) #Use sub to keep only letters in the text
    text = text.lower() #Convert letters to lower case
    text = text.split() #Convert str to list of words
    #Stemming is about taking root of the word Eg:Loved,Loving-->Love. Stemming is done to avoid too much sparcity.
    ps = PorterStemmer()
    text = [ps.stem(word) for word in text if not word in set(stopwords.words('english'))] #Return all the words in english which are NOT present in the stopwords list
    text = ' '.join(text)#Join words with spaces. It reverses back the list of words into a string
    corpus.append(text) 


# In[12]:


data['text'].head()


# In[14]:


#A matrix with a lot of 0s is called Sparse matrix.
#Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(corpus).toarray() #toarray will convert the list of reviews to matrix of fatures
y = data.iloc[:,0].values


# In[16]:


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


# ## Multinomial Naive Bayes

# In[24]:


# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
classifier.fit(X_train, y_train)


# In[25]:


# Predicting the Test set results
y_pred = classifier.predict(X_test)


# In[26]:


# Making the Confusion Matrix
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report


# In[27]:


accuracy_score(y_test,y_pred)


# ## Logistic Regression

# In[34]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train,y_train)


# In[35]:


y_pred = model.predict(X_test)


# In[36]:


accuracy_score(y_test,y_pred)


# ## K-NN classifier

# In[40]:


from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train,y_train)


# In[41]:


y_pred = model.predict(X_test)


# In[42]:


accuracy_score(y_test,y_pred)


# In[ ]:




