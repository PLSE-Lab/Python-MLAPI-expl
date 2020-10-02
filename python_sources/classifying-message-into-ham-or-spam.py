#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Importing Dataset

# In[ ]:


dataset = pd.read_csv("../input/spam.csv",encoding='latin-1')


# In[ ]:


##Checking the head
dataset.head()


# In[ ]:


dataset.describe()


# In[ ]:


dataset.groupby('v1').describe()


# In[ ]:


dataset['length'] = dataset['v2'].apply(len)


# # Data Visualization

# In[ ]:


dataset['length'].plot(bins=50, kind='hist') 


# In[ ]:


dataset.length.describe()


# In[ ]:


dataset[dataset['length'] == 910]['v2'].iloc[0]


# In[ ]:


dataset.hist(column='length', by='v1', bins=50,figsize=(12,4))


# **Dropping the last four columns,because that three columns don't have any values and length column doesn't have any values useful to build to a model**

# In[ ]:


dataset.drop(labels = ['Unnamed: 2','Unnamed: 3','Unnamed: 4','length'],axis = 1,inplace = True)


# In[ ]:


import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


# 1.  **Here we  are creating corpus,we are using re package to remove punctuation and special characters. **
# 
# 2. **We are converting all characters in the corpus into lower case**
# 
# 3. **Then we are splitting a message into words to remove stop words and to perform stemming**
# 
# 4.  **And then we are removing stopwords and we are performing stemming while removing stopwords,the word which is not a stop word will be converted to its root form.For example loved will be converted to love.This process is called stemming**
# 
# 5. **After this we are joining the words in the list again  to form a message without any stopwords and all words will be present in its root form**
# 
# 6. **After all this step we are appending refined message into  corpus**

# In[ ]:


portstemmer = PorterStemmer()
corpus = []
for i in range (0,len(dataset)):
    mess = re.sub('[^a-zA-Z]',repl = ' ',string = dataset['v2'][i])
    mess.lower()
    mess = mess.split()
    mess = [portstemmer.stem(word) for word in mess if word not in set(stopwords.words('english'))]
    mess = ' '.join(mess)
    corpus.append(mess)


# In[ ]:


corpus[1]


# In[ ]:


len(corpus)


# #### In this step we are vectorizing the words,We are creating sparse matrix here

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer


# In[ ]:


countvectorizer = CountVectorizer()


# In[ ]:


x = countvectorizer.fit_transform(corpus).toarray() #Independent Variable


# In[ ]:


y = dataset['v1'].values #Dependent Variable


# In[ ]:


x.shape


# #### Creating training and test set

# In[ ]:


from sklearn.model_selection import train_test_split 


# In[ ]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3)


# ## Using MultinomialNB to classify the message

# In[ ]:


from sklearn.naive_bayes import MultinomialNB


# In[ ]:


multinomialnb = MultinomialNB()


# In[ ]:


multinomialnb.fit(x_train,y_train)


# In[ ]:


y_pred = multinomialnb.predict(x_test)


# In[ ]:


y_pred


# In[ ]:


from sklearn.metrics import classification_report,confusion_matrix


# ## Classification Report

# In[ ]:


print(classification_report(y_test,y_pred))


# ## Confusion Matrix

# In[ ]:


confusion_matrix(y_test,y_pred)


# In[ ]:


from sklearn.metrics import accuracy_score


# ## Accuracy Score

# In[ ]:


accuracy_score(y_test,y_pred)


# In[ ]:




