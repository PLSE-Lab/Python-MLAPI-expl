#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


dataset = pd.read_csv('../input/Restaurant_Reviews.tsv',delimiter = '\t')


# In[4]:


dataset.head()


# In[5]:


dataset.describe()


# In[6]:


dataset.info()


# ## Checking for null values

# In[7]:


dataset.isnull().sum()


# In[8]:


sns.countplot(x = dataset['Liked'],data = dataset)


# In[9]:


dataset[dataset['Liked'] == 1]["Liked"].count()


# In[10]:


dataset[dataset['Liked'] == 0]['Liked'].count()


# In[11]:


from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import re


# ## Data Preprocessing

# In[12]:


stemmer = SnowballStemmer('english')
corpus = []
for i in range(0,1000):
    review = re.sub('[^a-zA-Z]',' ',dataset['Review'][i])
    review = review.lower()
    review = review.split()
    review = [stemmer.stem(word) for word in review if word not in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)


# In[13]:


corpus[1]


# In[14]:


len(corpus)


# In[15]:


corpus[999]


# ## Creating Bag of Words Model

# In[16]:


from sklearn.feature_extraction.text import CountVectorizer


# In[17]:


cv = CountVectorizer(max_features=1500)


# In[18]:


x = cv.fit_transform(corpus).toarray()


# ### Shape of the sparse matrix

# In[19]:


x.shape


# In[20]:


y = dataset['Liked'].values


# In[21]:


from sklearn.model_selection import train_test_split


# In[22]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 17)


# ## Naive Baye's Classifier(MultinomialNB)

# In[23]:


from sklearn.naive_bayes import MultinomialNB


# In[24]:


classifier = MultinomialNB()


# ### Training the classifier

# In[25]:


classifier.fit(x_train,y_train)


# ### Making Predictions

# In[26]:


y_pred = classifier.predict(x_test)


# In[27]:


y_train_pred = classifier.predict(x_train)


# ## Evaluating the classifier

# In[28]:


from sklearn.metrics import classification_report,confusion_matrix,accuracy_score


# In[29]:


print(classification_report(y_test,y_pred))


# In[30]:


confusion_matrix(y_test,y_pred)


# In[31]:


print('Training Accuray --->',accuracy_score(y_train,y_train_pred))
print('Testing Accuray --->',accuracy_score(y_test,y_pred))


# ### Random Forest Classifier

# In[32]:


from sklearn.ensemble import RandomForestClassifier


# In[52]:


rf = RandomForestClassifier(n_estimators=800)
rf.fit(x_train,y_train)


# In[53]:


print(classification_report(y_test,y_pred))


# In[54]:


confusion_matrix(y_test,y_pred)


# In[55]:


y_train_pred = rf.predict(x_train)


# In[56]:


print('Traning Accuracy --->',accuracy_score(y_train,y_train_pred))
print('Testing Accuracy --->',accuracy_score(y_test,y_pred))


# ## Support Vector Classification

# In[35]:


from sklearn.svm import SVC


# In[36]:


svc = SVC(gamma = 'scale')


# In[37]:


svc.fit(x_train,y_train)


# In[38]:


y_pred = svc.predict(x_test)


# ## Grid Search for SVC

# In[39]:


from sklearn.model_selection import GridSearchCV


# In[40]:


parameters = [{'C':[1,10,100,1000],'kernel':['linear']},{'C':[1,10,100,1000],'kernel':['rbf'],'gamma':[1,0.5,0.1,0.01,0.001]}]


# In[41]:


gs = GridSearchCV(estimator=SVC(),param_grid=parameters,scoring='accuracy',cv = 10)


# In[42]:


gs = gs.fit(x_train,y_train)


# In[43]:


gs


# In[44]:


y_pred = gs.predict(x_test)


# In[45]:


print(classification_report(y_test,y_pred))


# In[46]:


confusion_matrix(y_test,y_pred)


# In[47]:


y_train_pred = gs.predict(x_train)


# In[48]:


print('Training Accuracy --->',accuracy_score(y_train,y_train_pred))
print('Testing Accuracy --->',accuracy_score(y_test,y_pred))


# In[ ]:




