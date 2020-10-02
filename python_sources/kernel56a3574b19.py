#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
data = pd.read_csv('../input/yelp-reviews-dataset/yelp.csv')


# In[ ]:


data.head()


# In[ ]:


data.describe()


# In[ ]:


data['text length'] = data['text'].apply(len)


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
sns.boxplot(x='stars',y='text length',data = data)


# In[ ]:


sns.countplot(x='stars',data=data,palette = 'rainbow')


# In[ ]:


stars = data.groupby('stars').mean()
stars


# In[ ]:


sns.heatmap(stars.corr(),cmap='coolwarm',annot=True)


# In[ ]:



from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()


# In[ ]:


new_data = data[(data['stars']==1) | (data['stars']==5)]
new_data
X = new_data['text']
y = new_data['stars']
X = cv.fit_transform(X)


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.33)


# In[ ]:


#training a model
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
nb.fit(X_train,y_train)


# In[ ]:


predictions = nb.predict(X_test)


# In[ ]:


#creating a confusion matrix
from sklearn.metrics import confusion_matrix,classification_report
print(confusion_matrix(y_test,predictions))

print(classification_report(y_test,predictions))


# In[ ]:


from sklearn.feature_extraction.text import  TfidfTransformer


# In[ ]:


from sklearn.pipeline import Pipeline


# In[ ]:


#Creating pipeline
pipeline = Pipeline([
    ('bow', CountVectorizer()),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
])


# In[ ]:


X = new_data['text']
y = new_data['stars']
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3,random_state=101)


# In[ ]:


pipeline.fit(X_train,y_train)


# In[ ]:


predictions = pipeline.predict(X_test)


# In[ ]:


print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))

