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
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split
from sklearn import naive_bayes
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV


# In[ ]:


df = pd.read_csv('../input/spam.csv', encoding = "ISO-8859-1")

df.head()


# In[ ]:


df.describe()


# In[ ]:


df.isnull().sum()


# In[ ]:


# Majority of the values in Unnamed: 2, Unnamed: 3 & Unnamed: 4 are null values
# Dropping the three columns and renaming the columns v1 & v2

df.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1, inplace=True)
df.rename(columns={"v1":"label", "v2":"sms"}, inplace=True)

df.head()


# In[ ]:


df.label.value_counts()


# In[ ]:


# convert label to a numerical variable
df.label = pd.Categorical(df.label).codes

df.head()


# In[ ]:


# Train the classifier if it is spam or ham based on the text
# TFIDF Vectorizer
stopset = set(stopwords.words('english'))
vectorizer = TfidfVectorizer(use_idf=True, lowercase=True, strip_accents='ascii', stop_words=stopset)


# In[ ]:


vectorizer.fit(df)


# In[ ]:


y = df.label

X = vectorizer.fit_transform(df.sms)


# In[ ]:


X


# TF(t) = (Number of times term t appears in a document) / (Total number of terms in the document)
# 
# IDF(t) = log_e(Total number of documents / Number of documents with term t in it).
# 
# tf-idf score=TF(t)*IDF(t)

# In[ ]:


## Spliting the SMS to separate the text into individual words
splt_txt1=df.sms[0].split()
print(splt_txt1)


# In[ ]:


## Finding the most frequent word appearing in the SMS
max(splt_txt1)


# In[ ]:


## Count the number of words in the first SMS
len(splt_txt1)


# In[ ]:


X[0]


# It means in the first SMS there are 20 (len(splt_txt1)) words & out of which only 14 elements have been taken, that's why we'll get only 14 tf-idf values for the first the SMS.
# 
# Likewise elements or words of all other SMSes are taken into consideration

# In[ ]:


print(X)


# In[ ]:


## Spliting the SMS to separate the text into individual words
splt_txt2 = df.sms[1].split()
print(splt_txt2)
print(max(splt_txt2))


# In[ ]:


## The most freaquent word across all the SMSes
max(vectorizer.get_feature_names())


# In[ ]:


print (y.shape)
print (X.shape)


# In[ ]:


##Split the test and train
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10, test_size = 0.2)


# In[ ]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[ ]:


## Let us try different models, and check how thye accuracy is for each of the models

clf = naive_bayes.MultinomialNB()
model = clf.fit(X_train, y_train)


# In[ ]:


prediction = dict()
prediction['Naive_Bayes'] = model.predict(X_test)
accuracy_score(y_test, prediction["Naive_Bayes"])


# In[ ]:


models = dict()
models['Naive_Bayes'] = naive_bayes.MultinomialNB()
models['SVC'] = SVC()
models['KNC'] = KNeighborsClassifier()
models['RFC'] = RandomForestClassifier()
models['Adaboost'] = AdaBoostClassifier()
models['Bagging'] = BaggingClassifier()
models['ETC'] = ExtraTreesClassifier()
models['GBC'] = GradientBoostingClassifier()


# In[ ]:


results = dict()
accuracies = dict()

for key, value in models.items():
    value.fit(X_train, y_train)
    output = value.predict(X_test)
    accuracies[key] = accuracy_score(y_test, output)


# In[ ]:


accuracies


# In[ ]:


# With the default values, Gradient Boost sems to be performing the best
# Let's fine tune and make predictions

paramGrid = dict(n_estimators=np.array([50, 100, 200,400,600,800,900]))

model = GradientBoostingClassifier(random_state=10)

grid = GridSearchCV(estimator=model, param_grid=paramGrid)

grid_result = grid.fit(X_train, y_train)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

