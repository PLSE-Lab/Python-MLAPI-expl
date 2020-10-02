#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


# importing necessary libraries
import numpy as np
import pandas as pd
import nltk 
import matplotlib.pyplot as plt


# In[ ]:


datatrain = pd.read_csv(os.path.join(dirname, filenames[2]))
datatest = pd.read_csv(os.path.join(dirname, filenames[1]))


# In[ ]:


datatrain.head()


# In[ ]:


datatest.head()


# In[ ]:


def basic_info(data):
    print(data.shape)
    print(data.columns)
    print(data.info())


# In[ ]:


basic_info(datatest)


# In[ ]:


basic_info(datatrain)


# In[ ]:


from nltk.stem import PorterStemmer
ps = PorterStemmer()


# In[ ]:


import string
import re
remove_hash = re.compile("#")
stopwords = nltk.corpus.stopwords.words('english')
def preprocessing_and_cleaining(x):
    x = x.lower()
    #print(x)
    x = remove_hash.sub("", x)
    x = x.split(" ")
    x = [i for i in x if i not in stopwords]
    #print(x)
    x = [i for i in x if i not in string.punctuation]
    #print(x)
    x = " ".join([ps.stem(i) for i in x])
    #print(x)
    return x


# In[ ]:


#cleaning the train and test tweets
datatrain['Cleaned_tweet'] = datatrain['text'].apply(lambda x: preprocessing_and_cleaining(x))


# In[ ]:


datatest['Cleaned_tweet'] = datatest['text'].apply(lambda x: preprocessing_and_cleaining(x))


# In[ ]:


datatrain.head()


# In[ ]:


datatest.head()


# ### Let's use TF-IDF for vectorization and make first out Logistic Regression

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# In[ ]:


tfidf = TfidfVectorizer()
tfidf.fit(datatrain['Cleaned_tweet'])


# In[ ]:


X = tfidf.transform(datatrain['Cleaned_tweet'])
X_test = tfidf.transform(datatest['Cleaned_tweet'])


# In[ ]:


X_train, X_val, y_train, y_val = train_test_split(X, datatrain['target'], train_size = 0.75, random_state = 0)


# In[ ]:


for c in [0.01, 0.05, 0.25, 0.5, 0.75, 1]:
    lr = LogisticRegression(C=c)
    lr.fit(X_train, y_train)
    print("Accuracy for C = %s: %s" % (c, accuracy_score(y_val, lr.predict(X_val))))


# taking C = 1

# In[ ]:


final_tfidf = LogisticRegression(C = 1, max_iter=5000)
final_tfidf.fit(X, datatrain['target'])
#print("Final Accuracy: %s" %accuracy_score(datatrain['target'], final_tfidf.predict(X_test)))
prediction_df = pd.DataFrame({"id": datatest['id'],"target": final_tfidf.predict(X_test)})


# In[ ]:


prediction_df


# ### Let's make a random forest classifier

# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


rfc = RandomForestClassifier(random_state=0)


# In[ ]:


rfc.fit(X_train, y_train)


# In[ ]:


accuracy_score(y_val, lr.predict(X_val))


# In[ ]:


predictionrfr_df = pd.DataFrame({"id": datatest['id'],"target": rfc.predict(X_test)})


# In[ ]:


predictionrfr_df


# # From both Logistic and Random Forest, I got the same score. So I will be uploading both of them

# In[ ]:


prediction_df.to_csv("submission.csv", index = False)


# In[ ]:


predictionrfr_df.to_csv("submission2.csv", index = False)


# In[ ]:




