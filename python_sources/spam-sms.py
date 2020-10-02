#!/usr/bin/env python
# coding: utf-8

# The data contains one set of SMS messages in English of 5,574 messages, tagged according being ham or spam.
# The files contain one message per line. Each line is composed by two columns: v1 contains the label (ham or spam) and v2 contains the raw text.
# In this Naive Bayes is used to classify the class (ham,spam) of incoming sms.
# The accuracy of this model is 98%.

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


# **Importing Libraries**
# 

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


# **Creating Objects**

# In[ ]:


cv = CountVectorizer()
nb = MultinomialNB()


# **Importing Data**

# In[ ]:


spam_data = pd.read_csv("/kaggle/input/spam-text-message-classification/SPAM text message 20170820 - Data.csv")


# In[ ]:


spam_data.head()


# In[ ]:


ham = spam_data[spam_data["Category"]=="ham"]
ham.count()


# In[ ]:


spam_data.count()


# In[ ]:


spam = spam_data[spam_data["Category"]=="spam"]
spam.count()


# In[ ]:


sns.countplot(spam_data["Category"])


# In[ ]:


spamHam_count = cv.fit_transform(spam_data["Message"])
spamHam_count.toarray()


# In[ ]:


print(cv.get_feature_names())


# In[ ]:


label = spam_data["Category"].values
label


# In[ ]:


nb.fit(spamHam_count, label)


# In[ ]:


test_sample = ["hi i will call you later","you have won a reward worth rs 1lakh!!!"]
test_sample = cv.transform(test_sample)


# In[ ]:


test_predict = nb.predict(test_sample)
test_predict


# In[ ]:


X = spamHam_count
y = label


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2)


# In[ ]:


nb.fit(X_train,y_train)


# In[ ]:


y_train_predict = nb.predict(X_train)
y_train_predict


# In[ ]:


score = accuracy_score(y_train,y_train_predict)
score


# In[ ]:


cm = confusion_matrix(y_train,y_train_predict)
sns.heatmap(cm, annot = True)


# In[ ]:


y_test_predict = nb.predict(X_test)
score = accuracy_score(y_test,y_test_predict)
score


# In[ ]:


cm = confusion_matrix(y_test,y_test_predict)
sns.heatmap(cm , annot = True)


# In[ ]:


print(classification_report(y_test,y_test_predict))


# In[ ]:




