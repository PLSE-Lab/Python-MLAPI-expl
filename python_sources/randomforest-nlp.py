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



from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# In[ ]:


dataset1 = pd.read_csv('../input/Sheet_1.csv',usecols=['response_id','class','response_text'],encoding='latin-1')
dataset2 = pd.read_csv("../input/Sheet_2.csv",encoding='latin-1')


# In[ ]:


dataset1.head(5)


# In[ ]:


dataset2.head(5)


# In[ ]:


dataset1['class'].value_counts()


# In[ ]:


from wordcloud import WordCloud, STOPWORDS
from nltk.corpus import stopwords
stop = set(stopwords.words('english'))
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
def cloud(text):
    wordcloud = WordCloud(background_color="white",stopwords=stop).generate(" ".join([i for i in text.str.upper()]))
    plt.figure(figsize=(12,8))
    plt.imshow(wordcloud)
    
    plt.axis("off")
    plt.title("Chat Bot Response")
cloud(dataset1['response_text'])


# In[ ]:


from sklearn import preprocessing
Encode = preprocessing.LabelEncoder()

from sklearn import metrics
#from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


# In[ ]:


dataset1['Label'] = Encode.fit_transform(dataset1['class'])


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
vector = CountVectorizer()
x = dataset1.response_text
y = dataset1.Label
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=1)
x_train_dtm = vector.fit_transform(x_train)
x_test_dtm = vector.transform(x_test)
rf = RandomForestClassifier(max_depth=10,max_features=10)
rf.fit(x_train_dtm,y_train)
rf_predict = rf.predict(x_test_dtm)
metrics.accuracy_score(y_test,rf_predict)


# In[ ]:


dataset2["class"].value_counts()


# In[ ]:


def wordcloud(dataframe):
    stopwords = set(STOPWORDS)
    wordcloud = WordCloud(background_color="white",stopwords=stopwords).generate(" ".join([i for i in dataframe.str.upper()]))
    plt.figure(figsize=(12,8))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.title("bag_composition")

wordcloud(dataset2['resume_text'])


# In[ ]:


dataset2['Label'] = Encode.fit_transform(dataset2['class'])
x = dataset2.resume_text
y = dataset2.Label
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=1)
x_train_dtm = vector.fit_transform(x_train)
x_test_dtm = vector.transform(x_test)


# In[ ]:


rf = RandomForestClassifier(max_depth=10,max_features=10)
rf.fit(x_train_dtm,y_train)
rf_predict = rf.predict(x_test_dtm)
metrics.accuracy_score(y_test,rf_predict)


# In[ ]:




