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

import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
stop = set(stopwords.words('english'))
from wordcloud import WordCloud, STOPWORDS
import matplotlib as plty
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import sklearn


import seaborn as sns
import matplotlib.pyplot as plt
from subprocess import check_output

get_ipython().run_line_magic('matplotlib', 'inline')
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

import plotly.graph_objs as go
from sklearn import preprocessing
Encode = preprocessing.LabelEncoder()
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
tfid = TfidfVectorizer()
vect = CountVectorizer()
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.manifold import TSNE
NB = MultinomialNB()

import nltk
from nltk.corpus import stopwords
stopwords = stopwords.words("english")
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

print(check_output(["ls", "../input"]).decode("utf8"))


# In[ ]:


chatbot = pd.read_csv("../input/Sheet_1.csv",usecols=['response_id','class','response_text'],encoding='latin-1')
resume = pd.read_csv("../input/Sheet_2.csv",encoding='latin-1')


# ****Sheet_1.csv contains 80 user responses, in the response_text column, to a therapy chatbot. Bot said: 'Describe a time when you have acted as a resource for someone else'.  User responded. If a response is 'not flagged', the user can continue talking to the bot. If it is 'flagged', the user is referred to help. ****

# In[ ]:


chatbot.head()


# In[ ]:


chatbot['class'].value_counts()


# In[ ]:


def cloud(text):
    wordcloud = WordCloud(background_color="white",stopwords=stop).generate(" ".join([i for i in text.str.upper()]))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.title("Chat Bot Response")
cloud(chatbot['response_text'])


# **Model Building **

# In[ ]:


chatbot['Label'] = Encode.fit_transform(chatbot['class'])


# In[ ]:


chatbot['Label'].value_counts()
#not_flagged    55
#flagged        25


# **Naive Bayes**

# In[ ]:


x = chatbot.response_text
y = chatbot.Label
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=1)
x_train_dtm = vect.fit_transform(x_train)
x_test_dtm = vect.transform(x_test)
NB.fit(x_train_dtm,y_train)
y_predict = NB.predict(x_test_dtm)
metrics.accuracy_score(y_test,y_predict)





# **Random Forest**

# In[ ]:


rf = RandomForestClassifier(max_depth=10,max_features=10)
rf.fit(x_train_dtm,y_train)
rf_predict = rf.predict(x_test_dtm)
metrics.accuracy_score(y_test,rf_predict)


# In[ ]:


Chatbot_Text = chatbot["response_text"]
len(Chatbot_Text)


# In[ ]:


Tf_idf = CountVectorizer(max_features=256).fit_transform(Chatbot_Text.values)


# In[ ]:


Tf_idf


# In[ ]:


tsne = TSNE(
    n_components=3,
    init='random', # pca
    random_state=101,
    method='barnes_hut',
    n_iter=250,
    verbose=2,
    angle=0.5
).fit_transform(Tf_idf.toarray())



# As we can that,Random forest shows the higher successful rate of 80% as compared to Navie bayes, which is 70%.
# Though the minimum successful rate for such small data should be 95% then and then only it can be accepted to work better in real senario.
# I think fine tuning of certain parameters will help to imporve the success rate.
# Ludwig library by Uber can be the best option considered for improving the success rate.

# **Resume**

# In[ ]:


resume.head()


# In[ ]:


resume['class'].value_counts()


# In[ ]:


def cloud(text):
    wordcloud = WordCloud(background_color="white",stopwords=stop).generate(" ".join([i for i in text.str.upper()]))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.title("Resume Bot Response")
cloud(resume['resume_text'])





# In[ ]:


resume['Label'] = Encode.fit_transform(resume['class'])


# In[ ]:


xr = resume.resume_text
yr= resume.Label
xr_train,xr_test,yr_train,yr_test = train_test_split(xr,yr,random_state=1)
xr_train_dtm = vect.fit_transform(xr_train)
xr_test_dtm = vect.transform(xr_test)
NB.fit(xr_train_dtm,yr_train)
yr_predict = NB.predict(xr_test_dtm)
metrics.accuracy_score(yr_test,yr_predict)





# In[ ]:


rf = RandomForestClassifier(max_depth=10,max_features=10)
rf.fit(xr_train_dtm,yr_train)
rf_predict = rf.predict(xr_test_dtm)
metrics.accuracy_score(yr_test,rf_predict)


# In[ ]:


resume_Text = resume["resume_text"]
len(resume_Text)


# For resume data,it can be seen that the Navie Bayes shows higher success rate as compare to Random forest.
