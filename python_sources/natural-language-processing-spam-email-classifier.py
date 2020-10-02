#!/usr/bin/env python
# coding: utf-8

# <a href="https://www.facebook.com/codemakerz"><img src="https://scontent.ffjr1-4.fna.fbcdn.net/v/t1.0-9/36189148_736466693143793_2172101683281133568_n.png?_nc_cat=107&_nc_eui2=AeHzxv3SUcQBOfijLP-cEnHkX4z9XQXdeau__2MlErWZ1x07aZ1zx1PzJUDDxL6cpr7oPqYiifggXDptgtP8W5iCoDRjcdILDBYZ5Ig40dqi8Q&_nc_oc=AQmMCNXdzelFB2rdtpk8wN8nC410Wm2yKupYfYS1FxHNejTF0Jhr1G3WIZORKRF3TvFpohMB8Puw29Txxan8CW05&_nc_ht=scontent.ffjr1-4.fna&oh=7b13627e991a4d1b508923041bd7bc22&oe=5D8A7B03" />
# </a>
# Follow Us:
# Facebook: https://www.facebook.com/codemakerz
# 
# <h1>Natural Language Processing - Email Ham & Spam</h1>
# <h3>As a learner i am also looking for new things, Help us with your suggestion and ideas. </h3>

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from nltk import FreqDist
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # Load Data

# In[ ]:


df_mails = pd.read_csv('/kaggle/input/spam-and-ham/spam.csv',encoding= 'latin-1')


# In[ ]:


df_mails.head() # HAM - GOOD EMAILS, SPAM - BAD EMAILS


# # Missing Values

# In[ ]:


df_mails.isnull().sum()  #There are no missing value except unnamed column, we dont need those cols.


# # EDA

# In[ ]:


df_mails.v1.value_counts()  # We can see there are total 747 spam and 4825 ham


# In[ ]:


df_mails.v1.value_counts().plot(kind="bar");


# In[ ]:


# Find most popular spam token
df_spam = df_mails[df_mails.v1 == 'spam']
df_spam.head()


# In[ ]:


# Import Spacy for tokenization
import spacy
nlp = spacy.load("en_core_web_sm")


# In[ ]:


".".isalpha()


# In[ ]:


spam_token=[]
famous_keyword = []
for spam in np.array(df_spam.v2):
    doc = nlp(spam.lower())
    for token in doc:
        # add famous keywords
        if token.pos_ == "NOUN" or token.pos_ == 'PRON' or token.pos_ == 'PROPN':
            if not token.text in famous_keyword and not token.is_stop and token.text.isalpha():
                famous_keyword.append(token.text)
        # add all spam tokens                              
        if not token.is_stop and not token.text.isdigit() and token.text.isalpha():
            spam_token.append(token.text)


# In[ ]:


# So these are keywords which u will get usually in spam messages
famous_keyword[0:10]


# In[ ]:


spam_token[0:10] # so these are most common unique spam keywords


# In[ ]:


# Frequency Distribution
freq_spam = FreqDist(spam_token)
freq_spam


# In[ ]:


# so we can see that mostly spam keywrods are entry, free, prize, claim which make sense. In regular life when
# you recieve any spam message they include these keywords.
plt.figure(figsize=(15,10))
freq_spam.plot(50)


# # Split Train & Test

# In[ ]:


from sklearn.model_selection import train_test_split
import re # for regula rexpression


# In[ ]:


# before splitting data we will try to reduce the dimensionality of tfid matrix by filtering stop words and 
# lemmatization.

corpus = []
for i in range(df_mails.shape[0]):
    msg = re.sub('[^a-zA-Z]', ' ', df_mails.v2[i] ) # we will remove all the special characters
    msg = msg.lower() # change it to lower case
    doc = nlp(msg) # create spacy document
    # remove stop words and perform. lemmatization
    tokens_no_stop = [token.lemma_ for token in doc if not token.is_stop and not token.text.isspace()]
    msg = ' '.join(tokens_no_stop) # join all the tokens to make sentence.
    corpus.append(msg) # append to corpus list


# In[ ]:


# Above whole step is to reduce the dimensionality nd to provide only valid text to you model.
corpus[0:10]


# In[ ]:


X = corpus # Email
y = df_mails["v1"] # Result Ham or Spam


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state=34)


# # Term Frequency Inverse Document

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[ ]:


vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)


# # Train Classifier

# In[ ]:


from sklearn.svm import LinearSVC
clf = LinearSVC()
clf.fit(X_train_tfidf,y_train)


# # Pipeline
We applied vecotirzer for train dataset, same we have to apply for test dataset also, In our case there are less step. But there may be case where you have to apply tokenization, lemmatization, stemming, stop words. In this case you have apply all the step again which will take an extra effort. So better way to do it is defining a pipeline. So next time when we need to repeat those step, we can simply call that pipeline.
# In[ ]:


from sklearn.pipeline import Pipeline


# In[ ]:


text_clf = Pipeline([('tfidf', TfidfVectorizer()),
                     ('clf', LinearSVC()),
])

# Feed the training data through the pipeline
text_clf.fit(X_train, y_train)  


# # Predict

# In[ ]:


predictions = text_clf.predict(X_test)


# In[ ]:


predictions # See prediction is in text form whichis very good. IN other machine learning algorithm we
# need to apply labelencoder.


# # Confusion Matrix

# In[ ]:


from sklearn import metrics
cm = metrics.confusion_matrix(predictions, y_test)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True)


# # Classification Report

# In[ ]:


print(metrics.classification_report(y_test,predictions))


# In[ ]:


# So we can see we are getting a good prediction and recall for both the cases(ham & spam)


# # Accuracy

# In[ ]:


print(metrics.accuracy_score(y_test, predictions)) # we got an accuray of 98% which is really amazing.


# # Test New Email

# In[ ]:


text_clf.predict(["Weekly Lottery Participation. Win upto $10,000."])


# In[ ]:


text_clf.predict(["Hello Sir. How are you?"])


# In[ ]:


# So it is working as we expected. Try some more messages. !!! 
# Thank you... !! UPVOTE if you like the code. 

