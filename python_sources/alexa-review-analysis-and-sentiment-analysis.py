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


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


#Reading the dataset
alexa=pd.read_csv("../input/amazon_alexa.tsv",sep="\t")
alexa.head()


# In[ ]:


alexa.info()


# In[ ]:


alexa.describe()


# This shows that feedback columns is tilted towards the positive  reviews.

# In[ ]:


alexa.feedback.value_counts()


# Negative data are too less so high probability that if we go for model it may overfit. So we will try for both supervised way and unsupervised way

# > **BUT First let's analyze some data**

# In[ ]:


#Percentage of people who liked and disliked Alexa.
alexa.groupby("feedback").rating.count().plot(kind="pie",shadow=True, autopct='%1.1f%%',explode=(0.1,0.1));


# Seems Alexa is doing gud. I also should buy one

# In[ ]:


#Length of reviews given by people
alexa["length"]=alexa.verified_reviews.apply(len)
alexa.head()


# In[ ]:


plt.figure(figsize=(8,5))
alexa.length.plot(kind="box")


# In[ ]:


#Who expressed their feelings better. Either sad people or happy people?
alexa.groupby("feedback").length.mean().plot(kind="bar");
plt.title("Average word length by both happy and unhappy people");


# Seems unhappy people expressed with more words

# In[ ]:


alexa.groupby("rating").length.mean().plot(kind="bar");
plt.title("rating vs length");


# In[ ]:


#Ratings distribution
alexa.groupby("rating").feedback.count().plot(kind="pie",shadow=True,autopct='%1.1f%%',explode=(0.1,0.1,0.1,0.1,0.1))


# Ok so the people are happy about it

# Let us visualize some words which people use for expressing positive as well as negative reviews

# In[ ]:


#Positive words
good=alexa[alexa.feedback==1].verified_reviews.unique().tolist()
good=" ".join(good)
from wordcloud import WordCloud
cv=WordCloud().generate(good)
cv
plt.figure(figsize=(10,8))
plt.imshow(cv)


# In[ ]:


#Negative reviews
bad=alexa[alexa.feedback==0].verified_reviews.unique().tolist()
bad=" ".join(bad)
from wordcloud import WordCloud
cv=WordCloud().generate(bad)
cv
plt.figure(figsize=(10,8))
plt.imshow(cv)


# Hmmmmmm. Seems there are a lot similar words in them. So we need to remove these prominent common words so that it becomes easier for our model

# In[ ]:


#Comparing both wordcloud and creating a list of words that occur in both oftenly including stopwords
common=["Amazon","device","Alexa","one","Echo","work","product"]
from nltk.corpus import stopwords
stop=stopwords.words("english")
stop.extend(["Amazon","device","Alexa","one","Echo","work","product","amazon","alexa","thing","echo","dot","use"])


# There are a couple of extra non alphabetic characters and numbers present in the review section. So we need to remove those undesirable characters for creating a better model. 

# In[ ]:


#Converting to lower case
alexa.verified_reviews=alexa.verified_reviews.str.lower()


# In[ ]:


#Removing special characters ("[^a-z]">> This signifies that replace everything apart from lower case alphabets with white space)
alexa.verified_reviews=alexa.verified_reviews.str.replace("[^a-z]"," ")


# In[ ]:


#split into a list
alexa.verified_reviews=alexa.verified_reviews.str.split()


# In[ ]:


alexa.verified_reviews=alexa.verified_reviews.apply(lambda x:[word for word in x if word not in stop])


# In[ ]:


alexa.verified_reviews=alexa.verified_reviews.apply(lambda x: " ".join(word for word in x))


# Now the stopwords and common word has been removed. We can keep on adding irrelevant word to stop list and run the code again for better model creation

# Again Creating wordcloud

# In[ ]:


#Positive words
good=alexa[alexa.feedback==1].verified_reviews.unique().tolist()
good=" ".join(good)
from wordcloud import WordCloud
cv=WordCloud().generate(good)
cv
plt.figure(figsize=(10,8))
plt.imshow(cv)


# In[ ]:


#Negative reviews
bad=alexa[alexa.feedback==0].verified_reviews.unique().tolist()
bad=" ".join(bad)
from wordcloud import WordCloud
cv=WordCloud().generate(bad)
cv
plt.figure(figsize=(10,8))
plt.imshow(cv)


# Now our wordcloud look a bit better with different work, Removing irrelevant words can help in attaining great accuracy.

# **Now we can go for Sentiment Analysis using some common models like Decison tree, Random Forest and SVM and compare the accuracies**

# In[ ]:


alexa.verified_reviews=alexa.verified_reviews.str.split()


# In[ ]:


#Using wordlemmatizer to remove any plural word like "dogs" will become "dog"
from nltk.stem import WordNetLemmatizer
wll=WordNetLemmatizer()
alexa.verified_reviews=alexa.verified_reviews.apply(lambda x:[wll.lemmatize(word) for word in x])


# In[ ]:


#Using portstemmer to convert words to its base form
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
alexa.verified_reviews=alexa.verified_reviews.apply(lambda x:" ".join([ps.stem(word) for word in x]))


# In[ ]:


alexa.head(10)


# Now our data set seems to be ready for furthur process. Since the machine wont understand meaning of these words so we need to vectorize them.
# We will create multiple models. 
# 1. Supervised learning with positive and negative reviews
# 2. Unsupervised by clustering.

# 1. Supervised learning
# 

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
X=cv.fit_transform(alexa.verified_reviews)
X=X.toarray()


# In[ ]:


y=alexa.feedback.tolist()
y=np.asarray(y)


# In[ ]:


y.shape,X.shape


# Converted all words to vectors. Now we will split the data into training and test data

# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)


# In[ ]:


from sklearn.tree import DecisionTreeClassifier
dc=DecisionTreeClassifier()
dc.fit(X_train,y_train)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rc=RandomForestClassifier()
rc.fit(X_train,y_train)


# In[ ]:


y_pred_dc=dc.predict(X_test)


# In[ ]:


y_pred_rf=rc.predict(X_test)


# In[ ]:


from sklearn.metrics import accuracy_score
accuracy_score_dc=accuracy_score(y_test,y_pred_dc)


# In[ ]:


from sklearn.metrics import accuracy_score
accuracy_score_rf=accuracy_score(y_test,y_pred_rf)


# In[ ]:


from sklearn import svm
sv=svm.SVC()
sv.fit(X_train,y_train)
y_pred_sv=sv.predict(X_test)


# In[ ]:


accuracy_score_sv=accuracy_score(y_test,y_pred_sv)


# In[ ]:


print("Decision Tree Accuracy=",accuracy_score_dc)
print("Random Forest=",accuracy_score_rf)
print("SVM accuracy=",accuracy_score_sv)


# Stay tuned for Unsupervised learning and leave your opinion.

# In[ ]:




