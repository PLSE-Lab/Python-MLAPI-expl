#!/usr/bin/env python
# coding: utf-8

# # Disaster in Tweets with NLP
# These days, there are lots of **fake tweets** about disaster and alike subjects. <br>
# In this competition, we will build a ML model which **predicts if a disaster announcement tweet is real or fake**. <br>
# We will use a precious ML algorithm for these prediction, which is named as **NLP (Natural Language Processing).** <br>
# 
# 1. [Dataset import and preview](#1)
# 2. [Understanding from words](#2)
# 3. [Modeling](#3)
# 4. [Prediction and submission](#4)

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# ## Dataset import and preview <a id="1"></a>

# In[ ]:


train_df = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")
test_df = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")


# * Some of the tweets might be irrelevant to disasters.
# * We will check these tweets first.

# In[ ]:


train_df[train_df["target"] == 0]["text"].values[1]


# * As we can see here, we read an example of the irrelevant tweets.
# * After this, we will get an example of relevant tweets.

# In[ ]:


train_df[train_df["target"] == 1]["text"].values[1]


# * As we can see here, this example is relevant to a forest fire,which is a disaster.

# ## Understanding from words <a id="2"></a>
# * Actually, the logic is too simple.
# * The words in our tweets are going to tell if a tweet is fake or not in substance.
# * After here, we are going to use the precious library of Python, which is named as sklearn.
# * We will get our words, and turn them into the data which our ML model can process.

# In[ ]:


from sklearn import feature_extraction, linear_model, model_selection, preprocessing


# In[ ]:


count_vectorizer = feature_extraction.text.CountVectorizer() # this is our Vectorizer model.


# In[ ]:


## In this cell, we get the counts for the first 5 tweets in our training data.
exp_train_vectors = count_vectorizer.fit_transform(train_df["text"][0:5])


# In[ ]:


## We use .todense() to turn sparse into dense.
## The reason for this is because sparse only keeps non-zero indexes to save space.
print(exp_train_vectors[0].todense().shape)
print(exp_train_vectors[0].todense())


# * That tuple tells us that there are 54 words in the first 5 words in our training data.
# * That list tells us the unique words in our first tweet.All indexes in that list are words in our first 5 tweets. The non-zero indexes are in our first tweet.
# * Since this was an example, we aren't going to use these vectors. Let's build the real one.

# In[ ]:


train_vectors = count_vectorizer.fit_transform(train_df["text"])
test_vectors = count_vectorizer.transform(test_df["text"])


# ## Modeling <a id="3"></a>
# * As you can see above, we mentioned that the words in our tweets are going to tell if a tweet is real or not.
# * Particular words in our tweet might make a sign if that tweet is real or fake.
# * We suppose that we have a linear connection here, so we are going to build a linear model.

# In[ ]:


## Since our vectors are too big, we want to push our model weights
## towards 0 without decreasing all different words.
## Ridge Regression is an effective way for this.
clf = linear_model.RidgeClassifier()


# * Now, we will test our model's performance on training data.
# * For this test, we will be using Cross Validation.
# * Cross Validation is when we train a part of our data, and validate it with the left part of our data.
# * If we do cross validation more than one times, we will have a better view on how a particular model performs.

# In[ ]:


scores = model_selection.cross_val_score(clf, train_vectors, train_df["target"], cv=3, scoring="f1")
scores


# * We select our model, we get our vectors, and we cross-validate it three times.
# * Our scores might seem terrible to you, but actually they are not.
# * This shows us that our model will score around 0.64 in the leaderboard.

# ## Prediction and submission <a id="4"></a>

# In[ ]:


clf.fit(train_vectors,train_df["target"]) ## we fit our training data and vectors


# In[ ]:


test_pred = pd.Series(clf.predict(test_vectors),name="target").astype(int)
results = pd.concat([test_df["id"],test_pred],axis=1)
results.head()


# In[ ]:


results.to_csv("submission.csv",index=False)

