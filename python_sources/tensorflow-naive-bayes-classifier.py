#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import feature_extraction, linear_model, model_selection, preprocessing

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


# tensorflow hub
import tensorflow_hub as hub
# tensor flow module
import tensorflow as tf
import tensorflow_probability as tfp

# matplotlib
from matplotlib import colors
from matplotlib import pyplot as plt

# word vectorizor
# first converts the text into a matrix of word counts
# then transforms these counts by normalizing them based on the term frequency
from sklearn.feature_extraction.text import TfidfVectorizer

# used to create word encoders
from sklearn import preprocessing


# # Background
# In this competition we have a keyword(sometimes), location(sometimes), and text to represent a Tweet. Our goal is to predict whether the tweet is describing a real disaster (1) or not (0).
# 
# In this kernel I am going to focus on the text and keyword features to predict a target. First, I must convert the text into vectors to use in my model. I will use a multinominal Naive Bayes classifier to build my model given the text features. Then I will use Naive Bayes to get predictions based on keywords.

# Tensorflow code is inspired by github repo:
# https://github.com/nicolov/naive_bayes_tensorflow/blob/master/tf_iris.py
# 

# # Import Data

# read in data to pandas dataframes

# In[ ]:


train_df = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")
test_df = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")


# # Exploratory
# I just want to get a general idea of the data before we predict the targets

# count how many of each target is found in the training set 

# In[ ]:


train_df.groupby("target")["id"].nunique()


# This is amazing! **Our training set has no null targets and our targets are fairly balanced as well!**

# How many unique locations are duplicated in the tweets of the tweets that contain locations?

# In[ ]:


print(len(test_df.loc[test_df["location"].notnull(),"location"]))
print(len(test_df.loc[test_df["location"].notnull(),"location"].unique()))


# it looks like not even half match a previous tweet. We may eventually have to categorize locations using a different technique than string matching. For example, it may be useful to categorize cities verse suburban areas rather than just the city names.

# What about the key words?

# In[ ]:


print(len(test_df.loc[test_df["keyword"].notnull(),"keyword"]))
print(len(test_df.loc[test_df["keyword"].notnull(),"keyword"].unique()))


# These may actually be useful in a naive bayes model since each key word will have a conditional probability.

# # Tweet Text: Sentence Embedding

# Next we can convert each of the texts into vectors using a TensorFlow universal-sentence-encoder.

# In[ ]:


embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/3")
X_train_embeddings = embed(train_df["text"].values)
X_test_embeddings = embed(test_df["text"].values)


# # Tensor Flow Naiver Bayes

# Lets create numpy matricies for the features in the training and test set.

# In[ ]:


X_train_matrix = X_train_embeddings['outputs'].numpy()


# In[ ]:


X_test_matrix = X_test_embeddings['outputs'].numpy()


# The target is my y variable for my model.

# In[ ]:


y_train = tf.constant(train_df["target"])


# First let's create a python object that can fit and predict a naive bayes classifier

# In[ ]:


class TFNaiveBayesClassifier:
    dist = None
    
    # X is the matrix containing the vectors for each sentence
    # y is the list target values in the same order as the X matrix
    def fit(self, X, y):
        unique_y = np.unique(y) # unique target values: 0,1
        print(unique_y)
        # `points_by_class` is a numpy array the size of 
        # the number of unique targets.
        # in each item of the list is another list that contains the vector
        # of each sentence from the same target value
        points_by_class = np.asarray([np.asarray(
            [np.asarray(
                X.iloc[x,:]) for x in range(0,len(y)) if y[x] == c]) for c in unique_y])
        mean_list=[]
        var_list=[]
        for i in range(0, len(points_by_class)):
            mean_var, var_var = tf.nn.moments(tf.constant(points_by_class[i]), axes=[0])
            mean_list.append(mean_var)
            var_list.append(var_var)
        mean=tf.stack(mean_list, 0)
        var=tf.stack(var_list, 0)
        # Create a 3x2 univariate normal distribution with the 
        # known mean and variance
        self.dist = tfp.distributions.Normal(loc=mean, scale=tf.sqrt(var))
        
    def predict(self, X):
        assert self.dist is not None
        nb_classes, nb_features = map(int, self.dist.scale.shape)

        # uniform priors
        priors = np.log(np.array([1. / nb_classes] * nb_classes))
        
        # Conditional probabilities log P(x|c)
        # (nb_samples, nb_classes, nb_features)
        all_log_probs = self.dist.log_prob(
            tf.reshape(
                tf.tile(X, [1, nb_classes]), [-1, nb_classes, nb_features]))
        # (nb_samples, nb_classes)
        cond_probs = tf.reduce_sum(all_log_probs, axis=2)
        
        # posterior log probability, log P(c) + log P(x|c)
        joint_likelihood = tf.add(priors, cond_probs)

        # normalize to get (log)-probabilities
        norm_factor = tf.reduce_logsumexp(
            joint_likelihood, axis=1, keepdims=True)
        log_prob = joint_likelihood - norm_factor
        # exp to get the actual probabilities
        return tf.exp(log_prob)


# Here we initialize our naive bayes model and fit it using the training data

# In[ ]:


tf_nb = TFNaiveBayesClassifier()
tf_nb.fit(pd.DataFrame(X_train_matrix),
          y_train)


# predict probability of each target values in the test set

# In[ ]:


y_pred = tf_nb.predict(X_test_matrix)


# Create a dataframe containing the probability of each target given the text in each tweet.

# In[ ]:


predProbGivenText_df = pd.DataFrame(y_pred.numpy())
predProbGivenText_df.head()


# # Keyword Naive Bayes

# Let's look at our unique key words.

# In[ ]:


uniq_keywords = train_df["keyword"].unique()[1:]
print(len(uniq_keywords))
print(uniq_keywords)


# I noticed that there are some keywords that are very similar to each other so I decided to manually make them the same word.

# In[ ]:


def replace_keywords(df_og):
    df = df_og.copy()
    df["keyword"] = df["keyword"].replace("ablaze","blaze")
    df["keyword"] = df["keyword"].replace("blazing","blaze")
    df["keyword"] = df["keyword"].replace("annihilated","annihilation")
    df["keyword"] = df["keyword"].replace("attacked","attack")
    df["keyword"] = df["keyword"].replace("bioterror","bioterrorism")
    df["keyword"] = df["keyword"].replace("blown%20up","blew%20up")
    df["keyword"] = df["keyword"].replace("bloody","blood")
    df["keyword"] = df["keyword"].replace("bleeding","blood")
    df["keyword"] = df["keyword"].replace("body%20bags","body%20bag")
    df["keyword"] = df["keyword"].replace("body%20bagging","body%20bag")
    df["keyword"] = df["keyword"].replace("bombed","bomb")
    df["keyword"] = df["keyword"].replace("bombing","bomb")
    df["keyword"] = df["keyword"].replace("burning%20buildings","buildings%20burning")
    df["keyword"] = df["keyword"].replace("buildings%20on%20fire","buildings%20burning")
    df["keyword"] = df["keyword"].replace("burned","burning")
    df["keyword"] = df["keyword"].replace("casualties","casualty")
    df["keyword"] = df["keyword"].replace("catastrophe","catastrophic")
    df["keyword"] = df["keyword"].replace("collapse","collapsed")
    df["keyword"] = df["keyword"].replace("collide","collision")
    df["keyword"] = df["keyword"].replace("collided","collision")
    df["keyword"] = df["keyword"].replace("crash","crashed")
    df["keyword"] = df["keyword"].replace("crush","crushed")
    df["keyword"] = df["keyword"].replace("dead","death")
    df["keyword"] = df["keyword"].replace("deaths","death")
    df["keyword"] = df["keyword"].replace("deluge","deluged")
    df["keyword"] = df["keyword"].replace("demolished","demolish")
    df["keyword"] = df["keyword"].replace("demolition","demolish")
    df["keyword"] = df["keyword"].replace("derailment","derail")
    df["keyword"] = df["keyword"].replace("derailed","derail")
    df["keyword"] = df["keyword"].replace("desolation","desolate")
    df["keyword"] = df["keyword"].replace("destroyed","destroy")
    df["keyword"] = df["keyword"].replace("destruction","destroy")
    df["keyword"] = df["keyword"].replace("detonate","detonation")
    df["keyword"] = df["keyword"].replace("devastated","devastation")
    df["keyword"] = df["keyword"].replace("drowned","drown")
    df["keyword"] = df["keyword"].replace("drowning","drown")
    df["keyword"] = df["keyword"].replace("electrocute","electrocuted")
    df["keyword"] = df["keyword"].replace("evacuated","evacuate")
    df["keyword"] = df["keyword"].replace("evacuation","evacuate")
    df["keyword"] = df["keyword"].replace("explode","explosion")
    df["keyword"] = df["keyword"].replace("exploded","explosion")
    df["keyword"] = df["keyword"].replace("fatality","fatalities")
    df["keyword"] = df["keyword"].replace("floods","flood")
    df["keyword"] = df["keyword"].replace("flooding","flood")
    df["keyword"] = df["keyword"].replace("bush%20fires","forest%20fire")
    df["keyword"] = df["keyword"].replace("forest%20fires","forest%20fire")
    df["keyword"] = df["keyword"].replace("hailstorm","hail")
    df["keyword"] = df["keyword"].replace("hazardous","hazard")
    df["keyword"] = df["keyword"].replace("hijacking","hijack")
    df["keyword"] = df["keyword"].replace("hijacker","hijack")
    df["keyword"] = df["keyword"].replace("hostages","hostage")
    df["keyword"] = df["keyword"].replace("injured","injury")
    df["keyword"] = df["keyword"].replace("injures","injury")
    df["keyword"] = df["keyword"].replace("inundated","inundation")
    df["keyword"] = df["keyword"].replace("mass%20murderer","mass%20murder")
    df["keyword"] = df["keyword"].replace("obliterated","obliterate")
    df["keyword"] = df["keyword"].replace("obliteration","obliterate")
    df["keyword"] = df["keyword"].replace("panicking","panic")
    df["keyword"] = df["keyword"].replace("quarantined","quarantine")
    df["keyword"] = df["keyword"].replace("rescuers","rescue")
    df["keyword"] = df["keyword"].replace("rescued","rescue")
    df["keyword"] = df["keyword"].replace("rioting","riot")
    df["keyword"] = df["keyword"].replace("dust%20storm","sandstorm")
    df["keyword"] = df["keyword"].replace("screamed","screams")
    df["keyword"] = df["keyword"].replace("screaming","screams")
    df["keyword"] = df["keyword"].replace("sirens","siren")
    df["keyword"] = df["keyword"].replace("suicide%20bomb","suicide%20bomber")
    df["keyword"] = df["keyword"].replace("suicide%20bombing","suicide%20bomber")
    df["keyword"] = df["keyword"].replace("survived","survive")
    df["keyword"] = df["keyword"].replace("survivors","survive")
    df["keyword"] = df["keyword"].replace("terrorism","terrorist")
    df["keyword"] = df["keyword"].replace("thunderstorm","thunder")
    df["keyword"] = df["keyword"].replace("traumatised","trauma")
    df["keyword"] = df["keyword"].replace("twister","tornado")
    df["keyword"] = df["keyword"].replace("typhoon","hurricane")
    df["keyword"] = df["keyword"].replace("weapons","weapon")
    df["keyword"] = df["keyword"].replace("wild%20fires","wildfire")
    df["keyword"] = df["keyword"].replace("wounded","wounds")
    df["keyword"] = df["keyword"].replace("wrecked","wreckage")
    df["keyword"] = df["keyword"].replace("wreck","wreckage")
    return(df)


# In[ ]:


train_df = replace_keywords(train_df)
test_df = replace_keywords(test_df)


# Using the updated set of keywords let's get the probability for each target given a specific key word.

# In[ ]:


uniq_keywords = train_df["keyword"].unique()[1:]
kword_resArr = []
print(len(uniq_keywords))
for kword in uniq_keywords:
    kword_df = train_df.loc[train_df["keyword"] == kword,: ]
    total_kword = float(len(kword_df))
    target0_n = float(len(kword_df.loc[kword_df["target"]==0,:]))
    target1_n = float(len(kword_df.loc[kword_df["target"]==1,:]))
    kword_prob_df = pd.DataFrame({'keyword':[kword],
                                 "keywordPred0": [target0_n/total_kword],
                                 "keywordPred1": [target1_n/total_kword]})
    kword_resArr.append(kword_prob_df)
predProbGivenKeyWord_df= pd.concat(kword_resArr)
predProbGivenKeyWord_df.head()


# # Create submission file

# get probabilities given the tweet text

# In[ ]:


test_df["textprob0"]=predProbGivenText_df.loc[:,0].copy()
test_df["textprob1"]=predProbGivenText_df.loc[:,1].copy()
test_df.head()


# get the probabilities given the key words. Note that if there is no key word than the probability is 50% for both targets.

# In[ ]:


test_df = test_df.merge(predProbGivenKeyWord_df, how='left', on="keyword")


# In[ ]:


test_df["keywordPred0"]=test_df["keywordPred0"].fillna(0.5)
test_df["keywordPred1"]=test_df["keywordPred1"].fillna(0.5)


# Now let's calculate the probability given the text and the keyword. Then let's choose our prediction based on which target has the higher probability.

# In[ ]:


test_df["pred0"]=test_df["textprob0"]*test_df["keywordPred0"]
test_df["pred1"]=test_df["textprob1"]*test_df["keywordPred1"]
test_df["target"]=test_df["pred1"]>test_df["pred0"]
test_df["target"] = test_df["target"].astype(np.int)


# Create submission file

# In[ ]:


submission_df = test_df.loc[:,["id","target"]]
submission_df.head()


# In[ ]:


submission_df.to_csv("submission.csv",index=False)


# Note that in the code above I only look at the tweets themselves and the keywords, but many of the tweets do not have keywords. We could maybe create a model to infer keywords from tweets that do not have keywords. We also did not include location in the model.
