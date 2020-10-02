#!/usr/bin/env python
# coding: utf-8

# # Spam Detection
# ## Author: Ventsislav Yordanov

# <img src="https://media.giphy.com/media/2j5RA3SioKdck/giphy.gif" style="height:400px"/>
# Image Source: https://media.giphy.com/media/2j5RA3SioKdck/giphy.gif

# ## Loading the needed libraries

# In[ ]:


import numpy as np
import pandas as pd

# Preprocessing
import re
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

# Feature extraction, model evaluation and hyperparemter optimization
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Classifiers
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# ## Business Problem
# Can we use this dataset to build a protection model that will accurately classify which messages are spam? This application is widely used from the email service providers like Gmail, Yahoo, and so on.
# 
# <img src="https://i.gifer.com/Ou1t.gif" style="height:400px"/>
# Image Source: https://i.gifer.com/Ou1t.gif

# ## Dataset Information
# The SMS Spam Collection is a set of SMS tagged messages that have been collected for SMS Spam research. It contains one set of SMS messages in English of 5,574 messages, tagged according being ham (legitimate) or spam.
# 
# Source: https://www.kaggle.com/uciml/sms-spam-collection-dataset/home

# ## Loading the Dataset

# In[ ]:


df = pd.read_csv("../input/spam.csv", encoding = "latin-1")
df.head()


# ## Cleaning the Dataset
# We can see that we have 5 columns with very confusing names. However, it's easy to see that the first column contains the target. The second one contains the message text. The other columns may be some additional notes. Let's explore them a little bit.

# In[ ]:


print(sum(df.iloc[:, 2].notna()))
df.iloc[:, 2].unique()


# In[ ]:


print(sum(df.iloc[:, 3].notna()))
df.iloc[:, 3].unique()


# In[ ]:


print(sum(df.iloc[:, 4].notna()))
df.iloc[:, 4].unique()


# Well, it seems that these column contains some additional comments about the messages. However, they contain only a few values and there is no documentation about them in the source. I think it's safe to remove them and try to build our machine learning model only on the message text.

# In[ ]:


df = df[["v1", "v2"]]
df.head()


# The columns are still with confusing names, let's rename.

# In[ ]:


df.columns = ["class", "message"]
df.head()


# ### Cleaning the messages
# We want to keep only the important and useful words. To achieve these we will follow the steps:
# 1. **Keep only the words** in the message
# 2. Transform all words in **lower case**. We want **"Love"** and **"love"** to mean the same thing.
# 3. Remove all **stop words**. Stop words usually refers to the most common words in a language, for example: **"the", "a", "is", etc.** We don't need these words. They don't give us any insight.
# 4. Perform **stemming**. Stemming is a process in which we get the **root of the words**. We want all the different versions of the same word to be presented in one word. They all mean the same thing. Example: **"love", "loving", "lovely".**

# In[ ]:


# Download the last available version of the stopwords
nltk.download("stopwords")


# In[ ]:


def clean_message(message):
    """
    Receives a raw message and clean it using the following steps:
    1. Remove all non-words in the message
    2. Transform the message in lower case
    3. Remove all stop words
    4. Perform stemming

    Args:
        message: the raw message
    Returns:
        a clean message using the mentioned steps above.
    """
    
    message = re.sub("[^A-Za-z]", " ", message)
    message = message.lower()
    message = message.split()
    stemmer = PorterStemmer()
    message = [stemmer.stem(word) for word in message if word not in set(stopwords.words("english"))]
    message = " ".join(message)
    return message


# In[ ]:


# Testing how our function works
message = df.message[0]
print(message)

message = clean_message(message)
print(message)


# In[ ]:


corpus = []
for i in range(0, len(df)):
    message = clean_message(df.message[i])
    corpus.append(message)


# In[ ]:


corpus[:5]


# ## Exploring the Data
# Let's see what part of the messages are **spam** and what are legitimate (**ham**).

# In[ ]:


print(round(sum(df["class"] == "ham") / len(df) * 100, 2))
print(round(sum(df["class"] == "spam") / len(df) * 100, 2))


# ## Modelling the Data

# ### Creating a Bag of Words Model
# Bag of Words model is a very popular **NLP model** used to **preprocess the texts** to classify before fitting the classification algorithms.

# In[ ]:


count_vectorizer = CountVectorizer()
features = count_vectorizer.fit_transform(corpus).toarray()
features.shape


# In[ ]:


labels = df["class"].values
labels[:5]


# ### Splitting the Data into Test and Training Sets

# In[ ]:


features_train, features_test, labels_train, labels_test = train_test_split(features, labels, 
    test_size = 0.20, stratify = labels, random_state = 42)


# In[ ]:


print(count_vectorizer.get_feature_names()[:10])
print(count_vectorizer.get_feature_names()[-10:])


# ### Fitting a Multinomial Naive Bayes Classifier.
# The multinomial Naive Bayes classifier is suitable for classification with discrete features (e.g., word counts for text classification). The multinomial distribution normally requires integer feature counts. However, in practice, fractional counts such as tf-idf may also work. That's why we're going to try this model first.

# In[ ]:


nb_classifier = MultinomialNB()


# In[ ]:


k_fold = StratifiedKFold(n_splits = 10)
scores = cross_val_score(nb_classifier, features_train, labels_train, cv = k_fold)
print("mean:" , scores.mean(), "std:", scores.std())


# In[ ]:


nb_classifier.fit(features_train, labels_train)
labels_predicted = nb_classifier.predict(features_test)
accuracy_score(labels_test, labels_predicted)


# In[ ]:


confusion_matrix(labels_test, labels_predicted, labels = ["ham", "spam"])


# ## Model Selection and Improvement

# ### Fine-Tuning Multinomial Naive Bayes

# In[ ]:


kfold = StratifiedKFold(n_splits = 10)
parameters = {"alpha": np.arange(0, 1, 0.1)}
searcher = GridSearchCV(MultinomialNB(), param_grid = parameters, cv = kfold)
searcher.fit(features_train, labels_train)
best_multinomial_nb = searcher.best_estimator_

print("Best CV params", searcher.best_params_)
print("Best CV accuracy", searcher.best_score_)
print("Test accuracy of best grid search hypers:", searcher.score(features_test, labels_test))


# ## Warning: check me later
# There is something strange here after the fine-tuning we have a little bit more bad accuracy.
# **Is this the accuracy paradox?**

# ### Scores definitions
# To choose our model we'll use the accuracy, recall, precision, and f1 score. Here are some definitions for this metrics:
# * Accuracy: Overall, how often is the classifier correct?
# * Recall: When it's actually yes, how often does it predict yes?
# * Precision: When it predicts yes, how often is it correct?
# * F1 score: can be interpreted as a weighted average of the precision and recall.

# In[ ]:


labels_predicted = best_multinomial_nb.predict(features_test)
print("Accuracy Score:", accuracy_score(labels_test, labels_predicted))
print(classification_report(labels_test, labels_predicted))


# ### Other Classifiers: Logistic Regression, Decision Tree, Random Forest

# In[ ]:


get_ipython().run_cell_magic('time', '', '\nmodels = [LogisticRegression(), DecisionTreeClassifier(), RandomForestClassifier()]\nfor model in models:\n    model.fit(features_train, labels_train)\n\n    scores = cross_val_score(model, features_train, labels_train, cv = kfold)\n    print(type(model))\n    print("Mean score:" , scores.mean(), "Std:", scores.std())\n    print()\n\n    predictions = model.predict(features_test)\n    accuracy_score(labels_test, predictions)\n\n    labels_predicted = model.predict(features_test)\n    print("Test Accuracy Score:", accuracy_score(labels_test, labels_predicted))\n    print(classification_report(labels_test, labels_predicted))')


# ## Conclusions
# Well, it is controversial which is the best model. It depends on what's important for our spam detection. Personally, I think that the precision metric for the spam class is very important, but the recall is also important. In such a case when we don't know which classifier to choose. We can use the best f1 score. If some classifiers have exactly the same f1 score, we can choose the simpler one. So, if we follow this rule, we can see that the logistic regression give us the best score.

# # Resources:
# * https://www.dataschool.io/simple-guide-to-confusion-matrix-terminology/
# * https://en.wikipedia.org/wiki/Natural_language_processing
# * https://en.wikipedia.org/wiki/Stop_words
# * https://en.wikipedia.org/wiki/Stemming
# * http://scikit-learn.org/stable/index.html
# * https://stats.stackexchange.com/questions/250273/benefits-of-stratified-vs-random-sampling-for-generating-training-data-in-classi/250742#250742
# * https://stats.stackexchange.com/questions/117643/why-use-stratified-cross-validation-why-does-this-not-damage-variance-related-b/117649#117649?newreg=2a9d984517504dcbbf55fda2f11489b7
# * https://stats.stackexchange.com/questions/49540/understanding-stratified-cross-validation

# # TODOs
# * Try different values for test_size in the "train_test_split" function
# * Try CountVectorizer with some values for the "max_features" parameter
# * Use pickle to save the trained classifiers
# 
# # Future Ideas
# * Compare more classifiers
# * Try to use TFIDF and compare the results
# * Try to use Dimensionality Reduction
# 
# # Notes
# * Reduce the number of the features, because they are too many now and this may lead to overfitting
# * Add more data to avoid high bias
