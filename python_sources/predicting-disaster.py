#
# NLP with Disaster Tweets Kaggle Competition
#
# Author: Owen Bezick
# 

# Imports 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import feature_extraction, linear_model, model_selection, preprocessing

# Data Import 
train_df = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")
test_df = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")

# Instantiate count vectorizer
count_vectorizer = feature_extraction.text.CountVectorizer()

# Creating vectors for all tweets
train_vectors = count_vectorizer.fit_transform(train_df["text"])
test_vectors = count_vectorizer.transform(test_df["text"])

# The words contained in each tweet could be a good indicator of whether they're about a real disaster or not.
# The presence of particular word (or set of words) in a tweet might link directly to whether or not that tweet is real.
# So, I will use a linear model. 

# Ridge Regression Model
clf = linear_model.RidgeClassifier()

# Scoring the model based on f1
scores = model_selection.cross_val_score(clf, train_vectors, train_df["target"], cv=3, scoring="f1")
scores

# Can improve on model with: TFIDF, LSA, LSTM / RNNs

# Test data
clf.fit(train_vectors, train_df["target"])

# Sample Submission
sample_submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")

sample_submission["target"] = clf.predict(test_vectors)

# View head of sample submission
sample_submission.head()

# Write submission to .csv
sample_submission.to_csv("submission.csv", index=False)