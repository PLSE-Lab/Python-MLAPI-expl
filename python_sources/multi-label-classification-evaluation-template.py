#!/usr/bin/env python
# coding: utf-8

# This kernel serves as a template for evaluating your multi-label text classification algorithm. All you need to do is modify the classifier below.

# **Specify some classifier here!**
# 
# The classifier must fulfill the following requirements:
# 
# 1. adhere to the sklearn conventions, i.e., provide fit and predict methods.
# 2. must be able to accept and return an indicator matrix as ground truth / predictions
# 
# Furthermore, you must assign the classifier to the 'clf' variable name.

# In[ ]:


from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

# for demonstration, employ TFIDF with binary relevance logistic regression
clf = Pipeline([("vectorizer", TfidfVectorizer(max_features = 25000)), ("classifier", OneVsRestClassifier(LogisticRegression(), n_jobs = 4))])


# **Evaluate classifier**
# 
# By default, train on the EconBiz samples that also have a full-text available.
# By specifying ALL_TITLES = True, you may additionally use the samples that do not have a full-text for training as well.
# By specifying DATASET = "pubmed.csv", you may evaluate your approach on PubMed.
# 
# By default, evaluate on a single 90-10 training-test split. Perform a 10-fold crossvalidation over the samples that have a full-text by specifying SINGLE_FOLD = False.

# In[ ]:


SINGLE_FOLD = True
ALL_TITLES = False
DATASET = "econbiz.csv"

from sklearn.metrics import f1_score
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import MultiLabelBinarizer

def load_dataset(dataset_path, fold_i, all_titles = False):

    df = pd.read_csv("../input/" + dataset_path)
    if not all_titles:
        df = df[df["fold"].isin(range(0,10))]
    
    labels = df["labels"].values
    labels = [[l for l in label_string.split()] for label_string in labels]
    multilabel_binarizer = MultiLabelBinarizer(sparse_output = True)
    multilabel_binarizer.fit(labels)

    def to_indicator_matrix(some_df):
        some_df_labels = some_df["labels"].values
        some_df_labels = [[l for l in label_string.split()] for label_string in some_df_labels]
        return multilabel_binarizer.transform(some_df_labels)

    
    test_df = df[df["fold"] == fold_i]
    X_test = test_df["title"].values
    y_test = to_indicator_matrix(test_df)

    train_df = df[df["fold"] != fold_i]
    X_train = train_df["title"].values
    y_train = to_indicator_matrix(train_df)
    
    return X_train, y_train, X_test, y_test

def evaluate(dataset):
    
    scores = []
    for i in range(0, 10):
        train_df, y_train, test_df, y_test = load_dataset(dataset, i, all_titles = ALL_TITLES)
        clf.fit(train_df, y_train)
        y_pred = clf.predict(test_df)

        scores.append(f1_score(y_test, y_pred, average="samples"))

        if SINGLE_FOLD:
            break
    return np.mean(scores)

print("EconBiz average F-1 score:", evaluate(DATASET))

