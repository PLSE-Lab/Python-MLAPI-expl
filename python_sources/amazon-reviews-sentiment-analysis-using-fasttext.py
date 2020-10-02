#!/usr/bin/env python
# coding: utf-8

# # Amazon Product Reviews Sentiment Analysis using FastText
# 
# ## Description
# 
# This project aims to classify the Amazon Product reviews in three classes.
# 
# 1) Positive (If the stars are above 3). <br>
# 2) Neutral  (If the stars are equal to 3). <br>
# 3) Negative (If the stars are below 3).

# ## FastText Introduction
# 
# FastText as a library for efficient learning of word representations and sentence classification. It is written in C++ and supports multiprocessing during training. FastText allows you to train supervised and unsupervised representations of words and sentences. These representations (embeddings) can be used for numerous applications from data compression, as features into additional models, for candidate selection, or as initializers for transfer learning.

# The below script reads the data, creates the labels, does some minor text preprocessing using `multiprocessing`.

# In[ ]:


import re
import datetime
import fasttext
import numpy as np
import pandas as pd
import multiprocessing
from sklearn.model_selection import train_test_split


# In[ ]:


# Number of processes to spin up.
NUM_PROCESSES = multiprocessing.cpu_count()


# In[ ]:


DATA_PATH = "/kaggle/input/consumer-reviews-of-amazon-products/1429_1.csv"
MULTIPROC_FILES_PATH = "/kaggle/output/"


# In[ ]:


print("Started: ", datetime.datetime.now())
df = pd.read_csv(DATA_PATH)
print("Data Read: ", datetime.datetime.now(), df.shape)


# In[ ]:


def get_sentiment(star):
    """
    Function to return the sentiment from the stars given.
    :param star: Number of stars.
    :return sentiment: Sentiment based on stars.
    """

    if star == 3:
        return "neutral"
    elif star < 3:
        return "negative"
    else:
        return "positive"


# In[ ]:


def preprocess_review(review):
    """
    Function to preprocess the text review.
    :param review: Review.
    :return preprocessed_review: Preprocessed Review.
    """

    # Minor preprocessing.
    review = review.lower().strip()
    review = review.lstrip(",").rstrip(",").replace("\n", " ").replace("\t", " ")
    review = re.sub('[^.,a-zA-Z0-9 \n\.]', '', review)
    preprocced_review = review
    #TODO Add code to handle special chars by regex.

    # N - Gramming.
    # n = 4
    # review = review.replace('^"', '').replace('"$', '').replace('""', '"')
    # review = "^" + review.replace(" ", "*") + "$" # Padding for short strings.
    # preprocced_review = " ".join([review[i: i + n] for i in range(len(review) - n + 1)])

    return preprocced_review


# In[ ]:


def preprocess_data(df, f_ind):
    """
    Function to preprocess the data.
    :param df: Raw DataFrame.
    :return df: Preprocessed DataFrame.
    """

    df = df[["reviews.rating", "reviews.text"]]
    df = df.dropna()
    print("Dropped Missing Rows: ", datetime.datetime.now())
    df["sentiment"] = df["reviews.rating"].apply(lambda star: get_sentiment(star))
    print("Label Processed:", datetime.datetime.now())
    df = df.drop(["reviews.rating"], axis=1)
    print("Dropped Stars Column: ", datetime.datetime.now())
    df["reviews.text"] = df["reviews.text"].apply(lambda review: preprocess_review(review))
    print("Preprocessed Reviews (String Manipulations and N-Gramming): ", datetime.datetime.now(), df.shape)
    df.to_csv("./" + "amazon_multiproc_file_{}.csv".format(f_ind))


# In[ ]:


dfs = np.array_split(df, NUM_PROCESSES)
jobs = []
for i in range(NUM_PROCESSES):
    print("Process {} started".format(i))
    p = multiprocessing.Process(target=preprocess_data(dfs[i], i))
    jobs.append(p)
    p.start()


# In[ ]:


import os

temp_df = pd.read_csv("./" + "amazon_multiproc_file_0.csv", nrows=4)
columns = temp_df.columns
df = pd.DataFrame(columns=columns)
for file in os.listdir("./"):
    if file.endswith(".csv") and "multiproc" in file:
        print(f"Now, concatinating {file}.")
        ind_df = pd.read_csv(os.path.join("./", file))
        df = pd.concat([df, ind_df], axis=0)
df = df.loc[:, ~df.columns.str.match('Unnamed')]


# After the preprocessing is done, train-val-test files are to be created for the FastText model.
# 
# The format required by FastText is like `__label__positive    The restaurant was great.`
# 
# Notice the `__label__`. It's how FastText understands that `positive` is the label for the data `The restaurant was great.`
# 
# They could either be separated by a space or a tab.
# 
# The file extension does not matter. It could be any of the TXT/TSV/CSV or other extensions which can hold textual data.

# In[ ]:


X = df["reviews.text"]
y = df["sentiment"]
print("X and Y Split Done: ", datetime.datetime.now())


# In[ ]:


# Split into train-test sets.
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    stratify=y,
                                                    test_size=0.1)
print("Stratified Split Done. Writing Files now: ", datetime.datetime.now())


# In[ ]:


# Write the test file.
with open("./test.txt", "w") as test_file_handler:
    for X_test_entry, y_test_entry in zip(X_test, y_test):
        line_to_write = "__label__" + str(y_test_entry) + "\t" + str(X_test_entry) + "\n"
        try:
            test_file_handler.write(line_to_write)
        except:
            print(line_to_write)
            break
print("Test File Written: ", datetime.datetime.now())


# In[ ]:


# Write the train file.
with open("./train.txt", "w") as train_file_handler:
    for X_train_entry, y_train_entry in zip(X_train, y_train):
        line_to_write = "__label__" + str(y_train_entry) + "\t" + str(X_train_entry) + "\n"
        try:
            train_file_handler.write(line_to_write)
        except:
            print(line_to_write)
            break
print("Train File Written: ", datetime.datetime.now())


# Model Training starts here.
# 
# You can also get results/predictions for your test review by running the below file. 

# In[ ]:


model = fasttext.train_supervised(input="./train.txt")


# In[ ]:


def print_results(N, p, r):
    print("N\t" + str(N))
    print("P@{}\t{:.3f}".format(1, p))
    print("R@{}\t{:.3f}".format(1, r))

results = model.test("./test.txt")
print_results(*results)


# ## References and Sources
# 
# Thanks to the authors of these articles!
# 
# 1) <a href="https://towardsdatascience.com/fasttext-under-the-hood-11efc57b2b3">FastText: Under the Hood</a> (Medium Article). <br>
# 2) <a href="https://stackabuse.com/python-for-nlp-working-with-facebook-fasttext-library/">Python for NLP: Working with Facebook FastText Library</a> (StackAbuse Article). <br>
# 3) <a href="https://fasttext.cc/">FastText Official Documentation</a> 
