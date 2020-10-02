# THIS KERNEL SHOWS YOU THE BASIC PROCESS SIMPLE TF-IDF AND NAIVE BAYES
# TO PREDICT QUESTIONS OF QUORA

import numpy as np 
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# SET PANDAS DISPLAY OPTIONS
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None) 
pd.set_option('display.max_colwidth', -1)

# LOAD THE DATA
df_train = pd.read_csv("../input/train.csv")
print(df_train.head())

df_test = pd.read_csv("../input/test.csv")
print(df_test.head())

# INSPECT THE DATA
print("Training data, number of rows: %d"%len(df_train.index))
print("Testing data, number of rows: %d"%len(df_test.index))

# find out the percentage of data from each target class
print("Data from each target class in training data:")
print(df_train['target'].value_counts())

# with 1,225,312 data being in class '0' and other 80,810  being in class '1'
# we discovered an imbalance here

# INSPECT THE COLUMNS
df_train_columns = list(df_train.columns)
df_test_columns = list(df_test.columns)

print("Training data columns : %d"%len(df_train_columns))
print(df_train_columns)
print("Testing data columns : %d"%len(df_test_columns))
print(df_test_columns)

# READ TRAINING DATA AND SEPARATE INTO X AND y
X_train = df_train['question_text']
y_train = df_train['target']

X_test = df_test['question_text']

vectorizer = CountVectorizer()

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# MNB CLASSIFICATION
mnb = MultinomialNB()
mnb.fit(X_train,y_train)
mnb_prediction = mnb.predict(X_test)

mnb_results = np.array(list(zip(df_test['qid'],mnb_prediction)))
mnb_results = pd.DataFrame(mnb_results, columns=['qid', 'prediction'])
mnb_results.to_csv('submission.csv', index = False)