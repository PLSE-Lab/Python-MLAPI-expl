#!/usr/bin/env python
# coding: utf-8

# # Identifying Political Post-Bots on Twitter
# 
# During the 2016 USA election, certain candidates were accused of marketing their party's campaigns on social media using post bots sourced from Russia, namely Twitter and Facebook.
# 
# In this kernel, we inspect a dataset consisting of 3 million politically-related tweets around the time of the 2016 election, some of which were posted by the aforementioned post bots. We use Sci-Kit Learn and Tensorflow-Keras to build a classifier capable of identifying right-wing-leaning post bots based on account data and tweet contents.
# 
# ---
# 
# 1. Environment setup
# 2. Data pre-processing
# 3. Decision tree classification using Sci-Kit Learn
# 4. NLP clssification using Keras
# 
# ---
# 
# ### 1 | Environment Setup

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import os

from sklearn.feature_extraction.text import CountVectorizer
import nltk
import re
import keras.preprocessing
from collections import Counter
import keras
import tensorflow as tf

# Read input data
df = pd.read_csv("../input/3-million-russian-troll-tweets-538/IRAhandle_tweets_1.csv")
df2 = pd.read_csv("../input/3-million-russian-troll-tweets-538/IRAhandle_tweets_2.csv")
df3 = pd.read_csv("../input/3-million-russian-troll-tweets-538/IRAhandle_tweets_3.csv")
df4 = pd.read_csv("../input/3-million-russian-troll-tweets-538/IRAhandle_tweets_4.csv")

# Build a cumulative dataframe
df = df.append(df2)
df = df.append(df3)
df = df.append(df4)


# ---
# 
# ### 2 | Data Preprocessing
# 
# First, we extract the columns we wish to leave in the framework. Next, we reformat qualitative data into quantitative forms.
# 
# In order to train the model, we want to use the following fields:
# * ```content``` - text content of the tweet
# * ```region``` - world region from which the tweet was posted
# * ```language``` - language in which the tweet was written
# * ```following``` - number people followed by the account of the tweet's poster
# * ```followers``` - number people following the account of the tweet's poster
# * ```updates``` - number of tweets created by the poster's account
# * ```retweet``` - retweet count for the tweet
# * ```account_type``` - type of account **(response variable of interest)**
# * ```post_type``` - type of post

# In[ ]:


# Select columns to leave
columns_to_leave = ['content', 'region', 'language', 'following',
                    'followers', 'updates', 'retweet',
                    'account_type', 'post_type']
columns_as_label = ['account_category']

# Drop all languages besides English
df = df.loc[df['language'] == 'English']
df.drop(columns=['language'])

df.head()


# ------
# 
# #### 2.1 | Quantify Qualitative Data
# 
# The columns will be quantified by simple categorization, with each unique textual/qualitative field mapping to a unique number.
# 
# Columns to be quantified:
# * ```account_type```
# * ```region```
# * ```content```

# In[ ]:


# Convert account_type
account_type_map = {}
val_id = 0
for item in df.account_type.unique():
    account_type_map[item] = val_id
    val_id += 1
df['account_type'] = df['account_type'].apply(lambda x: account_type_map[x])

# Convert region
region_type_map = {}
val_id = 0
for item in df.region.unique():
    region_type_map[item] = val_id
    val_id += 1
df['region'] = df['region'].apply(lambda x: region_type_map[x])

# Convert language
language_type_map = {}
val_id = 0
for item in df.language.unique():
    language_type_map[item] = val_id
    val_id += 1
df['language'] = df['language'].apply(lambda x: language_type_map[x])


# ---
# 
# #### 2.2 | Train & Test Splitting
# 
# We split the dataset into training and testing sets.

# In[ ]:


# Splitting data into training and test sets
columns_to_leave_in_overall_data = columns_to_leave + columns_as_label
train, test = train_test_split(df[columns_to_leave_in_overall_data], test_size=0.2)

# Filter out unnecessary columns
train_X = train[columns_to_leave]
train_y = train[columns_as_label]
test_X = test[columns_to_leave]
test_y = test[columns_as_label]


# ---
# 
# ### 3 | Training a Model with Sci-Kit Learn
# 
# For the classification using decision trees, we attempt to use the fields ```region```, ```following```, ```followers```, ```updates```, and ```retweet``` to predict ```account_type```.
# 
# #### 3.1 | Trying a Model
# 
# Below, we train Sci-Kit Learn's basic ```DecisionTreeClassifier```.

# In[ ]:


train_columns = ['region', 'following', 'followers', 'updates', 'retweet']

clf = DecisionTreeClassifier()
clf.fit(train_X[train_columns], train_y)

predicted_array = clf.predict(test_X[train_columns])
test_y_list = list(test_y.account_category)


# Below, we validate the accuracy results from the ```DecisionTreeClassifier``` used above.

# In[ ]:


correct_values = 0
incorrect_map = {}

# Manually score tweet accuracy (without Sci-Kit Learn automatic scoring methods)
# This gives a breakdown of the misclassified data
for (predicted, correct) in zip(predicted_array, test_y_list):
    if predicted == correct:
        correct_values += 1
    else:
        if correct in incorrect_map:
            incorrect_map[correct] += 1
        else:
            incorrect_map[correct] = 1

# Output accuracy values
print("Overall accuracy: {}%".format(np.round(100 * correct_values / len(predicted_array), decimals=2)))
print("---")
for k in incorrect_map.keys():
    print("Misclassified \"{}\": {}% | count = {}".format(k, np.round(
        100 * incorrect_map[k] / test_y.account_category.value_counts()[k], decimals=2
    ), incorrect_map[k]))


# ---
# 
# #### 3.2 | Alternative Decision Tree Models
# 
# Clearly, the ```DecisionTreeClassifier``` does not perform as well as we had hoped. Below, we try to use other models: ```RandomForestClassifier``` and ```AdaBoostClassifier```.

# In[ ]:


# Apply a one-hot-style encoding to account_category
df[['account_category']] = df.account_category.apply(lambda x: 1 if x == 'RightTroll' else 0)

# Split the dataset once again for consistency
columns_to_leave_in_overall_data = columns_to_leave + columns_as_label
train, test = train_test_split(df[columns_to_leave_in_overall_data], test_size=0.2)
train_X = train[columns_to_leave]
train_y = train[columns_as_label]
test_X = test[columns_to_leave]
test_y = test[columns_as_label]

# Function for prediction of data with various parameters, for easy testing
def pred(classifier, lab, test_X, test_y, print_output):
    predicted_array = classifier.predict(test_X[train_columns])
    test_y_list = list(test_y.account_category)
    correct_values = 0

    for (predicted, correct) in zip(predicted_array, test_y_list):
        if predicted == correct:
            correct_values += 1
    
    # Output the results
    if print_output:
        print("---" + lab + "---")
        print("Accuracy: {}%".format(
            np.round(100 * correct_values / len(predicted_array), decimals=2)))
        
# Test set validation of decision tree models
def test_df(path_num, clfc, lab, print_output):
    df_test = pd.read_csv("../input/3-million-russian-troll-tweets-538/IRAhandle_tweets_" + str(path_num) + ".csv")
    region_type_map = {}
    val_id = 0
    for item in df_test.region.unique():
        region_type_map[item] = val_id
        val_id += 1

    # Update the test data file to match the style of the training data
    df_test['region'] = df_test['region'].apply(lambda x: region_type_map[x])
    df_test[['account_category']] = df_test.account_category.apply(lambda x: 1 if x == 'RightTroll' else 0)
    train_columns = ['region', 'following', 'followers', 'updates', 'retweet']
    
    # Predict values
    pred(clfc, lab, df_test[train_columns], df_test[['account_category']], print_output)
    
# Columns for model training
train_columns = ['region', 'following', 'followers', 'updates', 'retweet']

# Fit the various models
clf = DecisionTreeClassifier()
clf.fit(train_X[train_columns], train_y)
clf2 = RandomForestClassifier()
clf2.fit(train_X[train_columns], train_y)
clf3 = AdaBoostClassifier()
clf3.fit(train_X[train_columns], train_y)

# Output tests results on a never-before-seen test file
for (cl, lab) in [(clf, "DecisionTreeClassifier-Test"),
                  (clf2, "RandomForestClassifier-Test"),
                  (clf3, "AdaBoostClassifier-Test")]:
    test_df(5, cl, lab, True)


# ---
# 
# ### 4 | Training a Model with Keras
# 
# We attempt to build a Keras model, utilizing basic NLP techniques to extract insights from the tweet text contents.
# 
# #### 4.1 | Data & Text Cleaning
# 
# First, text data was pre-processed and cleaned, so that later it could be fed into a Keras neural network.

# In[ ]:


russian_bot_tweets = df.copy()

# One-hot-style encoding of all russian trolls once mor 
russian_bot_tweets["russian_bot"] = russian_bot_tweets["account_category"]

# Clean remove null texts
russian_bot_tweets = russian_bot_tweets[pd.notnull(russian_bot_tweets["content"])]

# Clean the text of "RT"
russian_bot_tweets["text"] = russian_bot_tweets["content"].apply(lambda x: x.replace("RT", ""))

# Combine the dataframes
tweets = russian_bot_tweets[["text", "russian_bot"]]

# Clean text
tweets["text"] = list(map(lambda x: nltk.word_tokenize(re.sub("[^a-zA-Z\s]", "", re.sub(r"http.?://[^\s]+[\s]?", "", (re.sub(r"@\w+", "", x)))).lstrip().rstrip().lower()), tweets["text"]))

# Select only the 50000 most used words, or else the input layer will be way too big
vocabulary = list(dict(Counter(list([i for l in tweets["text"] for i in l])).most_common(50000)).keys())

# Converting the word list to a dictionary (makes the next step faster)
vocabulary_dict = dict(zip(vocabulary, range(len(vocabulary))))

# Add a padding keyword in to the dictionary for future use
vocabulary_dict["<PAD>"] = len(vocabulary) + 1

# Convert tweet text to numeric, dictionary-specific form
tweets["text"] = [[vocabulary_dict[s] for s in i  if s in vocabulary_dict] for i in tweets["text"]]

# Split data to train and test
tweets_train, tweets_test, russian_bot_train, russian_bot_test = train_test_split(tweets["text"], tweets["russian_bot"], test_size=0.25, random_state=100)

# Pad all tweets so they are all the same size
max_size = len(max(tweets["text"], key=lambda x: len(x)))


# #### 4.2 Building the Keras model
# 
# Below, we bulid a simple neural network Keras model with 16 nodes.

# In[ ]:


# Convert the training and test sets to usable datasets
tweets_train = keras.preprocessing.sequence.pad_sequences(tweets_train,
                                                        value=vocabulary_dict["<PAD>"],
                                                        padding="post",
                                                        maxlen=max_size)

tweets_test = keras.preprocessing.sequence.pad_sequences(tweets_test,
                                                       value=vocabulary_dict["<PAD>"],
                                                       padding="post",
                                                       maxlen=max_size)

# Build a Keras model
model = keras.Sequential()
model.add(keras.layers.Embedding(len(vocabulary_dict) + 1, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))
model.summary()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

# Split the tweets
tweets_val = tweets_train[:20000]
partial_tweets_train = tweets_train[20000:]
russian_bot_val = russian_bot_train[:20000]
partial_russian_bot_train = russian_bot_train[20000:]

# Train the model
history = model.fit(partial_tweets_train, partial_russian_bot_train, epochs=40, batch_size=1000, validation_data=(tweets_val, russian_bot_val), verbose=1)


# #### 4.3 | Keras Model Validation
# 
# Below, we obtain the accuracy statistics for our Keras model.

# In[ ]:


# Validate the model
results = model.evaluate(tweets_test, russian_bot_test)

print("Model loss: {}\nModel accuracy: {}%".format(np.round(results[0], decimals=4), np.round(100 * results[1], decimals=2)))

