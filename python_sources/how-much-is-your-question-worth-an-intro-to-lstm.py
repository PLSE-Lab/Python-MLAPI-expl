#!/usr/bin/env python
# coding: utf-8

# # About this kernel
# 
# *Tell me, Mr. AI... how much is my question worth?*
# 
# Have you ever thought about an incredible piece of trivia, and wondered how much that would have been worth if it was asked in a game of Jeopardy? This is what we will try to predict today by using this list of 200k+ questions from the popular game show. To do that, we will use:
# 
# 1. Load the data, and create a train and test split.
# 2. A **Simple Linear Model** with a simple bag-of-words encoding.
# 3. A **Bidirectional LSTM** model, with GlobalMaxPooling, fully-connected layers, and softmax output.

# In[ ]:


import os

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalMaxPooling1D, LSTM, Bidirectional, Embedding, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# # 1. Preprocessing
# 
# The first step is to load the data (in CSV format), and split the data.

# In[ ]:


data_df = pd.read_csv('/kaggle/input/200000-jeopardy-questions/JEOPARDY_CSV.csv')
data_df = data_df[data_df[' Value'] != 'None']

print(data_df.shape)
data_df.head()


# ## Creating bins
# 
# Since the values could easily vary, this means we would have way too many classes to classify! Instead, we will bin it in this way: if the value is smaller than 1000, then we round to the nearest hundred. Otherwise, if it's between 1000 and 10k, we round it to nearest thousand. If it's greater than 10k, then we round it to the nearest 10-thousand.

# In[ ]:


data_df['ValueNum'] = data_df[' Value'].apply(
    lambda value: int(value.replace(',', '').replace('$', ''))
)


# In[ ]:


def binning(value):
    if value < 1000:
        return np.round(value, -2)
    elif value < 10000:
        return np.round(value, -3)
    else:
        return np.round(value, -4)

data_df['ValueBins'] = data_df['ValueNum'].apply(binning)


# In[ ]:


print("Total number of categories:", data_df[' Value'].unique().shape[0])
print("Number of categories after binning:", data_df['ValueBins'].unique().shape[0])
print("\nBinned Categories:", data_df['ValueBins'].unique())


# Then, we will split our data by randomly selected 20% of the shows, and use the questions from that show as what we will try to predict.

# In[ ]:


show_numbers = data_df['Show Number'].unique()
train_shows, test_shows = train_test_split(show_numbers, test_size=0.2, random_state=2019)

train_mask = data_df['Show Number'].isin(train_shows)
test_mask = data_df['Show Number'].isin(test_shows)

train_labels = data_df.loc[train_mask, 'ValueBins']
train_questions = data_df.loc[train_mask, ' Question']
test_labels = data_df.loc[test_mask, 'ValueBins']
test_questions = data_df.loc[test_mask, ' Question']


# # 2. Simple Linear Model

# ## Transform questions to bag-of-words
# 
# Bag of words is a very simple, but very convenient way of representing any type of freeform text using vectors. [This article on medium](https://machinelearningmastery.com/gentle-introduction-bag-words-model/) goes in depth about the subject.
# 
# In our model, we will limit ourselves to only using the top 2000 most frequent words as features, in order for the logistic regression model to not overfit on too many features. Further, we are removing **stop words**, which are very common words in English that we wish to remove in order to only keep relevant information. Feel free to try different values of `max_features` and `stop_words`!

# In[ ]:


get_ipython().run_cell_magic('time', '', "bow = CountVectorizer(stop_words='english', max_features=2000)\nbow.fit(data_df[' Question'])")


# In[ ]:


X_train = bow.transform(train_questions)
X_test = bow.transform(test_questions)

y_train = train_labels
y_test = test_labels

print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_test:", y_test.shape)


# ## Train the Logistic Regression model
# 
# Logistic Regression is perhaps the simplest regression model out there.

# In[ ]:


get_ipython().run_cell_magic('time', '', "lr = LogisticRegression(solver='saga', multi_class='multinomial', max_iter=200)\nlr.fit(X_train, y_train)")


# ## Evaluate the results

# In[ ]:


y_pred = lr.predict(X_test)

print(classification_report(y_test, y_pred))


# # 2. LSTM

# ## Tokenize & Pad
# 
# We are doing 3 things here:
# 
# 1. Train a tokenizer in all the text. This tokenizer will create an dictionary mapping words to an index, aka `tokenizer.word_index`.
# 2. Convert the questions (which are strings of text) into a list of list of integers, each representing the index of a word in the `word_index`.
# 3. Pad each "list of list" into a single numpy array. To do this, we use the `pad_sequences` function, and set a maximum length (50 is reasonable since most questions will be at most 20 words), after which any word is cutoff.
# 
# Note:
# * Tokenizer will take at most 50k words. Here, we are using more words than Logistic Regression since the input dimension does not account for **all** words, but only the words that are actually given in the sequence.

# In[ ]:


tokenizer = Tokenizer(num_words=50000)
tokenizer.fit_on_texts(data_df[' Question'])

train_sequence = tokenizer.texts_to_sequences(train_questions)
test_sequence = tokenizer.texts_to_sequences(test_questions)

print("Original text:", train_questions[0])
print("Converted sequence:", train_sequence[0])


# In[ ]:


X_train = pad_sequences(train_sequence, maxlen=50)
X_test = pad_sequences(test_sequence, maxlen=50)

print(X_train.shape)
print(X_test.shape)


# ## Encode labels as counts
# 
# Unlike Sklearn, Keras requires your labels to be either one-hot-encoded, or encoded using label encoders. For the former, you will need to use a `categorical_crossentropy` loss when you compile the model, and for the latter you need to use `sparse_categorical_crossentropy`. We will use the latter for simplicity, but if you want to learn more about one-hot-encoding you can check out [this user guide](https://scikit-learn.org/stable/modules/preprocessing.html#preprocessing-categorical-features).

# In[ ]:


le = LabelEncoder()
le.fit(data_df['ValueBins'])

y_train = le.transform(train_labels)
y_test = le.transform(test_labels)

print(y_train.shape)
print(y_test.shape)


# ## Building and running the model

# In[ ]:


num_words = tokenizer.num_words
output_size = len(le.classes_)


# In[ ]:


model = Sequential([
    Embedding(input_dim=num_words, 
              output_dim=200, 
              mask_zero=True, 
              input_length=50),
    Bidirectional(LSTM(150, return_sequences=True)),
    GlobalMaxPooling1D(),
    Dense(300, activation='relu'),
    Dropout(0.5),
    Dense(output_size, activation='softmax')
    
])

model.compile('adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.summary()


# ## Train the model

# In[ ]:


model.fit(X_train, y_train, epochs=10, batch_size=1024, validation_split=0.1)


# ## Evaluate the model

# In[ ]:


y_pred = model.predict(X_test, batch_size=1024).argmax(axis=1)
print(classification_report(y_test, y_pred))

