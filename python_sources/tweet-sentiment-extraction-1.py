#!/usr/bin/env python
# coding: utf-8

# This notebook is for tweet sentiment analysis- we are given the text of the tweet, as well as the sentiment, and are asked to generate the part of the tweet that embodies that sentiment.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import tensorflow as tf
import matplotlib.pyplot as plt


# In[ ]:


data = pd.read_csv("../input/tweet-sentiment-extraction/train.csv")


# In[ ]:


data.head()


# In[ ]:


print(len(data.text), len(data.textID))


# ## Clean text

# In[ ]:


for idx, value in enumerate(data.text): ## remove hyperlinks
    words = str(value).split()
    words = [x for x in words if not x.startswith("http")]
    data["text"][idx] = " ".join(words)


# In[ ]:


import string

def clean_text(dataset, field):
    for index, strin in enumerate(dataset[field]):
        if not strin:
            strin = strin.lower()
            strin = strin.replace("'", "")
            strin = strin.replace("\n", "")
            strin = strin.strip()
            strin = strin.replace('[{}]'.format(string.punctuation), '')
            dataset[field][index] = strin


clean_text(data, 'text')
clean_text(data, 'selected_text')


# In[ ]:


print(len(data.text), len(data.textID))


# In[ ]:


data = data[pd.notnull(data.selected_text)]


# In[ ]:


print(data.text[data.textID == "a88287bbda"])
print(len(data.text), len(data.textID))


# ## Separate into training and validation sets

# In[ ]:


from sklearn.model_selection import train_test_split

train, validation = train_test_split(data, test_size = 0.25)
print(len(data), len(train), len(validation))


# In[ ]:


train.head()


# In[ ]:


validation.head()


# In[ ]:


print(train.text[train.textID == "a88287bbda"])


# ## Tokenize and create vocabulary

# In[ ]:


vocab_size = 10000
embedding_dim = 16
max_length = 50
trunc_type='post'
pad_type='post'
oov_tok = "<OOV>"

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(train.text)
word_index = tokenizer.word_index


# In[ ]:


training_sequences = tokenizer.texts_to_sequences(np.array(train.text))
training_padded = pad_sequences(training_sequences,truncating=trunc_type, padding=pad_type)

max_length = len(training_padded[0])

validation_sequences = tokenizer.texts_to_sequences(np.array(validation.text))
validation_padded = pad_sequences(validation_sequences, padding=pad_type, maxlen = max_length)


# ## Prepare data

# The padded sequences are going to be the inputs. The output will be an array of the same length, but with ones at the indexes of the words that we keep (that embody the sentiment), and zeros for the rest of the words.

# In[ ]:


training_selected_sequences = tokenizer.texts_to_sequences(np.array(train.selected_text))
validation_selected_sequences = tokenizer.texts_to_sequences(np.array(validation.selected_text))


# In[ ]:


def get_list(padded, sequence):
    return np.array([1 if x in sequence else 0 for x in padded])


# So for this tweet:

# In[ ]:


training_padded[4]


# These words are important:

# In[ ]:


training_selected_sequences[4]


# So the output array will look like this:

# In[ ]:


get_list(training_padded[4], training_selected_sequences[4])


# In[ ]:


train_y = np.array([get_list(i,j) for i,j in zip(training_padded, training_selected_sequences)])
validate_y = np.array([get_list(i,j) for i,j in zip(validation_padded, validation_selected_sequences)])


# In[ ]:


train_y


# In[ ]:


np.array(train.sentiment).shape


# In[ ]:


training_padded.shape


# To get the phrases back from the predicted arrays:

# In[ ]:


rev_word_index = {v: k for k, v in word_index.items()}
def get_phrase(array_x, array_y, index): 
    return np.array([rev_word_index[i] for i in array_x[index][array_y.astype(bool)[index]]])


# In[ ]:


print(train.text.values[4])
print(str(get_phrase(training_padded, train_y, 4)))


# In[ ]:


train_x = np.copy(training_padded)
validate_x = np.copy(validation_padded)


# In[ ]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, Dropout


# In[ ]:


plt.style.use('dark_background')

def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.plot(history.history['val_'+string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  plt.show()
  
# plot_graphs(history, "accuracy")
# plot_graphs(history, "loss")


# I was having some trouble figuring out how to incorporate the sentiment and the input text into the model, but I noticed that for neutral tweets, in almost every case the selected text is just the whole tweet. So, I decided to just go with that, and make two separate models, one for negative and one for positive tweets.

# In[ ]:


#!pip install -U keras-tuner


# In[ ]:


training_padded = np.array(training_padded)
validation_padded = np.array(validation_padded)
train_y = np.array(train_y)
validate_y = np.array(validate_y)


# In[ ]:


training_padded


# In[ ]:


print(np.array(training_padded)[(train.sentiment == "positive")].shape)
print(training_padded[(train.sentiment == "neutral")].shape)
training_padded[(train.sentiment == "negative")].shape


# In[ ]:


train_positive_x = training_padded[(train.sentiment == "positive")]
train_neutral_x = training_padded[(train.sentiment == "neutral")]
train_negative_x = training_padded[(train.sentiment == "negative")]
train_positive_y = train_y[(train.sentiment == "positive")]
train_neutral_y = train_y[(train.sentiment == "neutral")]
train_negative_y = train_y[(train.sentiment == "negative")]

validate_positive_x = validation_padded[(validation.sentiment == "positive")]
validate_neutral_x = validation_padded[(validation.sentiment == "neutral")]
validate_negative_x = validation_padded[(validation.sentiment == "negative")]
validate_positive_y = validate_y[(validation.sentiment == "positive")]
validate_neutral_y = validate_y[(validation.sentiment == "neutral")]
validate_negative_y = validate_y[(validation.sentiment == "negative")]


# ## Tune and Train models

# I read at this link: (https://medium.com/huggingface/multi-label-text-classification-using-bert-the-mighty-transformer-69714fa3fb3d) that this is the correct loss function to use for multilabel classification, which I think is what I'm doing here.

# In[ ]:


loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)


# In[ ]:


get_ipython().system('pip install -U keras-tuner')


# In[ ]:


import kerastuner
def build_model(hp):
    model = Sequential()
    model.add(Embedding(vocab_size, hp.Int('units', min_value = 5, max_value = 200, step = 25), input_length=max_length))
    model.add(Dropout(0.5))
    model.add(Bidirectional(LSTM(20)))
    model.add(Dropout(0.5))
    model.add(Dense(max_length, activation='softmax'))
    model.compile(loss=loss, optimizer=tf.keras.optimizers.Adam(hp.Choice('learning_rate',
                      values=[1e-2, 1e-3, 1e-4])), metrics=['accuracy'])
    return model

tuner = kerastuner.tuners.RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=5,
    executions_per_trial=3)

    


# In[ ]:


tuner.search(train_positive_x, train_positive_y, epochs = 40,verbose = 2,validation_data = (validate_positive_x, validate_positive_y), callbacks=[tf.keras.callbacks.EarlyStopping('val_loss', patience=6)])


# In[ ]:


positive_model = tuner.get_best_models()[0]
positive_model.compile(loss=l, optimizer=tf.keras.optimizers.Adam(0.001), metrics=['accuracy'])
positive_history = positive_model.fit(np.array(train_positive_x), np.array(train_positive_y), epochs=60, verbose=2,
                    validation_data = (np.array(validate_positive_x), np.array(validate_positive_y)),callbacks=[tf.keras.callbacks.EarlyStopping('val_loss', patience=6)] )


# In[ ]:





# In[ ]:


# positive_model = Sequential()
# positive_model.add(Embedding(vocab_size, 16, input_length=max_length))
# positive_model.add(Dropout(0.5))
# positive_model.add(Bidirectional(LSTM(20)))
# positive_model.add(Dropout(0.5))
# positive_model.add(Dense(max_length, activation='softmax'))
# positive_model.compile(loss=l, optimizer=tf.keras.optimizers.Adam(0.001), metrics=['accuracy'])
# positive_history = positive_model.fit(np.array(train_positive_x), np.array(train_positive_y), epochs=60, verbose=2,
#                    validation_data = (np.array(validate_positive_x), np.array(validate_positive_y)))


# In[ ]:


plot_graphs(positive_history, "accuracy")
plot_graphs(positive_history, "loss")


# In[ ]:


tuner.search(train_negative_x, train_negative_y, epochs = 40,verbose = 2,validation_data = (validate_negative_x, validate_negative_y), callbacks=[tf.keras.callbacks.EarlyStopping('val_loss', patience=6)])


# In[ ]:


tuner.results_summary()


# In[ ]:


negative_model = tuner.get_best_models()[0]
negative_model.compile(loss=l, optimizer=tf.keras.optimizers.Adam(0.001), metrics=['accuracy'])
negative_history = negative_model.fit(train_negative_x, train_negative_y, epochs=100, verbose=2,
                   validation_data = (validate_negative_x, validate_negative_y), callbacks=[tf.keras.callbacks.EarlyStopping('val_loss', patience=6)])


# In[ ]:


# negative_model = Sequential()
# negative_model.add(Embedding(vocab_size, 16, input_length=max_length))
# negative_model.add(Dropout(0.5))
# negative_model.add(Bidirectional(LSTM(20)))
# negative_model.add(Dropout(0.5))
# negative_model.add(Dense(max_length, activation='softmax'))
# negative_model.compile(loss=l, optimizer=tf.keras.optimizers.Adam(0.001), metrics=['accuracy'])
# negative_history = negative_model.fit(train_negative_x, train_negative_y, epochs=60, verbose=2,
#                    validation_data = (validate_negative_x, validate_negative_y))


# In[ ]:


plot_graphs(negative_history, "accuracy")
plot_graphs(negative_history, "loss")


# In[ ]:


val_neg_preds = [np.round(negative_model.predict(item[np.newaxis])) for item in validate_negative_x]


# ## Predicting for test set

# In[ ]:


test = pd.read_csv("../input/tweet-sentiment-extraction/test.csv")


# In[ ]:


for idx, value in enumerate(test.text):
    words = str(value).split()
    words = [x for x in words if not x.startswith("http")]
    test["text"][idx] = " ".join(words)
test


# In[ ]:


clean_text(test, 'text')


# In[ ]:


test


# In[ ]:


test_sequences = tokenizer.texts_to_sequences(np.array(test.text))
test_padded = pad_sequences(test_sequences,truncating=trunc_type, maxlen = max_length,padding=pad_type)


# In[ ]:


def get_phrase(array_x, array_y, index): 
    return np.array([rev_word_index[i] for i in array_x[index][array_y.astype(bool)[index]]])


# In[ ]:


preds = []
for index, item in enumerate(test_padded):
    if test.sentiment[index] == "positive":
        p = np.round(positive_model.predict(item[np.newaxis]))
        preds.append(p)
    elif test.sentiment[index] == "negative":
        p = np.round(negative_model.predict(item[np.newaxis]))
        preds.append(p)
    else:
        #p = np.round(neutral_model.predict(item[np.newaxis]))
        preds.append(test_padded[index].astype(bool).astype(int)[np.newaxis])


# In[ ]:


def get_phrase(array_x, array_y, index): 
    return np.array([rev_word_index[i] for i in array_x[index][array_y.astype(bool)[index][0]]if i != 0])


# In[ ]:


test["prediction"] = np.zeros(len(test))

for index, item in enumerate(preds):
    test['prediction'][index] = str(get_phrase(test_padded, np.array(preds), index))


# In[ ]:


test.prediction[test.sentiment == "neutral"] = test.text[test.sentiment == "neutral"]


# In[ ]:


test.prediction = test.prediction.str.replace("[", "")
test.prediction = test.prediction.str.replace("]", "")
test.prediction = test.prediction.str.replace("'", "")
test.prediction = test.prediction.str.replace("<OOV>", "")


# In[ ]:


test.prediction[(test.prediction) == ''] = test.text[(test.prediction) == '']


# In[ ]:


test


# In[ ]:


evaluation = test.textID.copy().to_frame()


# In[ ]:


evaluation['selected_text'] = test['prediction']
evaluation


# In[ ]:


evaluation.to_csv("submission.csv", index=False)


# In[ ]:




