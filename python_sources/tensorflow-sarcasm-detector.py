#!/usr/bin/env python
# coding: utf-8

# <center><h1>Using Tensorflow for Sarcasm Detection in News Headlines</h1></center>

# <h2>Context of the Dataset</h2>

# Past studies in Sarcasm Detection mostly make use of Twitter datasets collected using hashtag based supervision but such datasets are noisy in terms of labels and language. Furthermore, many tweets are replies to other tweets and detecting sarcasm in these requires the availability of contextual tweets.
# 
# To overcome the limitations related to noise in Twitter datasets, this News Headlines dataset for Sarcasm Detection is collected from two news website. TheOnion aims at producing sarcastic versions of current events and we collected all the headlines from News in Brief and News in Photos categories (which are sarcastic). We collect real (and non-sarcastic) news headlines from HuffPost.
# 
# This new dataset has following advantages over the existing Twitter datasets:
# 
# * Since news headlines are written by professionals in a formal manner, there are no spelling mistakes and informal usage. This reduces the sparsity and also increases the chance of finding pre-trained embeddings.
# 
# * Furthermore, since the sole purpose of TheOnion is to publish sarcastic news, we get high-quality labels with much less noise as compared to Twitter datasets.
# 
# * Unlike tweets which are replies to other tweets, the news headlines we obtained are self-contained. This would help us in teasing apart the real sarcastic elements.

# <h2>Inspiration</h2>

# To identify sarcastic sentences and to distinguish between fake news and legitimate news?

# In[ ]:


import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split


# In[ ]:


df = pd.read_json("../input/news-headlines-dataset-for-sarcasm-detection/Sarcasm_Headlines_Dataset_v2.json", lines=True)
df.head()


# ## Visualize the Data

# In[ ]:


import string
from nltk.corpus import stopwords
import plotly.graph_objects as go
import plotly_express as px
from plotly.subplots import make_subplots
from wordcloud import WordCloud, ImageColorGenerator
from PIL import Image


# In[ ]:


def remove_stopwords(text):
    final_text = []
    for i in text.split():
        if i.strip().lower() not in stop:
            final_text.append(i.strip())
    return " ".join(final_text)


# In[ ]:


stop = set(stopwords.words('english'))
punctuation = list(string.punctuation)
stop.update(punctuation)


# In[ ]:


df['headline']=df['headline'].apply(remove_stopwords)
wc_0 = WordCloud(max_words = 2000, background_color="white", width = 1000 , height = 800).generate(" ".join(df[df.is_sarcastic == 0].headline))
wc_1 = WordCloud(max_words = 2000, background_color="white", width = 1000 , height = 800).generate(" ".join(df[df.is_sarcastic == 1].headline))
fig = make_subplots(1, 2, subplot_titles=("Non-Sarcastic", "Sarcastic"))
fig.update_layout(
    title="Word Clouds")
fig.add_trace(go.Image(z = wc_0), 1, 1)
fig.add_trace(go.Image(z = wc_1), 1, 2)
fig.show()


# ## Modelling a Sarcasm Classifier

# In[ ]:


vocab_size = 10000
embedding_dim = 16
max_length = 100
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"
training_size = 20000
epochs = 5


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(df['headline'], df['is_sarcastic'], train_size = 0.9, random_state = 73)

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(x_train)
word_index = tokenizer.word_index

train_sequences = tokenizer.texts_to_sequences(x_train)
x_train_padded = pad_sequences(train_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

test_sequences = tokenizer.texts_to_sequences(x_test)
x_test_padded = pad_sequences(test_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)


# In[ ]:


tf.keras.backend.clear_session()
tf.random.set_seed(51)
np.random.seed(51)
model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64,recurrent_dropout = 0.3 , dropout = 0.3, return_sequences = True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32,recurrent_dropout = 0.1 , dropout = 0.1)),
    tf.keras.layers.Dense(512, activation = "relu"),
    tf.keras.layers.Dense(1, activation = "sigmoid")
])

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['accuracy'])


# In[ ]:


history = model.fit(x_train_padded,y_train, batch_size = 128, epochs = epochs, validation_data = (x_test_padded, y_test))


# In[ ]:


train_acc = history.history['accuracy']
train_loss = history.history['loss']
val_acc = history.history['val_accuracy']
val_loss = history.history['val_loss']

fig_model = make_subplots(1, 2, subplot_titles=("Accuracy over Time", "Loss over Time"))
fig_model.add_trace(go.Scatter(x = np.arange(epochs),
                                 y = train_acc,
                                 mode = "lines+markers",
                                 name = "Training Accuracy"),
                   row = 1, col = 1)
fig_model.add_trace(go.Scatter(x = np.arange(epochs),
                                 y = val_acc,
                                 mode = "lines+markers",
                                 name = "Validation Accuracy"),
                   row = 1, col = 1)
fig_model.add_trace(go.Scatter(x = np.arange(epochs),
                                 y = train_loss,
                                 mode = "lines+markers",
                                 name = "Training Loss"),
                   row = 1, col = 2)
fig_model.add_trace(go.Scatter(x = np.arange(epochs),
                                 y = val_loss,
                                 mode = "lines+markers",
                                 name = "Validation Loss"),
                   row = 1, col = 2)
fig_model.show()


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report


# In[ ]:


pred = model.predict_classes(x_test_padded)


# In[ ]:


print(classification_report(y_test, pred, target_names = ['Non-Sarcastic','Sarcastic']))


# In[ ]:


cm = confusion_matrix(y_test, pred)
cm = pd.DataFrame(cm , index = ['Non-Sarcastic','Sarcastic'] , columns = ['Predicted Non-Sarcastic','Predicted Sarcastic'])
plt.figure(figsize = (10,8))
sns.heatmap(cm,cmap = "BuGn", linecolor = 'black' , linewidth = 1 , annot = True, fmt='' , xticklabels = ['Predicted Non-Sarcastic','Predicted Sarcastic'] , yticklabels = ['Non-Sarcastic','Sarcastic'])
plt.show()


# <center><h1> The End </h1></center>

# Thank you for going through my notebook. Hopefully you have gained valuable insights into the dataset.
# I would appreciate if you Upvote this notebook as well! Kindly let me know if I can improve my work in any aspect
