#!/usr/bin/env python
# coding: utf-8

# ## About the Data

# This News Headlines dataset for Sarcasm Detection is collected from two news website. 
#     
# 1. **TheOnion**   
# > TheOnions aims at producing sarcastic versions of current events and the data are headlines from News in Brief and News in Photos categories *(which are sarcastic)*.   
#       
#        
# 2. **HUffPost**  
# > Real *(and non-sarcastic)* news headlines are from **HuffPost**

# ## Why Classifying ?

# > 1. Can you identify sarcastic sentences? 
# > 2. Can you distinguish between fake news and legitimate news?

# ## Import Libraries

# In[ ]:


# to load, access, process and dump json files
import json
# regular repression
import re
# to parse HTML contents
from bs4 import BeautifulSoup

# for numerical analysis
import numpy as np 
# to store and process in a dataframe
import pandas as pd 

# for ploting graphs
import matplotlib.pyplot as plt
# advancec ploting
import seaborn as sns
# to create word clouds
from wordcloud import WordCloud, STOPWORDS 

# To encode values
from sklearn.preprocessing import LabelEncoder
# Convert a collection of text documents to a matrix of token counts
from sklearn.feature_extraction.text import CountVectorizer
# confusion matrix
from sklearn.metrics import confusion_matrix
# train test split
from sklearn.model_selection import train_test_split

# for deep learning 
import tensorflow as tf
# to tokenize text
from tensorflow.keras.preprocessing.text import Tokenizer
# to pad sequence 
from tensorflow.keras.preprocessing.sequence import pad_sequences


# ## Data

# In[ ]:


# import data
df = pd.read_json("../input/news-headlines-dataset-for-sarcasm-detection/Sarcasm_Headlines_Dataset.json", lines=True)
# show first few rows
df.head()


# ## Stopwords

# In[ ]:


# Stopwords list from https://github.com/Yoast/YoastSEO.js/blob/develop/src/config/stopwords.js
stopwords = [ "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's", "its", "itself", "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's", "should", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "we'd", "we'll", "we're", "we've", "were", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves" ]


# ## Utility Function

# In[ ]:


# to plot n-gram
# ==============

def plot_ngram(is_sarcastic, n):
    
    temp_df = df[df['is_sarcastic'] == is_sarcastic]
    
    word_vectorizer = CountVectorizer(ngram_range=(n, n), analyzer='word')
    sparse_matrix = word_vectorizer.fit_transform(temp_df['headline'])
    
    frequencies = sum(sparse_matrix).toarray()[0]
    
    return pd.DataFrame(frequencies, index=word_vectorizer.get_feature_names(), columns=['frequency'])            .sort_values(by='frequency', ascending=False)             .reset_index()             .head(10)


# In[ ]:


# to plot wordcloud
# =================

def plot_wordcloud(headlines, cmap):
    fig, ax = plt.subplots(figsize=(8, 6))
    wc = WordCloud(max_words = 1000, background_color ='white', stopwords = stopwords, 
                   min_font_size = 10, colormap=cmap)
    wc = wc.generate(headlines)
    plt.axis('off')
    plt.imshow(wc)


# In[ ]:


# to plot model accuracy and loss
# ===============================

def plot_history(history):
    
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy', c='dodgerblue', lw='2')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', c='orange', lw='2')
    plt.title('Accuracy', loc='left', fontsize=16)
    plt.xlabel("Epochs")
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss', c='dodgerblue', lw='2')
    plt.plot(history.history['val_loss'], label='Validation Loss', c='orange', lw='2')
    plt.title('Loss', loc='left', fontsize=16)
    plt.xlabel("Epochs")
    plt.ylabel('Loss')
    plt.legend()

    plt.show()


# In[ ]:


# to plot confusion matrix
# ========================

def plot_cm(pred):
    
    pred = pred.ravel()
    pred = np.round(pred)
    
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    cm = confusion_matrix(validation_labels, pred)
    sns.heatmap(cm, annot=True, cbar=False, fmt='1d', cmap='Blues', ax=ax)

    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_yticklabels(['Non-Sarcastic', 'Sarcastic', ])
    ax.set_xticklabels(['Non-Sarcastic', 'Sarcastic'])

    plt.show()


# ## is_sarcastic column
# > ***0***  - indicates real / Non saracastic headline   
# > ***1***  - indicates sarcastic headline

# ### No. of headlines in each category

# In[ ]:


sns.set_style('darkgrid')
plt.figure(figsize=(4, 5))
sns.countplot(df['is_sarcastic'], palette='Dark2')
plt.title('No. of non sarcastic and sarcastic headlines')
plt.xlabel("")
plt.ylabel("")
plt.show()


# In[ ]:


sarc = df[df['is_sarcastic'] == 1]
non_sarc = df[df['is_sarcastic'] != 1]

print('No. of saracastic headlines :', len(sarc))
print('No. of non-saracastic headlines :', len(non_sarc))


# > Disparity in the no. headlines in each category would lead to a biased model.  
# > We need to make sure that there are equal no. of sarcastic and non sarcastic headlines.  
# > We will have drop some non sarcastic headlines.  

# ### Equalize no. of headlines

# In[ ]:


non_sarc = non_sarc.sample(len(sarc))
df = pd.concat([sarc, non_sarc])
df = df.sample(frac=1)

print('No. of saracastic headlines :', len(sarc))
print('No. of non-saracastic headlines :', len(non_sarc))

df.head()


# ## Data Preprocessing

# From : https://www.kaggle.com/madz2000/sarcasm-detection-with-glove-word2vec-82-accuracy
# 

# In[ ]:


# removing non alphanumeric character
def alpha_num(text):
    return re.sub(r'[^A-Za-z0-9 ]', '', text)

# removing the stopwords from text
def remove_stopwords(text):
    final_text = []
    for i in text.split():
        if i.strip().lower() not in stopwords:
            final_text.append(i.strip())
    return " ".join(final_text)


# In[ ]:


# apply preprocessing steps

df['headline'] = df['headline'].str.lower()
df['headline'] = df['headline'].apply(alpha_num)
df['headline'] = df['headline'].apply(remove_stopwords)

df.head()


# ## Plot

# In[ ]:


# word cloud of saracastic headlines
sarcastic = ' '.join(df[df['is_sarcastic']==1]['headline'].to_list())
plot_wordcloud(sarcastic, 'Reds')


# In[ ]:


# word cloud of non-saracastic headlines
non_sarcastic = ' '.join(df[df['is_sarcastic']==0]['headline'].to_list())
plot_wordcloud(non_sarcastic, 'Blues')


# In[ ]:


# n-grams of non-saracastic headlines

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Non-Sarcastic Headlines', ha='left', fontsize=16)
plt.subplots_adjust(wspace=0.7)
axes = axes.flatten()

titles = ['Unigram', 'Bigram', 'Trigram']

for i in range(3):
    sns.barplot(data=plot_ngram(0, i+1), y='index', x='frequency', ax=axes[i])
    axes[i].set_xlabel('')
    axes[i].set_ylabel('')
    axes[i].set_title(titles[i], loc='left')


# In[ ]:


# n-grams of non-saracastic headlines

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Sarcastic Headlines', ha='left', fontsize=16)
plt.subplots_adjust(wspace=0.7)
axes = axes.flatten()

titles = ['Unigram', 'Bigram', 'Trigram']

for i in range(3):
    sns.barplot(data=plot_ngram(1, i+1), y='index', x='frequency', ax=axes[i])
    axes[i].set_xlabel('')
    axes[i].set_ylabel('')
    axes[i].set_title(titles[i], loc='left')


# ## Get headlines and labels

# In[ ]:


# container for sentences
headlines = np.array([headline for headline in df['headline']])

# container for labels
labels = np.array([label for label in df['is_sarcastic']])


# ## Train-test split

# In[ ]:


# train-test split
train_sentences, validation_sentences, train_labels, validation_labels = train_test_split(headlines, labels, 
                                                                                          test_size=0.33, 
                                                                                          stratify=labels)


# ## Model parameters

# In[ ]:


# model parameters
vocab_size = 1200
embedding_dim = 16
max_length = 120
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"


# ## Tokenize and Sequence text

# In[ ]:


# tokenize sentences
tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(train_sentences)
word_index = tokenizer.word_index

# convert train dataset to sequence and pad sequences
train_sequences = tokenizer.texts_to_sequences(train_sentences)
train_padded = pad_sequences(train_sequences, padding=padding_type, maxlen=max_length)

# convert validation dataset to sequence and pad sequences
validation_sequences = tokenizer.texts_to_sequences(validation_sentences)
validation_padded = pad_sequences(validation_sequences, padding=padding_type, maxlen=max_length)


# ## With Word Embedding

# In[ ]:


# model initialization
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# compile model
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# model summary
print(model.summary())


# In[ ]:


# fit model
num_epochs = 20
history = model.fit(train_padded, train_labels, 
                    epochs=num_epochs, verbose=1,
                    validation_split=0.3)

# predict values
pred = model.predict(validation_padded)


# In[ ]:


# plot history
plot_history(history)


# In[ ]:


# plot confusion matrix
plot_cm(pred)


# In[ ]:


# reviews on which we need to predict
sentence = ["Breathing oxygen related to staying alive", 
            "Safety meeting ends in accident"]

# convert to a sequence
sequences = tokenizer.texts_to_sequences(sentence)

# pad the sequence
padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

# preict the label
print(model.predict(padded))


# ## With LSTM

# In[ ]:


# model initialization
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# compile model
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# model summary
model.summary()


# In[ ]:


# fit model
num_epochs = 20
history = model.fit(train_padded, train_labels, 
                    epochs=num_epochs, verbose=1,
                    validation_split=0.3)

# predict values
pred = model.predict(validation_padded)


# In[ ]:


# plot history
plot_history(history)


# In[ ]:


# plot confusion matrix
plot_cm(pred)


# ## With Convolution

# In[ ]:


# model initialization
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.Conv1D(128, 5, activation='relu'),
    tf.keras.layers.GlobalMaxPooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# compile model
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# model summary
model.summary()


# In[ ]:


# fit model
num_epochs = 20
history = model.fit(train_padded, train_labels, 
                    epochs=num_epochs, verbose=1,
                    validation_split=0.3)

# predict values
pred = model.predict(validation_padded)


# In[ ]:


# plot history
plot_history(history)


# In[ ]:


# plot confusion matrix
plot_cm(pred)


# ## Glove

# ## BERT

# In[ ]:




