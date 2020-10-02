#!/usr/bin/env python
# coding: utf-8

# # Tensorflow Keras Tutorial - Basics of NLP (Part 5)
# 
# **What is Keras?** Keras is a wrapper that allows you to implement Deep Neural Network without getting into intrinsic details of the Network. It can use Tensorflow or Theano as backend. This tutorial series will cover Keras from beginner to intermediate level.
# 
# YOU CAN CHECK OUT REST OF THE TUTORIALS OF THIS SERIES.
# 
# [PART 1](https://www.kaggle.com/akashkr/tf-keras-tutorial-neural-network-part-1)<br>
# [PART 2](https://www.kaggle.com/akashkr/tf-keras-tutorial-cnn-part-2)<br>
# [PART 3](https://www.kaggle.com/akashkr/tf-keras-tutorial-binary-classification-part-3)<br>
# [PART 4](https://www.kaggle.com/akashkr/tf-keras-tutorial-pretrained-models-part-4)
# 
# In the previous notebooks we worked on image data. Now we are going to see text data. The common places where NLP is applied is Document Classification, Sentiment Analysis, Chat-bots etc.
# 
# In this tutorial we are going to see,
# * Tokenization
# * padding
# * Embedding
# * Modelling
# * Visualizing Embedding weights

# # Importing Libraries and reading data

# In[ ]:


from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import plotly.express as px
import tensorflow as tf
import pandas as pd


# In[ ]:


train_df = pd.read_csv('../input/nlp-getting-started/train.csv')


# # Overview of Dataset
# 
# ### Data Format
# 
# Each sample in the train and test set has the following information:
# 
# * The text of a tweet
# * A keyword from that tweet (although this may be blank!)
# * The location the tweet was sent from (may also be blank)
# 
# ### Target
# 
# **You are predicting whether a given tweet is about a real disaster or not**. If so, predict a 1. If not, predict a 0.
# 
# ### Columns
# 
# id - a unique identifier for each tweet
# text - the text of the tweet
# location - the location the tweet was sent from (may be blank)
# keyword - a particular keyword from the tweet (may be blank)
# target - in train.csv only, this denotes whether a tweet is about a real disaster (1) or not (0)
# 
# > NOTE: **We will be using just the text and target features of the data**

# In[ ]:


train_df.head()


# In[ ]:


print(f'Shape of data: {train_df.shape}')


# In[ ]:


# Find the number of missing values
print(train_df.info())


# # Tokenization
# 
# Lets suppose you have two statements you need to classify as positive or negative.
# 
# | **text**         | **sentiment** |
# |------------------|---------------|
# | I am happy today | positive      |
# | He hit me today! | negative      |
# 
# The neural network does't understands words like `I, am, today`. To feed these into the neural network, each word must be converted into a unique number or **token**.
# 
# > {'i': 1, 'am': 2, 'happy': 3, 'today': 5, 'he': 4, 'hit': 6, 'me': 7}
# 
# Now the sentences becomes-<br>
# `I am happy today` -> `(1, 2, 3, 5)`<br>
# `He hit me today!` -> `(4, 6, 7, 5)`<br>
# 
# Note that `today` has the same token in both the sentences. By default symbols are stripped and every character is converted to lower case. This array of tokens is **sequence**.
# 
# ### Tokenizer
# > * **num_words** Number of unique tokens to be used while creating embeddings from text. If more words are present in the data and number of tokens are less, it takes the most common words to generate embeddings<br>
# * **oov_token** Token to be used if some new value is encountered while converting text to sequence or embedding
# 
# ### Tokenizer.fit_on_text()
# > fits the tokenizer of the given input text
# 
# ### Tokenizer.text_to_sequences(texts)
# > converts `texts` to sequences
# 
# Lets try the tokenization code on just 5 rows of data.

# In[ ]:


# Fitting to the input data
tokenizer = Tokenizer(num_words=700, oov_token='OOV')
tokenizer.fit_on_texts(train_df.head()['text'])


# In[ ]:


# Tokens 
word_index = tokenizer.word_index
print(word_index)


# In[ ]:


# Generate text sequences
sequences = tokenizer.texts_to_sequences(train_df.head()['text'])
print(sequences)


# ### pad_sequence
# Pads zeros in sequence to make the length of sequences equal
# > * **sequences** list of sequences to be padded
# * **maxlen** Maximum length of all the sequence (if not passed, sequence will be padded to the length of the longest sequence
# * **padding** `pre` or `post`, pad before or after the sequence
# * **truncating** `pre` or `post`, remove values from sequences larger than `maxlen`, either at the beginning or at the end of the sequences.

# In[ ]:


padded = pad_sequences(sequences, padding='post')
print(padded)


# # Preprocessing

# In[ ]:


# Splitting the data into 2/3 as train and 1/3 as test
X_train, X_test, y_train, y_test = train_test_split(train_df['text'], train_df['target'], test_size=0.33, random_state=42)


# In[ ]:


vocab_size = 500
embedding_dim = 16
max_length = 50
trunc_type='post'
oov_tok = "<OOV>"

# Tokenization
tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(X_train)

word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(X_train)
testing_sequences = tokenizer.texts_to_sequences(X_test)

# Padding
padded = pad_sequences(sequences, maxlen=max_length, truncating=trunc_type)  # 
testing_padded = pad_sequences(testing_sequences, maxlen=max_length)  # , maxlen=max_length


# In[ ]:


padded.shape


# In[ ]:


testing_padded.shape


# # Embedding
# A word embedding is a learned representation for text where words that have the same meaning have a similar representation. As in tokenization, a word is represented as a number, in Embedding a word is represented as a vector/set of numbers which represent meaning of the word i.e. similar words will have similar embedding. This preserves the sentiment of the words.

# # Modelling
# 
# ### tf.keras.layers.Embedding
# `tf.keras.layers.Embedding` Turns positive integers (indexes) into dense vectors of fixed size.
# 
# > * **input_dim** size of the vocabulary
# * **output_dim** Dimension of the dense embedding
# * **input_length** Length of input sequences, when it is constant.This argument is required if you are going to connect
#     `Flatten` then `Dense` layers upstream
#     (without it, the shape of the dense outputs cannot be computed).

# In[ ]:


model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()


# In[ ]:


num_epochs = 10
history = model.fit(padded, y_train, epochs=num_epochs, validation_data=(testing_padded, y_test))


# As you can see, after 3-4 epochs the model is overfitting. Try tweaking the model or early stopping (discussed is Part 2) to avoid that.

# # Visualization
# Now lets plot the embedding into a 3D space to see if the embedding is able to separate features in words.
# 
# We will be using **PCA-Principal Component Analysis** to reduce the dimensionality of the data with minimal loss in features. The theory of PCA is beyond the scope of this tutorial. We will be using PCA to reduce 16 points to 3 to facilitate plot.

# In[ ]:


# Getting weights from the embedding
e = model.layers[0]
weights = e.get_weights()[0]
print(weights.shape)

# Reverse mapping function from token to word
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])


# In[ ]:


# Combining embedding and words into a DataFrame
embedding_df = pd.DataFrame()
for word_num in range(1, vocab_size):
    word = reverse_word_index[word_num]
    embeddings = weights[word_num]
    embedding_df = embedding_df.append(pd.Series({'word':word, 'em':embeddings}), ignore_index=True)
    
embedding_df = pd.concat([embedding_df['em'].apply(pd.Series), embedding_df['word']], axis=1)


# In[ ]:


# Using PCA to map 16 embedding values to 3 to plot
p = PCA(n_components=3)
principal_components = p.fit_transform(embedding_df.filter(regex=r'\d'))
embedding_df[['x', 'y', 'z']] = pd.DataFrame(principal_components)

embedding_df.shape


# In[ ]:


fig = px.scatter_3d(embedding_df, x='x', y='y', z='z', hover_name='word', color='x')
fig.show()


# You can hover to the points on the +X side and see words like **hiroshima, derailment, earthquake, drought, suicide** etc and **bags, better, full** etc on the -X side. This shows how well the embedding layer is trained to separate words which might possibiliy be tweet of disaster. Try increasing the `vocab_size` to visualize more words.

# In[ ]:


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()

plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend(loc=0)
plt.show()


# Think of loss as the **error** in prediction. **With increasing epochs the training accuracy increases constantly while the validation accuracy increases then slowly decreases as overfitting occurs.
# With increasing epoch the training loss decreases constantly while validation loss decreases first than slowly increases.**
