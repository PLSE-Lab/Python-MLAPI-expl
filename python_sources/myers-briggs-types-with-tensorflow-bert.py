#!/usr/bin/env python
# coding: utf-8

# The Myers-Briggs Type Indicator (MBTI) is one of the most widely-used personality tests. It classifies people into 16 4-letter categories. For more information on the MBTI, see: https://www.16personalities.com/personality-types
# 
# This dataset contains a person's Myers-Briggs type and the text of their online posts. So, can we use deep learning to predict someone's personality type based on what they write online?

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


data = pd.read_csv("../input/mbti-type/mbti_1.csv")


# In[ ]:


data.head()


# The first thing I want to do is turn the personality types into numbered categories, ranging from 0 to 15. These will serve as the output of the model when it is categorizing people.

# In[ ]:


types = np.unique(data.type.values)


# In[ ]:


def get_type_index(string):
    return list(types).index(string)


# In[ ]:


data['type_index'] = data['type'].apply(get_type_index)


# In[ ]:


data.posts.values[0]


# Now, I want to clean the text to get rid of hyperlinks, puncuation, and anything else that's cluttering up the text. Specifically, the pipe (|) character seems like it separates different posts, but without spaces between them. First, I'm going to replace the pipes with spaces so that the tokenizer won't parse those parts as one long word. Then, I'll clean up the rest.

# In[ ]:


import string
import re

def clean_text(text):
    regex = re.compile('[%s]' % re.escape('|'))
    text = regex.sub(" ", text)
    words = str(text).split()
    words = [i.lower() + " " for i in words]
    words = [i for i in words if not "http" in i]
    words = " ".join(words)
    words = words.translate(words.maketrans('', '', string.punctuation))
    return words


# In[ ]:


data['cleaned_text'] = data['posts'].apply(clean_text)


# In[ ]:


data.cleaned_text.values[0]


# In[ ]:


data.head()


# Now, we split the data into training, testing, and validation sets,

# In[ ]:


from sklearn.model_selection import train_test_split

train, test = train_test_split(data)
train, val = train_test_split(train)


# Keras has a great tokenizer that we can use to turn sequences of words into arrays of numbers. For more information, see: https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/text/Tokenizer

# In[ ]:


import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

vocab_size = 10000
trunc_type = "post"
pad_type = "post"
oov_tok = "<OOV>"
tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(data.cleaned_text.values)


# In[ ]:


maxlen = 1500
train_sequences = tokenizer.texts_to_sequences(train.cleaned_text.values)
train_padded = pad_sequences(train_sequences, maxlen = maxlen, truncating = trunc_type, padding = pad_type)

val_sequences = tokenizer.texts_to_sequences(val.cleaned_text.values)
val_padded = pad_sequences(val_sequences, maxlen = maxlen, truncating = trunc_type, padding = pad_type)


# In[ ]:


train_padded


# So, our model is going to take in these arrays of numbers that represent the text, and it's going to output the personality type that it thinks is associated with it. Here, I'm going to convert the personality types to one-hot-encoded labels. This simply means that to represent a particular category, we make an array with the length of the total possible number of categories, and make all of the values zero except at the index of the category we're trying to represent. 

# In[ ]:


one_hot_labels = tf.keras.utils.to_categorical(train.type_index.values, num_classes=16)
val_labels= tf.keras.utils.to_categorical(val.type_index.values, num_classes=16)


# In[ ]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, Flatten, Dropout, Conv1D, GlobalMaxPooling1D

def create_model():
    op = tf.keras.optimizers.Adam(learning_rate=0.00001)

    model = Sequential()
    model.add(Embedding(vocab_size, 256, input_length=maxlen-1))
    model.add(Dropout(0.3))
    model.add(Bidirectional(LSTM(200, return_sequences=True)))
    model.add(Dropout(0.3))
    model.add(Bidirectional(LSTM(20)))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(16, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=op, metrics=['accuracy'])
    return model


# Using a TPU can greatly reduce the amount of time spent training the model.

# In[ ]:


use_tpu = True
if use_tpu:
    # Create distribution strategy
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)

    # Create model
    with strategy.scope():
        model = create_model()
else:
    model = create_model()
    
model.summary()


# In[ ]:


model.fit(train_padded, one_hot_labels, epochs =20, verbose = 1, 
          validation_data = (val_padded, val_labels),  callbacks = [tf.keras.callbacks.EarlyStopping(patience = 3)])


# This model didn't do very well, only achieving around 20% accuracy. This is a difficult challenge- to classify people into 16 different categories based on text that may loosely correlate with those categories. Let's see if we can do a bit better by incorporating a transformer. I used the one from this Keras example: https://keras.io/examples/nlp/text_classification_with_transformer/
# 
# For more information about transformers in general, see: https://medium.com/inside-machine-learning/what-is-a-transformer-d07dd1fbec04

# In[ ]:


from tensorflow.keras import layers
from tensorflow import keras
class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
        self.projection_dim = embed_dim // num_heads
        self.query_dense = layers.Dense(embed_dim)
        self.key_dense = layers.Dense(embed_dim)
        self.value_dense = layers.Dense(embed_dim)
        self.combine_heads = layers.Dense(embed_dim)

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        # x.shape = [batch_size, seq_len, embedding_dim]
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)  # (batch_size, seq_len, embed_dim)
        key = self.key_dense(inputs)  # (batch_size, seq_len, embed_dim)
        value = self.value_dense(inputs)  # (batch_size, seq_len, embed_dim)
        query = self.separate_heads(
            query, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        key = self.separate_heads(
            key, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        value = self.separate_heads(
            value, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(
            attention, perm=[0, 2, 1, 3]
        )  # (batch_size, seq_len, num_heads, projection_dim)
        concat_attention = tf.reshape(
            attention, (batch_size, -1, self.embed_dim)
        )  # (batch_size, seq_len, embed_dim)
        output = self.combine_heads(
            concat_attention
        )  # (batch_size, seq_len, embed_dim)
        return output


# In[ ]:


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


# In[ ]:


class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, emded_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=emded_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=emded_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions


# In[ ]:


embed_dim = 32  # Embedding size for each token
num_heads = 2  # Number of attention heads
ff_dim = 32  # Hidden layer size in feed forward network inside transformer

def create_model(): 
    inputs = layers.Input(shape=(maxlen,))
    embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
    x = embedding_layer(inputs)
    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
    x = transformer_block(x)
    # x = layers.GlobalAveragePooling1D()(x)
    # x = layers.Dropout(0.1)(x)
    x = (Bidirectional(LSTM(200, return_sequences=True)))(x)
    x = (Dropout(0.3))(x)
    x = (Bidirectional(LSTM(20)))(x)
    x = layers.Dense(20, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(16, activation="softmax")(x)
    
    op = tf.keras.optimizers.Adam(learning_rate=0.00001)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(op, 'categorical_crossentropy', metrics = ['accuracy'])
    return model


# In[ ]:


use_tpu = True
if use_tpu:
    # Create distribution strategy
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)

    # Create model
    with strategy.scope():
        model = create_model()
else:
    model = create_model()
    
model.summary()


# In[ ]:


model.fit(train_padded, one_hot_labels, epochs =30, verbose = 1, 
          validation_data = (val_padded, val_labels), callbacks = [tf.keras.callbacks.EarlyStopping(patience = 3)])


# That one didn't work too well, either. Looks like I'm going to have to bring out the big guns. BERT is arguably the most powerful transformer out there right now. For more information: https://towardsdatascience.com/bert-explained-state-of-the-art-language-model-for-nlp-f8b21a9b6270

# In[ ]:


import transformers
tokenizer = transformers.AutoTokenizer.from_pretrained('bert-large-uncased')


# In[ ]:


maxlen = 1500

train_input_ids = [tokenizer.encode(str(i), max_length = maxlen , pad_to_max_length = True) for i in train.cleaned_text.values]
val_input_ids = [tokenizer.encode(str(i), max_length = maxlen , pad_to_max_length = True) for i in val.cleaned_text.values]


# In[ ]:


def create_model(): 
    input_word_ids = tf.keras.layers.Input(shape=(maxlen,), dtype=tf.int32,
                                           name="input_word_ids")
    bert_layer = transformers.TFBertModel.from_pretrained('bert-large-uncased')
    bert_outputs = bert_layer(input_word_ids)[0]
    pred = tf.keras.layers.Dense(16, activation='softmax')(bert_outputs[:,0,:])
    
    model = tf.keras.models.Model(inputs=input_word_ids, outputs=pred)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(
    learning_rate=0.00001), metrics=['accuracy'])
    return model


# In[ ]:


use_tpu = True
if use_tpu:
    # Create distribution strategy
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)

    # Create model
    with strategy.scope():
        model = create_model()
else:
    model = create_model()
    
model.summary()


# In[ ]:


batch_size = 16

model.fit(np.array(train_input_ids), one_hot_labels,validation_data = (np.array(val_input_ids), val_labels),
          verbose = 1, epochs = 20, batch_size = batch_size,  callbacks = [tf.keras.callbacks.EarlyStopping(patience = 5)])


# In[ ]:


test_input_ids = [tokenizer.encode(str(i), max_length = maxlen , pad_to_max_length = True) for i in test.cleaned_text.values]
test_labels= tf.keras.utils.to_categorical(test.type_index.values, num_classes=16)


# In[ ]:


model.evaluate(np.array(test_input_ids), test_labels)


# The BERT model isn't perfect, but it's much better than the other models. Looks like there is some relationship between your personality type and what you post online, after all. 

# In[ ]:




