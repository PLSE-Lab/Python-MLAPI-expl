#!/usr/bin/env python
# coding: utf-8

# # NLP Challange (part 3)
# 
# I have challanged myself to learn NLP in one week and this is my third notebook for challange.
# 
# [Here](https://www.kaggle.com/maunish/nlp-challenge-part-1?scriptVersionId=32391796) is first one.
# [Here](https://www.kaggle.com/maunish/nlp-challenge-part-2?scriptVersionId=32391910) is second one.
# 
# In first notebook we learned about topics like
# 
# 1. What is NLP?
# 2. Tf (term frequency) and idf (inverse document frequency)
# 3. CountVectorizer
# 4. TfidfVectorizer and TfidfTransformer
# 5. Training Logistic Regression, SVM, XGboost, Navie Bayes,
# 6. GridSearch.
# 
# In second notebook we covered some advance NLP topics like
# 
# 1. What is Word vectors and word embeddings?
# 2. Simple NN
# 3. RNN
# 4. LSTM
# 5. GRU
# 6. Bidirectional LSTM
# 
# In third Notebook we are going to learn about topics like.
# 1. Attention models
# 2. What is transformer
# 3. BERT.

# # Sequence Modeling
# 
# There are basically various types of architecture in which RNN works.
# 
# ![image.png](attachment:image.png)
# 
# **one to one**: one to one means one input is given and it produces one output<br/>
# it is just work like basic Neural Network.
# 
# **Many to One**: In many to one architecture RNN is provided series of inputs and at the end<br/>
# of series it gives one output. This type of architecture could be used in sentiment analysis <br/> 
# where we have to predict negativity or positivity of sentence.
# 
# **One to Many**: This type of architecture just take one input and based on that generates many output.<br/>
# One example on one to many is image captioning where we provide one single image and based on that.<br/>
# It generates a series of words.
# 
# **Many to Many**: This type of architecture basically takes sequence and outputs sequence.<br/>
# Mostly this type of architecture is used in language translation.
# 
# 
# Watch the below video to understand differnt types of sequence models.<br/>
# [Sequence Models](https://www.coursera.org/lecture/nlp-sequence-models/different-types-of-rnns-BO8PS)
# 
# I am not writing code for sequence translation or generation.<br/>
# But various resources of implementation are available on internet.
# 

# # Encoder-Decoder and Attention
# 
# Encoder-Decoder are an amazing method of sequence modeling. When I read about it in various places, I was enthalled by<br/>
# How amazing this things works. 
# 
# **NOTE**: My way of explaining encoder-decoder is slight different from what I have read, so you may castigate me for my weired explanation<br/>
# 
# **Encoder-Decoder**.
# Let's say encoder and decoder are persons we will call them Dalinar and Kaladin.<br/>
# Dalinar lives in a place named Alethkar and Kaladin lives in a place named Hearthstone and they are traders.<br/>
# Both the places speaks different language and so people of both the places used to face problem of language while trading<br/>.
# Now as Dalinar and Kaladin known each other through trading, they decided to solve this problem.<br/>
# The idea was that each individual will create drawings of various processes of trading.<br/>
# and will write down the basic sentence used for that process in both the languages and will create a book<br/>
# so everyone can learn form it. So that's the story of encoder and decoder.
# 
# As Dalinar and Kaladin used image as their intermediate language, encoder-decoder uses NN hidden layers as their<br/>
# intermediate language.
# 
# But Encoder-Decoder are no real person so, how do they learn an intermediate language ?<br/>
# 
# Do you remember word embeddings are gist of whole word in vector form.<br/>
# 
# Similarly Enocder-Decoder are RNNs in which  **encoder** figures out gist of the Sequence<br/>
# and sends it to **Decoder** to convert that into desired Sequence.
# 
# Now to more serious explanation.<br/>
# 
# Let's understand what trained encoder-decoder do.<br/>
# 
# Suppose we want to translate english sentence "Hey, How are you?" to french "Salut, comment vas tu" <br/>
# Encoder's RNN will take each word one by one in vector form and use embeddings to convert it to word embedding, and <br/>
# create an hidden state pass it as input along with next word to next RNN iteration.<br/>
# 
# It will continue this for last word "you" so we will get total four hidden state for each iteration<br/>.
# Aa RNN's are passed previous state, last state is dependent on all previous state indirectly, so last hidden state.<br/>
# acts like gist of whole sentence.<br/>
# 
# And so last hidden state is passed to decoder which is also an RNN which will Generate a words step by step in French language.
# 
# **Attention Models**
# 
# What are attention models and how are they differnt from encoder-decoder ?<br/>
# 
# Well as we saw in encoder-decoder we only passed last hidden state in encoder to decoder.<br/>
# In Attention models each hidden state of encoder is passed to decoder.
# 
# Now what decoder will do is it has **weight** attach to each of the **hidden-state** passed to it by encoder.<br/>
# 
# So it will multipy each **hidden-state** by those **weights** to each hidden state, calculate **softmax** for each them, so it will highlight.<br/>
# most important **hidden-state** of them all and then each **softmax value** is multiplied again to **orignal-state**<br/>
# and then all of them are added to get single **hidden-state** which is used to output single word in sequence.<br/>
# 
# steps perfomed by decoder
# 
# 1. w1 x h1, w2 x h2, w3 x h3 <br/>
# 2. s1 = Softmax(w1xh1),s2 = Softmax(w2xh3),s3 = Softmax(w3xh3)<br/>
# 3. H1 = s1 x h1, H2 = s2 x h2, H3= s3 x h3<br/>
# 4. H_final = (H1 + H2 + H3)
# 
# This H_final will be used to generate one word.<br/>
# Now there will be other set of weights which will be connected to this hidden states for gererating next word.
# 
# You might have got the idea that what Attention model basically does is while outputting each word in a sentence<br/>
# It will look at which part of the orignal sentence is important for the translation and which part is not.<br/>
# So Decoder will focus on the words in orignal sentence which are important to it.
# 
# Attention model works better because it gets to see whole sentence as oppose to encoder-decoder where only.<br/>
# last hidden state was passed to it.
# 
# **Note**: If you did't get what I explained here please read 1st article it has far better explanation.
# 
# 
# videos
# 
# 1. [Attention in Neural Networks](https://www.youtube.com/watch?v=W2rWgXJBZhU)
# 
# Articles
# 
# 1. [Visualizing A Neural Machine Translation Model](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/)
# 2. [Sequence Models](https://www.coursera.org/lecture/nlp-sequence-models/attention-model-intuition-RDXpX)
# 3. [How Does Attention Work in Encoder-Decoder Recurrent Neural Networks](https://machinelearningmastery.com/how-does-attention-work-in-encoder-decoder-recurrent-neural-networks/)
# 
# 

# # Transformer and BERT
# 
# As explaining Tansromfer and BERT is far above my writting skills i will link the videos and articles below.
# 
# 
# ### Transformer
# 1. [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
# 
# Links  to videos.
# 1. [Transformer Neural Networks - EXPLAINED! (Attention is all you need)](https://www.youtube.com/watch?v=TQQlZhbC5ps)
# 
# Orignal paper "Attention is all you need"
# 1. [Attention is all you need](https://arxiv.org/pdf/1706.03762v4.pdf)
# 
# ### BERT
# 
# Link to articles
# 1. [The Illustrated BERT, ELMo, and co. ](http://jalammar.github.io/illustrated-bert/)
# 2. [A Visual Guide to Using BERT for the First Time](http://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/)
# 
# 

# # BERT MODEL

# In[ ]:


import os

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from kaggle_datasets import KaggleDatasets
import transformers
from tqdm.notebook import tqdm
from tokenizers import BertWordPieceTokenizer


# In[ ]:


def fast_encode(texts, tokenizer, chunk_size=256, maxlen=512):
    """
    https://www.kaggle.com/xhlulu/jigsaw-tpu-distilbert-with-huggingface-and-keras
    """
    tokenizer.enable_truncation(max_length=maxlen)
    tokenizer.enable_padding(max_length=maxlen)
    all_ids = []
    
    for i in tqdm(range(0, len(texts), chunk_size)):
        text_chunk = texts[i:i+chunk_size].tolist()
        encs = tokenizer.encode_batch(text_chunk)
        all_ids.extend([enc.ids for enc in encs])
    
    return np.array(all_ids)


# In[ ]:


def build_model(transformer, max_len=512):
    """
    https://www.kaggle.com/xhlulu/jigsaw-tpu-distilbert-with-huggingface-and-keras
    """
    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    sequence_output = transformer(input_word_ids)[0]
    cls_token = sequence_output[:, 0, :]
    out = Dense(3, activation='sigmoid')(cls_token)
    
    model = Model(inputs=input_word_ids, outputs=out)
    model.compile(Adam(lr=1e-5), loss='categorical_crossentropy')
    
    return model


# In[ ]:


# Detect hardware, return appropriate distribution strategy
try:
    # TPU detection. No parameters necessary if TPU_NAME environment variable is
    # set: this is always the case on Kaggle.
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    # Default distribution strategy in Tensorflow. Works on CPU and single GPU.
    strategy = tf.distribute.get_strategy()

print("REPLICAS: ", strategy.num_replicas_in_sync)


# In[ ]:


AUTO = tf.data.experimental.AUTOTUNE

# Data access
GCS_DS_PATH = KaggleDatasets().get_gcs_path()

# Configuration
EPOCHS = 3
BATCH_SIZE = 16 * strategy.num_replicas_in_sync
MAX_LEN = 192


# In[ ]:




# First load the real tokenizer
tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-multilingual-cased')
# Save the loaded tokenizer locally
tokenizer.save_pretrained('.')
# Reload it with the huggingface tokenizers library
fast_tokenizer = BertWordPieceTokenizer('vocab.txt', lowercase=False)
fast_tokenizer


# In[ ]:


from sklearn import preprocessing
from sklearn.model_selection import train_test_split

PATH = '../input/spooky-author-identification'
train = pd.read_csv(f'{PATH}/train.zip')
test = pd.read_csv(f'{PATH}/test.zip')
sample = pd.read_csv(f'{PATH}/sample_submission.zip')


#data preprocssing
encoder = preprocessing.LabelEncoder()
y = encoder.fit_transform(train["author"].values)

#data split
x_train, x_valid, y_train, y_valid = train_test_split(train.text.values,y,random_state=42,test_size=0.1,shuffle=True)


# In[ ]:


X_train = fast_encode(x_train.astype(str), fast_tokenizer, maxlen=MAX_LEN)
X_valid = fast_encode(x_valid.astype(str), fast_tokenizer, maxlen=MAX_LEN)
X_test = fast_encode(test.text.astype(str), fast_tokenizer, maxlen=MAX_LEN)


# In[ ]:


train_dataset = (
    tf.data.Dataset
    .from_tensor_slices((X_train, y_train))
    .repeat()
    .shuffle(2048)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

valid_dataset = (
    tf.data.Dataset
    .from_tensor_slices((X_valid, y_valid))
    .batch(BATCH_SIZE)
    .cache()
    .prefetch(AUTO)
)

test_dataset = (
    tf.data.Dataset
    .from_tensor_slices(X_test)
    .batch(BATCH_SIZE)
)


# In[ ]:


get_ipython().run_cell_magic('time', '', "with strategy.scope():\n    transformer_layer = (\n        transformers.TFDistilBertModel\n        .from_pretrained('distilbert-base-multilingual-cased')\n    )\n    model = build_model(transformer_layer, max_len=MAX_LEN)\nmodel.summary()")


# In[ ]:


n_steps = X_train.shape[0] // BATCH_SIZE
train_history = model.fit(
    train_dataset,
    steps_per_epoch=n_steps,
    validation_data=valid_dataset,
    epochs=EPOCHS
)


# In[ ]:


# sub['toxic'] = model.predict(test_dataset,verbose=1)


# In[ ]:


sub = model.predict(test_dataset,verbose=1)


# In[ ]:


sample.iloc[:,-3:] = sub 


# In[ ]:


sample.to_csv("submission.csv",index=False)


# In[ ]:




