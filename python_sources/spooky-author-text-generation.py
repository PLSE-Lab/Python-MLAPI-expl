#!/usr/bin/env python
# coding: utf-8

# # Generating text using LSTM

# Simple LSTM based text generator.
# ## Flow
# 1. **Preprocessing**  - Selecting texts of a specific author.<br>
# 2. **Tokenization** - Converting texts to word tokens using keras tokenizer.<br>
# 3. **Sliding Window** - Generating training sequences by combining all the text and making sliding windows<br>
# 4. ** Model**  - Simple LSTM Keras model<br>
# 5. ** Generator**  - Padds the user input to start generation, and generates text till set word limit is reached<br>

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
from tensorflow.contrib.learn import preprocessing
from keras.callbacks import ModelCheckpoint
import re
from pickle import dump


# ## Preprocessing
# Here we choose text from a particular author. The LSTM will generate text and act like an author in himself.<br>
# **Why single author ??** *This is because the choice of words, styles and other things vary from person to person.*

# In[ ]:


train_df = pd.read_csv('../input/train.csv')
author = train_df[train_df['author'] == 'EAP']["text"]
author[:5]


# ## Tokenization
# We will convert all the text to sequences using Keras tokenizer. The padding of sequences is not needed here. We will see that in next cell

# In[ ]:


max_words = 5000 # Max size of the dictionary
tok = keras.preprocessing.text.Tokenizer(num_words=max_words)
tok.fit_on_texts(author.values)
sequences = tok.texts_to_sequences(author.values)
print(sequences[:5])


# Now we will combine all the above generated tokens or well said as sequences to a single one. This is so that we can apply the sliding windows for training.

# In[ ]:


text = [item for sublist in sequences for item in sublist]
len(text)


# ## Generating sequences for training data
# We create sequencs using sliding window method. On every iteration we move the frame by 1 stride(distance), then we consider the n-1 evements of the frame for training and the nth evement is predicted. Here the value of n is 20.<br>
# <br>
# **Example: **<br>
# **Iteration 1**<br>
# sentence --> "i am a author whose books don't get published"<br>
# trainX --> "i am a author whose books don't get"<br>
# Y --> "published"<br>
# **Iteration 2**<br>
# sentence --> "am a author whose books don't get published easily"<br>
# trainX --> "am a author whose books don't get published"<br>
# Y --> "easily"<br>
# ![img](https://eli.thegreenplace.net/images/2018/markov-chain-window.png)

# In[ ]:


sentence_len = 20
pred_len = 1
train_len = sentence_len - pred_len
seq = []
# Sliding window to generate test and train data
for i in range(len(text)-sentence_len):
    seq.append(text[i:i+sentence_len])
# Reverse dictionary so as to decode tokenized sequences back to words and sentences
reverse_word_map = dict(map(reversed, tok.word_index.items()))
dump(tok, open('tokenizer.pkl', 'wb'))


# In[ ]:


trainX = []
trainy = []
for i in seq:
    trainX.append(i[:train_len])
    trainy.append(i[-1])
#print("Training on : "," ".join(map(lambda x: reverse_word_map[x], trainX[0])),"\nTo predict : "," ".join(map(lambda x: reverse_word_map[x], trainy[0])))


# In[ ]:


model = keras.Sequential()
model.add(keras.layers.Embedding(max_words,100,input_length=train_len))
model.add(keras.layers.LSTM(256, dropout=0.6, recurrent_dropout=0.2))
model.add(keras.layers.Dense(1024,activation="relu"))
model.add(keras.layers.Dense(4999,activation="softmax"))
model.summary()


# In[ ]:


model.compile(optimizer=keras.optimizers.Adam(lr=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
filepath = "./weight_tr5.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
history = model.fit(np.asarray(trainX),
         pd.get_dummies(np.asarray(trainy)),
         epochs = 500,
         batch_size = 10240,
         callbacks = callbacks_list,
         verbose = 2)


# ## Generator
# Here this function iteratively predicts and generates sentences.<br>
# It takes initial words as input and the total length of text to be generated. It padds the input and then predicts the next word. Then appends the predicted word to the input sentence and this continues iteratively till the specified sentence length is reached.

# In[ ]:


def gen(seq,max_len = 20):
    sent = tok.texts_to_sequences([seq])
    #print(sent)
    while len(sent[0]) < max_len:
        sent2 = keras.preprocessing.sequence.pad_sequences(sent[-19:],maxlen=19)
        op = model.predict(np.asarray(sent2).reshape(1,-1))
        sent[0].append(op.argmax()+1)
    return " ".join(map(lambda x : reverse_word_map[x],sent[0]))


# # Testing it up
# ### Can it write a 600 word school literature answer ??

# In[ ]:


start = [("i am curious of",26),("is this why he was ",32),
         ("he was scared of such ",24),("sea was blue like nothing else ",20),
        ("the last day i colud ever enjoy",50),("could you stop doing all this you trouble me a lot",600)]
# Last one was Describe in 600 words
for i in range(len(start)):
    print("<<-- Sentence %d -->>\n"%(i),gen(start[i][0],start[i][1]))

