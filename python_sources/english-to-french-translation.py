#!/usr/bin/env python
# coding: utf-8

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


from matplotlib import pyplot as plt
import seaborn as sns


# In[ ]:


lines =[]
with open('/kaggle/input/frenchenglish-bilingual-pairs/fra-eng/fra.txt','r') as f:
    lines.extend(f.readline() for i in range(5))


# In[ ]:


lines


# In[ ]:


df = pd.read_csv('/kaggle/input/frenchenglish-bilingual-pairs/fra-eng/fra.txt', sep = '\t', header = None)


# In[ ]:


df.head()


# In[ ]:


df.tail()


# In[ ]:


#https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.rename.html
df.columns = ['English','French']


# In[ ]:


df


# basic eda

# In[ ]:


#How to calculate number of words in a string in DataFrame:
#https://stackoverflow.com/a/37483537/4084039
word_count = df['English'].str.split().apply(len).value_counts()


# In[ ]:


type(word_count)


# In[ ]:


word_count


# In[ ]:


word_dict = dict(word_count)
word_dict = dict(sorted(word_dict.items(), key=lambda kv: kv[1]))


# In[ ]:


word_dict


# In[ ]:


index  = np.arange(len(word_dict))


# In[ ]:


values1 = word_dict.values()


# In[ ]:


values1


# In[ ]:


plt.figure(figsize=(20,5))
plt.bar(index,values1)
plt.xlabel('length of sentences in english')
plt.ylabel('occurances')
plt.xticks(index,word_dict.keys())
plt.show()


# In[ ]:


word_count = df['French'].str.split().apply(len).value_counts()


# In[ ]:


word_count


# In[ ]:


word_dict = dict(word_count)
word_dict = dict(sorted(word_dict.items(), key=lambda kv: kv[1]))


# In[ ]:


word_dict


# In[ ]:


index  = np.arange(len(word_dict))
values2 = word_dict.values()


# In[ ]:


plt.figure(figsize=(20,5))
plt.bar(index,values2)
plt.xlabel('length of sentences in english')
plt.ylabel('occurances')
plt.xticks(index,word_dict.keys())
plt.show()


# In[ ]:


len(word_dict)


# In[ ]:


word_count = df['French'].str.split().apply(len)


# In[ ]:


word_count


# In[ ]:


french_lenght_of_sentences = word_count.values


# In[ ]:


word_count = df['English'].str.split().apply(len)


# In[ ]:


english_lenght_of_sentences = word_count.values


# In[ ]:


plt.boxplot([english_lenght_of_sentences,french_lenght_of_sentences])
plt.xticks([1,2],['English','French'])
plt.ylabel('Lenght of sentences')
plt.grid()
plt.show()


# In[ ]:


plt.figure(figsize=(10,3))
sns.kdeplot(english_lenght_of_sentences,label="english sentences")
sns.kdeplot(french_lenght_of_sentences,label="french sentences")
plt.legend()
plt.show()


# In[ ]:


import numpy as np
import tensorflow as tf
import keras
import tqdm as tqdm
from tensorflow.keras.layers import Dense,concatenate,Activation,Dropout,Input,LSTM,Embedding,Flatten,Conv1D,BatchNormalization
from tensorflow.keras.models import Model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


# In[ ]:


batch_size = 64  # Batch size for training.
epochs = 20  # Number of epochs to train for.
latent_dim = 256  # Latent dimensionality of the encoding space.
num_samples = 15437  # Number of samples to train on.
# Path to the data txt file on disk.
data_path = '/kaggle/input/frenchenglish-bilingual-pairs/fra-eng/fra.txt'


# In[ ]:


# Vectorize the data.
input_texts = []
target_texts = []
input_characters = set()
target_characters = set()


# In[ ]:


with open(data_path, 'r', encoding='utf-8') as f:
    lines = f.read().split('\n')


# In[ ]:


for line in lines[: min(num_samples, len(lines) - 1)]:
    input_text, target_text, = line.split('\t')
    # We use "tab" as the "start sequence" character
    # for the targets, and "\n" as "end sequence" character.
    target_text = '\t' + target_text + '\n'
    input_texts.append(input_text)
    target_texts.append(target_text)
    for char in input_text:
        if char not in input_characters:
            input_characters.add(char)
    for char in target_text:
        if char not in target_characters:
            target_characters.add(char)


# In[ ]:


target_characters


# In[ ]:


input_characters = sorted(list(input_characters))
target_characters = sorted(list(target_characters))
num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)
max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])


# In[ ]:


print('Number of samples:', len(input_texts))
print('Number of unique input tokens:', num_encoder_tokens)
print('Number of unique output tokens:', num_decoder_tokens)
print('Max sequence length for inputs:', max_encoder_seq_length)
print('Max sequence length for outputs:', max_decoder_seq_length)


# In[ ]:


input_token_index = dict(
    [(char, i) for i, char in enumerate(input_characters)])
target_token_index = dict(
    [(char, i) for i, char in enumerate(target_characters)])


# In[ ]:


encoder_input_data = np.zeros(
    (len(input_texts), max_encoder_seq_length, num_encoder_tokens),
    dtype='float32')
decoder_input_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')
decoder_target_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')


# In[ ]:


for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    for t, char in enumerate(input_text):
        encoder_input_data[i, t, input_token_index[char]] = 1.
    encoder_input_data[i, t + 1:, input_token_index[' ']] = 1.
    for t, char in enumerate(target_text):
        # decoder_target_data is ahead of decoder_input_data by one timestep
        decoder_input_data[i, t, target_token_index[char]] = 1.
        if t > 0:
            # decoder_target_data will be ahead by one timestep
            # and will not include the start character.
            decoder_target_data[i, t - 1, target_token_index[char]] = 1.
    decoder_input_data[i, t + 1:, target_token_index[' ']] = 1.
    decoder_target_data[i, t:, target_token_index[' ']] = 1.


# In[ ]:


# Define an input sequence and process it.
encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]


# In[ ]:


# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None, num_decoder_tokens))
# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                     initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)


# In[ ]:


# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)


# In[ ]:


model.summary()


# In[ ]:


# Run training
model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2)


# In[ ]:


# Save model
model.save('s2s.h5')


# In[ ]:


from keras.models import load_model

model = load_model("s2s.h5")


# In[ ]:



# Next: inference mode (sampling).
# Here's the drill:
# 1) encode input and retrieve initial decoder state
# 2) run one step of decoder with this initial state
# and a "start of sequence" token as target.
# Output will be the next target token
# 3) Repeat with the current target token and current states

# Define sampling models
encoder_model = Model(encoder_inputs, encoder_states)


# In[ ]:


decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)


# In[ ]:


# Reverse-lookup token index to decode sequences back to
# something readable.
reverse_input_char_index = dict(
    (i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict(
    (i, char) for char, i in target_token_index.items())


# In[ ]:


def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, target_token_index['\t']] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == '\n' or
           len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]

    return decoded_sentence


# In[ ]:


import warnings
warnings.simplefilter("ignore")


# In[ ]:


from nltk.translate.bleu_score import sentence_bleu
for seq_index in range(10,12):
    # Take one sequence (part of the training set)
    # for trying out decoding.
    input_seq = encoder_input_data[seq_index: seq_index + 1]
    decoded_sentence = decode_sequence(input_seq)
    print('-')
    print('Input sentence:', input_texts[seq_index])
    print('Decoded sentence:', decoded_sentence)
    
    score = sentence_bleu(input_texts[seq_index], target_texts[seq_index])
    print('from original sentence',score)
    
    score = sentence_bleu(input_texts[seq_index], decoded_sentence)
    print('from converted sentence',score)


# In[ ]:


df[10:12]


# # now instead of using characters we will be using word embeddings

# In[ ]:


df  = df[0:12000]


# In[ ]:


df.shape


# In[ ]:


df['Decoder'] = df['French'] + ' \n'


# In[ ]:


df['French']  = '\t ' + df['French']  + ' \n'


# In[ ]:


df[0:10]


# In[ ]:


df=  df.applymap(str.lower)


# In[ ]:


df[0:10]


# In[ ]:


#y = df['Decoder']
#X = df.drop(['Decoder'],axis =1)


# In[ ]:


#print(y.shape)
#print(X.shape)


# In[ ]:


#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,random_state=42)


# In[ ]:


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


# In[ ]:


max_english_sentence_lenght = 0
for i in range(df['English'].shape[0]):
    a = len(df['English'].iloc[i].split())
    if a > max_english_sentence_lenght:
        max_english_sentence_lenght =a


# In[ ]:


max_english_sentence_lenght


# In[ ]:


tokenizer = Tokenizer( filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
tokenizer.fit_on_texts(df['English'])


# In[ ]:


sequence_train = tokenizer.texts_to_sequences(df['English'])
#sequence_test = tokenizer.texts_to_sequences(X_test['English'])


# In[ ]:


english_dic = tokenizer.word_index


# In[ ]:


english_vocab = len(tokenizer.word_index) + 1


# In[ ]:


english_vocab


# In[ ]:


sequence_train[0:10]


# In[ ]:


#need to cover the case if state that is present in test data may not be present in train data by padding although X_train 
# consists all the state
#english_test_text = pad_sequences(sequence_test, max_english_sentence_lenght,padding ='post')
english_train_text = pad_sequences(sequence_train, max_english_sentence_lenght,padding = 'post')


# In[ ]:


english_train_text[0:10]


# In[ ]:


max_french_sentence_lenght = 0
for i in range(df['French'].shape[0]):
    a = len(df['French'].iloc[i].split())
    if a > max_french_sentence_lenght:
        max_french_sentence_lenght =a


# In[ ]:


max_french_sentence_lenght


# In[ ]:


tokenizer = Tokenizer( filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~')
tokenizer.fit_on_texts(df['French'])


# In[ ]:


sequence_train = tokenizer.texts_to_sequences(df['French'])
#sequence_test = tokenizer.texts_to_sequences(X_test['French'])


# In[ ]:


decoder_train = tokenizer.texts_to_sequences(df['Decoder'])
#decoder_test = tokenizer.texts_to_sequences(y_test)


# In[ ]:


french_dic = tokenizer.word_index


# In[ ]:


french_dic


# In[ ]:


french_vocab = len(tokenizer.word_index) +1


# In[ ]:


french_vocab


# In[ ]:


sequence_train[0:10]


# In[ ]:


decoder_train[0:10]


# In[ ]:


#need to cover the case if state that is present in test data may not be present in train data by padding although X_train 
# consists all the state
#french_test_text = pad_sequences(sequence_test, max_french_sentence_lenght,padding = 'post')
french_train_text = pad_sequences(sequence_train, max_french_sentence_lenght,padding = 'post')


# In[ ]:


#need to cover the case if state that is present in test data may not be present in train data by padding although X_train 
# consists all the state
#decoder_test_text = pad_sequences(decoder_test, max_french_sentence_lenght,padding = 'post')
decoder_train_text = pad_sequences(decoder_train, max_french_sentence_lenght,padding = 'post')


# In[ ]:


french_train_text[0:10]


# In[ ]:


decoder_train_text[0:10]


# In[ ]:


shape_of_decoder = decoder_train_text.shape


# In[ ]:


np.amax(decoder_train_text)


# In[ ]:


shape_of_decoder[0]


# In[ ]:


decoder_encoded_array = np.zeros((12000,11,5202))


# In[ ]:


for i in range(12000):
    for j in range(11):
        decoder_encoded_array[i,j,decoder_train_text[i,j]] =1 


# In[ ]:


#decoder_input_array = np.zeros((8000,10,4063))


# In[ ]:


#for i in range(8000):
 #   for j in range(10):
  #      decoder_input_array[i,j,french_train_text[i,j]] =1


# In[ ]:


#encoder_input_array = np.zeros((8000,5,2056))


# In[ ]:


#for i in range(8000):
 #   for j in range(5):
  #      encoder_input_array[i,j,english_train_text[i,j]] =1


# In[ ]:


#decoder_encoded_array


# In[ ]:


from tensorflow.keras.layers import Dense,concatenate,Activation,Dropout,Input,LSTM,Embedding,Flatten,Conv1D,BatchNormalization
from keras.models import Model


# In[ ]:


encoder_input = Input(shape =(5,))#adding voc


# In[ ]:


#encoder_embedding = Embedding(english_vocab,256)(encoder_input)
#encoder_lstm,state_h,state_c = LSTM(256,return_state = True)(encoder_embedding)


# In[ ]:


encoder_embedding = Embedding(english_vocab,256)(encoder_input)
encoder = LSTM(256, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_embedding)


# In[ ]:


encoder_states = [state_h,state_c]


# In[ ]:


state_h.shape


# In[ ]:


decoder_input =Input(shape=(11,))


# In[ ]:


#decoder_embedding = Embedding(french_vocab,256)(decoder_input)
#decoder_lstm = LSTM(256,return_sequences = True)(decoder_embedding , initial_state =encoder_states)
#decoder_dense_output = Dense(french_vocab,activation ='softmax')(decoder_lstm)


# In[ ]:


decoder_embedding = Embedding(french_vocab,256)(decoder_input)
decoder_lstm = LSTM(256, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding,initial_state=encoder_states)
decoder_dense = Dense(french_vocab, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)


# In[ ]:


model = Model([encoder_input,decoder_input],decoder_outputs)


# In[ ]:


model.summary()


# In[ ]:


import tensorflow as tf
tf.keras.utils.plot_model(
model,
to_file="model1.png",
show_shapes=False,
show_layer_names=True,
rankdir="TB",
expand_nested=False,
dpi=96,
)


# In[ ]:


model.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics = ['accuracy'])


# In[ ]:


batch_size =100
epochs = 10


# In[ ]:


model.fit([english_train_text, french_train_text], decoder_encoded_array,
          batch_size=batch_size,
          epochs=epochs,
          validation_split = 0.2)


# similarly we can get the sentence in french from english as we have done using decode_sequence fuction in characters representation

# source : https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html
