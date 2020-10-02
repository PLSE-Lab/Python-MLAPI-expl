#!/usr/bin/env python
# coding: utf-8

# ### Neural Machine Translation using seq2seq

# please upvote it if you like it

# In[ ]:


import os,sys
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Model
from keras.layers import Input,LSTM,GRU,Dense,Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical


# In[ ]:


# config variables
BATCH_SIZE = 64
EPOCHS = 100
LATENT_DIM = 256
NUM_SAMPLES = 10000
MAX_SEQUENCE_LENGTH = 100
MAX_NUM_WORDS = 20000
EMBEDDING_DIM = 100


# In[ ]:


input_texts = [] # sentence in original language
target_texts = [] # sentence in target language
target_texts_inputs = [] # sentence in target language offset by 1


# In[ ]:


n = 0
for line in open('/kaggle/input/hin.txt'):
    if n!=10:
        print(line)
        n+=1


# In[ ]:


# load in the data
t = 0
for line in open('/kaggle/input/hin.txt'):
  t+=1
  if t>NUM_SAMPLES:
    break
  # input and target are seperated by '\t'
  if '\t' not in line:
    continue
  input_text, translation = line.split('\t')
  target_text = translation + ' <eos>'
  target_text_input = '<sos> ' + translation
  input_texts.append(input_text)
  target_texts.append(target_text)
  target_texts_inputs.append(target_text_input)
print('num samples:',len(input_texts))


# In[ ]:


tokenizer_inputs = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer_inputs.fit_on_texts(input_texts)
input_sequences = tokenizer_inputs.texts_to_sequences(input_texts)


# In[ ]:


# word to index mapping for input language
word2idx_inputs = tokenizer_inputs.word_index
print('Found %s unique input tokens.'%len(word2idx_inputs))


# In[ ]:


# max length input seq
max_len_input = max(len(s) for s in input_sequences)


# In[ ]:


# tokenize the outputs
tokenizer_outputs = Tokenizer(num_words=MAX_NUM_WORDS, filters='')
tokenizer_outputs.fit_on_texts(target_texts + target_texts_inputs)
target_sequences = tokenizer_outputs.texts_to_sequences(target_texts)
target_sequences_inputs = tokenizer_outputs.texts_to_sequences(target_texts_inputs)


# In[ ]:


# word to index mapping for output language
word2idx_outputs = tokenizer_outputs.word_index
print('Found %s unique output tokens.'%len(word2idx_outputs))


# In[ ]:


num_words_output = len(word2idx_outputs)+1


# In[ ]:


# max length output seq
max_len_target = max(len(s) for s in target_sequences)


# In[ ]:


# padding the sequences
encoder_inputs = pad_sequences(input_sequences, maxlen=max_len_input)
print('Encoder data shape:',encoder_inputs.shape)

decoder_inputs = pad_sequences(target_sequences_inputs, maxlen=max_len_target, padding='post')
print('Decoder data shape:',decoder_inputs.shape)

decoder_targets = pad_sequences(target_sequences, maxlen=max_len_target, padding='post')


# In[ ]:


get_ipython().system('wget http://nlp.stanford.edu/data/glove.6B.zip')


# In[ ]:


get_ipython().system('unzip glove.6B.zip')


# In[ ]:


# loading pre-trained word vectors
word2vec = {}
with open('glove.6B.%sd.txt'%EMBEDDING_DIM) as f:
  for line in f:
    values = line.split()
    word = values[0]
    vec = np.asarray(values[1:],dtype='float32')
    word2vec[word] = vec
print('Found %s word vectors.'%len(word2vec))


# In[ ]:


# prepare embedding matrix
num_words = min(MAX_NUM_WORDS, len(word2idx_inputs)+1)
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word2idx_inputs.items():
  if i < MAX_NUM_WORDS:
    embedding_vector = word2vec.get(word)
    if embedding_vector is not None:
      embedding_matrix[i] = embedding_vector


# In[ ]:


# creating embedding layer
embedding_layer = Embedding(num_words,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=max_len_input,
                            trainable=False)


# In[ ]:


decoder_targets_one_hot = np.zeros((len(input_texts),
                                   max_len_target,
                                   num_words_output),dtype='float32')
for i,d in enumerate(decoder_targets):
  for t,word in enumerate(d):
    decoder_targets_one_hot[i, t, word] = 1


# In[ ]:


# build the model
encoder_inputs_placeholder = Input(shape=(max_len_input,))
x = embedding_layer(encoder_inputs_placeholder)
encoder = LSTM(LATENT_DIM, return_state=True, dropout=0.5)
encoder_outputs, h, c = encoder(x)


# In[ ]:


encoder_states = [h,c] # keeping state to pass into decoder


# In[ ]:


decoder_inputs_placeholder = Input(shape=(max_len_target,))
decoder_embedding = Embedding(num_words_output, LATENT_DIM)
decoder_inputs_x = decoder_embedding(decoder_inputs_placeholder)


# In[ ]:


decoder_lstm = LSTM(LATENT_DIM, return_sequences=True, return_state=True,
                    dropout=0.5)
decoder_outputs,_,_ = decoder_lstm(decoder_inputs_x,
                                   initial_state=encoder_states)


# In[ ]:


decoder_dense = Dense(num_words_output,activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)


# In[ ]:


model = Model([encoder_inputs_placeholder,decoder_inputs_placeholder],
              decoder_outputs)
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# In[ ]:


r = model.fit([encoder_inputs, decoder_inputs],decoder_targets_one_hot,
              batch_size=BATCH_SIZE,
              epochs=70,
              validation_split=0.2)


# In[ ]:


# plot the results
plt.plot(r.history['loss'],label='loss')
plt.plot(r.history['val_loss'],label='val_loss')
plt.legend()
plt.show()


# In[ ]:


encoder_model = Model(encoder_inputs_placeholder, encoder_states)

decoder_state_input_h = Input(shape=(LATENT_DIM,))
decoder_state_input_c = Input(shape=(LATENT_DIM,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]


# In[ ]:


decoder_inputs_single = Input(shape=(1,))
decoder_inputs_single_x = decoder_embedding(decoder_inputs_single)


# In[ ]:


decoder_outputs, h, c = decoder_lstm(decoder_inputs_single_x,
                                     initial_state=decoder_states_inputs)


# In[ ]:


decoder_states = [h, c]
decoder_outputs = decoder_dense(decoder_outputs)


# In[ ]:


decoder_model = Model([decoder_inputs_single]+decoder_states_inputs,
                      [decoder_outputs]+decoder_states)


# In[ ]:


# mapping indices back to words
idx2word_eng = {v:k for k,v in word2idx_inputs.items()}
idx2word_trans = {v:k for k,v in word2idx_outputs.items()}


# In[ ]:


def decode_sequence(input_seq):
  states_value = encoder_model.predict(input_seq)
  # generating empty target seq of len 1
  target_seq = np.zeros((1,1))
  target_seq[0,0] = word2idx_outputs['<sos>']
  
  eos = word2idx_outputs['<eos>']
  output_sentence = []
  for _ in range(max_len_target):
    output_tokens,h,c = decoder_model.predict([target_seq]+states_value)
    
    idx = np.argmax(output_tokens[0,0,:])
    if eos == idx:
      break
    
    word = ''
    if idx>0:
      word = idx2word_trans[idx]
      output_sentence.append(word)
    
    target_seq[0,0] = idx
    states_value = [h,c]
    
  return ' '.join(output_sentence) 


# In[ ]:


i = np.random.choice(len(input_texts))
input_seq = encoder_inputs[i:i+1]
translation = decode_sequence(input_seq)
print('Input:',input_texts[i])
print('Translation:',translation)


# In[ ]:




