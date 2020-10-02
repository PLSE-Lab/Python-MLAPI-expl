#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import re
import time
import os
from keras.preprocessing.sequence import pad_sequences
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding, Bidirectional, concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
print(os.listdir("../input"))


# Resources:
# 
# https://github.com/Currie32/Chatbot-from-Movie-Dialogue
# 
# https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html
# 
# https://medium.com/@dev.elect.iitd/neural-machine-translation-using-word-level-seq2seq-model-47538cba8cd7
# https://github.com/devm2024/nmt_keras/blob/master/base.ipynb

# **Loading the data**
# 
# **Matching conv lines to ID**

# In[ ]:


# Load the data
lines = open('../input/cornell-moviedialog-corpus/movie_lines.txt', encoding='utf-8', errors='ignore').read().split('\n')
conv_lines = open('../input/cornell-moviedialog-corpus/movie_conversations.txt', encoding='utf-8', errors='ignore').read().split('\n')


# In[ ]:


# The sentences that we will be using to train our model.
lines[:5]


# In[ ]:


# The sentences' ids, which will be processed to become our input and target data.
conv_lines[:5]


# In[ ]:


# Create a dictionary to map each line's id with its text
id2line = {}
for line in lines:
    _line = line.split(' +++$+++ ')
    if len(_line) == 5:
        id2line[_line[0]] = _line[4]


# In[ ]:


# Create a list of all of the conversations' lines' ids.
convs = []
for line in conv_lines[:-1]:
    _line = line.split(' +++$+++ ')[-1][1:-1].replace("'","").replace(" ","")
    convs.append(_line.split(','))


# In[ ]:


#id and conversation sample
for k in convs[300]:
    print (k, id2line[k])


# ** Creating qns inputs and answer targets**

# In[ ]:


# Sort the sentences into questions (inputs) and answers (targets)
questions = []
answers = []

for conv in convs:
    for i in range(len(conv)-1):
        questions.append(id2line[conv[i]])
        answers.append(id2line[conv[i+1]])


# In[ ]:


# Compare lengths of questions and answers
print(len(questions))
print(len(answers))


# Cleaning text

# In[ ]:


def clean_text(text):
    '''Clean text by removing unnecessary characters and altering the format of words.'''

    text = text.lower()
    
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "that is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"how's", "how is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"n'", "ng", text)
    text = re.sub(r"'bout", "about", text)
    text = re.sub(r"'til", "until", text)
    text = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", text)
    text = " ".join(text.split())
    return text


# In[ ]:


# Clean the data
clean_questions = []
for question in questions:
    clean_questions.append(clean_text(question))
    
clean_answers = []    
for answer in answers:
    clean_answers.append(clean_text(answer))


# In[ ]:


r = np.random.randint(1,len(questions))
print ('original text......')
for i in range(r, r+3):
    print(questions[i])
    print(answers[i])
    print()
print ('cleaned text......')
for i in range(r, r+3):
    print(clean_questions[i])
    print(clean_answers[i])
    print()


# **Selecting qns and answers with appropriate length (<20 words)**

# In[ ]:


# Find the length of sentences
lengths = []
for question in clean_questions:
    lengths.append(len(question.split()))
for answer in clean_answers:
    lengths.append(len(answer.split()))
# Create a dataframe so that the values can be inspected
lengths = pd.DataFrame(lengths, columns=['counts'])


# In[ ]:


lengths['counts'].describe()


# In[ ]:


print(np.percentile(lengths, 80))
print(np.percentile(lengths, 85))
print(np.percentile(lengths, 90))
print(np.percentile(lengths, 95))
print(np.percentile(lengths, 99))


# In[ ]:


# Remove questions and answers that are shorter than 1 word and longer than 20 words.
min_line_length = 2
max_line_length = 20

# Filter out the questions that are too short/long
short_questions_temp = []
short_answers_temp = []

for i, question in enumerate(clean_questions):
    if len(question.split()) >= min_line_length and len(question.split()) <= max_line_length:
        short_questions_temp.append(question)
        short_answers_temp.append(clean_answers[i])

# Filter out the answers that are too short/long
short_questions = []
short_answers = []

for i, answer in enumerate(short_answers_temp):
    if len(answer.split()) >= min_line_length and len(answer.split()) <= max_line_length:
        short_answers.append(answer)
        short_questions.append(short_questions_temp[i])
        
print(len(short_questions))
print(len(short_answers))


# In[ ]:


r = np.random.randint(1,len(short_questions))

for i in range(r, r+3):
    print(short_questions[i])
    print(short_answers[i])
    print()


#  **Preprocessing for word based model**

# In[ ]:


#choosing number of samples
num_samples = 60000  # Number of samples to train on.
short_questions = short_questions[:num_samples]
short_answers = short_answers[:num_samples]


# In[ ]:


#append start and end tokens for the answers
short_answers2 = []
for ans in short_answers:
    ans = '<SOS> ' + ans + ' <EOS>'
    short_answers2.append(ans)


# In[ ]:


# Create a dictionary for the frequency of the vocabulary
# Create 
vocab = {}
for question in short_questions:
    for word in question.split():
        if word not in vocab:
            vocab[word] = 1
        else:
            vocab[word] += 1
            
for answer in short_answers2:
    for word in answer.split():
        if word not in vocab:
            vocab[word] = 1
        else:
            vocab[word] += 1


# In[ ]:


# Remove rare words from the vocabulary.
# We will aim to replace fewer than 5% of words with <UNK>
# You will see this ratio soon.
threshold = 20
count = 0
for k,v in vocab.items():
    if v >= threshold:
        count += 1


# In[ ]:


print("Size of total vocab:", len(vocab))
print("Size of vocab we will use:", count)


# In[ ]:


#we will create dictionaries to provide a unique integer for each word.
vocab_to_int = {}

word_num = 0
for word, count in vocab.items():
    if count >= threshold:
        vocab_to_int[word] = word_num
        word_num += 1


# In[ ]:


# Add the unique tokens (pad and unknown vocab) to the vocabulary dictionaries.
codes = ['<PAD>','<UNK>']
for code in codes:
    code_int = len(vocab_to_int)
    vocab_to_int[code] = code_int


# In[ ]:


#switch <PAD> value to well's value of 0 for padding purposes later
print (vocab_to_int['<PAD>'])
for i, v in vocab_to_int.items():
    if v == 0:
        print (i)


# In[ ]:


for i, v in vocab_to_int.items():
    if v == 0:
        vocab_to_int[i] = vocab_to_int['<PAD>']
vocab_to_int['<PAD>'] = 0


# In[ ]:


# Create dictionaries to map the unique integers to their respective words.
# i.e. an inverse dictionary for vocab_to_int.
int_to_vocab = {v_i: v for v, v_i in vocab_to_int.items()}


# In[ ]:


# Check the length of the dictionaries.
print(len(vocab_to_int))
print(len(int_to_vocab))


# In[ ]:


# Convert the text to integers with paddings
# Replace any words that are not in the respective vocabulary with <UNK> 
questions_int = []
for question in short_questions:
    ints = []
    for word in question.split():
        if word not in vocab_to_int:
            ints.append(vocab_to_int['<UNK>'])
        else:
            ints.append(vocab_to_int[word])
    questions_int.append(ints)
    
answers_int = []
for answer in short_answers2:
    ints = []
    for word in answer.split():
        if word not in vocab_to_int:
            ints.append(vocab_to_int['<UNK>'])
        else:
            ints.append(vocab_to_int[word])
    answers_int.append(ints)
# Check the lengths
print(len(questions_int))
print(len(answers_int))


# In[ ]:


# Calculate what percentage of all words have been replaced with <UNK># Calcul 
word_count = 0
unk_count = 0

for question in questions_int:
    for word in question:
        if word == vocab_to_int["<UNK>"]:
            unk_count += 1
        word_count += 1
    
for answer in answers_int:
    for word in answer:
        if word == vocab_to_int["<UNK>"]:
            unk_count += 1
        word_count += 1
    
unk_ratio = round(unk_count/word_count,4)*100
    
print("Total number of words:", word_count)
print("Number of times <UNK> is used:", unk_count)
print("Percent of words that are <UNK>: {}%".format(round(unk_ratio,3)))


# In[ ]:


#include padding
encoder_input_data = pad_sequences(questions_int, maxlen=max_line_length, value=vocab_to_int['<PAD>'], padding='post') #pad to max_line_length
decoder_input_data = pad_sequences(answers_int, maxlen=max_line_length+2, value=vocab_to_int['<PAD>'], padding='post') #pad to max_line_length + start and end tokens


# In[ ]:


#decoder target is 1 timestep (word) ahead of decoder input, in a 3-d array
decoder_target_data = np.zeros(
    (len(answers_int), max_line_length+2, len(vocab_to_int)), #memory error occurs after 3500
    dtype='float32')


# In[ ]:


for i, target_seq in enumerate(answers_int):
    for t, seq in enumerate(target_seq):
        if t > 0:
            # decoder_target_data will be ahead by one timestep
            # and will not include the start character.
            decoder_target_data[i, t - 1, seq] = 1.


# In[ ]:


print (encoder_input_data.shape)
print (decoder_input_data.shape)
print (decoder_target_data.shape)


# In[ ]:


#include embedding size
embedding_size = 200


# In[ ]:


encoder_inputs = Input(shape=(None,))
en_x=  Embedding(len(vocab_to_int), embedding_size)(encoder_inputs)
encoder = Bidirectional(LSTM(100, return_state=True))
encoder_outputs, state_h_1, state_c_1, state_h_2, state_c_2 = encoder(en_x)


# In[ ]:


state_h = concatenate([state_h_1, state_h_2], axis=1)
state_c = concatenate([state_c_1, state_c_1], axis=1)
# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]


# In[ ]:


#non-bidirectional approach
# encoder_inputs = Input(shape=(None,))
# en_x=  Embedding(len(vocab_to_int), embedding_size)(encoder_inputs)
# encoder = LSTM(50, return_state=True)
# encoder_outputs, state_h, state_c = encoder(en_x)
# # We discard `encoder_outputs` and only keep the states.
# encoder_states = [state_h, state_c]


# In[ ]:


# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None,))
dex=  Embedding(len(vocab_to_int), embedding_size)
final_dex= dex(decoder_inputs)
decoder_lstm = LSTM(200, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(final_dex,
                                     initial_state=encoder_states)
decoder_dense = Dense(len(vocab_to_int), activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc']) #sparse_categorical_crossentropy as labels in a single integer array


# In[ ]:


model.summary()


# In[ ]:


#early stopping & saving
# earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
mcp_save = ModelCheckpoint('Best_weights_movie_word.hdf5', save_best_only=True, monitor='val_loss', mode='min')
# reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, min_delta=1e-4, mode='min')


# In[ ]:


# from keras.models import load_model
# model = load_model('../input/movie30/s2s_movie_word_30.h5')


# In[ ]:


model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=128,
          epochs=40,
          validation_split=0.05,
          callbacks= [mcp_save]
#           callbacks=[earlyStopping, mcp_save, reduce_lr_loss] #change from 0.1)))
         )


# In[ ]:


model.save('s2s_movie_word_40.h5')


# In[ ]:


encoder_model = Model(encoder_inputs, encoder_states)
encoder_model.summary()


# In[ ]:


#Create sampling model
decoder_state_input_h  = Input(shape=(200,))
decoder_state_input_c = Input(shape=(200,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

final_dex2= dex(decoder_inputs)

decoder_outputs2, state_h2, state_c2 = decoder_lstm(final_dex2, initial_state=decoder_states_inputs)
decoder_states2 = [state_h2, state_c2]
decoder_outputs2 = decoder_dense(decoder_outputs2)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs2] + decoder_states2)


# In[ ]:


# For non-bidirectional 
# #Create sampling model
# decoder_state_input_h  = Input(shape=(50,))
# decoder_state_input_c = Input(shape=(50,))
# decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

# final_dex2= dex(decoder_inputs)

# decoder_outputs2, state_h2, state_c2 = decoder_lstm(final_dex2, initial_state=decoder_states_inputs)
# decoder_states2 = [state_h2, state_c2]
# decoder_outputs2 = decoder_dense(decoder_outputs2)
# decoder_model = Model(
#     [decoder_inputs] + decoder_states_inputs,
#     [decoder_outputs2] + decoder_states2)


# In[ ]:


def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)
    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1,1))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0] = vocab_to_int['<SOS>']

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = int_to_vocab[sampled_token_index]
        decoded_sentence += ' '+sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == '<EOS>' or
           len(decoded_sentence) > 52):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1,1))
        target_seq[0, 0] = sampled_token_index

        # Update states
        states_value = [h, c]

    return decoded_sentence


# In[ ]:


for i in range(50):
    seq_index = np.random.randint(1, len(encoder_input_data))
    # Take one sequence (part of the training set)
    # for trying out decoding.
    input_seq = encoder_input_data[seq_index: seq_index + 1]
    decoded_sentence = decode_sequence(input_seq)
    print('-')
    print('Input sentence:', short_answers[seq_index: seq_index + 1])
    print('Decoded sentence:', decoded_sentence)

