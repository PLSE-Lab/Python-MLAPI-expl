#!/usr/bin/env python
# coding: utf-8

# # About This Kernel
# 
# This is the training kernel for the [Tensorflow 2.0 Question Answering](https://www.kaggle.com/c/tensorflow2-question-answering) Competition. I also created [an inference kernel](https://www.kaggle.com/xhlulu/tf2-qa-lstm-inference-kernel/) that shows how to use the trained model to make a submission.
# 
# My approach is heavily inspired from [Oleg's kernel](https://www.kaggle.com/opanichev/tf2-0-qa-binary-classification-baseline), i.e. I break down each document into parts corresponding to each of the candidate long answers. Then, I label each of the long answer to be `1` if it is the true long answer, or `0` if not. For each of row, I also include the question and the `example_id`. Then, I train a LSTM to predict that label.
# 
# The sections are broken down in the following way:
# 
# 1. **Build Dataset:** Two different utility functions for creating ready-to-use dataframes for both training and testing data. The `build_train` function only loads a subset of all training data, and set a sampling rate `S`, such that we only keep `1/S` of all the negative-labelled data (and keep all positive-labelled data).
# 
# 2. **Preprocessing:** Train a keras `Tokenizer` to encode the text and questions into list of integers (tokenization), then pad them to a fixed length to form a single numpy array for text and one for questions.
# 
# 3. **Modelling:**
#     * Generate a fasttext embedding (directly using [FAIR's Official Python API](https://github.com/facebookresearch/fastText/tree/master/python)) based on the index of the tokenizer. 
#     * Build two 2-layer bidirectional LSTM; one to read the questions, and one to read the text. 
#     * Concatenate the output of the LSTM and feed in 2-layer fully-connected neural networks.
#     * Predict the binary output using Sigmoid activation.
#     * Optimize using Adam and binary cross-entropy loss.
# 
# 4. **Save Model:** Due to the submission time limit, it is better to import the model we just trained in a separate kernel to infer and submit. In the inference kernel, we first remove all the rows with less than 0.5 confidence, then for each `example_id` we only keep the one with the highest confidence to be the output.

# In[ ]:


import os
import json
import gc
import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm_notebook as tqdm
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, SpatialDropout1D, concatenate, Masking
from tensorflow.keras.layers import LSTM, Bidirectional, GlobalMaxPooling1D, Dropout
from tensorflow.keras.preprocessing import text, sequence
from tqdm import tqdm_notebook as tqdm
import fasttext


# # Build Dataset

# In[ ]:


def build_train(train_path, n_rows=300000, sampling_rate=20):
    with open(train_path) as f:
        processed_rows = []

        for i in tqdm(range(n_rows)):
            line = f.readline()
            if not line:
                break

            line = json.loads(line)

            text = line['document_text'].split(' ')
            question = line['question_text']
            annotations = line['annotations'][0]
            example_id = line['example_id']

            for i, candidate in enumerate(line['long_answer_candidates']):
                label = i == annotations['long_answer']['candidate_index']

                start = candidate['start_token']
                end = candidate['end_token']

                if label or (i % sampling_rate == 0):
                    processed_rows.append({
                        'text': " ".join(text[start:end]),
                        'is_long_answer': label,
                        'question': question,
                        'example_id': example_id,
                        'annotation_id': annotations['annotation_id']
                    })

        train = pd.DataFrame(processed_rows)
        
        return train


# In[ ]:


def build_test(test_path):
    with open(test_path) as f:
        processed_rows = []

        for line in tqdm(f):
            line = json.loads(line)

            text = line['document_text'].split(' ')
            question = line['question_text']
            example_id = line['example_id']

            for candidate in line['long_answer_candidates']:
                start = candidate['start_token']
                end = candidate['end_token']

                processed_rows.append({
                    'text': " ".join(text[start:end]),
                    'question': question,
                    'example_id': example_id,
                    'sequence': f'{start}:{end}'

                })

        test = pd.DataFrame(processed_rows)
    
    return test


# In[ ]:


directory = '/kaggle/input/tensorflow2-question-answering/'
train_path = directory + 'simplified-nq-train.jsonl'
test_path = directory + 'simplified-nq-test.jsonl'

train = build_train(train_path)
test = build_test(test_path)


# In[ ]:


print(train.shape)
print("Total number of positive labels:", train.is_long_answer.sum())
print("Number of anno:", train.is_long_answer.sum())
train.head()


# In[ ]:


print(test.shape)
test.head()


# # Preprocessing

# In[ ]:


def compute_text_and_questions(train, test, tokenizer):
    train_text = tokenizer.texts_to_sequences(train.text.values)
    train_questions = tokenizer.texts_to_sequences(train.question.values)
    test_text = tokenizer.texts_to_sequences(test.text.values)
    test_questions = tokenizer.texts_to_sequences(test.question.values)
    
    train_text = sequence.pad_sequences(train_text, maxlen=300)
    train_questions = sequence.pad_sequences(train_questions)
    test_text = sequence.pad_sequences(test_text, maxlen=300)
    test_questions = sequence.pad_sequences(test_questions)
    
    return train_text, train_questions, test_text, test_questions


# In[ ]:


tokenizer = text.Tokenizer(lower=False, num_words=80000)

for text in tqdm([train.text, test.text, train.question, test.question]):
    tokenizer.fit_on_texts(text.values)


# In[ ]:


train_target = train.is_long_answer.astype(int).values


# In[ ]:


train_text, train_questions, test_text, test_questions = compute_text_and_questions(train, test, tokenizer)
del train


# # Modelling

# In[ ]:


def build_embedding_matrix(tokenizer, path):
    embedding_matrix = np.zeros((tokenizer.num_words + 1, 300))
    ft_model = fasttext.load_model(path)

    for word, i in tokenizer.word_index.items():
        if i >= tokenizer.num_words - 1:
            break
        embedding_matrix[i] = ft_model.get_word_vector(word)
    
    return embedding_matrix


# In[ ]:


def build_model(embedding_matrix):
    embedding = Embedding(
        *embedding_matrix.shape, 
        weights=[embedding_matrix], 
        trainable=False, 
        mask_zero=True
    )
    
    q_in = Input(shape=(None,))
    q = embedding(q_in)
    q = SpatialDropout1D(0.2)(q)
    q = Bidirectional(LSTM(100, return_sequences=True))(q)
    q = GlobalMaxPooling1D()(q)
    
    
    t_in = Input(shape=(None,))
    t = embedding(t_in)
    t = SpatialDropout1D(0.2)(t)
    t = Bidirectional(LSTM(150, return_sequences=True))(t)
    t = GlobalMaxPooling1D()(t)
    
    hidden = concatenate([q, t])
    hidden = Dense(300, activation='relu')(hidden)
    hidden = Dropout(0.5)(hidden)
    hidden = Dense(300, activation='relu')(hidden)
    hidden = Dropout(0.5)(hidden)
    
    out1 = Dense(1, activation='sigmoid')(hidden)
    
    model = Model(inputs=[t_in, q_in], outputs=out1)
    model.compile(loss='binary_crossentropy', optimizer='adam')

    return model


# In[ ]:


path = '/kaggle/input/fasttext-crawl-300d-2m-with-subword/crawl-300d-2m-subword/crawl-300d-2M-subword.bin'
embedding_matrix = build_embedding_matrix(tokenizer, path)


# In[ ]:


model = build_model(embedding_matrix)
model.summary()


# In[ ]:


train_history = model.fit(
    [train_text, train_questions], 
    train_target,
    epochs=5,
    validation_split=0.2,
    batch_size=1024
)


# # Save Model

# In[ ]:


# saving
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)


# In[ ]:


model.save('model.h5')

