#!/usr/bin/env python
# coding: utf-8

# # Memory network for question answering with Keras on Babi Tasks(~97% accuracy)

# **In this example, we will build a memory network for question answering.**
# 
# This exemple was simply taken in the book **Deep Learning with Keras,** *by Antonio Gulli and Sujit Pal, 2017.* 
# 
# * Memory networks are a specialized architecture that consist of a memory unit in addition to other learnable units, usually RNNs. Each input updates the memory state and the final output is computed by using the memory along with the output from the learnable unit. 
# * This architecture was suggested in **2014** via the paper ***(Memory Networks, by J. Weston, S. Chopra, and A. Bordes, arXiv:1410.3916, 2014)***. A year later, another paper ***(Towards AIComplete Question Answering: A Set of Prerequisite Toy Tasks, by J. Weston, arXiv:1502.05698, 2015)*** put forward the idea of a synthetic dataset and a standard set of 20 question answering tasks, each with a higher degree of difficulty than the previous one, and applied various deep learning networks to solve these tasks. Of these, the memory network achieved the best results across all the tasks. 
# * This dataset was later made available to the general public through **Facebook's bAbI** project (https://research.fb.com/projects/babi/). The implementation of our memory network resembles most closely the one described in this paper (***End-To-End Memory Networks, by S. Sukhbaatar, J. Weston, and R. Fergus, Advances in Neural Information Processing Systems, 2015)***, in that all the training happens jointly in a single network. It uses the bAbI dataset to solve the first question answering task.
# 
# ### Table of interest:
# * 1. Get the data.
# * 2. Pre-processing.
# * 3. Model Building and Training.
# * 4. Model Evaluation and tests.
# 
# 
# Let's get started.

# In[ ]:


from keras.layers import Input
from keras.layers.core import Activation, Dense, Dropout, Permute
from keras.layers.embeddings import Embedding
from keras.layers.merge import add, concatenate, dot
from keras.layers.recurrent import LSTM
from keras.models import Model, Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
from keras.utils.data_utils import get_file

import collections
import itertools
import nltk
from nltk.tokenize import word_tokenize
import numpy as np
import matplotlib.pyplot as plt
import os
import re
import tarfile
from functools import reduce

get_ipython().run_line_magic('matplotlib', 'inline')


# ## 1. Get the data.
# The bAbI data for the first question answering task consists of 10,000 short sentences each for the training and the test sets. A story consists of two to three sentences, followed by a question. The last sentence in each story has the question and the answer appended to it at the end. The following block of code parses each of the training and test files into a list of triplets of story, question and answer:

# In[ ]:


path = get_file('babi-tasks-v1-2.tar.gz', origin='https://s3.amazonaws.com/text-datasets/babi_tasks_1-20_v1-2.tar.gz')

tar = tarfile.open(path)


# ## 2. Pre-processing
# ### 2.1 Getting User Stories : 
# 
# This is a crucial step where we need to ingest the train and test file and extract stories out of it. The raw format in which text stories, questions and answers are kept is aforementioned. Initially, we will write some helper functions.

# In[ ]:


def tokenize(sent):
    '''
    argument: a sentence string
    returns a list of tokens(words)
    '''
    return word_tokenize(sent)
 
def parse_stories(lines):
    '''
    - Parse stories provided in the bAbI tasks format
    - A story starts from line 1 to line 15. Every 3rd line,
      there is a question &amp;amp;amp;amp;amp; answer.
    - Function extracts sub-stories within a story and
      creates tuples
    '''
    data = []
    story = []
    for line in lines:
        line = line.decode('utf-8').strip()
        nid, line = line.split(' ', 1)
        nid = int(nid)
        if nid == 1:
            # reset story when line ID=1 (start of new story)
            story = []
        if '\t' in line:
            # this line is tab separated Q, A &amp;amp;amp;amp;amp; support fact ID
            q, a, supporting = line.split('\t')
            # tokenize the words of question
            q = tokenize(q)
            # Provide all the sub-stories till this question
            substory = [x for x in story if x]
            # A story ends and is appended to global story data-set
            data.append((substory, q, a))
            story.append('')
        else:
            # this line is a sentence of story
            sent = tokenize(line)
            story.append(sent)
    return data
 
def get_stories(f):
    '''
    argument: filename
    returns list of all stories in the argument data-set file
    '''
    # read the data file and parse 10k stories
    data = parse_stories(f.readlines())
    # lambda func to flatten the list of sentences into one list
    flatten = lambda data: reduce(lambda x, y: x + y, data)
    # creating list of tuples for each story
    data = [(flatten(story), q, answer) for story, q, answer in data]
    return data


# In[ ]:


challenge = 'tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_{}.txt'
print('Extracting stories for the challenge: single_supporting_fact_10k')
# Extracting train stories
train_stories = get_stories(tar.extractfile(challenge.format('train')))
# Extracting test stories
test_stories = get_stories(tar.extractfile(challenge.format('test')))


# In[ ]:


print('Number of training stories:', len(train_stories))
print('Number of test stories:', len(test_stories))
train_stories[0]


# ### 2.2. Vectorization of stories.
# As previously, the input to our RNNs is a sequence of word IDs. So we need to use our vocabulary dictionary to convert the (story, question, and answer) triplet into a sequence of integer word IDs. The next block of code does this and zero pads the resulting sequences of story and answer to the maximum sequence lengths we computed previously. At this point, we have lists of padded word ID sequences for each triplet in the training and test sets:

# In[ ]:


def vectorize_stories(data, word_idx, story_maxlen, query_maxlen):
    # story vector initialization
    X = []
    # query vector initialization
    Xq = []
    # answer vector intialization
    Y = []
    for story, query, answer in data:
        # creating list of story word indices
        x = [word_idx[w] for w in story]
        # creating list of query word indices
        xq = [word_idx[w] for w in query]
        # let's not forget that index 0 is reserved
        y = np.zeros(len(word_idx) + 1)
        # creating label 1 for the answer word index
        y[word_idx[answer]] = 1
        X.append(x)
        Xq.append(xq)
        Y.append(y)
    return (pad_sequences(X, maxlen=story_maxlen),
            pad_sequences(Xq, maxlen=query_maxlen), np.array(Y))


# ### 2.3. Build a vocab.
# Our next step is to run through the texts in the generated lists and build our vocabulary. This should be quite familiar to us by now, since we have used a similar idiom a few times already. Unlike the previous time, our vocabulary is quite small, only 22 unique words, so we will not have any out of vocabulary words:

# In[ ]:


# creating vocabulary of words in train and test set
vocab = set()
for story, q, answer in train_stories + test_stories:
    vocab |= set(story + q + [answer])

# sorting the vocabulary
vocab = sorted(vocab)
 
# Reserve 0 for masking via pad_sequences
vocab_size = len(vocab) + 1
 
# calculate maximum length of story
story_maxlen = max(map(len, (x for x, _, _ in train_stories + test_stories)))
 
# calculate maximum length of question/query
query_maxlen = max(map(len, (x for _, x, _ in train_stories + test_stories)))
 
# creating word to index dictionary
word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
 
# creating index to word dictionary
idx_word = dict((i+1, c) for i,c in enumerate(vocab))
 
# vectorize train story, query and answer sentences/word using vocab
inputs_train, queries_train, answers_train = vectorize_stories(train_stories,
                                                               word_idx,
                                                               story_maxlen,
                                                               query_maxlen)
# vectorize test story, query and answer sentences/word using vocab
inputs_test, queries_test, answers_test = vectorize_stories(test_stories,
                                                            word_idx,
                                                            story_maxlen,
                                                            query_maxlen)


# ### 2.4.Find out the maximum length of the sequence.
# The memory network is based on RNNs, where each sentence in the story and question is treated as a sequence of words, so we need to find out the maximum length of the sequence for our story and question. The following block of code does this. We find that the maximum length of a story is 14 words and the maximum length of a question is just 4 words:

# In[ ]:


print('-------------------------')
print('Vocabulary:\n',vocab,"\n")
print('Vocab size:', vocab_size, 'unique words')
print('Story max length:', story_maxlen, 'words')
print('Query max length:', query_maxlen, 'words')
print('Number of training stories:', len(train_stories))
print('Number of test stories:', len(test_stories))
print('-------------------------')


# ## 3. Define and Train the model
# ### 3.1. Define the model
# The definition is longer than we have seen previously, so it may be convenient to refer to the diagram as you look through the definition:
# 
# ![model.PNG](attachment:model.PNG)
# 
# The model is trained using the RMSprop optimizer and categorical cross-entropy as the loss function:
# 
# Now let's review the code of our model building.

# In[ ]:


# number of epochs to run
train_epochs = 100
# Training batch size
batch_size = 32
# Hidden embedding size
embed_size = 50
# number of nodes in LSTM layer
lstm_size = 64
# dropout rate
dropout_rate = 0.30


# In[ ]:


# placeholders
input_sequence = Input((story_maxlen,))
question = Input((query_maxlen,))
 
print('Input sequence:', input_sequence)
print('Question:', question)
 
# encoders
# embed the input sequence into a sequence of vectors
input_encoder_m = Sequential()
input_encoder_m.add(Embedding(input_dim=vocab_size,
                              output_dim=embed_size))
input_encoder_m.add(Dropout(dropout_rate))
# output: (samples, story_maxlen, embedding_dim)
 
# embed the input into a sequence of vectors of size query_maxlen
input_encoder_c = Sequential()
input_encoder_c.add(Embedding(input_dim=vocab_size,
                              output_dim=query_maxlen))
input_encoder_c.add(Dropout(dropout_rate))
# output: (samples, story_maxlen, query_maxlen)
 
# embed the question into a sequence of vectors
question_encoder = Sequential()
question_encoder.add(Embedding(input_dim=vocab_size,
                               output_dim=embed_size,
                               input_length=query_maxlen))
question_encoder.add(Dropout(dropout_rate))
# output: (samples, query_maxlen, embedding_dim)
 
# encode input sequence and questions (which are indices)
# to sequences of dense vectors
input_encoded_m = input_encoder_m(input_sequence)
print('Input encoded m', input_encoded_m)
input_encoded_c = input_encoder_c(input_sequence)
print('Input encoded c', input_encoded_c)
question_encoded = question_encoder(question)
print('Question encoded', question_encoded)
 
# compute a 'match' between the first input vector sequence
# and the question vector sequence
# shape: `(samples, story_maxlen, query_maxlen)
match = dot([input_encoded_m, question_encoded], axes=-1, normalize=False)
print(match.shape)
match = Activation('softmax')(match)
print('Match shape', match)
 
# add the match matrix with the second input vector sequence
response = add([match, input_encoded_c])  # (samples, story_maxlen, query_maxlen)
response = Permute((2, 1))(response)  # (samples, query_maxlen, story_maxlen)
print('Response shape', response)
 
# concatenate the response vector with the question vector sequence
answer = concatenate([response, question_encoded])
print('Answer shape', answer)
 
answer = LSTM(lstm_size)(answer)  # Generate tensors of shape 32
answer = Dropout(dropout_rate)(answer)
answer = Dense(vocab_size)(answer)  # (samples, vocab_size)
# we output a probability distribution over the vocabulary
answer = Activation('softmax')(answer)


# In[ ]:


# build the final model
model = Model([input_sequence, question], answer)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
              metrics=['accuracy'])
 
print(model.summary())


# We train this network for 100 epochs with a batch size of 32 and achieve an accuracy of over 96% on the validation set:
# 
# ### 3.2. Training the Model

# In[ ]:


# start training the model
history = model.fit([inputs_train, queries_train], answers_train, 
                    batch_size=batch_size, 
                    epochs=train_epochs,
                    validation_data=([inputs_test, queries_test], answers_test))
 
# save model
model.save('model.h5')


# ## 4. Model evaluation.
# 
# ### 4.2. Visualization of history.

# In[ ]:


def plotmodelhistory(history): 
    fig, axs = plt.subplots(1,2,figsize=(15,5)) 
    # summarize history for accuracy
    axs[0].plot(history.history['accuracy']) 
    axs[0].plot(history.history['val_accuracy']) 
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy') 
    axs[0].set_xlabel('Epoch')
    axs[0].legend(['train', 'validate'], loc='upper left')
    # summarize history for loss
    axs[1].plot(history.history['loss']) 
    axs[1].plot(history.history['val_loss']) 
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss') 
    axs[1].set_xlabel('Epoch')
    axs[1].legend(['train', 'validate'], loc='upper left')
    plt.show()

# list all data in history
print(history.history.keys())

plotmodelhistory(history)


# ### 4.2. Test of model.
# We ran the model against the first 10 stories from our test set to verify how good the predictions were:

# In[ ]:


for i in range(0,10):
    current_inp = test_stories[i]
    current_story, current_query, current_answer = vectorize_stories([current_inp], word_idx, story_maxlen, query_maxlen)
    current_prediction = model.predict([current_story, current_query])
    current_prediction = idx_word[np.argmax(current_prediction)]
    print(' '.join(current_inp[0]), ' '.join(current_inp[1]), '| Prediction:', current_prediction, '| Ground Truth:', current_inp[2])
    print("--------------------------------------------------------------")


# ### References:
# * **Deep Learning with Keras**, *by Antonio Gulli and Sujit Pal, 2017.*
# * https://becominghuman.ai/q-a-system-deep-learning-2-2-c0ad60800e3
# * https://appliedmachinelearning.blog/2019/05/01/developing-factoid-question-answering-system-on-babi-facebook-data-set-python-keras-part-1/
# * https://appliedmachinelearning.blog/2019/05/02/building-end-to-end-memory-network-for-question-answering-system-on-babi-facebook-data-set-python-keras-part-2/
# 

# **Hope that you find this notebook helpful. More to come.**
# 
# **Please upvote this, to keep me motivate for doing better.**
# 
# **Thanks.**
# 
