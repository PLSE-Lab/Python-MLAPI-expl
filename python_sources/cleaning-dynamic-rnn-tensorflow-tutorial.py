#!/usr/bin/env python
# coding: utf-8

# ### WARNING: Contains Obscene Words
# 
# To go along with this tutorial, read the code comments.
# 
# ## PART 0: Loading the data

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # graphs
get_ipython().run_line_magic('matplotlib', 'inline')
from tqdm import tqdm # progress bar
import copy # deepcopy not refrencing
import re # Regex
from collections import Counter # counter

# Machine learning
import tensorflow as tf

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[2]:


data_train = pd.read_csv('../input/train.csv')
data_test = pd.read_csv('../input/test.csv')


# In[3]:


data_train.head(10)


# In[4]:


data_test.head(10)


# In[5]:


# data
train_sentences = data_train['comment_text'].fillna("_na_").values.tolist()
test_sentences = data_test['comment_text'].fillna("_na_").values.tolist()

# labels
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y = data_train[list_classes].values

labels_ = np.zeros([len(data_train), 6])
for i in range(len(y)):
    for j in range(6):
        labels_[i][j] = y[i][j]


# In[6]:


print(len(train_sentences))
print(len(test_sentences))


# ## PART 1: Cleaning the text

# In[7]:


# some parameters
maxlen_sent = 200 # maximum length of the sentence, explained below


# In[8]:


# I will be using the previous method I used in my notebook
# https://www.kaggle.com/ybonde/cleaning-word2vec-lstm-working

# making a list of total sentences
total_ = copy.deepcopy(train_sentences)
total_.extend(test_sentences)
print('[*]Training Sentences:', len(train_sentences))
print('[*]Test Sentences:', len(test_sentences))
print('[*]Total Sentences:', len(total_))

# converting the text to lower
for i in range(len(total_)):
    total_[i] = str(total_[i]).lower()

# convert into list of words remove unecessary characters, split into words,
# no hyphens and other special characters, split into words
def sentence_to_wordlist(raw):
    clean = re.sub("[^a-zA-Z0-9]"," ", raw)
    words = clean.split()
    return words

# tokenising the lowered corpus
clean_ = []
for i in tqdm(range(len(total_))):
    clean_.append(sentence_to_wordlist(total_[i]))


# In[9]:


# Getting the tokens
tokens_ = []
for s in clean_:
    tokens_.extend(s)
print("[*]total number of tokens:",len(tokens_))
tokens_ = sorted(list(set(tokens_)))
print("[*]total number of unique tokens:",len(tokens_))


# In[10]:


# making the word2id dictionary
word2id = dict((c,i) for i,c in enumerate(tokens_))
id2word = dict((i,c) for c,i in word2id.items())


# In[11]:


# now we need to make a graph to understand the length distribution
# number of sentence whose lengths is less than 100
less_than_100 = 0
less_than_200 = 0
for s in clean_:
    if len(s) <= 100:
        less_than_100 += 1
    if len(s) <= 200:
        less_than_200 += 1
# thus 82% is <= 100
print(less_than_100/len(clean_) * 100)
# thus 93% is <= 200
print(less_than_200/len(clean_) * 100)
# equal to 1
equal_to_1 = 0
for s in clean_:
    if len(s) == 1:
        equal_to_1 += 1
print(equal_to_1)


# In[12]:


# Thus we select 200 as our length of data, that is all the data will be clipped
# to last 200 words. Note that we previously defined maxlen_sent
clean_last_ = [s[-maxlen_sent:] for s in clean_]

# converting the text to numbers
data_ = np.array([[word2id[i] for i in s] for s in clean_last_])


# In[13]:


print(total_[12])
print()
print(clean_[12])
print()
print(clean_last_[12])
print()
print(data_[12])


# In[14]:


# we now calculate the sequences lengths of each sentence as we will pass those through
# the tf.nn.dynamic_rnn() function
seqlens = np.array([len(s) for s in data_])

# removing the sentences with length less than 1
zero_length_indices = [i for i,c in enumerate(data_) if len(c) == 0]
# we will save this for later use

# removing those indices from the data
data_ = np.array([data_[i] for i in range(len(data_)) if i not in zero_length_indices])
seqlens_ = np.array([seqlens[i] for i in range(len(seqlens)) if i not in zero_length_indices])
labels_ = np.array([labels_[i] for i in range(len(labels_)) if i not in zero_length_indices])


# In[15]:


# all the sentences with 0 length are in testing set, so we don't need to worry when we slice he 
print(len(train_sentences) < zero_length_indices[0])


# In[16]:


# now we pad the sequences to proper length, other wise the tensorflow won't be able
# to input it via placeholder, though these zeros won't be processed.
for i in tqdm(range(len(data_))):
    data_[i] = np.hstack([data_[i], np.zeros(maxlen_sent - len(data_[i]), dtype = np.int32)])


# In[17]:


# thus the padding was succesfull
print(data_[0].shape)


# In[18]:


# now splitting the data into training, validation and testing data
# training
train_data_ = data_[:int(len(train_sentences) * 0.9)]
train_labels_ = labels_[:int(len(train_sentences) * 0.9)]
train_seqlens_ = seqlens_[:int(len(train_sentences) * 0.9)]

# validation
train_data_v = data_[int(len(train_sentences) * 0.9): len(train_sentences)]
train_labels_v = labels_[int(len(train_sentences) * 0.9): len(train_sentences)]
train_seqlens_v = seqlens_[int(len(train_sentences) * 0.9): len(train_sentences)]

# testing
test_data_ = data_[len(train_sentences):]
test_seqlens_ = seqlens_[len(train_sentences):]

# reconfig the zero indices list
zero_length_indices_corrected = np.array(zero_length_indices) - len(train_sentences)

'''
Now this is a pretty good time to save these arrays as they are, so we can load them later and
use as per requirement. You can train your own word2vec model, and then use it. Or as shown
here, directly feed them into tensorflow, using supervised embeddings.
'''


# ## PART 2: Build the model
# We are using tensorflow as our machine learning API library. For learning how to make a dynamic_rnn graph go [here](https://github.com/yashbonde/basic-utils/blob/master/Dynamic%20LSTM%20in%20Tensorflow.ipynb).

# In[29]:


# parameters
batch_size = 256 # batch size
n_epochs = 1000 # number of training epochs
num_classes = 6 # number of output classes
e_dim = 128 # dimension of embeddings
vocab_size = len(tokens_) # total number of words
n_hidden_lstm = 256 # hidden size of LSTM network
disp_step = 10 # disply status after every disp_step epochs


# In[20]:


# Defining the placeholders for IO
_x = tf.placeholder(tf.int32, [batch_size, maxlen_sent])
_y = tf.placeholder(tf.float32, [batch_size, num_classes])
_seqlens = tf.placeholder(tf.int32, [batch_size])


# In[21]:


with tf.name_scope("embeddings"):
    embeddings = tf.Variable(tf.random_uniform([vocab_size, e_dim]), name = 'embedding_matrix')
    embed = tf.nn.embedding_lookup(embeddings, _x)


# In[22]:


with tf.variable_scope("lstm"):
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_lstm, forget_bias = 1.0)
    outputs, states = tf.nn.dynamic_rnn(lstm_cell, embed,
                                        sequence_length = _seqlens,
                                        dtype = tf.float32)


# In[23]:


# defining the dense layers weights and biases
# first layer
W1 = tf.truncated_normal([n_hidden_lstm, 128], name = 'W1')
b1 = tf.truncated_normal([128], name = 'b1')

# second layer
W2 = tf.truncated_normal([128, num_classes], name = 'W2')
b2 = tf.truncated_normal([num_classes], name = 'b2')


# In[24]:


op_1 = tf.nn.relu(tf.matmul(states[1], W1) + b1) # output dense layer 1
y_op = tf.matmul(op_1, W2) + b2 # output dense layer 2


# In[25]:


# defining the cross_entropy, the competition uses log loss so that is what we are
# going to implememt the same
sigmoid_diff = tf.nn.sigmoid_cross_entropy_with_logits(labels = _y, logits = y_op)
cross_entropy = tf.reduce_mean(sigmoid_diff)


# ## PART 3: Training the model

# In[26]:


train_step = tf.train.AdamOptimizer().minimize(cross_entropy)


# In[27]:


def get_batch(batch_size, data_x, data_y, data_seqlen):
    instance_indices = list(range(len(data_x)))
    np.random.shuffle(instance_indices)
    batch = instance_indices[:batch_size]
    x = [data_x[i] for i in batch]
    y = [data_y[i] for i in batch]
    seqlens = [data_seqlen[i] for i in batch]
    return x,y,seqlens


# In[33]:


# tensorflow session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# loss_list
loss_list = []
val_loss_list = []

for e in range(n_epochs):
    # get batch for training
    x_batch, y_batch, seqlen_batch = get_batch(batch_size, train_data_,
                                               train_labels_, train_seqlens_)
    # make feed dictionary
    feed_dict = {_x:x_batch, _y:y_batch, _seqlens:seqlen_batch}
    # find cross_entropy
    ce = sess.run(cross_entropy, feed_dict = feed_dict)
    loss_list.append(ce)
    # run one step
    sess.run(train_step, feed_dict = feed_dict)
    if e%disp_step == 0:
        # getting a batch for validation task
        val_x_batch, val_y_batch, val_seqlen_batch = get_batch(batch_size, train_data_v,
                                                               train_labels_v,
                                                               train_seqlens_v)
        # making feed dictionary
        feed_dict_val = {_x: val_x_batch, _y:val_y_batch, _seqlens:val_seqlen_batch}
        # getting loss
        val_loss = sess.run(cross_entropy, feed_dict = feed_dict_val)
        val_loss_list.append(val_loss)
        # printing the result
        print("Epoch: {0}, Validation Loss: {1}".format(e, val_loss))
        # training on validation data
        sess.run(train_step, feed_dict = feed_dict_val)


# In[34]:


plt.figure(figsize = (10,10))
plt.plot(loss_list)


# In[35]:


plt.figure(figsize = (10,10))
plt.plot(val_loss_list)


# Clearly we need to optimize the model, but the results are sufficient as a tutorial on how to use Dynamic LSTM for analysis of text.
