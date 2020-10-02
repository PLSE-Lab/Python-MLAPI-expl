#!/usr/bin/env python
# coding: utf-8

# # The Knowledge - a RNN covid-19 knowledge base
# The final model will try to build a knowledge base that answer question from user input. The answer is a resume from english papers that already collected from the competition provider.
# 
# ## Steps
# Mainly, there are two big step to reproduce the model.
# 1. Data Preparation
# This step need the most time. in Kaggle it will take at least 4-5 hours to finish.
# The succesful of the model is mainly due the succesfullness of this task. The most importance one is the regex part.
# 2. model build
# the model use RNN model like the one from tensorflow page. however instead of using char, it use word level.
# each epoch need 7 minute to finisg eith GPU or 3,5 haours without gpu. 
# 
# Total time to run with GPU for 20 epoch is 4-5 hours
# 
# ## Pros and cons
# ### Pros
# The model takes all text from the the dataset. To handle this, the data preparation task need to create temporary files and del objects regurally. 
# 
# The model takes only english literature by using pycld2 library.
# 
# ### Cons
# The data need to be tuned with more powerful machine and will takes time. tehr esult is not the best and i belive can be improved better.
# 
# The regular expression need to be tuned to get better words.
# 
# 

# ## Data Preparation
# Using common data preparation routine to properly extract the data.
# ### Import library

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from nltk.stem import WordNetLemmatizer   
lemmatizer = WordNetLemmatizer() 

from nltk.tokenize import word_tokenize 
import re
import json

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# ### Import pycld2 library
# install pycld2 library because it is not available by default in kaggle environment. This is a library that will be used to extract english only papers.

# In[ ]:


get_ipython().system('pip install -U pycld2')


# In[ ]:


import pycld2 as cld2


# Build class to extract english only papers. I assumed that doc with 70% english is an english document.

# In[ ]:


def checkEnglish(text):
    isReliable, textBytesFound, details, vectors = cld2.detect(text, returnVectors=True)
    best = 0
    for each in vectors:
        if (each[-1] == 'en'):
            best += each[1]
    if best != 0 :
        if (best/len(text) > 0.7): # only if the doc has many recognized english
            return True
    return False


# ### Import List of Files
# import list of file from kaggle dataset. this will takes everything without any filetering.

# In[ ]:


import os
json_file = []
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        if filename.split(".")[1] == 'json' :
            json_file.append(os.path.join(dirname, filename))
#         print(os.path.join(dirname, filename))


# In[ ]:


len(json_file)


# ### Text filtering
# Gets each text document that contain the text paper from each file,the do several treatment:
# 1. regex filtering
# 2. transform to lower case
# 3. check english
# 
# Note the check english is the last one, it is because i hope the previous treatment can extract the words better.
# After several trial, the regex filtering can improve the model alot. As better regex result in better and more human readable prediction.

# In[ ]:


total= len(json_file)
i = 0
db = []
for item in json_file:
    i += 1
    with open(item,"r") as output:
        file_1 = json.load(output)
#     print ( "Process Doc # " + str(i) + " left " + str(total - i) + " docs.")
#     file_1= pd.read_json(item,orient = 'index')''

    for texts in file_1["body_text"]:
#         print(texts['text'])
        text_temp = re.sub("(\s\.)|(\.\s)"," . ", texts['text'])
        text_temp = ' '.join([item for sublist in re.findall("(\/{2,}[\w\d\/]+[-.\w\d]+)|([\w\d]+[\s.?,]{0,})|(\d+)|(\w)|(\s[.,?])",
                                        text_temp) for item in sublist if len(item) > 0]
                           ).lower()
        text_temp = re.sub("\s{2,}|,"," ", text_temp)
        text_temp = re.sub("coronavirus","corona virus", text_temp)
        if(checkEnglish(text_temp)):
            db.append(text_temp)


# In[ ]:


len(db)


# > free memory

# In[ ]:


import gc
gc.collect()


# ### Save Memory by Splitting
# To save the memory the text object (db) is splitted and dumped into several files. After this all task will use the load and unload files method. we use pandas as panads support gzip compression with pickle.

# In[ ]:


# split dataset to save memory
i = 0
while i < len(db) :
    pd.DataFrame(db[i:i+100000-1]).to_pickle("/kaggle/working/df_" + str(i), compression='gzip')
    i += 100000
    
# del db object is it is very big
del db


# free memory

# In[ ]:


import gc
gc.collect()


# ### Lemmatize
# We use lemmatize to get the base word.This can be helpful to decrease words count. 

# In[ ]:


# helper function to lemmatize
def text_process(text):
    temp = word_tokenize(text)
    for i in range(len(temp)):
        temp[i] = lemmatizer.lemmatize(temp[i])
    return temp


# In[ ]:


# apply lemitizer
set_temp = ()
for dirname, _, filenames in os.walk('/kaggle/working'):
    for filename in filenames:
        if filename[:3] == 'df_' :
            print("Processing file : " + os.path.join(dirname, filename))
            df_temp = pd.read_pickle(os.path.join(dirname, filename), compression='gzip')
            df_temp = df_temp.apply(lambda x : word_tokenize(x.values[0]), axis=1)
            df_temp.to_pickle(os.path.join(dirname, filename), compression='gzip')
            
                                     
#             json_file.append(os.path.join(dirname, filename))


# In[ ]:


# del df_temp
import gc
gc.collect()


# ### Generate BoW( Bag of Words)
# Generate the set of words for first initialisation of BoW, the sort it to make it understandable. 

# In[ ]:


# generate set of word, in case needed
set_temp = set()
for dirname, _, filenames in os.walk('/kaggle/working'):
    for filename in filenames:
        if filename[:3] == 'df_' :
            print("Processing file : " + os.path.join(dirname, filename))
            df_temp = pd.read_pickle(os.path.join(dirname, filename), compression='gzip')
            df_temp.apply(lambda x : set_temp.update(set(x)))
            del df_temp
#             print(set_temp)
            


# Sort it for easy understanding of the document and set.

# In[ ]:


set_temp = sorted(list(set_temp))


# In[ ]:


len(set_temp)


# get insight of the BoW

# In[ ]:


i = 100
" ".join(set_temp[i:i+100])


# ### Dimension reduction
# As part of the task, we reduce the word list. For the model i pick only words that exist in more than 1% or the documnet.
# by doing this, i hope i can remove the typo and reduce train time.

# In[ ]:


char2count = {u:0 for i, u in enumerate(set_temp)}


# function to help generate BoW

# In[ ]:


def char2count_helper(text):
    global total_count
    global char2count
    total_count += 1
    text = set(text)
    for each in text:
        char2count[each] += 1


# Generate first BoW

# In[ ]:


total_count = 0
for dirname, _, filenames in os.walk('/kaggle/working'):
    for filename in filenames:
        if filename[:3] == 'df_' :
            print("Processing file : " + os.path.join(dirname, filename))
            df_temp = pd.read_pickle(os.path.join(dirname, filename), compression='gzip')
            df_temp.apply(lambda x : char2count_helper(x))
            del df_temp


# In[ ]:


total_count


# Generate final BoW that only contain 2-100% words in docs

# In[ ]:


words_count = 0
char2count_clean = []
for key, item in char2count.items():
    if ((item > total_count * 0.01) and (item < total_count*1) ): 
        words_count += 1
        char2count_clean.append(key)
print(words_count)

#BoW
char2idx = {u:i for i, u in enumerate(char2count_clean)}
# idx2char = np.array(vocab)
idx2char = {i:u for i, u in enumerate(char2count_clean)}


# check some words of bow

# In[ ]:


"viruses" in char2idx


# ## Build the model
# we user RNN word level model to capture the history of a word / words. The train and label file is huge, it is not possible to load and train the data at the same time. The train is 100 seq , and the label also 100 seq. To manage this, i use one single dataset that consist 101 seq each row. A helper class is also created to handle train and label dataset, 

# In[ ]:


import psutil
psutil.virtual_memory()


# ## Import the library

# In[ ]:


import pandas as pd
import numpy as np
import os
import re
# import scipy.sparse as sps
# import sparse
import random as rd
from tqdm.notebook import tqdm
import sys
import pandas as pd
import tensorflow as tf
 

import numpy as np
import os
import time


# ### Set the parameters

# In[ ]:



# Batch size
BATCH_SIZE = 128 * 2

# len char of prediction
lenData = 100

# Buffer size to shuffle the dataset
# (TF data is designed to work with possibly infinite sequences,
# so it doesn't attempt to shuffle the entire sequence in memory. Instead,
# it maintains a buffer in which it shuffles elements).
BUFFER_SIZE = 10000
# Length of the vocabulary in chars
vocab_size = len(char2count_clean)

# The embedding dimension
embedding_dim = 256

# Number of RNN units
rnn_units = 1024


# ### Helper class to generate the dataset
# The class generate dataset for x and label for the model. Each row of the original dataset consist of 101 seq. The details are :
# 1. X is seq 1 to 100
# 2. Label is  seq 2 to 101

# In[ ]:


class helperDf():
  def __init__(self, df, batch_size, coll_size):
    # self.df = df.sample(frac=1).reset_index(drop=True)
    self.df =df
    self.batch_size = batch_size
    self.coll_size = coll_size
    self.pointer = 0
    self.batch_idx = batch_size
    self.epocs = 0

  def _set_index_array(self):
        self.index_array = np.arange(self.getDf)
        if self.shuffle:
            self.index_array = np.random.permutation(self.df)
  
  def __iter__(self):
    self.pointer = 0
    self.reset()
    return self

  def __next__(self):
    if (self.batch_size + self.batch_idx > self.df.shape[0]) : self.reset()
    return self.batch()

  def __len__(self):
    return self.df.shape[0]//self.batch_idx

  def take(self, count):
    counter = 0
    self.pointer = 0
    self.reset()
    while counter < count :
      yield self.batch()
      counter += 1

  def getDf(self):
    return self.df

    #  use list
  # def data(self):
  #   return np.array([sublist[:self.coll_size] for sublist in self.df[self.pointer:self.batch_size]])

  # def target(self):
  #   return np.array([sublist[1:self.coll_size+1] for sublist in self.df[self.pointer:self.batch_size]])


# use pandas df
  def data(self):
    return self.df.loc[self.pointer:(self.batch_size-1),:(self.coll_size-1)].values

  def target(self):
    return self.df.loc[self.pointer:(self.batch_size-1 ),1:self.coll_size].values

  # use numpy array
  # def data(self):
  #   return self.df[self.pointer:(self.batch_size),:(self.coll_size)]

  # def target(self):
  #   return self.df[self.pointer:(self.batch_size ),1:self.coll_size+1]

  def addPointer(self):
    self.pointer = self.batch_size
    self.batch_size += self.batch_idx
    self.epocs +=1

  def reset(self):
    self.pointer = 0
    self.batch_size = self.batch_idx

  def batch(self):
    x = self.data()
    y = self.target()
    self.addPointer()
    return (x,y)

  def count(start=0, step=1):
    # count(10) --> 10 11 12 13 14 ...
    # count(2.5, 0.5) -> 2.5 3.0 3.5 ...
    n = start
    while True:
        yield n
        n += step
        
  def repeat(self, times=None):
    if times is None:
        while True:
            yield self.df
    else:
      for i in xrange(times):
        yield self.df

  def on_epoch_end():
    self._set_index_array()


# ### Helper class for sampling
# previous tries need sampling to reduce the dataset number. it is because the machine can not handle more than 1M rows.

# In[ ]:


class buildData():
  def __init__(self,lenRow):
    self.count = lenRow
    if (lenRow < 1000000) :
        self.maxCount = lenRow
    else:
        self.maxCount = 1000000
    self.df = pd.DataFrame(columns=["input","target"])
    self.df = []
    self.pbar = tqdm(total=self.maxCount)
    self.pbar_total = tqdm(total=self.count)

    print("generating sample (to save memory)")
    self.sampled = sorted(rd.sample(range(self.count),k=self.maxCount))
    self.counter = 0
    self.idxSampled = 0

  def getCounter(self):
    return self.counter

  def getList(self):
    return self.df

  def build(self,text, lentext):
    text = " ".join(text)
    text = re.sub('coronavirus', 'corona virus', text)
    text = text.split()
    step = lentext + 1
    text = [item for item in text if item in char2idx]
    for i in range(0,len(text),step):
      if (len(text[i:]) > step) :
        if (len(self.sampled) <= self.idxSampled) : break
        if (self.counter == self.sampled[self.idxSampled]):
          # for c in self.text[i:i+step]:
            # self.df.append(char2idx[c])
          input = np.array([char2idx[c] for c in text[i:i+step]])
          # target = np.array([char2idx[c] for c in self.text[i+1:i+101]])
          # self.df = self.df.append(pd.DataFrame([[input,target]]))
          self.df.append(input)
          self.pbar.update(1)
          self.idxSampled += 1
        self.counter += 1
        self.pbar_total.update(1)


# ### generate sample dataset

# In[ ]:


# del generate
generate = buildData(774905)
for dirname, _, filenames in os.walk('/kaggle/working'):
    for filename in filenames:
        if filename[:3] == 'df_' :
            print("Processing file : " + os.path.join(dirname, filename))
            # df_temp = 0
            df_temp = pd.read_pickle(os.path.join(dirname, filename), compression='gzip')
            # print()
            df_temp.apply(lambda x : generate.build(x, lenData))
            del df_temp


# ### Generate dataset with Generation Helper Class
# The dataset object is the final dataset to be used for the model.

# In[ ]:


df = pd.DataFrame(generate.getList())
del generate
dataset = helperDf(df, BATCH_SIZE,lenData)
del df


# ### Build model
# Build RNN simple model. it is very basic model as i am not sure that complicated model can be handle with kaglle machine. 

# In[ ]:


def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
  model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim,
                              batch_input_shape=[batch_size, None]),
    tf.keras.layers.GRU(rnn_units,
                        return_sequences=True,
                        stateful=True,
                        recurrent_initializer='glorot_uniform'),
    tf.keras.layers.Dense(vocab_size)
  ])
  return model


# In[ ]:


def loss(labels, logits):
  return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)


# In[ ]:


# with tpu_strategy.scope():
    model = build_model(
      vocab_size = len(char2idx),
      embedding_dim=embedding_dim,
      rnn_units=rnn_units,
      batch_size=BATCH_SIZE)
    model.compile(optimizer='adam', loss=loss)


# In[ ]:


model.summary()


# In[ ]:


checkpoint_dir = '/kaggle/working' 
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)


# ### Train the model
# train the wodel with 10 epoch. 

# In[ ]:


EPOCHS=10
history = model.fit(dataset,steps_per_epoch = len(dataset), epochs=EPOCHS, callbacks=[checkpoint_callback])


# In[ ]:


tf.train.latest_checkpoint(checkpoint_dir)


# ## Prediction
# The model accept word/words and generate 100 words as a paragraph. This model is not the best due to resource limitation. This model can answer almost anything as long as the word/word is included in BoW.

# In[ ]:


model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)

model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

model.build(tf.TensorShape([1, None]))

# model.summary()

import tensorflow as tff
def generate_text(model, start_string):
  # Evaluation step (generating text using the learned model)
  # Number of characters to generate
  num_generate = 100
  start_string = start_string.split()

  # Converting our start string to numbers (vectorizing)
  input_eval = [char2idx[s] for s in start_string if s in char2idx]
  input_eval = tf.expand_dims(input_eval, 0)

  # Empty string to store our results
  text_generated = start_string

  # Low temperatures results in more predictable text.
  # Higher temperatures results in more surprising text.
  # Experiment to find the best setting.
  temperature = 1.0

  # Here batch size == 1
  model.reset_states()
  for i in range(num_generate):
      predictions = model(input_eval)
      # remove the batch dimension
      predictions = tff.squeeze(predictions, 0)

      # using a categorical distribution to predict the character returned by the model
      predictions = predictions / temperature
      predicted_id = tff.random.categorical(predictions, num_samples=1)[-1,0].numpy()

      # We pass the predicted character as the next input to the model
      # along with the previous hidden state
      input_eval = tff.expand_dims([predicted_id], 0)

      text_generated.append(idx2char[predicted_id])
#   print(text_generated)
  return (' '.join(text_generated))


# to try the model, words " corona virus" are given. The prediction is quite useful consider it only took 10 epoch.

# In[ ]:


print(generate_text(model, start_string=u"corona virus "))


# # Finish
# Thanks You
