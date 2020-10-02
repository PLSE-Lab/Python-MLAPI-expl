#!/usr/bin/env python
# coding: utf-8

# In[58]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[59]:


from __future__ import absolute_import, division, print_function

import tensorflow as tf
tf.enable_eager_execution()

import numpy as np
import os
import time


# In[60]:


path_to_file = tf.keras.utils.get_file('98.txt', 'http://www.gutenberg.org/files/98/98.txt')


# In[61]:


# Read, then decode for py2 compat.
text1 = open(path_to_file, 'rb').read().decode()
# length of text is the number of characters in it
print ('Length of text: {} characters'.format(len(text1)))


# In[62]:


# Take a look at the first 250 characters in text
print(text1[:250])


# In[63]:


# The unique characters in the file
vocabulary = sorted(set(text1))
print ('{} unique characters'.format(len(vocabulary)))


# In[64]:


# Creating a mapping from unique characters to indices
char_2_idx = {u:i for i, u in enumerate(vocabulary)}
idx_2_char = np.array(vocabulary)

text_as_int = np.array([char_2_idx[c] for c in text1])


# In[65]:


print('{')
for char,_ in zip(char2idx, range(20)):
    print('  {:4s}: {:3d},'.format(repr(char), char2idx[char]))
print('  ...\n}')


# In[66]:


# Show how the first 13 characters from the text are mapped to integers
print ('{} ---- characters mapped to int ---- > {}'.format(repr(text1[:13]), text_as_int[:13]))


# In[67]:


# The maximum length sentence we want for a single input in characters
seq_length = 100
examples_per_epoch = len(text1)//seq_length

# Create training examples / targets
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

for i in char_dataset.take(5):
  print(idx2char[i.numpy()])


# In[68]:


sequences = char_dataset.batch(seq_length+1, drop_remainder=True)

for item in sequences.take(5):
    print(repr(''.join(idx2char[item.numpy()])))


# In[69]:


def splitinputtarget(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

dataset = sequences.map(splitinputtarget)


# In[70]:


for inputexample, targetexample in  dataset.take(1):
    print ('Input data: ', repr(''.join(idx2char[inputexample.numpy()])))
    print ('Required data:', repr(''.join(idx2char[targetexample.numpy()])))


# In[71]:


for i, (inputidx, targetidx) in enumerate(zip(inputexample[:5], targetexample[:5])):
    print("Step {:4d}".format(i))
    print("  input: {} ({:s})".format(inputidx, repr(idx2char[inputidx])))
    print("  expected output: {} ({:s})".format(targetidx, repr(idx2char[targetidx])))


# In[72]:


# Batch size 
BATCH_SIZE = 64
steps_per_epoch = examples_per_epoch//BATCH_SIZE

# Buffer size to shuffle the dataset
# (TF data is designed to work with possibly infinite sequences, 
# so it doesn't attempt to shuffle the entire sequence in memory. Instead, 
# it maintains a buffer in which it shuffles elements).
BUFFER_SIZE = 10000

dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

dataset


# In[73]:


# Length of the vocabulary in chars
vocab_size = len(vocabulary)

# The embedding dimension 
embedding_dim = 256

# Number of RNN units
rnn_units = 1024


# In[74]:


if tf.test.is_gpu_available():
  rnn= tf.keras.layers.CuDNNGRU
else:
  import functools
  rnn = functools.partial(
    tf.keras.layers.GRU, recurrent_activation='sigmoid')


# In[75]:


def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
  model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, 
                              batch_input_shape=[batch_size, None]),
    rnn(rnn_units,
        return_sequences=True, 
        recurrent_initializer='glorot_uniform',
        stateful=True),
    tf.keras.layers.Dense(vocab_size)
  ])
  return model


# In[76]:


model = build_model(
  vocab_size = len(vocabulary), 
  embedding_dim=embedding_dim, 
  rnn_units=rnn_units, 
  batch_size=BATCH_SIZE)


# In[77]:


for input_example_batch, target_example_batch in dataset.take(1): 
  example_batch_predictions = model(input_example_batch)
  print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")


# In[78]:


model.summary()


# In[79]:


sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
sampled_indices = tf.squeeze(sampled_indices,axis=-1).numpy()


# In[80]:


sampled_indices


# In[81]:


print("Input: \n", repr("".join(idx2char[input_example_batch[0]])))
print()
print("Next Char Predictions: \n", repr("".join(idx2char[sampled_indices ])))


# In[82]:


# Train the Model

def loss(labels, logits):
  return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

example_batch_loss  = loss(target_example_batch, example_batch_predictions)
print("Prediction shape: ", example_batch_predictions.shape, " # (batch_size, sequence_length, vocab_size)") 
print("scalar_loss:      ", example_batch_loss.numpy().mean())


# In[83]:


model.compile(
    optimizer = tf.train.AdamOptimizer(),
    loss = loss)


# In[84]:


# Directory where the checkpoints will be saved
checkpoint_dir = './training_checkpoints'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)


# In[85]:


EPOCHS=10

history = model.fit(dataset.repeat(), epochs=EPOCHS, steps_per_epoch=steps_per_epoch, callbacks=[checkpoint_callback])


# In[86]:


tf.train.latest_checkpoint(checkpoint_dir)

model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)

model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

model.build(tf.TensorShape([1, None]))


model.summary()


# In[87]:


def generate_text(model, start_string):
  # Evaluation step (generating text using the learned model)

  # Number of characters to generate
  num_generate = 200

  # Converting our start string to numbers (vectorizing) 
  input_eval = [char2idx[s] for s in start_string]
  input_eval = tf.expand_dims(input_eval, 0)

  # Empty string to store our results
  text_generated = []

  # Low temperatures results in more predictable text.
  # Higher temperatures results in more surprising text.
  # Experiment to find the best setting.
  temperature = 1.0

  # Here batch size == 1
  model.reset_states()
  for i in range(num_generate):
      predictions = model(input_eval)
      # remove the batch dimension
      predictions = tf.squeeze(predictions, 0)

      # using a multinomial distribution to predict the word returned by the model
      predictions = predictions / temperature
      predicted_id = tf.multinomial(predictions, num_samples=1)[-1,0].numpy()
      
      # We pass the predicted word as the next input to the model
      # along with the previous hidden state
      input_eval = tf.expand_dims([predicted_id], 0)
      
      text_generated.append(idx2char[predicted_id])

  return (start_string + ''.join(text_generated))


# In[88]:


print(generate_text(model, start_string=u"Tale"))
