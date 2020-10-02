#!/usr/bin/env python
# coding: utf-8

# # Mini project 2: Text generation with TensorFlow
# 
# This mini project is really guided by various tutorials on RNN text generation, mainly Max Woolf's and Tensforlow. I had already trained and used textgenrnn, which is a Python module for text generation built on top of Keras/Tensorflow but wanted to learn how to build my own model and train it.
# 
# I tried to give a detailed account of each step, which was a good way to make sure I understood what was going on, at least at a high level.
# 
# There's also a little bit of NLTK experimentation, which I read is a clunky but powerful Python NLP library.
# 
# https://colab.research.google.com/drive/1mMKGnVxirJnqDViH7BDJxFqWrsXlPSoK <br>
# https://www.tensorflow.org/tutorials

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


# # Training data selection and exploration
# Let's import TensorFlow and json so that I can look at my data via the json file that contains all the database metadata.

# In[ ]:


import tensorflow as tf
import json
from pandas.io.json import json_normalize
metadata = pd.read_json("../input/gutenberg-dammit/gutenberg-dammit-files/gutenberg-metadata.json")
print(metadata.shape)
metadata.head()


# In[ ]:


metadata.loc[1000:1050]


# I tried to browse the json file with Pandas's **groupby** command to see if it would show me say all the unique 'title' values for each author. 
# 
# > group = metadata.groupby('Author') <br>
# > byauthor = group.apply(lambda x: x['Title'].unique())
# 
# But I kept getting the error **unhashable type: 'list'**. Apparently a list object cannot be used as key because it's not hashable.
# 
# Soooo I chose a text 'at random', based on interest and what some crude segmenting of the file showed me. I picked Jack London's *The Sea Wolf* as my training dataset. Here's what the 'text as string' version looks like.
# 

# In[ ]:


with open('../input/gutenberg-dammit/gutenberg-dammit-files/010/01074.txt', 'r') as text:
    data = text.read().replace('\n', ' ')
    print(data)


# Below is the text 'as is'. I somehow couldn't perform an analysis on this version of the text because it registered as an 'open file object' and not as strings.

# In[ ]:


open_text = open("../input/gutenberg-dammit/gutenberg-dammit-files/010/01074.txt", "r")
print(open_text.read())


# In[ ]:


print ('Length of text: {} characters'.format(len(data)))


# Then let's import the NLTK and do some text analysis.

# In[ ]:


import nltk
from nltk.tokenize import sent_tokenize
tokenized_text=sent_tokenize(data)
print(tokenized_text)


# Let's first tokenize the text.

# In[ ]:


from nltk.tokenize import word_tokenize
tokenized_word=word_tokenize(data)
print(tokenized_word)


# Now let's look at the frequency distribution.

# In[ ]:


from nltk.probability import FreqDist
fdist = FreqDist(tokenized_word)
print(fdist)


# In[ ]:


#What are the two most common words?
fdist.most_common(2)


# In[ ]:


#What are the ten most common?
fdist.most_common(10)


# In[ ]:


#What are the 10 least common?
fdist.most_common()[-10:]


# Now let's import **matplotlib** to do some data viz.

# In[ ]:


import matplotlib.pyplot as plt
fdist.plot(30,cumulative=False) #show me the 30 most common words
plt.show()


# In[ ]:


fdist.plot(50,cumulative=False) #show me the 50 most common words
plt.show()


# Now let's look at stop words. Stop words are commonly used words that are like 'noise' in the text.

# In[ ]:


from nltk.corpus import stopwords
stop_words=set(stopwords.words("english"))
print(stop_words)


# In[ ]:


filtered_sent=[]  #create an empty list for the text empty of stop words
for w in tokenized_text:     #go through every word in the text
    if w not in stop_words:  #if it's not a stop word
        filtered_sent.append(w)  #add it to the list
print("Tokenized Sentence:",tokenized_text)  #print the text


# In[ ]:


print("Filtered Sentence:",filtered_sent) #print the text without stop words


# I can't see much of a difference between the tokenized and filtered texts. Let's print out their characters to see if it actually did anything.

# In[ ]:


print ('Length of text: {} characters'.format(len(data))) 
print ('Length of filtered text: {} characters'.format(len(filtered_sent)))


# In[ ]:


vocab = sorted(set(data))
print ('{} unique characters'.format(len(vocab)))


# # Vectorization
# 
# The first step is to vectorize the text, i.e. map strings to a numerical representation.

# In[ ]:


#Each unique character gets assigned a number
char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

text_as_int = np.array([char2idx[c] for c in data])

#print the table
print('{')
for char,_ in zip(char2idx, range(20)):
    print('  {:4s}: {:3d},'.format(repr(char), char2idx[char]))
print('  ...\n}')


# In[ ]:


#print the first 13 characters as an array
print ('{} ---- characters mapped to int ---- > {}'.format(repr(data[:13]), text_as_int[:13]))


# # Data prep
# Some data prep is needed before we start training the model.
# 
# The first step is to divide the text into sequences (**seq_length**) with the **tf.data.Dataset.from_tensor_slices** function, which turns the text vector into a stream of character indices.
# 
# The idea, as I understand it, is to use batches as training sets. For each set, we're asking the model: given all the characters seen so far, what is the most probable next character?
# 

# In[ ]:


# The maximum length sentence we want for a single input in characters
seq_length = 100
examples_per_epoch = len(data)//(seq_length+1)

# Create training examples / targets
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

# Print the first 5 characters
for i in char_dataset.take(5): 
  print(idx2char[i.numpy()])


# It works! Now let's try with bigger batches. Here the **char_dataset** is turned into a sequence using **batch**. We'll print 10 sequences.

# In[ ]:


# Turn the character set into a sequence using .batch
sequences = char_dataset.batch(seq_length+1, drop_remainder=True)

# Print 10 sequences
for item in sequences.take(10):
  print(repr(''.join(idx2char[item.numpy()])))


# Now that we can divide up the text into sequences, we need to **map** the text to input and target sequences. The targets contain the same length of text except shifted to one character to the right. 
# 
# Example: <br>
# Text: 'Hello'<br>
# Input seq: 'Hell'<br>
# Target seq: 'ello'
# 
# We can do this (map the text) by applying a function to each sequence.

# In[ ]:


# Input & target text mapping function
def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

# Apply the function to each of my sequences
dataset = sequences.map(split_input_target)

# Print one mapped sequence
for input_example, target_example in  dataset.take(1):
  print ('Input data: ', repr(''.join(idx2char[input_example.numpy()])))
  print ('Target data:', repr(''.join(idx2char[target_example.numpy()])))


# Each character of the sequence is processed one step at a time. In the first step, the model will process "T" and then try to predict "h", and then do the same thing for all the following characters. But being a RNN, it also keeps a memory of the previous steps in addition to the current input character. 
# 
# Let's see if it does this correctly.

# In[ ]:


# for each of the first five characters of the sequence, print out the input and the target
for i, (input_idx, target_idx) in enumerate(zip(input_example[:5], target_example[:5])):
    print("Step {:4d}".format(i))
    print("  input: {} ({:s})".format(input_idx, repr(idx2char[input_idx])))
    print("  expected output: {} ({:s})".format(target_idx, repr(idx2char[target_idx])))


# Now, let's shuffle the data and pack it into batches. <br>
# (I tried to understand why the data needs to be shuffled and it seems like it has to do with the statefulness of the model. If not shuffled, the samples are reset at each sequence and not propagated to the next bach. Not sure why this matters but there is an interesting discussion here: http://philipperemy.github.io/keras-stateful-lstm/)

# In[ ]:


# Batch size
BATCH_SIZE = 64

# Buffer size to shuffle the dataset
# (TF data is designed to work with possibly infinite sequences,
# so it doesn't attempt to shuffle the entire sequence in memory. Instead,
# it maintains a buffer in which it shuffles elements).
BUFFER_SIZE = 10000

dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

dataset


# # Building the model
# Now let's build the model.
# 
# We'll use **tf.keras.Sequential** to build the model, which groups a linear stack of layers. We'll use 3 layers for this model:
# 
# **tf.keras.layers.Embedding** --> the input layer (a table that will map the number of each character to a vector) <br>
# **tf.keras.layers.GRU** --> a type of RNN (could also have used a LSTM layer)<br>
# **tf.keras.layers.Dense** --> the output later 

# In[ ]:


# Length of the vocabulary in characters
vocab_size = len(vocab)

# The embedding dimension
embedding_dim = 256

# Number of RNN units
rnn_units = 1024

# Build the model using 3 layers.
def build_model(vocab_size, embedding_dim, rnn_units, batch_size): #let's specify the dimensions
  model = tf.keras.Sequential([  #used to embed all three layers
    tf.keras.layers.Embedding(vocab_size, embedding_dim, #input later
                              batch_input_shape=[batch_size, None]),
    tf.keras.layers.GRU(rnn_units, #RNN
                        return_sequences=True,
                        stateful=True,  #it needs tobe stateful! 
                        recurrent_initializer='glorot_uniform'),
    tf.keras.layers.Dense(vocab_size) #output layer
  ])
  return model

model = build_model(
  vocab_size = len(vocab),
  embedding_dim=embedding_dim,
  rnn_units=rnn_units,
  batch_size=BATCH_SIZE)


# Ok, the model's built! Now let's try it.
# 
# First we need to check the shape of the ouput.

# In[ ]:


for input_example_batch, target_example_batch in dataset.take(1):
  example_batch_predictions = model(input_example_batch)
  print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")


# Now let's look at a model summary.

# In[ ]:


model.summary()


# Let's look at a sample.
# 

# In[ ]:


sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
sampled_indices = tf.squeeze(sampled_indices,axis=-1).numpy()
sampled_indices


# These are the predictions for the next character in each timestep of the sample. Let's decode the array to see what the current state of the model's prediction is. 

# In[ ]:


print("Input: \n", repr("".join(idx2char[input_example_batch[0]])))
print()
print("Next Char Predictions: \n", repr("".join(idx2char[sampled_indices ])))


# Lol. Let's train that model.

# # Training the model
# First step: attach a loss function using **tf.keras.losses.sparse_categorical_crossentropy**, which computes the sparse categorical crossentropy loss.

# In[ ]:


# Function that sets the from_logits flag
def loss(labels, logits):
  return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

example_batch_loss = loss(target_example_batch, example_batch_predictions)
print("Prediction shape: ", example_batch_predictions.shape, " # (batch_size, sequence_length, vocab_size)")
print("scalar_loss:      ", example_batch_loss.numpy().mean())


# Configure the training procedure with using the **tf.keras.Model.compile** method (which compiles an optimizer, the loss function, the loss weights, and other stuff). Here the optimizer is 'adam'.

# In[ ]:


model.compile(optimizer='adam', loss=loss)


# Now I need to configure checkpoints to ensure they are saved during the training. Checkpoints are used to save a model or weights at every determined interval so that the model can be loaded later to continue the training from the state saved.
# 
# https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ModelCheckpoint

# In[ ]:


# Directory where the checkpoints will be saved
checkpoint_dir = './training_checkpoints'

# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)


# Now the actual training!!! We'll keep it at 10 epochs first so that it doesn't take forever. I foresee so-so results but let's try it first.

# In[ ]:


EPOCHS=10
history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])


# In[ ]:


#start training from the latest checkpoint
tf.train.latest_checkpoint(checkpoint_dir)


# In[ ]:


model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)

model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

model.build(tf.TensorShape([1, None]))

model.summary()


# In[ ]:


def generate_text(model, start_string):
  # Evaluation step (generating text using the learned model)

  # Number of characters to generate
  num_generate = 1000

  # Converting the first string to numbers (vectorizing)
  input_eval = [char2idx[s] for s in start_string]
  input_eval = tf.expand_dims(input_eval, 0)

  # Empty string to store our results
  text_generated = []

  # Temperature is the 'creativity' variable:
  # Low temperatures result in more predictable text
  # Higher temperatures result in more surprising text
  temperature = 1.0

  # Here batch size == 1
  model.reset_states()
  for i in range(num_generate):
      predictions = model(input_eval)
      # remove the batch dimension
      predictions = tf.squeeze(predictions, 0)

      # using a categorical distribution to predict the character returned by the model
      predictions = predictions / temperature
      predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

      # We pass the predicted character as the next input to the model
      # along with the previous hidden state
      input_eval = tf.expand_dims([predicted_id], 0)

      text_generated.append(idx2char[predicted_id])

  return (start_string + ''.join(text_generated))

print(generate_text(model, start_string=u"The")) #specifiy the first word of the text as a prompt

