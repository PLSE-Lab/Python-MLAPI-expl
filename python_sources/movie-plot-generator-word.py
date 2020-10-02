#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
tf.enable_eager_execution()

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))

import tensorflow as tf
from keras.utils.data_utils import get_file
import tarfile

import numpy as np
import os
import time


# # Data preprocessing

# In[ ]:


filename='MovieSummaries.tar.gz' #download data
path = get_file(
    filename,
    origin='https://www.cs.cmu.edu/~ark/personas/data/MovieSummaries.tar.gz')
statinfo = os.stat(path)
print('Succesfully downloaded', filename, ', size: ', statinfo.st_size/1024/1024, 'Mbytes.')


# In[ ]:


tar = tarfile.open(path) #exctract files
tar.extractall(path="movies")
tar.close()


# In[ ]:


dir_dat=os.path.realpath('movies')
path=dir_dat+'/MovieSummaries/'
os.listdir(path)


# In[ ]:


plots=pd.read_fwf(path+'plot_summaries.txt', delimiter='\n', header=None) # load movie plots to pandas
plots.columns=['movies']
plots = plots.movies.str.split('\t', 1, expand=True)
plots.columns=['ID','plot']
plots.head()


# In[ ]:


# id is object type, so convert it into numeric
plots['ID']=plots['ID'].apply(pd.to_numeric, errors='coerce')


# In[ ]:


# load movie metadata
tags=[ 'ID',
'Freebase movie ID',
'Movie name',
'release date',
'box office revenue',
'runtime',
'Movie languages',
'Movie countries',
'genres']
metadat=pd.read_csv(path+'movie.metadata.tsv', sep='\t', lineterminator='\n', names=tags)
mov_names=metadat[['ID', 'Movie name', 'genres']]
plot_genre=pd.merge(plots, mov_names, how='left', on='ID') # merge both tables on ID
print(plot_genre.shape)
plot_genre.head()


# In[ ]:


plot_genre.dropna(inplace=True)


# In[ ]:


import re # regex for cleaning the genre strings
def clean_genres(line):
    #print(line)
    a=re.sub('[""{}]', '', line)
    b=re.split(': |,',a)
    new=[]
    [new.append(b[i]) for i in range(len(b))   if i % 2 != 0]
    str1=''
    str1 = ','.join(str(e) for e in new)
    return str1


# In[ ]:


plot_genre['genres']=plot_genre['genres'].apply(lambda x: clean_genres(x))


# In[ ]:





# In[ ]:


import re
import string
from string import digits
plot_genre['plot']=plot_genre['plot'].apply(lambda x: re.sub(r"[^a-zA-Z,.]",' ', x)).apply(lambda x: re.sub(",", ' , ', x)).apply(lambda x: re.sub("\.", ' . ', x))
plot_genre['genres']=plot_genre['genres'].apply(lambda x: re.sub(r"[^a-zA-Z,.]",' ', x)).apply(lambda x: re.sub(",", ' , ', x)).apply(lambda x: re.sub("\.", ' . ', x)) 
remove_digits = str.maketrans('', '', digits)
plot_genre['plot']=plot_genre['plot'].apply(lambda x: x.translate(remove_digits)).apply(lambda x: x.lower())
plot_genre['genres']=plot_genre['genres'].apply(lambda x: x.translate(remove_digits)).apply(lambda x: x.lower())
plot_genre['plot']=plot_genre['plot'].apply(lambda x: re.sub('\'',' \' ', x))
del plots, metadat,mov_names
print(plot_genre.shape)
plot_genre.head()


# # word level, no encoder decoder using tf example

# In[ ]:


plot1="family film|thriller|comedy"
plots=plot_genre[plot_genre['genres'].str.contains(plot1)]['plot']
print('Number of movies with genres: ',plot1, plots.shape)

sub=plots.sample(n=20000, random_state=14)
mask = (sub.str.len() <= 1000)
not_huge_plots = sub.loc[mask]
print('Number of movies with less than 1000 chars in plots: ',(not_huge_plots).shape)


# In[ ]:


#not_huge_plots.to_csv('plots.txt', sep='\t', encoding='utf-8', index=False, header=False)


# In[ ]:


#os.listdir('../working')
strings=not_huge_plots.values.T.tolist()
plotz = ''.join(str(e) for e in strings)


# In[ ]:


print ('Length of text: {} characters'.format(len(plotz)))


# In[ ]:


all_plots=set() #create vocabulary of unique words

for txt in plotz.split():
    for word in txt.split():
        if word not in all_plots:
            all_plots.add(word)


# In[ ]:


len(all_plots) # number of unique words in plots


# In[ ]:


vocab = sorted(list(all_plots))
word2idx = dict(
    [(word, i) for i, word in enumerate(vocab)])


# In[ ]:


for word,_ in zip(word2idx, range(20)):
    print('{:6s} ---> {:4d}'.format(repr(word), word2idx[word]))


# In[ ]:


idx2word = np.array(vocab)

text_as_int = np.array([word2idx[c] for c in plotz.split()])


# In[ ]:


print ('{} ---- characters mapped to int ---- > {}'.format(plotz.split()[:10], text_as_int[:10]))


# In[ ]:


# The maximum length sentence we want for a single input in characters
seq_length = 100

# Create training examples / targets
chunks = tf.data.Dataset.from_tensor_slices(text_as_int).batch(seq_length+1, drop_remainder=True)

for item in chunks.take(2):
  print(repr(' '.join(idx2word[item.numpy()])))


# In[ ]:


def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

dataset = chunks.map(split_input_target)


# In[ ]:


for input_example, target_example in  dataset.take(1):
  print ('Input data: ', repr(' '.join(idx2word[input_example.numpy()])))
  print ('Target data:', repr(' '.join(idx2word[target_example.numpy()])))


# In[ ]:


for i, (input_idx, target_idx) in enumerate(zip(input_example[:5], target_example[:5])):
    print("Step {:4d}".format(i))
    print("  input: {} ({:s})".format(input_idx, repr(idx2word[input_idx])))
    print("  expected output: {} ({:s})".format(target_idx, repr(idx2word[target_idx])))


# In[ ]:


BATCH_SIZE = 64
BUFFER_SIZE = 10000


# In[ ]:


dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)


# In[ ]:


class Model(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, units):
    super(Model, self).__init__()
    self.units = units

    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)

    if tf.test.is_gpu_available():
      self.gru = tf.keras.layers.CuDNNGRU(self.units, 
                                          return_sequences=True, 
                                          recurrent_initializer='glorot_uniform',
                                          stateful=True)
    else:
      self.gru = tf.keras.layers.GRU(self.units, 
                                     return_sequences=True, 
                                     recurrent_activation='sigmoid', 
                                     recurrent_initializer='glorot_uniform', 
                                     stateful=True)

    self.fc = tf.keras.layers.Dense(vocab_size)
        
  def call(self, x):
    embedding = self.embedding(x)
    
    # output at every time step
    # output shape == (batch_size, seq_length, hidden_size) 
    output = self.gru(embedding)
    
    # The dense layer will output predictions for every time_steps(seq_length)
    # output shape after the dense layer == (seq_length * batch_size, vocab_size)
    prediction = self.fc(output)
    
    # states will be used to pass at every step to the model while training
    return prediction


# In[ ]:


vocab_size = len(vocab)

# The embedding dimension 
embedding_dim = 256

# Number of RNN units
units = 512

model = Model(vocab_size, embedding_dim, units)


# In[ ]:


# Using adam optimizer with default arguments
optimizer = tf.train.AdamOptimizer()

# Using sparse_softmax_cross_entropy so that we don't have to create one-hot vectors
def loss_function(real, preds):
    return tf.losses.sparse_softmax_cross_entropy(labels=real, logits=preds)


# In[ ]:


model.build(tf.TensorShape([BATCH_SIZE, seq_length]))
model.summary()


# In[ ]:


# Directory where the checkpoints will be saved
checkpoint_dir = './training_checkpoints'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")


# In[ ]:


EPOCHS=300
# Training loop
for epoch in range(EPOCHS):
    start = time.time()
    
    # initializing the hidden state at the start of every epoch
    # initally hidden is None
    hidden = model.reset_states()
    
    for (batch, (inp, target)) in enumerate(dataset):
          with tf.GradientTape() as tape:
              # feeding the hidden state back into the model
              # This is the interesting step
              predictions = model(inp)
              loss = loss_function(target, predictions)
              
          grads = tape.gradient(loss, model.variables)
          optimizer.apply_gradients(zip(grads, model.variables))

          #if (epoch+1) % 2 == 0:
          #    print ('Epoch {} Batch {} Loss {:.4f}'.format(epoch+1, batch,loss))
    # saving (checkpoint) the model every 5 epochs
    if (epoch + 1) % 10 == 0:
      model.save_weights(checkpoint_prefix)
      print ('Epoch {} Loss {:.4f}'.format(epoch+1, loss))
      print ('Time taken for 10 epoch {} sec\n'.format(time.time() - start))


# In[ ]:


model.save_weights(checkpoint_prefix)


# In[ ]:


model = Model(vocab_size, embedding_dim, units)

model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

model.build(tf.TensorShape([1, None]))


# In[ ]:


num_generate = 400

# You can change the start string to experiment
start_string = 'now'

# Converting our start string to numbers (vectorizing) 
input_eval = [word2idx[s] for s in start_string]
input_eval = tf.expand_dims(input_eval, 0)

# Empty string to store our results
text_generated = []

# Low temperatures results in more predictable text.
# Higher temperatures results in more surprising text.
# Experiment to find the best setting.
temperature = 1.0


# In[ ]:


#Evaluation loop.

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
   
   text_generated.append(idx2word[predicted_id])


# In[ ]:


# make the text readable by capitalizing each sentence
generated=' '.join(text_generated)
new=re.sub(" ,", ',', generated)
new=re.sub(" \.", '.', new)
sentence=new.split('. ')
for i, line in enumerate(sentence):
    sentence[i]=line.capitalize()
print('. '.join(sentence))


# # to be continued

# 

# In[ ]:


import shutil
shutil.rmtree(dir_dat)

