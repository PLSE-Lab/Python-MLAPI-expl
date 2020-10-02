#!/usr/bin/env python
# coding: utf-8

# # Ch. 18 - Temporal Order Matters
# 
# In language, the order of words matters. The sentences 'The dog lies on the couch' and 'The couch lies on the dog' contain the exact same words yet they describe two very different situations. Our previous model did not take the order of words into account. In this chapter we will take a look at two methods to ensure that your model can access information from the order of words.
# 
# ## 1D Convolutions
# You might remember convolutional neural networks from computer vision week. In computer vision, convolutional filters slide over the image two dimensionally. There is also a version of convolutional filters that can slide over a sequence one dimensionally. The output is another sequence, much like the output of a two dimensional convolution was another 'image'. Everything else about 1D convolutions is exactly the same as 2D convolutions. 
# 
# To make it a bit easier we can download the IMDB dataset directly through Keras with tokenization already done:

# In[ ]:


# Some prep for getting the dataset to work in Kaggle
from os import listdir, makedirs
from os.path import join, exists, expanduser

cache_dir = expanduser(join('~', '.keras'))
if not exists(cache_dir):
    makedirs(cache_dir)
datasets_dir = join(cache_dir, 'datasets')
if not exists(datasets_dir):
    makedirs(datasets_dir)

# If you have multiple input files, change the below cp commands accordingly, typically:
# !cp ../input/keras-imdb/imdb* ~/.keras/datasets/
get_ipython().system('cp ../input/imdb* ~/.keras/datasets/')


# In[ ]:


from keras.datasets import imdb
from keras.preprocessing import sequence

max_words = 10000  # Our 'vocabulary of 10K words
max_len = 500  # Cut texts after 500 words

# Get data from Keras
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_words)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')


# In[ ]:


word_index = imdb.get_word_index()


# In[ ]:


# Pad sequences
x_train = sequence.pad_sequences(x_train, maxlen=max_len)
x_test = sequence.pad_sequences(x_test, maxlen=max_len)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)


# ## Building the conv model
# 
# Now we build our convolutional model. You will notice a couple new layers next to ``Conv1D``
# 
# - [``MaxPooling1D``](https://keras.io/layers/pooling/#maxpooling1d) works exactly like ``MaxPooling2D`` which we used earlier. It takes a piece of the sequence with specified length and returns the maximum element in the sequence much like it returned the maximum element of a small window in 2D convolutional networks. Note that MaxPooling always returns the maximum element for each channel. 
# - [``GlobalMaxPooling2D``](https://keras.io/layers/pooling/#globalmaxpooling1d) returns the maximum over the entire sequence. 
# 
# You can see the difference between the two in the model summary below. While ``MaxPooling1D`` significantly shortens the sequence, ``GlobalMaxPooling2D`` removes the temporal dimension entirely:

# In[ ]:


embedding_dim = 100


# In[ ]:


from keras.models import Sequential
from keras.layers import Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Dense

model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=max_len)) # We train our own embeddings
model.add(Conv1D(32, 7, activation='relu')) # 1D Convolution, 32 channels, windows size 7
model.add(MaxPooling1D(5)) # Pool windows of size 5
model.add(Conv1D(32, 7, activation='relu')) # Another 1D Convolution, 32 channels, windows size 7
model.add(GlobalMaxPooling1D()) # Global Pooling
model.add(Dense(1)) # Final Output Layer

model.summary()


# In[ ]:


model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['acc'])


# In[ ]:


history = model.fit(x_train, y_train,
                    epochs=2,
                    batch_size=128,
                    validation_split=0.2)


# That does not look too bad! 1D Convolutions also have relatively few parameters so they are quick to train.
# 
# ## Reocurrent Neural Networks
# 
# Another method to make order matter in neural networks is to give the network some kind of memory. So far, all of our networks did a forward pass without any memory of what happened before or after the pass. It is time to change that with reocurrent neural networks.
# 
# ![Simple RNN](https://storage.googleapis.com/aibootcamp/Week%204/assets/simple_rnn.png)

# Reocurrent neural networks contain reocurrent layers. Reocurrent layers can remember their last activation and use it as their own input.
# 
# $$A_{t} = activation( W * in + U * A_{t-1} + b)$$
# 
# A reocurrent layer takes a sequence as an input. For each element, it then computes a matrix multiplication ($W * in$) just like a ``Dense`` layer and runs the result through an activation function like e.g. ``relu``. It then retains it's own activation. When the next item of the sequence arrives, it performs the matrix multiplication as before but it also multiplies it's previous activation with a second matrix ($U * A_{t-1}$). It adds the result of both operations together and passes it through it's activation function again. In Keras, we can use a simple rnn like this:

# In[ ]:


from keras.layers import SimpleRNN


# In[ ]:


model = Sequential()
# No need to specify the sequence length anymore
model.add(Embedding(max_words, embedding_dim)) # We train our own embeddings
# RNN's only need their size as a parameter, just like Dense layers
model.add(SimpleRNN(32, activation='relu'))
# Dense output for final classification
model.add(Dense(1))

model.summary()


# The attuned reader might have noticed that we no longer specify an input length in the embeddings layer. That is because RNN's can work with sequences of arbitrary length! If not specified otherwise, a RNN layer will only pass the last output on to the next layer, which is why they have no trouble working with Dense layers. If we want to stack RNN layers, we need to tell them to pass on the entire output sequence so that the following layer has something to work with.

# In[ ]:


model = Sequential()
# No need to specify the sequence length anymore
model.add(Embedding(max_words, embedding_dim)) # We train our own embeddings
# This one returns the full sequence
model.add(SimpleRNN(32, activation='relu', return_sequences=True))
# This one just the last sequence element
model.add(SimpleRNN(32, activation='relu'))
# Dense output for final classification
model.add(Dense(1))

model.summary()


# In practice, it is still common to cut sequences after a certain length. Some sequences might just be extremely long and not contain much more valuable information after a certain point. Cutting them off saves on computing power.

# In[ ]:


model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['acc'])


# In[ ]:


history = model.fit(x_train, y_train,
                    epochs=5,
                    batch_size=128,
                    validation_split=0.2)


# You can see a RNN trains just like other neural nets, although this particular setup performs quite poorly on this task. A problem of RNN's is that their memory is quite short term. While they should in theory be able to tweak their outputs to retain long term memory, they are only really able to retain information about the last one or two words. In the next chapter, we will look at ``LSTM``s that do not have this issue.
# 
# ## Summary
# 
# In this chapter you have learned about two methods to take the order in sequences into account. 1 dimensional convolution works very similar to convolution as we know it from computer vision. It is also quite fast and uses few parameters. RNN's on the other hand use more parameters but can work with sequences of arbitrary length.

# In[ ]:




