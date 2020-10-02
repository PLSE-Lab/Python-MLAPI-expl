#!/usr/bin/env python
# coding: utf-8

# **Accompanying Slides**
# 
# https://docs.google.com/presentation/d/1NQpJtkD8PhMmER4fw4WO0VwaD9I-s20gWnypwbkx5wk/edit
# 
# **Inspiration**
# 
# https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
# 
# https://www.kaggle.com/jannesklaas/19-lstm-for-email-classification

# In[ ]:


import os
import numpy as np
from keras.layers import Activation, Conv1D, Dense, Embedding, Flatten, Input, MaxPooling1D
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.datasets import fetch_20newsgroups
from sklearn.datasets.base import get_data_home
from keras.metrics import categorical_accuracy

data_home = get_data_home()
twenty_home = os.path.join(data_home, "20news_home")

if not os.path.exists(data_home):
    os.makedirs(data_home)
    
if not os.path.exists(twenty_home):
    os.makedirs(twenty_home)
    
get_ipython().system('cp ../input/20-newsgroup-sklearn/20news-bydate_py3* /tmp/scikit_learn_data')


# ## Preprocessing the data
# 
# You already learned that we have to tokenize the text before we can feed it into a neural network. This tokenization process will also remove some of the features of the original text, such as all punctuation or words that are less common.

# In[ ]:


# http://qwone.com/~jason/20Newsgroups/
dataset = fetch_20newsgroups(subset='all', shuffle=True, download_if_missing=False)

texts = dataset.data # Extract text
target = dataset.target # Extract target


# In[ ]:


print (target[:10])

print (len(texts))
print (len(target))
print (len(texts[0].split()))
print (texts[0])
print (target[0])
print (dataset.target_names[target[0]])


# Remember we have to specify the size of our vocabulary. Words that are less frequent will get removed. In this case we want to retain the 20,000 most common words.

# In[ ]:


vocab_size = 20000

tokenizer = Tokenizer(num_words=vocab_size) # Setup tokenizer
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts) # Generate sequences


# In[ ]:


print (tokenizer.texts_to_sequences(['Hello King, how are you?']))

print (len(sequences))
print (len(sequences[0]))
print (sequences[0])


# In[ ]:


word_index = tokenizer.word_index
print('Found {:,} unique words.'.format(len(word_index)))


# Our text is now converted to sequences of numbers. It makes sense to convert some of those sequences back into text to check what the tokenization did to our text. To this end we create an inverse index that maps numbers to words while the tokenizer maps words to numbers.

# In[ ]:


# Create inverse index mapping numbers to words
inv_index = {v: k for k, v in tokenizer.word_index.items()}

# Print out text again
for w in sequences[0]:
    x = inv_index.get(w)
    print(x,end = ' ')


# ### Measuring text length
# 
# Let's ensure all sequences have the same length.

# In[ ]:


# Get the average length of a text
avg = sum(map(len, sequences)) / len(sequences)

# Get the standard deviation of the sequence length
std = np.sqrt(sum(map(lambda x: (len(x) - avg)**2, sequences)) / len(sequences))

avg,std


# You can see, the average text is about 300 words long. However, the standard deviation is quite large which indicates that some texts are much much longer. If some user decided to write an epic novel in the newsgroup it would massively slow down training. So for speed purposes we will restrict sequence length to 100 words. You should try out some different sequence lengths and experiment with processing time and accuracy gains.

# In[ ]:


print(pad_sequences([[1,2,3]], maxlen=5))
print(pad_sequences([[1,2,3,4,5,6]], maxlen=5))


# In[ ]:


max_length = 100
data = pad_sequences(sequences, maxlen=max_length)


# ## Turning labels into One-Hot encodings
# 
# Labels can quickly be encoded into one-hot vectors with Keras:

# In[ ]:


from keras.utils import to_categorical
labels = to_categorical(np.asarray(target))
print('Shape of data:', data.shape)
print('Shape of labels:', labels.shape)

print (target[0])
print (labels[0])


# ## Loading GloVe embeddings
# 

# In[ ]:


glove_dir = '../input/glove-global-vectors-for-word-representation' # This is the folder with the dataset

embeddings_index = {} # We create a dictionary of word -> embedding

with open(os.path.join(glove_dir, 'glove.6B.100d.txt')) as f:
    for line in f:
        values = line.split()
        word = values[0] # The first value is the word, the rest are the values of the embedding
        embedding = np.asarray(values[1:], dtype='float32') # Load embedding
        embeddings_index[word] = embedding # Add embedding to our embedding dictionary

print('Found {:,} word vectors in GloVe.'.format(len(embeddings_index)))


# In[ ]:


print (embeddings_index['frog'])
print (len(embeddings_index['frog']))


# In[ ]:


print (np.linalg.norm(embeddings_index['man'] - embeddings_index['woman']))
print (np.linalg.norm(embeddings_index['man'] - embeddings_index['cat']))

# https://nlp.stanford.edu/projects/glove/
print (np.linalg.norm(embeddings_index['frog'] - embeddings_index['toad']))
print (np.linalg.norm(embeddings_index['frog'] - embeddings_index['man']))

print (np.linalg.norm(embeddings_index['frog'] - embeddings_index['fog']))

print (np.linalg.norm(embeddings_index['frog'] - embeddings_index['fork']))
print (np.linalg.norm(embeddings_index['frog'] - embeddings_index['skyscraper']))


# In[ ]:


embedding_dim = 100 # We use 100 dimensional glove vectors

word_index = tokenizer.word_index
nb_words = min(vocab_size, len(word_index)) # How many words are there actually

embedding_matrix = np.zeros((nb_words, embedding_dim))

# The vectors need to be in the same position as their index. 
# Meaning a word with token 1 needs to be in the second row (rows start with zero) and so on

# Loop over all words in the word index
for word, i in word_index.items():
    # If we are above the amount of words we want to use we do nothing
    if i >= vocab_size: 
        continue
    # Get the embedding vector for the word
    embedding_vector = embeddings_index.get(word)
    # If there is an embedding vector, put it in the embedding matrix
    if embedding_vector is not None: 
        embedding_matrix[i] = embedding_vector


# In[ ]:


print (embedding_matrix[100])


# In[ ]:


model = Sequential()
model.add(Embedding(vocab_size, 
                    embedding_dim, 
                    input_length=max_length, 
                    weights = [embedding_matrix], 
                    trainable = False))
model.add(Conv1D(128, 3, activation='relu'))
model.add(MaxPooling1D(3))
model.add(Conv1D(128, 3, activation='relu'))
model.add(MaxPooling1D(3))
model.add(Conv1D(128, 3, activation='relu'))
model.add(MaxPooling1D(3))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(20, activation='softmax'))
model.summary()


# In[ ]:


# model.compile(optimizer='adam',
#               loss='binary_crossentropy',  # https://stackoverflow.com/questions/42081257/keras-binary-crossentropy-vs-categorical-crossentropy-performance
#               metrics=['accuracy'])

# https://stackoverflow.com/questions/42081257/keras-binary-crossentropy-vs-categorical-crossentropy-performance
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[categorical_accuracy])

model.fit(data, labels, validation_split=0.2, epochs=10)


# Our model achieves 63% accuracy on the validation set. Systems like these can be used to assign emails in customer support centers, suggest responses, or classify other forms of text like invoices which need to be assigned to an department. Let's take a look at how our model classified one of the texts:

# In[ ]:


example = data[400] # get the tokens
print (texts[400])

# Print tokens as text
for w in example:
    x = inv_index.get(w)
    print(x,end = ' ')


# In[ ]:


# Get prediction
pred = model.predict(example.reshape(1,100))


# In[ ]:


# Output predicted category
dataset.target_names[np.argmax(pred)]

