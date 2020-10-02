#!/usr/bin/env python
# coding: utf-8

# # Read the file

# In[ ]:


import pandas as pd

train = pd.read_csv("../input/train.csv", nrows = 1000)
test = pd.read_csv("../input/test.csv", nrows = 1000)
print(train.head())


# What is the fraction of insincere questions?

# In[ ]:


import numpy as np

print(np.mean(train.target))


# What is the maximum length of a sentence (in terms of number of words)?

# In[ ]:


max_words = np.max([len(i.split(" ")) for i in train.question_text])
print(max_words)


# # Import the embedding

# In[ ]:


import numpy as np

# loading embedding: https://www.kaggle.com/sudalairajkumar/a-look-at-different-embeddings
EMBEDDING_FILE = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'
def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE))

words_embedding = set(embeddings_index.keys())
print(len(embeddings_index))


# In[ ]:


print(embeddings_index["the"])


# In[ ]:


print(len(embeddings_index["the"]))


# # Checking the coverage

# In[ ]:


list_train_words = " ".join(train.question_text).split(" ")
words_training = set(list_train_words)
print(len(words_training.intersection(words_embedding))/len(words_training))


# Printing the missing words (those not present in the embedding):

# In[ ]:


print(words_training.difference(words_embedding))


# # Count the frequencies

# In[ ]:


from collections import Counter

frequencies = Counter(words_training.difference(words_embedding))
print(frequencies)


# Sorting by the number of appearences (following [stackoverflow](https://stackoverflow.com/questions/613183/how-do-i-sort-a-dictionary-by-value)):

# In[ ]:


import operator

sorted_frequencies = sorted(frequencies.items(), key=operator.itemgetter(1), reverse=True)
print(sorted_frequencies)


# # Build the embedding matrix
# 
# Following the [Keras blog](https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html), an embedding matrix can be created. The first step is to allocate space for the embedding matrix. The columns are the same as in the embedding index (300 in this case) and the rows are as many as the number of the training records. Words not found in embedding will be all-zeros.

# In[ ]:


embedding_matrix = np.zeros((len(words_training), 300))
mapping = {}

for index, word in enumerate(words_training):
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector
        mapping[word] = index
        
print(embedding_matrix)


# In[ ]:


print(embedding_matrix.__class__)
print(embedding_matrix.shape)


# # Prepare the input
# 
# Prepare the data taking all tokens (for the moment, only splitting on spaces):

# In[ ]:


labels = train.target
sentences = [i.split(" ") for i in train.question_text]
vocab_size = len(set([item for sublist in sentences for item in sublist]))
print(vocab_size)


# In[ ]:


print(sentences[0])


# In[ ]:


print(mapping[sentences[0][0]])


# In[ ]:


for i in sentences[0:2]:
    for j in i:
        if j in mapping.keys():
            print(mapping[j])


# In[ ]:


for i in sentences[0:2]:
    print([mapping[j] for j in i if j in mapping.keys()])


# In[ ]:


input_sequences = [[mapping[j] for j in i if j in mapping.keys()] for i in sentences]


# Following [Machine Mastery](https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/), the input file is padded in order to have all input sequences of equal size padding after (*post*) until a maximum of *maxwords* (50):

# In[ ]:


from keras.preprocessing.sequence import pad_sequences

padded_docs = pad_sequences(input_sequences, maxlen=max_words, padding='post')


# # Train the model
# 
# Following [Theo Viel's kernel](https://www.kaggle.com/theoviel/improve-your-score-with-text-preprocessing-v2) the f1 metric is implemented:

# In[ ]:


from keras import backend as K

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


# The model can be then sketched. For the moment, a very simple network:

# In[ ]:


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding

model = Sequential()
model.add(Embedding(vocab_size, 300, input_length=max_words, weights=[embedding_matrix], trainable=False))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[f1])
print(model.summary())


# In[ ]:


model.fit(padded_docs, labels, epochs=0)


# In[ ]:


loss, f1 = model.evaluate(padded_docs, labels, verbose=0)
print('F1: %f' % f1)


# In[ ]:




