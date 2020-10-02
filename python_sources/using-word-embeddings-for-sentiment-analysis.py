#!/usr/bin/env python
# coding: utf-8

# # Introduction
# In my previous kernel, I explored [deep learning with Keras](https://www.kaggle.com/bertcarremans/deep-learning-for-sentiment-analysis) on the [TwitterAirline data set](https://www.kaggle.com/crowdflower/twitter-airline-sentiment). The model with Dropout layers produced the best results. However, it was still outperformed by the LogisticRegression model. Let's try if we can do better than the LogisticRegression with a deep learning model that uses word embeddings. 
# 
# In this kernel we will:
# * use the Embedding layer of Keras to create word embeddings from the training data
# * use pretrained word embeddings trained on a much larger corpus

# # Word Embeddings
# When applying one-hot encoding to the words in the tweets, we end up with sparse vectors of high dimensionality (here the number of words). On larger data sets this could cause performance issues. Additionally, one-hot encoding does not take into account the semantics of the words. For instance, *plane* and *aircraft* are different words but have a similar meaning. 
# 
# Word embeddings reduce these two issues. Word embeddings are dense vectors with a much lower dimensionality. Secondly, the semantic relationships between words are reflected in the distance and direction of the vectors.  

# # Set-up of the project

# In[ ]:


# Basic packages
import pandas as pd 
import numpy as np
import re
import collections
import matplotlib.pyplot as plt
from pathlib import Path

# Packages for data preparation
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import LabelEncoder

# Packages for modeling
from keras import models
from keras import layers
from keras import regularizers


# In[ ]:


NB_WORDS = 10000  # Parameter indicating the number of words we'll put in the dictionary
VAL_SIZE = 1000  # Size of the validation set
NB_START_EPOCHS = 20  # Number of epochs we usually start to train with
BATCH_SIZE = 512  # Size of the batches used in the mini-batch gradient descent
MAX_LEN = 24  # Maximum number of words in a sequence
GLOVE_DIM = 50  # Number of dimensions of the GloVe word embeddings
INPUT_PATH = '../input'  # Path where all input files are stored


# In[ ]:


root = Path('../')
input_path = root / 'input/' 
ouput_path = root / 'output/'
source_path = root / 'source/'


# # Some helper functions

# In[ ]:


def deep_model(model, X_train, y_train, X_valid, y_valid):
    '''
    Function to train a multi-class model. The number of epochs and 
    batch_size are set by the constants at the top of the
    notebook. 
    
    Parameters:
        model : model with the chosen architecture
        X_train : training features
        y_train : training target
        X_valid : validation features
        Y_valid : validation target
    Output:
        model training history
    '''
    model.compile(optimizer='rmsprop'
                  , loss='categorical_crossentropy'
                  , metrics=['accuracy'])
    
    history = model.fit(X_train
                       , y_train
                       , epochs=NB_START_EPOCHS
                       , batch_size=BATCH_SIZE
                       , validation_data=(X_valid, y_valid)
                       , verbose=1)
    return history


def eval_metric(history, metric_name):
    '''
    Function to evaluate a trained model on a chosen metric. 
    Training and validation metric are plotted in a
    line chart for each epoch.
    
    Parameters:
        history : model training history
        metric_name : loss or accuracy
    Output:
        line chart with epochs of x-axis and metric on
        y-axis
    '''
    metric = history.history[metric_name]
    val_metric = history.history['val_' + metric_name]

    e = range(1, NB_START_EPOCHS + 1)

    plt.plot(e, metric, 'bo', label='Train ' + metric_name)
    plt.plot(e, val_metric, 'b', label='Validation ' + metric_name)
    plt.legend()
    plt.show()

def test_model(model, X_train, y_train, X_test, y_test, epoch_stop):
    '''
    Function to test the model on new data after training it
    on the full training data with the optimal number of epochs.
    
    Parameters:
        model : trained model
        X_train : training features
        y_train : training target
        X_test : test features
        y_test : test target
        epochs : optimal number of epochs
    Output:
        test accuracy and test loss
    '''
    model.fit(X_train
              , y_train
              , epochs=epoch_stop
              , batch_size=BATCH_SIZE
              , verbose=0)
    results = model.evaluate(X_test, y_test)
    
    return results

def remove_stopwords(input_text):
    '''
    Function to remove English stopwords from a Pandas Series.
    
    Parameters:
        input_text : text to clean
    Output:
        cleaned Pandas Series 
    '''
    stopwords_list = stopwords.words('english')
    # Some words which might indicate a certain sentiment are kept via a whitelist
    whitelist = ["n't", "not", "no"]
    words = input_text.split() 
    clean_words = [word for word in words if (word not in stopwords_list or word in whitelist) and len(word) > 1] 
    return " ".join(clean_words) 
    
def remove_mentions(input_text):
    '''
    Function to remove mentions, preceded by @, in a Pandas Series
    
    Parameters:
        input_text : text to clean
    Output:
        cleaned Pandas Series 
    '''
    return re.sub(r'@\w+', '', input_text)


# # Data Preparation
# ### Reading and cleaning data

# In[ ]:


tweets_dir = "twitter-airline-sentiment/"
df = pd.read_csv(input_path / tweets_dir / 'Tweets.csv')
df = df.reindex(np.random.permutation(df.index))  
df = df[['text', 'airline_sentiment']]
df.text = df.text.apply(remove_stopwords).apply(remove_mentions)


# ### Train-Test split

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(df.text, df.airline_sentiment, test_size=0.1, random_state=37)
print('# Train data samples:', X_train.shape[0])
print('# Test data samples:', X_test.shape[0])
assert X_train.shape[0] == y_train.shape[0]
assert X_test.shape[0] == y_test.shape[0]


# ### Converting words to numbers

# In[ ]:


tk = Tokenizer(num_words=NB_WORDS,
               filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
               lower=True,
               split=" ")
tk.fit_on_texts(X_train)

X_train_seq = tk.texts_to_sequences(X_train)
X_test_seq = tk.texts_to_sequences(X_test)


# ### Creating word sequences of equal length
# Before we can compute the word embeddings, we need to make sure the sequences are of equal length. In the example below, we truncate sequences to length MAX_LEN, or pad them with zeroes to achieve this. First, we'll have a look at the length of the (cleaned) tweets.

# In[ ]:


seq_lengths = X_train.apply(lambda x: len(x.split(' ')))
seq_lengths.describe()


# Based on the figures above we will set MAX_LEN to 24. So this means we will not be truncating any words, only pad with zeros. This is to avoid to lose information as the tweets are rather short.

# In[ ]:


X_train_seq_trunc = pad_sequences(X_train_seq, maxlen=MAX_LEN)
X_test_seq_trunc = pad_sequences(X_test_seq, maxlen=MAX_LEN)


# In[ ]:


X_train_seq_trunc[10]  # Example of padded sequence


# ### Converting the target classes to numbers

# In[ ]:


le = LabelEncoder()
y_train_le = le.fit_transform(y_train)
y_test_le = le.transform(y_test)
y_train_oh = to_categorical(y_train_le)
y_test_oh = to_categorical(y_test_le)


# ### Splitting off validation data

# In[ ]:


X_train_emb, X_valid_emb, y_train_emb, y_valid_emb = train_test_split(X_train_seq_trunc, y_train_oh, test_size=0.1, random_state=37)

assert X_valid_emb.shape[0] == y_valid_emb.shape[0]
assert X_train_emb.shape[0] == y_train_emb.shape[0]

print('Shape of validation set:',X_valid_emb.shape)


# # Modeling

# ### Training word embeddings
# Keras provides an **Embedding layer** which helps us to train specific word embeddings based on our training data. It will convert the words in our vocabulary to multi-dimensional vectors. 

# In[ ]:


emb_model = models.Sequential()
emb_model.add(layers.Embedding(NB_WORDS, 8, input_length=MAX_LEN))
emb_model.add(layers.Flatten())
emb_model.add(layers.Dense(3, activation='softmax'))
emb_model.summary()


# In[ ]:


emb_history = deep_model(emb_model, X_train_emb, y_train_emb, X_valid_emb, y_valid_emb)


# We have a validation accuracy of about 77%. The number of words in the tweets is rather low, so this result is rather good. 
# 
# By comparing the training and validation accuracy and loss, we see that the model starts overfitting from epoch 6..

# In[ ]:


eval_metric(emb_history, 'acc')


# In[ ]:


eval_metric(emb_history, 'loss')


# In[ ]:


emb_results = test_model(emb_model, X_train_seq_trunc, y_train_oh, X_test_seq_trunc, y_test_oh, 6)
print('/n')
print('Test accuracy of word embeddings model: {0:.2f}%'.format(emb_results[1]*100))


# This test result is quite ok, but still not better than the other deep learning model with Dropout layers, nor the LogisticRegression. Let's see if we can improve with pretrained word embeddings.
# 
# ### Using pre-trained word embeddings
# Because the training data is not so big, the model might not be able to learn good embeddings for the sentiment analysis. Luckily we can load pre-trained word embeddings built on a much larger training data. 
# 
# The [GloVe database](https://nlp.stanford.edu/projects/glove/) contains multiple pre-trained word embeddings, and more specific embeddings trained on tweets.

# In[ ]:


glove_file = 'glove.twitter.27B.' + str(GLOVE_DIM) + 'd.txt'
glove_dir = 'glove-global-vectors-for-word-representation/'
emb_dict = {}
glove = open(input_path / glove_dir / glove_file)
for line in glove:
    values = line.split()
    word = values[0]
    vector = np.asarray(values[1:], dtype='float32')
    emb_dict[word] = vector
glove.close()


# Let's test if we can find some airline related words in the dictionary.

# In[ ]:


airline_words = ['airplane', 'airline', 'flight', 'luggage']
for w in airline_words:
    if w in emb_dict.keys():
        print('Found the word {} in the dictionary'.format(w))


# To feed this into an Embedding layer, we need to build a matrix containing the words in the tweets and their representative word embedding. So this matrix will be of shape (NB_WORDS, GLOVE_DIM)

# In[ ]:


emb_matrix = np.zeros((NB_WORDS, GLOVE_DIM))

for w, i in tk.word_index.items():
    # The word_index contains a token for all words of the training data so we need to limit that
    if i < NB_WORDS:
        vect = emb_dict.get(w)
        # Check if the word from the training data occurs in the GloVe word embeddings
        # Otherwise the vector is kept with only zeros
        if vect is not None:
            emb_matrix[i] = vect
    else:
        break


# In[ ]:


glove_model = models.Sequential()
glove_model.add(layers.Embedding(NB_WORDS, GLOVE_DIM, input_length=MAX_LEN))
glove_model.add(layers.Flatten())
glove_model.add(layers.Dense(3, activation='softmax'))
glove_model.summary()


# With the *set_weights* method we load the pre-trained embeddings in the Embedding layer (here layer 0). By setting the *trainable* attribute to False, we make sure not to change the pre-trained embeddings.

# In[ ]:


glove_model.layers[0].set_weights([emb_matrix])
glove_model.layers[0].trainable = False


# In[ ]:


glove_history = deep_model(glove_model, X_train_emb, y_train_emb, X_valid_emb, y_valid_emb)


# In[ ]:


eval_metric(glove_history, 'loss')


# In[ ]:


eval_metric(glove_history, 'acc')


# In[ ]:


glove_results = test_model(glove_model, X_train_seq_trunc, y_train_oh, X_test_seq_trunc, y_test_oh, 3)
print('/n')
print('Test accuracy of word glove model: {0:.2f}%'.format(glove_results[1]*100))


# The model overfits fast, after 3 epochs. Moreover the validation accuracy is lower compared to the embeddings trained on the training data. 
# 
# As a final exercise, let's see what results we get when we train the embeddings with the same number of dimensions as the GloVe data.
# 
# ### Training word embeddings with more dimensions

# In[ ]:


emb_model2 = models.Sequential()
emb_model2.add(layers.Embedding(NB_WORDS, GLOVE_DIM, input_length=MAX_LEN))
emb_model2.add(layers.Flatten())
emb_model2.add(layers.Dense(3, activation='softmax'))
emb_model2.summary()


# In[ ]:


emb_history2 = deep_model(emb_model2, X_train_emb, y_train_emb, X_valid_emb, y_valid_emb)


# In[ ]:


eval_metric(glove_history, 'loss')


# In[ ]:


eval_metric(glove_history, 'acc')


# In[ ]:


emb_results2 = test_model(emb_model2, X_train_seq_trunc, y_train_oh, X_test_seq_trunc, y_test_oh, 3)
print('/n')
print('Test accuracy of word embedding model 2: {0:.2f}%'.format(emb_results2[1]*100))


# This result is very close to the model with 8-dimensional word embeddings. So there is no strong improvement. 
# 
# # Conclusion
# The best result is achieved with 8-dimensional word embeddings that are trained on the available data. This even outperforms the use of word embeddings that were trained on a much larger Twitter corpus. 
# 
# Until now we have just put a Dense layer on the flattened embeddings. By doing this, we do not take into account the relationships between the words in the tweet. This can be done with a recurrent neural network or a 1D convolutional network, which I'll cover in another kernel.
# 
