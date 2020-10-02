#!/usr/bin/env python
# coding: utf-8

# In this kernel we will use LSTM for sentiment analysis prediction. Since LSTM is good to use when you are dealing with sequential data. And in NLP we have sequential data like words, sentences, etc.
# 
# https://towardsdatascience.com/understanding-lstm-and-its-quick-implementation-in-keras-for-sentiment-analysis-af410fd85b47
# 
# Before we begin, we will set the seed for the components involved. That is, plain ol' Python, Numpy and Tensorflow. 

# In[ ]:


seed = 0

import random
import numpy as np
from tensorflow import set_random_seed

random.seed(seed)
np.random.seed(seed)
set_random_seed(seed)


# Now we will read our data:

# In[ ]:


import pandas as pd

train = pd.read_csv('../input/train.tsv',  sep="\t")
test = pd.read_csv('../input/test.tsv',  sep="\t")


# In[ ]:


train.head()


# The mean and max lengths of phrases are the following:

# In[ ]:


train['Phrase'].str.len().mean()


# In[ ]:


train['Phrase'].str.len().max()


# Let's take a look at how sentiments are distributed.

# In[ ]:


train['Sentiment'].value_counts()


# The values correspond to sentiments as follows:
# 
# ```
# 0 - negative
# 1 - somewhat negative
# 2 - neutral
# 3 - somewhat positive
# 4 - positive
# ```
# 
# We can see that most phrases are neutral, followed by the 'somewhats' (somewhat positive - somewhat negative). Then, far behind are positive and negative phrases. This (seemingly) makes classification harder, since most phrases are congregated towards the middle/neutral.
# 
# To train our model, we need to format our data.
# 
# Since we are dealing with text, we will first convert everything to lowercase. Then, we will tokenize our text. Currently, we build the tokenizer only on the training data. We could add to the ingredients the testing data, but results may go up or down. Testing is needed to determine whether or not adding the testing data will help. After the tokenization, we also need to pad the rows with zeros.
# 
# Apart from that, we need to convert the numerical output to categorical. Specifically, we need to one-hot encode the labels.
# 
# Finally, we need to shuffle our data as well.

# In[ ]:


def format_data(train, test, max_features, maxlen):
    """
    Convert data to proper format.
    1) Shuffle
    2) Lowercase
    3) Sentiments to Categorical
    4) Tokenize and Fit
    5) Convert to sequence (format accepted by the network)
    6) Pad
    7) Voila!
    """
    from keras.preprocessing.text import Tokenizer
    from keras.preprocessing.sequence import pad_sequences
    from keras.utils import to_categorical
    
    train = train.sample(frac=1).reset_index(drop=True)
    train['Phrase'] = train['Phrase'].apply(lambda x: x.lower())
    test['Phrase'] = test['Phrase'].apply(lambda x: x.lower())

    X = train['Phrase']
    test_X = test['Phrase']
    Y = to_categorical(train['Sentiment'].values)

    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(list(X))

    X = tokenizer.texts_to_sequences(X)
    X = pad_sequences(X, maxlen=maxlen)
    test_X = tokenizer.texts_to_sequences(test_X)
    test_X = pad_sequences(test_X, maxlen=maxlen)

    return X, Y, test_X


# In[ ]:


maxlen = 125
max_features = 10000

X, Y, test_X = format_data(train, test, max_features, maxlen)


# Let's take a look at how the data looks:

# In[ ]:


X


# As you can see each row is zero-padded on the left.

# In[ ]:


Y


# In[ ]:


test_X


# With the formatted data at hand, we move to split our training set to training and validation. The validation set will help as determine whether our model generalizes well or not.

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.25, random_state=seed)


# In[ ]:


from keras.layers import Dense, Embedding,LSTM
from keras.models import Sequential


# In[ ]:


model = Sequential()

# Input / Embdedding
model.add(Embedding(max_features,100,mask_zero=True))
model.add(LSTM(64,dropout=0.4, recurrent_dropout=0.4,return_sequences=True))
model.add(LSTM(32,dropout=0.5, recurrent_dropout=0.5,return_sequences=False))

# Output layer
model.add(Dense(5, activation='sigmoid'))

model.summary()


# Having build our network, we will start training.
# 
# Since this problem is multi-class classification, we will optimize the categorical crossentropy loss function. The optimizer we will use is ADAM.

# In[ ]:


epochs = 5
batch_size = 32


# In[ ]:


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=epochs, batch_size=batch_size, verbose=1)


# Finally, we will make our predictions on the test set.

# In[ ]:


sub = pd.read_csv('../input/sampleSubmission.csv')

sub['Sentiment'] = model.predict_classes(test_X, batch_size=batch_size, verbose=1)
sub.to_csv('sub_lstm.csv', index=False)

