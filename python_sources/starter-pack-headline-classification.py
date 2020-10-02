#!/usr/bin/env python
# coding: utf-8

# ## Introduction
# 
# In this kernel we will build a model to discriminate between real and fake news headlines. The dataset we will use is [Fake News Net](https://github.com/KaiDMML/FakeNewsNet), and contains headlines from political news articles, real and fake. After performing some basic pre-processing, we will train a Recurrent Neural Network (specifically, a bidirectional GRU) to classify between the different types of headlines. To evaluate the network's performance, we will measure percentile and F1-score accuracy metrics.

# ## Reading Data
# 
# First we are going to read the data for real and fake headlines, and then we are going to concatenate the two dataframes.

# In[ ]:


import pandas as pd

d_fake = pd.read_csv('../input/fnn_politics_fake.csv')
headlines_fake = d_fake.drop(['id', 'news_url', 'tweet_ids'], axis=1).rename(columns={'title': 'headline'})
headlines_fake['fake'] = 1

d_real = pd.read_csv('../input/fnn_politics_real.csv')
headlines_real = d_real.drop(['id', 'news_url', 'tweet_ids'], axis=1).rename(columns={'title': 'headline'})
headlines_real['fake'] = 0

data = pd.concat([headlines_fake, headlines_real])


# Let's shuffle our data and print its head:

# In[ ]:


data = data.sample(frac=1).reset_index(drop=True)
data.head()


# We'll print the number of items in each of the two classes, to see how imbalanced our dataset is.

# In[ ]:


data['fake'].value_counts()


# ## Data Processing
# 
# We also need to format the data. We will split the dataset into features `X` and target `Y`. For `Y`, we simply store the label at the target column. For `X`, we are first going to tokenise and pad our text input before storing it.

# In[ ]:


import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

def format_data(data, max_features, maxlen):
    data = data.sample(frac=1).reset_index(drop=True)
    data['headline'] = data['headline'].apply(lambda x: x.lower())

    Y = data['fake'].values # 0: Real; 1: Fake
    X = data['headline']

    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(list(X))

    X = tokenizer.texts_to_sequences(X)
    X = pad_sequences(X, maxlen=maxlen)

    return X, Y


# The `max_features` and `max_len` variables denote the length of each vector and the vocabulary length.

# In[ ]:


max_features, max_len = 3500, 25
X, Y = format_data(data, max_features, max_len)


# We'll now split the data into training and testing.

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=0)
len(X_train), len(X_test)


# ## Model
# 
# The model we will use is based around a bi-directional GRU, with dropout and two types of pooling. After pooling the GRU sequences, a single densely-connected layer will classify our input using the sigmoid activation function.

# In[ ]:


from keras.layers import Input, Dense, Bidirectional, GRU, Embedding, Dropout
from keras.layers import concatenate, SpatialDropout1D, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.models import Model

# Input shape
inp = Input(shape=(max_len,))

# Embedding and GRU
x = Embedding(max_features, 300)(inp)
x = SpatialDropout1D(0.33)(x)
x = Bidirectional(GRU(50, return_sequences=True))(x)

# Pooling
avg_pool = GlobalAveragePooling1D()(x)
max_pool = GlobalMaxPooling1D()(x)
conc = concatenate([avg_pool, max_pool])

# Output layer
output = Dense(1, activation='sigmoid')(conc)

model = Model(inputs=inp, outputs=output)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# model.load_weights('Weights/gru5.h5')
model.fit(X_train, Y_train, epochs=5, batch_size=32, verbose=1)


# Time to compute our results!

# In[ ]:


results = model.predict(X_test, batch_size=1, verbose=1)


# Since we are using the functional Keras model structure, our outputs are not classes, but probabilities in `[0, 1]`. Therefore, we need to convert these probabilities to actual class predictions. All probabilities above a certain threshold will be classified as `1` (fake) and anything below as `0` (real).

# In[ ]:


def convert_to_preds(results):
    """Converts probabilistic results in [0, 1] to
    binary values, 0 and 1."""
    return [1 if r > 0.5 else 0 for r in results]

preds = convert_to_preds(results)


# Use the prediction list from above, we are going to measure the percentile accuracy of each class, as well as the overall accuracy.

# In[ ]:


def accuracy_percentile(preds, Y_validate):
    """Return the percentage of correct predictions for each class and in total"""
    real_correct, fake_correct, total_correct = 0, 0, 0
    _, (fake_count, real_count) = np.unique(Y_validate, return_counts=True)

    for i, r in enumerate(preds):
        if r == Y_validate[i]:
            total_correct += 1
            if r == 0:
                fake_correct += 1
            else:
                real_correct += 1

    print('Real Accuracy:', real_correct/real_count * 100, '%')
    print('Fake Accuracy:', fake_correct/fake_count * 100, '%')
    print('Total Accuracy:', total_correct/(real_count + fake_count) * 100, '%')

accuracy_percentile(preds, Y_test)


# Finally, we are going to compute the F1 score as well.

# In[ ]:


from sklearn.metrics import f1_score

def accuracy_f1(preds, correct):
    """Returns F1-Score for predictions"""
    return f1_score(preds, correct, average='micro', labels=[0, 1])

accuracy_f1(preds, Y_test)

