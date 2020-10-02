#!/usr/bin/env python
# coding: utf-8

# # IMDB Sentiments
# 
# ## Introduction
# 
# This notebook follows the Text Classification guide from Google Machine Learning Guides.<br/>
# This notebook contains all the code that the guide shows in the tutorial and not in its github repo. Hope this guide helps you as you follow the Text Classification guide.
# 
# Link to the Guide: https://developers.google.com/machine-learning/guides/text-classification/
# 
# In this notebook, we see how to perform sentiment analysis using IMDB Movie Reviews Dataset. We will classify reviews into `2` labels: _positive(`1`)_ and _negetive(`0`)_. And we will encode the data using tf-idf and feed into a Multi-layer Perceptron. We will use tensorflow, with Keras API.

# ## Loading the required modules
# 
# Let;s get started by loading all the required modules and defining all the constants and variables that we will be needing all throughout the notebook

# In[ ]:


import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

from tensorflow.python.keras import models
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Dropout

path = './../input/aclimdb/'


# ## Load the Dataset
# 
# In this section, let's load the dataset and shuffle it so to make ready for analysis.

# In[ ]:


def shuffle(X, y):
    perm = np.random.permutation(len(X))
    X = X[perm]
    y = y[perm]
    return X, y


# In[ ]:


def load_imdb_dataset(path):
    imdb_path = os.path.join(path, 'aclImdb')

    # Load the dataset
    train_texts = []
    train_labels = []
    test_texts = []
    test_labels = []
    for dset in ['train', 'test']:
        for cat in ['pos', 'neg']:
            dset_path = os.path.join(imdb_path, dset, cat)
            for fname in sorted(os.listdir(dset_path)):
                if fname.endswith('.txt'):
                    with open(os.path.join(dset_path, fname)) as f:
                        if dset == 'train': train_texts.append(f.read())
                        else: test_texts.append(f.read())
                    label = 0 if cat == 'neg' else 1
                    if dset == 'train': train_labels.append(label)
                    else: test_labels.append(label)

    # Converting to np.array
    train_texts = np.array(train_texts)
    train_labels = np.array(train_labels)
    test_texts = np.array(test_texts)
    test_labels = np.array(test_labels)

    # Shuffle the dataset
    train_texts, train_labels = shuffle(train_texts, train_labels)
    test_texts, test_labels = shuffle(test_texts, test_labels)

    # Return the dataset
    return train_texts, train_labels, test_texts, test_labels


# Now, lets load the dataset and perform some analysis on the dataset!

# In[ ]:


trX, trY, ttX, ttY = load_imdb_dataset(path)

print ('Train samples shape :', trX.shape)
print ('Train labels shape  :', trY.shape)
print ('Test samples shape  :', ttX.shape)
print ('Test labels shape   :', ttY.shape)


# Okay, that's 25K samples in each train and test sets! Now, from here on we will do analysis only on the train set (we want no snooping bias!)
# 
# Alright, now we have `2` classes as we divided them, one for positive `1` and one for negetive `0`. Let's just verify that!
# 
# And let's also verify the number of samples that are present in each class.

# In[ ]:


uniq_class_arr, counts = np.unique(trY, return_counts=True)

print ('Unique classes :', uniq_class_arr)
print ('Number of unique classes : ', len(uniq_class_arr))

for _class in uniq_class_arr:
    print ('Counts for class ', uniq_class_arr[_class], ' : ', counts[_class])


# Okay, so that's expected! So, everything's fine!
# 
# Now, let;s take a few random samples and check if the labels are expected!
# 
# And, the counts for each class are also even!! Each class has `12500` samples! Alright!

# In[ ]:


size_of_samp = 10
rand_samples_to_check = np.random.randint(len(trX), size=size_of_samp)

for samp_num in rand_samples_to_check:
    print ('============================================================')
    print (trX[samp_num], '||', trY[samp_num])
    print ('============================================================')


# Okay, so reading the reviews, the labels are expected, so we are good!
# 
# Now, let's see the average number of words per sample!

# In[ ]:


plt.figure(figsize=(15, 10))
plt.hist([len(sample) for sample in list(trX)], 50)
plt.xlabel('Length of samples')
plt.ylabel('Number of samples')
plt.title('Sample length distribution')
plt.show()


# Let's now plot a frequency distribution plot of the most seen words in the corpus.

# In[ ]:


kwargs = {
    'ngram_range' : (1, 1),
    'dtype' : 'int32',
    'strip_accents' : 'unicode',
    'decode_error' : 'replace',
    'analyzer' : 'word'
}

vectorizer = CountVectorizer(**kwargs)
vect_texts = vectorizer.fit_transform(list(trX))
all_ngrams = vectorizer.get_feature_names()
num_ngrams = min(50, len(all_ngrams))
all_counts = vect_texts.sum(axis=0).tolist()[0]

all_ngrams, all_counts = zip(*[(n, c) for c, n in sorted(zip(all_counts, all_ngrams), reverse=True)])
ngrams = all_ngrams[:num_ngrams]
counts = all_counts[:num_ngrams]

idx = np.arange(num_ngrams)

plt.figure(figsize=(30, 30))
plt.bar(idx, counts, width=0.8)
plt.xlabel('N-grams')
plt.ylabel('Frequencies')
plt.title('Frequency distribution of ngrams')
plt.xticks(idx, ngrams, rotation=45)
plt.show()


# Well, the highest frequency words are the stop words. We not consider them while performing our analysis, as they don't provide insights as to what the sentiment of the document might be or to which class a document might belong.

# ## Prepare the data
# 
# Let's now prepare the data to feed into the model. For the data preparation step we will get bigrams and unigrams from the data and encode it using tf-idf. And will select the top `20000` features from the vector of tokens. Discard features that occurs less than two times, and will `f_classif` to get feature importance.

# In[ ]:


NGRAM_RANGE = (1, 2)
TOP_K = 20000
TOKEN_MODE = 'word'
MIN_DOC_FREQ = 2

def ngram_vectorize(train_texts, train_labels, val_texts):
    kwargs = {
        'ngram_range' : NGRAM_RANGE,
        'dtype' : 'int32',
        'strip_accents' : 'unicode',
        'decode_error' : 'replace',
        'analyzer' : TOKEN_MODE,
        'min_df' : MIN_DOC_FREQ,
    }
    
    # Learn Vocab from train texts and vectorize train and val sets
    tfidf_vectorizer = TfidfVectorizer(**kwargs)
    x_train = tfidf_vectorizer.fit_transform(train_texts)
    x_val = tfidf_vectorizer.transform(val_texts)
    
    # Select best k features, with feature importance measured by f_classif
    selector = SelectKBest(f_classif, k=min(TOP_K, x_train.shape[1]))
    selector.fit(x_train, train_labels)
    x_train = selector.transform(x_train).astype('float32')
    x_val = selector.transform(x_val).astype('float32')
    return x_train, x_val


# ## Build, Train and Evaluate the model
# 
# First, let's create a function that returns the appropriate number of units and the activation for the last layer.

# In[ ]:


def get_last_layer_units_and_activation(num_classes):
    if num_classes == 2:
        activation = 'sigmoid'
        units = 1
    else:
        activation = 'softmax'
        units = num_classes
    return units, activation


# Let's now create the model using the Keras API from tensorflow

# In[ ]:


def mlp_model(layers, units, dropout_rate, input_shape, num_classes):
    op_units, op_activation = get_last_layer_units_and_activation(num_classes)
    model = models.Sequential()
    model.add(Dropout(rate=dropout_rate, input_shape=input_shape))
    
    for _ in range(layers-1):
        model.add(Dense(units=units, activation='relu'))
        model.add(Dropout(rate=dropout_rate))
        
    model.add(Dense(units=op_units, activation=op_activation))
    return model


# Now, let's train the model

# In[ ]:


def train_ngram_model(data, learning_rate=1e-3, epochs=1000, batch_size=128, layers=2, units=64, 
                      dropout_rate=0.2):
    
    num_classes = 2
    
    # Get the data
    trX, trY, ttX, ttY = data
    
    # Verify the validation labels
    '''
    unexpected_labels = [v for v in ttY if v not in range(num_classes)]
    if len(unexpected_labels):
        raise ValueError('Unexpected label values found in the validation set:'
                         ' {unexpected_labels}. Please make sure that the labels'
                         ' in the validation set are in the same range as '
                         'training labels.'.format(unexpected_labels=unexpected_labels))
    '''
    
    # Vectorize the data
    x_train, x_val = ngram_vectorize(trX, trY, ttX)
    
    # Create model instance
    model = mlp_model(layers, units=units, dropout_rate=dropout_rate,
                      input_shape=x_train.shape[1:], num_classes=num_classes)
    
    # Compile model with parameters
    if num_classes == 2:
        loss = 'binary_crossentropy'
    else:
        loss = 'sparse_categorical_crossentropy'
    optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
    model.compile(optimizer=optimizer, loss=loss, metrics=['acc'])
    
    # Create callback for early stopping on validation loss. If the loss does
    # not decrease on two consecutive tries, stop training
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)]
    
    # Train and validate model
    history = model.fit(x_train, trY, epochs=epochs, validation_data=(x_val, ttY),
                        verbose=2, batch_size=batch_size, callbacks=callbacks)
    
    # Print results
    history = history.history
    val_acc = history['val_acc'][-1]
    val_loss = history['val_loss'][-1]
    print ('Validation accuracy: {acc}, loss: {loss}'.format(
            acc=val_acc, loss=val_loss))
    
    # Save model
    model.save('IMDB_mlp_model_' + str(val_acc) + '_' + str(loss) + '.h5')
    return val_acc, val_loss


# Alright!!!!<br/>
# Let's now call `train_ngram_model` and build the model!!

# In[ ]:


results = train_ngram_model((trX, trY, ttX, ttY))

print ('With lr=1e-3 | val_acc={results[0]} | val_loss={results[1]}'.format(results=results))
print ('===========================================================================================')


# In[ ]:


results


# Above we can see we are getting a whooping `90`% accuracy!!!<br/>
# But take a look at the `acc`, its getting a `97`% accuracy! This indicates a clear overfitting problem.

# ## Tune Hyperparameters
# 
# The above model is not tuned. So, take your time to find the best set of hyperparameters!<br/>
# See this page for more details: https://developers.google.com/machine-learning/guides/text-classification/step-5

# ## Deploy your model
# 
# Now, go deploy your model. See here: https://developers.google.com/machine-learning/guides/text-classification/step-6
