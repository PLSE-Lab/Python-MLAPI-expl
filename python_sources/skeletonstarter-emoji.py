#!/usr/bin/env python
# coding: utf-8

# # Welcome to the Emojify Challenge!

# In[ ]:


##################################################
# Imports
##################################################

import numpy as np
import cv2
import os
import pandas as pd
import matplotlib.pyplot as plt
import emoji


##################################################
# Params
##################################################

DATA_BASE_FOLDER = '/kaggle/input/emojify-challenge'


##################################################
# Utils
##################################################

def label_to_emoji(label):
    """
    Converts a label (int or string) into the corresponding emoji code (string) ready to be printed
    """
    return emoji.emojize(emoji_dictionary[str(label)], use_aliases=True)


# # Dataset

# In[ ]:


##################################################
# Load dataset
##################################################

df_train = pd.read_csv(os.path.join(DATA_BASE_FOLDER, 'train.csv'))
y_train = df_train['class']
df_validation = pd.read_csv(os.path.join(DATA_BASE_FOLDER, 'validation.csv'))
y_validation = df_validation['class']
emoji_dictionary = {
    '0': '\u2764\uFE0F',
    '1': ':baseball:',
    '2': ':smile:',
    '3': ':disappointed:',
    '4': ':fork_and_knife:'
}

# See some data examples
print('EXAMPLES:\n####################')
for idx in range(10):
    print(f'{df_train["phrase"][idx]} -> {label_to_emoji(y_train[idx])}')


# # Word embeddings
# 
# Words can be represented as n-dimentional vectors where the distance between points has a correspondence respect to similarity between word semantics (similar words are closer, while dissimilar ones are distant). This representation is known as word embeddings and here is extrapolated and pre-computed from the [GloVe](https://nlp.stanford.edu/projects/glove/) model. 
# 
# Here is depicted an example of bi-dimensional word embeddings:
# ![word embedding](https://shanelynnwebsite-mid9n9g1q9y8tt.netdna-ssl.com/wp-content/uploads/2018/01/word-vector-space-similar-words.jpg)
# 
# In our case a single word is represented by a vector of length 25.
# 
# # Phrase representation
# 
# All the phrases are padded to the phrase of maximum length, in this case `max_len = 10`, and each phrase is represented by the concatenation of his word embeddings (each phrase thus is a 10 * 25 = 250 dimentional vector).

# In[ ]:


# Load phrase representation
x_train = np.load(
    os.path.join(DATA_BASE_FOLDER, 
                 'train.npy')).reshape(len(df_train), -1)
x_validation = np.load(
    os.path.join(DATA_BASE_FOLDER, 
                 'validation.npy')).reshape(len(df_validation), -1)
print(f'Word embedding size: {x_train.shape[-1]}')


# # Model
# 
# Here you have to implement a model (or more models, for finding the most accurate) for classification.
# 
# You can use the sklearn (or optionally other more advanced frameworks such as pytorch or tensorflow) package that contains a pool of models already implemented that perform classification. (SVMs, NNs, LR, kNN, ...)

# In[ ]:


##################################################
# Implement you model here
##################################################




# # Evaluation

# In[ ]:


##################################################
# Evaluate the model here
##################################################

# Use this function to evaluate your model
def accuracy(y_pred, y_true):
    '''
    input y_pred: ndarray of shape (N,)
    input y_true: ndarray of shape (N,)
    '''
    return (1.0 * (y_pred == y_true)).mean()

# Report the accuracy in the train and validation sets.







# # Send the submission for the challenge

# In[ ]:


##################################################
# Save your test prediction in y_test_pred
##################################################

y_test_pred = None

# Create submission
submission = pd.read_csv(os.path.join(DATA_BASE_FOLDER, 'sample_submission.csv'))
x_test = np.load(os.path.join(DATA_BASE_FOLDER, 'test.npy')).reshape(len(submission), -1)
if y_test_pred is not None:
    submission['class'] = y_test_pred
submission.to_csv('my_submission.csv', index=False)

