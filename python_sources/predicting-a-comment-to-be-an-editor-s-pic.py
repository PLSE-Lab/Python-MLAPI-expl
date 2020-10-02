#!/usr/bin/env python
# coding: utf-8

# 

# ## Helper Methods

# <b>The code( from github) in below cell has been used in my code
# It has some really usefull method for preparing the training data
# 
# https://github.com/alexander-rakhlin/CNN-for-Sentence-Classification-in-Keras/blob/master/data_helpers.py

# In[ ]:




import numpy as np
import re
import itertools
from collections import Counter

"""
Original taken from https://github.com/dennybritz/cnn-text-classification-tf
"""

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

   
def pad_sentences(sentences, padding_word="<PAD/>"):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    sequence_length = max(len(x) for x in sentences)
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    return padded_sentences


def build_vocab(sentences):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return [vocabulary, vocabulary_inv]


def build_input_data(sentences, labels, vocabulary):
    """
    Maps sentencs and labels to vectors based on a vocabulary.
    """
    x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
    y = np.array(labels)
    return [x, y]


def load_data(resampled_data):
    """
    Loads and preprocessed data for the MR dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    sentences, labels = load_data_and_labels(resampled_data)
    sentences_padded = pad_sentences(sentences)
    vocabulary, vocabulary_inv = build_vocab(sentences_padded)
    x, y = build_input_data(sentences_padded, labels, vocabulary)
    return [x, y, vocabulary, vocabulary_inv]


# ## My code beyond this point.

# In[ ]:


def load_data_and_labels(resampled_data):
        x_text = [clean_str(sent) for sent in resampled_data.commentBody]
        x_text = [s.split(" ") for s in x_text]
        # Generate labels
        y = resampled_data.editorsSelection
        return [x_text, y]
    


# In[ ]:


import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore')


# In[ ]:


import os
print(os.listdir("../input"))
comments = pd.read_csv("../input/CommentsMarch2018.csv")


# ## Analyzing the data

# In[ ]:


comments.describe()


# In[ ]:


comments.loc[1:6, 'commentBody':'editorsSelection']


# In[ ]:


comments.info()


# So we can see there are about 250k comments and 34 columns for the month of March 2018, huge data, now lets see how does the editor's pick distribution looks like.

# **Let's take a look at some word clouds that I found interesting **

# In[ ]:


from wordcloud import WordCloud, STOPWORDS
# Defining the wordCloud method.
def generate_wordCloud(text, title):
    stopwords = set(STOPWORDS)
    extra_stopwords = {'one', 'al','et', 'br', 'Po', 'th', 'sayi', 'fr','wi', 'Unknown','co'}
    stopwords = stopwords.union(extra_stopwords)
    wc = WordCloud(stopwords=stopwords,
                  max_font_size=100,
                  max_words=500,
                  random_state=30,
                  background_color='white',mask=None).generate(str(text))

    plt.figure(figsize=(10,6))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis('off') # dont show the axes
    plt.title(title, fontdict={'size': 35,'color':"brown" ,
                                  'verticalalignment': 'bottom'})
    plt.show()


# In[ ]:


generate_wordCloud(comments.commentBody, "Most Common words in March 2018 Comments")


# In[ ]:


editor_selected_comments = comments[comments.editorsSelection==1]
non_editor_selected_comments =comments[comments.editorsSelection==0]


# In[ ]:


generate_wordCloud(non_editor_selected_comments.commentBody, "Common words in Non Editor's picked Comments")


# In[ ]:


generate_wordCloud(editor_selected_comments.commentBody,"Common words in Editor's picked Comments")


# **Looks like the editor's selection mainly involves comments mostly related to Trump, which is interesting.**

# In[ ]:


# plotting the comments vs editor's Selection in bar plot to see how data is distributed.
x = comments.editorsSelection.unique()
plt.bar(x, comments.editorsSelection.value_counts().values)
plt.xticks(x, [False,True])
plt.ylabel("Comments Frequency")
plt.xlabel("Editors Pick")
plt.title("Comments Freq Vs Editor's pick for March 2018", color="b")
plt.show()


# In[ ]:


len(editor_selected_comments)


# In[ ]:


(len(editor_selected_comments)/len(comments.editorsSelection))*100


#     1. So we can see that only about 2% of the comments are Editor Picks, about 98% are normal comments
#     2. To train any model we have to adjust our training data and test data to make sure we don't have any false positives.
#         that is resampling of the data. There are two methods of resampling.
#         a. Undersampling the data
#         b. Oversampling the data

# Before we do any of that, I want to see how do the recommendations looks in these two data sets(just curious)

# In[ ]:


plt.figure(figsize=(10,6))
plt.subplot(121)
editor_selected_comments.recommendations.plot.hist(title="Editor's Selected Comments")
plt.subplot(122)
non_editor_selected_comments.recommendations.plot.hist(title="Normal Comments")


# <b>It is interesting to see that none of the comments that have not been Editor's pick have any recommendations.

# ## Preparing the data

# In[ ]:


# Now lets try to do the undersampling which is rather straightforward in this case.
# We can get n times the editor's selected data for our normal comments data set.

normal_comments_indices = np.array(non_editor_selected_comments.index)
selected_comments_indices = np.array(editor_selected_comments.index)

#print("Selected Comments Indices Sample :: ",selected_comments_indices[:5])
#print("Normal Comments Indices Sample :: ",normal_comments_indices[:5])

normal_indices_undersample = np.array(np.random.choice(normal_comments_indices, 3*len(selected_comments_indices)))

print("Length of underSampled normal indices :: ",len(normal_indices_undersample))

resampled_data_indices = np.concatenate([selected_comments_indices, normal_indices_undersample])


resampled_data = comments.iloc[resampled_data_indices]

print("Length of resampled Data :: ",len(resampled_data))


# In[ ]:


sentences, labels, vocabulary, vocabulary_env= load_data(resampled_data)
    


# In[ ]:


from itertools import islice
def take(n, iterable):
    "Return first n items of the iterable as a list"
    return list(islice(iterable, n))

n_items = take(5, vocabulary.items())
print("Showing first 5 items of Vocabulary")
print(n_items)


# In[ ]:


resampled_data.commentBody[1:2].values[0]


# In[ ]:


print("Mapping of above comment into vector embedding")
sentences[1]


# In[ ]:


train_len = int(len(sentences) * 0.9)
print("Number of training samples ",train_len)
x_train = sentences[:train_len]
y_train = labels[:train_len]
x_test = sentences[train_len:]
y_test = labels[train_len:]

print("Number of test samples ", len(sentences)-train_len)


# ## Building the model

# 
# <b>Alright enough.. let's build the model now.. 
# 

# In[ ]:


import keras
print("Keras version is ", keras.__version__)


# In[ ]:


from keras.models import Sequential
from keras.layers import Dense, Flatten, Convolution1D, MaxPooling1D
from keras.layers.embeddings import Embedding

input_length = len(sentences[1])
input_dim = len(vocabulary)
print("Vocabulary length is ",input_dim)
print("Input Length is ",input_length)


model = Sequential()
# Adding the Embedding Layer which will return an embedding layer with weights.
model.add(Embedding(input_dim=input_dim, output_dim=100,input_length=input_length))
# Adding Concolution1D layer with kernal_size of 3, we are using 1D Convolutional later since
# we are dealing with text words,

model.add(Convolution1D(filters=100, kernel_size=3, activation="relu"))
# Max Pooling the output from above layer,pool_size is supposed to reduced them in half.
model.add(MaxPooling1D(pool_size=2))

# Flattening the output before passing it to Dense layer
model.add(Flatten())
# activation layer relu which will give us the best of 
model.add(Dense(250, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])


# In[ ]:


model.summary()


# In[ ]:


model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=128, verbose=2)
# Final evaluation of the model
scores = model.evaluate(x_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))


# <b> Try adding a 20% dropout

# In[ ]:


## We noticed that after 5 epochs the accuracy is 100%, this could be due to overfitting.

## lets try adding a dropout layer 

from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Dropout

model = Sequential()

# Adding the Embedding Layer which will return an embedding layer with weights.
model.add(Embedding(input_dim=input_dim, output_dim=100,input_length=input_length))

# Adding Concolution1D layer with kernal_size of 3, we are using 1D, since we are dealing with text words,
model.add(Convolution1D(filters=100, kernel_size=3, activation="relu"))

# Max Pooling the output from above layer,pool_size is supposed to reduced them in half.
model.add(MaxPooling1D(pool_size=2))

model.add(Flatten())

model.add(Dropout(rate=.2)) # adding a dropout layer for 20%

model.add(Dense(500, activation="linear"))
model.add(LeakyReLU(alpha=.01)) 
model.add(Dense(1, activation="sigmoid"))

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])


# In[ ]:


model.summary()


# In[ ]:


model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=5, batch_size=128, verbose=2)
# Final evaluation of the model
scores = model.evaluate(x_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))


# ## Conclusion

# The initial result that I got was 88.09%, which is pretty decent. I would need to train this with more data set to see how it behaves.
# and I am going to try Word2Vec and GLove as well and then compare the results
# 
# Any feedback to improve the model is more than welcome, I would love to optimize it and see how it behaves over the complete data set

# In[ ]:




