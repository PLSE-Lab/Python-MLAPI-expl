#!/usr/bin/env python
# coding: utf-8

# # Generating text for NLP using SimpleRNN with keras
# 
# ### Table of interest:
# 1. Introduction.
# 2. Import data and preprocessing.
# 3. Model Building.
# 4. Model Training and Prediction.
# 5. Conclusion.
# 
# This exemple was simply taken in the book **Deep Learning with Keras**, *by Antonio Gulli and Sujit Pal, 2017.*
# 

# ## 1. Introduction.
# 
# RNNs have been used extensively by the **natural language processing (NLP)** community for various applications. One such application is building language models. A language model allows us to predict the probability of a word in a text given the previous words. Language models are important for various higher level tasks such as machine translation, spelling correction, and so on.
# 
# A side effect of the ability to predict the next word given previous words is a generative model that allows us to generate text by ***sampling from the output probabilities***. In language modeling, our input is typically a sequence of words and the output is a sequence of predicted words. The training data used is existing unlabeled text, where we set the label **y(t)** at time **t** to be the input **x(t+1)** at time **t+1**.
# 
# We will train a character based language model on the text of ***Alice in Wonderland*** to predict the next character given **10** previous characters. We have chosen to build a character-based model here because it has a smaller vocabulary and trains quicker. The idea is the same as using a **word-based language model**, except we use characters instead of words. We will then use the trained model to generate some text in the same style.
# 
# Let's get started.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


from __future__ import print_function
from keras.layers import Dense, Activation
from keras.layers.recurrent import SimpleRNN
from keras.models import Sequential


# ## 2. Import data and preprocessing.
# 
# We read our input text from the text of ***Alice in Wonderland*** on the Project Gutenberg website (https://www.gutenberg.org/files/11/11.txt). The file contains line breaks and non-ASCII characters, so we do some preliminary cleanup and write out the contents into a variable called text.

# In[ ]:


fin = open("../input/alice_in_wonderland.txt", 'rb')
lines = []
for line in fin:
    line = line.strip().lower()
    line = line.decode("ascii", "ignore")
    if len(line) == 0:
        continue
    lines.append(line)
fin.close()
text = " ".join(lines)


# In[ ]:


# set of characters that occur in the text
chars = set([c for c in text])
# Total items in our vocabulary
nb_chars = len(chars)
# lookup tables to deal with indexes of characters rather than the characters themselves.
char2index = dict((c, i) for i, c in enumerate(chars))
index2char = dict((i, c) for i, c in enumerate(chars))


# The next step is to create the input and label texts. We do this by stepping through the text by a numberof characters given by the `STEP` variable and then extracting a span of text whose size is determined by the `SEQLEN` variable. The next character after the span is our label character.

# In[ ]:


SEQLEN = 10
STEP = 1
input_chars = []
label_chars = []
for i in range(0, len(text) - SEQLEN, STEP):
    input_chars.append(text[i:i + SEQLEN])
    label_chars.append(text[i + SEQLEN])


# The next step is to vectorize these input and label texts. Each row of the input to the RNN
# corresponds to one of the input texts shown previously. There are `SEQLEN` characters in this input, and since our vocabulary size is given by `nb_chars`, we represent each input character as a **one-hot encoded vector** of size (`nb_chars`). Thus each input row is a tensor of size (`SEQLEN` and `nb_chars`). 
# 
# Our output label is a single character, so similar to the way we represent each character of our input, it is represented as a **one-hot vector of size** (`nb_chars`). Thus, the shape of each label is `nb_chars`.

# In[ ]:


X = np.zeros((len(input_chars), SEQLEN, nb_chars), dtype=np.bool)
y = np.zeros((len(input_chars), nb_chars), dtype=np.bool)
for i, input_char in enumerate(input_chars):
    for j, ch in enumerate(input_char):
        X[i, j, char2index[ch]] = 1
    y[i, char2index[label_chars[i]]] = 1


# ## 3. Model Building.
# 
# * We define the RNN's output dimension to have a size of **128**. This is a hyper-parameter that needs to be determined by experimentation. In general, if we choose too small a size, then the model does not have sufficient capacity for generating good text, and you will see long runs of repeating characters or runs of repeating word groups. On the other hand, if the value chosen is too large, the model has too many parameters and needs a lot more data to train effectively.
# * We want to return a single character as output, not a sequence of characters, so `return_sequences=False`.
# * In addition, we set `unroll=True` because it improves performance on the TensorFlow backend.
# * The RNN is connected to a dense (fully connected) layer. The dense layer has (nb_char) units, which emits scores for each of the characters in the vocabulary. The activation on the dense layer is a softmax, which normalizes the scores to probabilities. The character with the highest probability is chosen as the prediction.
# * We compile the model with the categorical cross-entropy loss function, a good loss function for categorical outputs, and the RMSprop optimizer.
# 
# Now let's review the code of model building.
# 

# In[ ]:


HIDDEN_SIZE = 128
BATCH_SIZE = 128
NUM_ITERATIONS = 25
NUM_EPOCHS_PER_ITERATION = 1
NUM_PREDS_PER_EPOCH = 100

model = Sequential()
model.add(SimpleRNN(HIDDEN_SIZE, return_sequences=False, 
                    input_shape=(SEQLEN, nb_chars), 
                    unroll=True))
model.add(Dense(nb_chars))
model.add(Activation("softmax"))

model.compile(loss="categorical_crossentropy", optimizer="rmsprop")


# ## 4. Model Training and Prediction
# * Our training approach is a little different from what we have seen so far. So far our approach has been to train a model for a fixed number of epochs, then evaluate it against a portion of held-out test data. Since we don't have any labeled data here, we train the model for an epoch(`NUM_EPOCHS_PER_ITERATION=1`) then test it. We continue training like this for **25** (`NUM_ITERATIONS=25`) iterations, stopping once we see intelligible output. So effectively, we are training for NUM_ITERATIONS epochs and testing the model after each epoch.
# 
# * Our test consists of generating a character from the model given a random input, then dropping the first character from the input and appending the predicted character from our previous run, and generating another character from the model. We continue this 100 times (`NUM_PREDS_PER_EPOCH=100`) and generate and print the resulting string. The string gives us an indication of the quality of the model.
# 
# Let's review the code.

# In[ ]:


for iteration in range(NUM_ITERATIONS):
    print("=" * 50)
    print("Iteration #: %d" % (iteration))
    model.fit(X, y, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS_PER_ITERATION)
    test_idx = np.random.randint(len(input_chars))
    test_chars = input_chars[test_idx]
    print("\nGenerating from seed: %s" % (test_chars))
    print(test_chars, end="")
    for i in range(NUM_PREDS_PER_EPOCH):
        Xtest = np.zeros((1, SEQLEN, nb_chars))
        for i, ch in enumerate(test_chars):
            Xtest[0, i, char2index[ch]] = 1
        pred = model.predict(Xtest, verbose=0)[0]
        ypred = index2char[np.argmax(pred)]
        print(ypred, end="")
        # move forward with test_chars + ypred
        test_chars = test_chars[1:] + ypred
print()


# As you can see, by the end of the 25th epoch, it has learned to spell reasonably well, although it has trouble expressing coherent thoughts. The amazing thing about this model is that it is character-based and has no knowledge of words, yet it learns to spell words that look like they might have come from the original text.

# ## 5. Conclusion.
# 
# Generating the next character or next word of text is not the only thing you can do with this sort of model. This kind of model has been successfully used to make **stock predictions** (for more information refer to the article: ***Financial Market Time Series Prediction with Recurrent Neural Networks, by A. Bernal, S. Fok, and R. Pidaparthi, 2012***) and generate classical music (for more information refer to the article: ***DeepBach: A Steerable Model for Bach Chorales Generation, by G. Hadjeres and F. Pachet, arXiv:1612.01010, 2016***), to name a few interesting applications. **Andrej Karpathy** covers a few other fun examples, such as generating fake Wikipedia pages, algebraic geometry proofs, and Linux source code in his blog post at: ***The Unreasonable Effectiveness of Recurrent Neural Networks at http://karpathy.github.io/2015/05/21/rnn-effectiveness/.***
# 
# ### References:
# * **Deep Learning with Keras**, *by Antonio Gulli and Sujit Pal, 2017.*

# **Hope that you find this notebook helpful. More to come.**
# 
# **Please upvote this, to keep me motivate for doing better.**
# 
# **Thanks.**
# 
