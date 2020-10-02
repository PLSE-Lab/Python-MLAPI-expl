#!/usr/bin/env python
# coding: utf-8

# Let's see if we can train a Tensorflow model to play blackjack, a popular card game. For those of you who don't know, these are the rules:
# 
# -Each player starts out with 2 cards. In each round, a dealer will offer them a card, and they will say "hit" (to take it), or "pass". 
# 
# -The object of the game is to have the sum of your cards as close to 21 as possible, without going over.
# 
# -Face cards (jack, queen, king) all count as 10.
# 
# -Ace can count as 1 or 11, whichever the player wants.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in theread-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# First, we define a function to calculate the score of the hand. This function assumes Ace counts as 1 (more with that later)

# In[ ]:


def score(hand):
    new_hand = []
    for i in hand:
        if i < 11:
            new_hand.append(i)
        else:
            new_hand.append(10)
    score = sum(new_hand)
    return score


# If your score is 21 or over, you shouldn't draw any more cards.

# In[ ]:


def stillin(array):
    if score(array) < 21:
        if 1 in array and score(array) == 11:
            return False
        else:
            return True
    return False
            


# I'm going to create data to train the model with. Basically, we are generating random hands of cards (here Ace is 1, Jack/Queen/King are 11/12/13) that are "still in" (meaning their score is not over 21).

# In[ ]:


train = []

for i in range(1000): ## initialize hands
    hand = [np.random.randint(1, 14), np.random.randint(1, 14)]
    while stillin(hand):
        train.append(hand.copy())
        hand.append(np.random.randint(1,14))


# In[ ]:


len(train)


# Now, we generate the result- whether or not the model should draw a card. I'm doing this by having it "draw another card" (adding another number), and seeing it it pushes the score over 21 or not.

# In[ ]:


results = []
for hand in train:
    hit = np.random.randint(1,14)
    if score(hand) + hit > 21:
        results.append(0)
    else:
        results.append(1)


# In[ ]:


len(results)


# All the input for Tensorflow has to be the same shape.

# In[ ]:


maxlen = max([len(i) for i in train])
for hand in train:
    hand += [0] * (maxlen - len(hand))


# In[ ]:


np.shape(train)


# Here, we're training a simple Tensorflow 2 Sequential model with Dense layers to learn to play. For more information, see https://www.tensorflow.org/api_docs/python/tf/keras/Sequential?version=nightly

# In[ ]:


import tensorflow as tf

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(64, input_shape = (maxlen,), activation = 'relu'))
model.add(tf.keras.layers.Dense(64, activation = 'relu'))
model.add(tf.keras.layers.Dense(1, activation = 'sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics = ['accuracy'])


# In[ ]:


history = model.fit(np.array(train), np.array(results),epochs=50)


# We can also use the KerasTuner to help us find the optimal parameters for our model. For more information, visit: https://www.tensorflow.org/tutorials/keras/keras_tuner

# In[ ]:


get_ipython().system('pip install keras-tuner')


# In[ ]:


import kerastuner as kt
def build_model(hp):
    model = tf.keras.Sequential()
    hp_units = hp.Int('units', min_value = 16, max_value = 512, step = 16)
    
    model.add(tf.keras.layers.Dense(hp_units,activation = 'relu', input_shape = (maxlen,)))
    model.add(tf.keras.layers.Dense(16, activation = 'relu'))
    model.add(tf.keras.layers.Dense(1, activation = 'sigmoid'))
    
    hp_learning_rate = hp.Choice('learning_rate', values = [1e-3, 1e-4, 1e-5]) 
    
    model.compile(loss = 'binary_crossentropy', optimizer = tf.keras.optimizers.Adam(hp_learning_rate),
                  metrics = ['accuracy'])
    return model


# In[ ]:


tuner = kt.Hyperband(build_model,
                     objective = 'accuracy', 
                     max_epochs = 10,
                     factor = 3)  


# In[ ]:


tuner.search(train, results, epochs = 100, verbose = 2)


# Now, we can grab the best model and train it.

# In[ ]:


best_hps = tuner.get_best_hyperparameters(num_trials = 1)[0]
model = tuner.hypermodel.build(best_hps)
history = model.fit(np.array(train), np.array(results),epochs=50, verbose = 2)


# In[ ]:


def print_hand(hand):
    cards = {1: 'A', 2: '2', 3:'3', 4:'4', 5:'5', 6:'6', 7:'7', 8:'8', 9:'9', 
            10:'10', 11:'J', 12:'Q', 13:'K'}
    print("Hand: ", end = "")
    for i in hand:
        if i > 0:
            print(cards[i], end= " ")
    print("\nScore =" , score(hand), end = " ")
    if 1 in hand:
        print("or", score(hand) + 10)
    print("")
    if score(hand) > 21:
        print("Model lost!")
        return False
    return True


# In[ ]:


hit_dict = {0: "Pass", 1:"Hit"}
def play():
    hand = [np.random.randint(1,14),np.random.randint(1,14)] + [0] * (maxlen - 2)
    hit = True
    while(hit):
        if (print_hand(hand)):
            hit = np.round(model.predict([hand]))
            print("Model: " + hit_dict[hit[0][0]] + "\n")
            hand[np.count_nonzero(hand)] = np.random.randint(1,14)
        else:
            break


# Now let's play!

# In[ ]:


play()


# In[ ]:


play()


# In[ ]:


play()


# In[ ]:




