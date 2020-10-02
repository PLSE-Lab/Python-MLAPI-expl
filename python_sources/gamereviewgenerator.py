#!/usr/bin/env python
# coding: utf-8

# # Creating a game review generator using Metacritic game reviews
# 
# In this notebook I create a review generator based on game reviews from the "Metacritic Video Game Comments" dataset from Kaggle.
# Most of the code here is based on the following tutorial on text generation: https://www.analyticsvidhya.com/blog/2018/03/text-generation-using-python-nlp/

# In[ ]:


import numpy as np
import pandas as pd
import os
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import RNN
from keras.utils import np_utils
import re
from langdetect import detect
import matplotlib.pyplot as plt


# Loading the data

# In[ ]:


metacritic_game_user_comments = pd.read_csv("../input/metacritic-video-game-comments/metacritic_game_user_comments.csv")
metacritic_game_user_comments.head()


# # Data selection
# I chose to use only the Xenoblade Chronicles 2 reviews that gave a rating of 10. I chose to train on reviews from this game only, because this leaves a smaller dataset (+/- 150 reviews) which is needed to reduce training time. I chose to generate positive reviews only. I think this should result in a model trained for a more specific purpose and therefore should generate better texts.
# * Filter out all reviews except those of Xenoblade Chronicles 2.
# * Filter out all reviews that give a rating of 10. 
# 

# In[ ]:


xc2_reviews = metacritic_game_user_comments[(metacritic_game_user_comments['Title'] == 'Xenoblade Chronicles 2')]
xc2_reviews = xc2_reviews[(xc2_reviews['Userscore'] == 10)]
print(xc2_reviews.shape)


# Using only the Xenoblade Chronicles 2 reviews with a rating of 10 results in a dataset of 159 reviews.

# ### Cleaning the data
# * Remove the spoilers warning from the input reviews containing spoilers.
# * Detecting the language of reviews and selecting only English reviews.
# * Concatenating all column values into a single lowercase string.

# In[ ]:


# Print the last row to view the noice of the "This review contains spoilers.... " line
# And remove that part of the strings containing the substring
print(xc2_reviews.tail(1))
xc2_reviews['Comment'] = xc2_reviews['Comment'].str.replace('            This review contains spoilers, click expand to view.        ', '')

# Detect language and select only English reviews.
xc2_reviews['Language'] = xc2_reviews['Comment'].apply(detect)
xc2_reviews = xc2_reviews[(xc2_reviews['Language'] == 'en')]['Comment']
print(xc2_reviews.shape)


# As shown above, filtering out the non-english reviews removes 31 reviews from the dataset, leaving us with a nice, smaller dataset of 128 english reviews.

# In[ ]:


# Convert all column values into one lowercase string
xc2_reviews_string = '\n'.join(xc2_reviews.values).lower()
print(xc2_reviews_string[:500])


# Checking how many words this text contains to get an insight on the amount of training data.

# In[ ]:


# to count words in string 
res = len(re.findall(r'\w+', xc2_reviews_string)) 
res


# ### Character mapping
# Mapping the characters to numbers for our model.

# In[ ]:


characters = sorted(list(set(xc2_reviews_string)))

n_to_char = {n:char for n, char in enumerate(characters)}
char_to_n = {char:n for n, char in enumerate(characters)}


# ### Data Preprocessing
# Turning our data into sequences for the LSTM layers.

# In[ ]:


X = []
Y = []
input_length = len(xc2_reviews_string)
seq_length = 100

# Loop through the entire input string and create sequences of characters
for i in range(0, input_length - seq_length, 1):
    sequence = xc2_reviews_string[i:i + seq_length]
    label = xc2_reviews_string[i + seq_length]
    
    X.append([char_to_n[char] for char in sequence])
    Y.append(char_to_n[label])

print("Total Patterns:", len(X))


# In[ ]:


X_modified = np.reshape(X, (len(X), seq_length, 1))
X_modified = X_modified / float(len(characters))
Y_modified = np_utils.to_categorical(Y)


# ### Creating the model
# Adding an extra LSTM layer and another dropout layer resulted in the text generator starting to repeat itself. This is probably due to the model needing more training time, because the model became bigger. I will now train my model with 2 LSTM layers of 600 neurons isntead. Since the model is bigger, I will give it more training time.

# In[ ]:


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import RNN

model = Sequential()
model.add(LSTM(600, input_shape=(X_modified.shape[1], X_modified.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(600))
model.add(Dropout(0.2))
model.add(Dense(Y_modified.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')


# ### Training the model
# I've reset the batch size to 100, as it heavily speeds up training and didn't show any significant improvements regarding loss or the text generation. I've found out that the maximum runtime for a commit is actually 9 hours instead of 6, therefore I can let it train longer. 
# 

# In[ ]:


history = model.fit(X_modified, Y_modified, epochs=100, batch_size=100)
model.save_weights('xc2_review_generator_model_with_bigger_layers.h5')


# After fitting the model, we use the model history to plot the loss values during the training.

# In[ ]:


# https://keras.io/visualization/
# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train'], loc='upper left')
plt.show()


# ### Creating a method to generate text
# The first method creates a single string from the seperate characters.
# The second method generates a text of 400 characters, given an starting input text of 100 characters.

# In[ ]:


def combine_string(character_array):
    return_string = ""
    for char in character_array:
        return_string = return_string + char
    return return_string

def generate_review(starting_sequence):
    generated_review_mapped = [n_to_char[value] for value in starting_sequence]
    print(f'Starting sequence: {combine_string(generated_review_mapped)}')
    
    for i in range(400):
        x = np.reshape(starting_sequence,(1,len(starting_sequence), 1))
        x = x / float(len(characters))

        pred_index = np.argmax(model.predict(x, verbose=0))
        seq = [n_to_char[value] for value in starting_sequence]
        generated_review_mapped.append(n_to_char[pred_index])

        starting_sequence.append(pred_index)
        starting_sequence = starting_sequence[1:len(starting_sequence)]

    return generated_review_mapped


# ## Generating some reviews
# Now its time to generate some reviews. We generate reviews from some starting sequences from our dataset. 

# In[ ]:


generated_review_1 = generate_review(X[99].copy())
print(combine_string(generated_review_1))


# In[ ]:


generated_review_2 = generate_review(X[0].copy())
print(combine_string(generated_review_2))


# In[ ]:


generated_review_3 = generate_review(X[10].copy())
print(combine_string(generated_review_3))


# Finally, we generate a review from a custom starting sequence to see what result that leads to.

# In[ ]:


custom_starting_sequence = list('xenoblade chronicles 2 is a great game, like all games, it has its issues, but it runs smoothly and ')
custom_starting_sequence_mapped = [char_to_n[value] for value in custom_starting_sequence]

generated_custom_review = generate_review(custom_starting_sequence_mapped)
print(combine_string(generated_custom_review))

