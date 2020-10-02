#!/usr/bin/env python
# coding: utf-8

# # Character-level Text Generation using LSTM

# In my previous kernel, I have explained the process of text generation at word level. Some of the limitations of that method is, the generated text makes sense only if you have huge amount of text to train with and also the problem of increase in dimensionality of the label due to the increase in number of words in the vocabulary.
# 
# To overcome these problems, you can create a Character lever text generation model, which considers only the unique characters present in your text. If you are considering English language, this would be A-Z(26) and some other puncutations. So, hardly your vocabulary limit will not exceed at max 100-150 (if your text has different symbols and punctuations) campared to the tens of thousands of words in a word level model.
# 
# Here, I will walk you through the process of building character level text generation using LSTM. The main task here is to predict the next character given all the previous characters in a sequence of data i.e., generates text character by character.
# 
# First, let's import the necessary libraries.

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import RegexpTokenizer
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import layers
from keras.models import Sequential
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping


# In[ ]:


# Loading the dataset
data = pd.read_json('/kaggle/input/quotes-dataset/quotes.json')
print(data.shape)
data.head()


# In[ ]:


# Dropping duplicates and creating a list containing all the quotes
quotes = data['Quote'].drop_duplicates()
print(f"Total Unique Quotes: {quotes.shape}")

# Considering only top 5000 quotes
quotes_filt = quotes[:4000]
print(f"Filtered Quotes: {quotes_filt.shape}")
all_quotes = list(quotes_filt)
all_quotes[:2]


# For text generation with character, we want our model to learn probabilities about what character will come next given the seed text/character. We then join these to our required length to form a sentence. 
# 
# We need to know what are all the different characters present in our text data. Since the Neural networks understand only numbers, we need to create a dictionary to store the character with an integer index, so that we can use the index as the proxy for the character (commonly known as Vocabulary).

# In[ ]:


# Converting the list of quotes into a single string
processed_quotes = " ".join(all_quotes)
processed_quotes = processed_quotes.lower()
processed_quotes[:100]


# In[ ]:


# Tokeinization
tokenizer = Tokenizer(char_level=True, oov_token='<UNK>')
seq_len = 100  #(Length of the training sequence)
sentences = []
next_char = []

# Function to create the sequences
def generate_sequences(corpus):
    tokenizer.fit_on_texts(corpus)
    total_vocab = len(tokenizer.word_index) + 1
    print(f"Total Length of characters in the Text: {len(corpus)}")
    print(f"Total unique characters in the text corpus: {total_vocab}")
    
    # Loop through the entire corpus to create input sentences of fixed length 100 and the next character which comes next with a step of 1
    for i in range(0, len(corpus) - seq_len, 1):
        sentences.append(corpus[i:i+seq_len])
        next_char.append(corpus[i+seq_len])
        
            
    return sentences, next_char, total_vocab

# Generating sequences
sentences, next_char, total_vocab = generate_sequences(processed_quotes)

print(len(sentences))
print(len(next_char))
print(sentences[:1])
print(next_char[:1])


# Now, we have the data in a more structured format, but still we cannot input as it is. We need convert the text to numbers and reshape the data in **[samples, timesteps, features]** format as expected by the LSTM.
# * First we will create a matrix of zeros for the required shape
# * Then we will create a one-hot encoding of the input sequence by looping over throug each character in the sequence.

# In[ ]:


# Create a matrix of required shape
X_t = np.zeros((len(sentences), seq_len, total_vocab), dtype=np.bool)
y_t = np.zeros((len(sentences), total_vocab), dtype=np.bool)

# Loop through each sentences and each character in the sentence and replace the respective position with value 1.
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X_t[i, t, tokenizer.word_index[char]] = 1
        y_t[i, tokenizer.word_index[next_char[i]]] = 1


# In[ ]:


print(f"The shape of X: {X_t.shape}")
print(f"The shape of Y: {y_t.shape}")


# In[ ]:


print(f"Corpus Length: {len(processed_quotes)}")
print(f"Vocab Length: {total_vocab}")
print(f"Total Sequences: {len(sentences)}")


# Thus, finally the data is now ready to feed into the model. The network architecture is:
# * Single LSTM layer with 256 units and dropout
# * One Dense hidden layer with 64 units
# * Final output Dense layer with units equal to our total vocabulary and a **softmax** activation
# * Since it is a multi-class classification we will use **categorial crossentropy** as loss and **adam** as optimizer.

# In[ ]:


# Building the model
def create_model():
    model = Sequential()
    model.add(layers.LSTM(256, dropout=0.5, input_shape = (X_t.shape[1], X_t.shape[2])))
    model.add(layers.Dense(total_vocab, activation='softmax'))
    
    # compiling the model
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

model = create_model()
model.summary()


# In[ ]:


import gc
gc.collect()


# In[ ]:


# Training the model
callback = EarlyStopping(monitor='loss', patience=2)
model.fit(X_t, y_t, epochs=100, batch_size=64, callbacks=[callback])


# In[ ]:


model.save("char_text_gen_quotesV2.h5")


# In[ ]:


# # Loading the model
# from keras.models import load_model

# char_model = load_model("../input/char-trained-model/char_text_gen_quotesV2.h5")


# In[ ]:


# char_model.summary()


# Now, we will create a prediction sampling function. What this function does is, it takes the prediction for the next character and scales it by the temperature value. Then recomputes the probability and samples the final prediction from a multimomial distribution simulation.
# 
# NOTE: A temperature < 1, makes the model to have high confident for the next character whereas temperature > 1 allows the model to be creative (also this leads to higher mistakes). A temperature value of 1, has no scaling and hence it is same as the prediction from the network.

# In[ ]:


# Prediction sampling
def sample(preds, temperature = 0.5):
    preds = np.asarray(preds).astype("float64")
    scaled_pred = np.log(preds)/temperature
    scaled_pred = np.exp(scaled_pred)
    scaled_pred = scaled_pred/np.sum(scaled_pred)
    scaled_pred = np.random.multinomial(1, scaled_pred, 1)
    return np.argmax(scaled_pred)


# In[ ]:


start_index = np.random.randint(0, len(processed_quotes) - seq_len - 1)
generated = ""
sentence = processed_quotes[start_index : start_index + seq_len].lower()

for i in range(1000):
    x_pred = np.zeros((1, seq_len, total_vocab))
    for t, char in enumerate(sentence):
        x_pred[0, t, tokenizer.word_index[char]] = 1.0
    preds = model.predict(x_pred, verbose=0)[0]
    next_index = sample(preds)
    next_char = tokenizer.index_word[next_index]
    sentence = sentence[1:] + next_char
    generated += next_char
    
print(generated)


# The model generates a decent understandable text. Try using different text corpus to get exciting results.
# 
# Thanks for reading. Happy Learning!!

# In[ ]:




