#!/usr/bin/env python
# coding: utf-8

# # This is a notebook to analyze the sentiment of a sentence using the IMDb movie review dataset. Although this model is trained on reviews on IMDb but it could be generalised to the general English sentences as well. Feel free to play around with this model!

# ### Import necessary libraries

# In[ ]:


from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense
from keras.layers.embeddings import Embedding
from keras.layers import Flatten
from keras.callbacks import EarlyStopping
import string
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# ### Instantiate top 10,000 words from the dataset.

# In[ ]:


top_words = 10000
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=top_words)


# ### Handle the reviews with unknown words.

# In[ ]:


word_dict = imdb.get_word_index()
word_dict = { key:(value + 3) for key, value in word_dict.items() }
word_dict[''] = 0                                                    # Padding
word_dict['>'] = 1                                                   # Start
word_dict['?'] = 2                                                   # Unknown word
reverse_word_dict = { value:key for key, value in word_dict.items() }
print(' '.join(reverse_word_dict[id] for id in x_train[0]))


# Restrict the max_review_length to 500

# In[ ]:


max_review_length = 500
x_train = sequence.pad_sequences(x_train, maxlen=max_review_length)
x_test = sequence.pad_sequences(x_test, maxlen=max_review_length)


# ### Train a sequential model.

# In[ ]:


embedding_vector_length = 32
model = Sequential()
model.add(Embedding(top_words, embedding_vector_length, input_length=max_review_length))
model.add(Flatten())
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())


# ### Fit the model with EarlyStopping enabled.

# In[ ]:


es = EarlyStopping(monitor='val_loss', verbose=1, mode='min', patience=100)
hist = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=1000, batch_size=128, callbacks=[es], verbose=0)

_, train_acc = model.evaluate(x_train, y_train, verbose=1)
_, test_acc = model.evaluate(x_test, y_test, verbose=1)
print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))


# ### Accuracy Plot

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')

sns.set()
acc = hist.history['accuracy']
val = hist.history['val_accuracy']
epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, '-', label='Training accuracy')
plt.plot(epochs, val, ':', label='Validation accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='upper left')
plt.plot()


# ### Loss Plot

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')

sns.set()
loss = hist.history['loss']
val = hist.history['val_loss']
epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, '-', label='Training loss')
plt.plot(epochs, val, ':', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper left')
plt.plot()


# In[ ]:


scores = model.evaluate(x_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1] * 100))


# ### A function taking an English sentence as input and invoking the trained model to analyze the sentiment of the input sentence.

# In[ ]:


def analyze(text):
    # Prepare the input by removing punctuation characters, converting
    # characters to lower case, and removing words containing numbers
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)
    text = text.lower().split(' ')
    text = [word for word in text if word.isalpha()]

    # Generate an input tensor
    input = [1]
    for word in text:
        if word in word_dict and word_dict[word] < top_words:
            input.append(word_dict[word])
        else:
            input.append(2)
    padded_input = sequence.pad_sequences([input], maxlen=max_review_length)

    # Invoke the model and return the result
    result = model.predict(np.array([padded_input][0]))[0][0]
    return result


# ### Play with random sentences to check the sentiment score between 0 and 1, closer to 0 being a negative sentiment and closer to 1 being a positive sentiment.

# In[ ]:


analyze('Easily the most stellar experience I have ever had.')


# In[ ]:


analyze('This is the shittiest thing I have ever heard!') # Perhaps "shittiest" would be an outlier. (Need to take care of in future models!) 


# In[ ]:


analyze('I had a really bad experience with the customer service.')


# In[ ]:


analyze('This film is a once-in-a-lifetime opportunity')

