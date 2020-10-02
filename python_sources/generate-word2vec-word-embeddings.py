#!/usr/bin/env python
# coding: utf-8

# Started on 30 May 2019

# # Introduction

# #### Here I explore using word embeddings on the "Spooky Author Identification" datasets.
# #### I am using Gensim's Word2Vec to generate the word representations from all the text in the training and test sets.

# In[ ]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import re
import gensim 
from gensim.models import Word2Vec


# # Load data

# In[ ]:


train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
print(train_df.shape, test_df.shape)


# #### Let's clean up the text; make lowercase & remove punctuations. Then split the text into words. This prepares the text for use in Gensim's Word2Vec.

# In[ ]:


def clean_text(text):
    """
    Convert all to lowercase and remove punctuations
    """
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text) # remove everything that isn't word or space
    text = re.sub(r'\_', '', text)      # remove underscore
    return text


# In[ ]:


# clean train_df['text']
train_df['text'] = train_df['text'].map(lambda x: clean_text(x))
train_df['text'] = train_df['text'].map(lambda x: x.strip().split())
train_df.head()


# In[ ]:


# clean test_df['text']
test_df['text'] = test_df['text'].map(lambda x: clean_text(x))
test_df['text'] = test_df['text'].map(lambda x: x.strip().split())
test_df.head()


# # Generate word embeddings with Word2Vec

# #### Create corpus from all the words in the train & test datasets, and use Word2Vec to generate the word embeddings.

# In[ ]:


data = []  
# iterate through each row in train_df 
for i in range(len(train_df)):
    data.append(train_df['text'][i])
for j in range(len(test_df)):
    data.append(test_df['text'][j])


# In[ ]:


print(len(data))


# In[ ]:


# Create Word2Vec model using CBOW (sg=0)
# Set min_count to 1 so as to include all words
embedding = gensim.models.Word2Vec(data, size=50, window=10, min_count=1, sg=0)


# In[ ]:


print(embedding)


# #### The size of the vocabulary is 28,727 words. I am using a smaller dimension of 50 as I thought the vocab size is small.

# In[ ]:


# train & generate the embeddings
embedding.train(data,total_examples=len(data),epochs=30)


# # Examine the generated word embeddings

# #### Let's examine the generated Word2Vec model we have created.

# In[ ]:


words = list(embedding.wv.vocab)
print(len(words))


# #### Each word in the above corpus is expressed as a 50-dimension vector as shown below.

# In[ ]:


print(embedding['capered'])


# #### We can try pick random words in the corpus and find out what are the most similar words according to this generated Word2Vec model. Let's try a few examples.

# In[ ]:


embedding.most_similar('dark', topn=5)


# In[ ]:


embedding.most_similar('shocked', topn=5)


# In[ ]:


embedding.most_similar('sprang', topn=5)


# In[ ]:


embedding.most_similar('pride', topn=5)


# #### Looking at the above words listed as similar (vectors which are closer to each other), I thought the word embeddings are quite decent.

# # Use the word embeddings

# #### Let's use the generated Word2Vec embeddings for the "Spooky Authors Identification" problem.

# * One-hot encode the target variable to facilitate modelling.

# In[ ]:


# convert author labels into one-hot encodings
train_df['author'] = pd.Categorical(train_df['author'])
df_Dummies = pd.get_dummies(train_df['author'], prefix='author')
train_df = pd.concat([train_df, df_Dummies], axis=1)
# Check the conversion
train_df.head()


# ## Baseline model
# #### One simple way to use word embeddings would be to average the word embeddings of words in the text, and then feed into a softmax layer (3 classes) for training. To predict the class, I would pass the average word embeddings of the test text and determine the class with the highest probability.

# * Create X and Y from train_df, test_df, limiting to first 100 words.

# In[ ]:


X = train_df['text'].str[:100]
Y = train_df[['author_EAP', 'author_HPL', 'author_MWS']].values
print(X.shape, X[0], Y.shape, Y[0])


# In[ ]:


X_test = test_df['text'].str[:50]
print(X_test.shape, X_test[0])


# In[ ]:


def text_to_avg(text):
    """
    Given a list of words, extract the respective word embeddings
    and average the values into a single vector encoding the text meaning.
    """
    # initialize the average word vector
    avg = np.zeros((50,))
    # average the word vector by looping over the words in text
    for w in text:
        avg += embedding[w]
    avg = avg/len(text)
    return avg


# In[ ]:


X_avg = np.zeros((X.shape[0], 50)) # initialize X_avg
for i in range(X.shape[0]):
    X_avg[i] = text_to_avg(X[i])


# In[ ]:


print(X_avg.shape)
print(X_avg[0])


# In[ ]:


X_test_avg = np.zeros((X_test.shape[0], 50)) # initialize X_test_avg
for i in range(X_test.shape[0]):
    X_test_avg[i] = text_to_avg(X_test[i])


# In[ ]:


print(X_test_avg.shape)
print(X_test_avg[0])


# ### Split train data into a train and a dev set

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_dev, Y_train, Y_dev = train_test_split(X_avg, Y, test_size=0.2, random_state=123)
print(X_train.shape, Y_train.shape, X_dev.shape, Y_dev.shape)


# ### Build the dense neural network model from keras

# In[ ]:


from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(512, activation='relu', input_shape=(50,)))
model.add(layers.Dense(3, activation='softmax'))

model.summary()


# In[ ]:


# compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# ### Train the model

# In[ ]:


# train and validate the model
epochs = 50
history = model.fit(X_train, Y_train, epochs=epochs, batch_size=128, validation_data=(X_dev, Y_dev))


# In[ ]:


# plot and visualise the training and validation losses
loss = history.history['loss']
dev_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'bo', label='training loss')
plt.plot(epochs, dev_loss, 'b', label='validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# ## Re-train with the entire training set

# In[ ]:


model = models.Sequential()
model.add(layers.Dense(512, activation='relu', input_shape=(50,)))
model.add(layers.Dense(3, activation='softmax'))
# compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# In[ ]:


# train the model
epochs = 10
model.fit(X_avg, Y, epochs=epochs, batch_size=128)


# ### Predict on the test set and compute the probabilities for submission

# In[ ]:


# predict on test set
preds = model.predict(X_test_avg)
print(preds.shape)
print(preds[7])


# In[ ]:


# set the predicted labels to be the one with the highest probability
pred_labels = []
for i in range(len(X_test_avg)):
    pred_label = np.argmax(preds[i])
    pred_labels.append(pred_label)


# In[ ]:


print(pred_labels[7])


# # Create submission file

# In[ ]:


result = pd.DataFrame(preds, columns=['EAP','HPL','MWS'])
result.insert(0, 'id', test_df['id'])
result.head()


# In[ ]:


# Generate submission file in csv format
result.to_csv('submission.csv', index=False, float_format='%.20f')


# #### Thank you for reading this.
# #### Please upvote if you find it useful. Cheers!

# In[ ]:




