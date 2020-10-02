#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import math
from tqdm import tqdm
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split


# # Setup

# In[ ]:


train_df = pd.read_csv("../input/ndsc-beginner/train.csv")
train_df = train_df.sample(frac=1.)
val_df = train_df[:1000]
train_df = train_df[1000:]
val_df.head()


# In[ ]:


# Embdedding setup, save it in a dictionary for easier queries
embeddings_index = {}
f = open('../input/tutorial-how-to-train-your-custom-word-embedding/custom_glove_100d.txt')
for line in tqdm(f):
    values = line.split(" ")
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))


# In[ ]:


# Convert values to embeddings
def text_to_array(text):
    empyt_emb = np.zeros(100)
    text = text[:-1].split()[:100]
    embeds = [embeddings_index.get(x, empyt_emb) for x in text]
    embeds+= [empyt_emb] * (100 - len(embeds))
    return np.array(embeds)


# In[ ]:


val_vects = np.array([text_to_array(X_text) for X_text in (val_df["title"][:])])
val_y_labels = np.array(val_df["Category"])
# val_y = np.zeros((len(val_y_labels), 58))
# val_y[np.arange(len(val_y_labels)), val_y_labels] = 1


# In[ ]:


# Understand what a batch is made of
batch_size = 128
i = 0
texts = train_df.iloc[i*batch_size:(i+1)*batch_size, 1]
text_arr = np.array([text_to_array(text) for text in texts])
batch_labels = np.array(train_df["Category"][i*batch_size:(i+1)*batch_size])
batch_targets = np.zeros((batch_size, 58))
batch_targets[np.arange(batch_size), batch_labels] = 1
print(np.shape(text_arr))
print(np.shape(batch_targets))
print(text_arr)
print(batch_targets)


# In[ ]:


# Write generator, which 
batch_size = 128

def batch_gen(train_df):
    n_batches = math.floor(len(train_df) / batch_size)
    while True: 
        train_df = train_df.sample(frac=1.)  # Shuffle the data.
        for i in range(n_batches):
            texts = train_df.iloc[i*batch_size:(i+1)*batch_size, 1]
            text_arr = np.array([text_to_array(text) for text in texts])
            batch_labels = np.array(train_df["Category"][i*batch_size:(i+1)*batch_size])
            yield text_arr, batch_labels


# # Training

# In[ ]:


from keras.models import Sequential
from keras.layers import CuDNNLSTM, Dense, Bidirectional, Activation, Dropout


# In[ ]:


import matplotlib.pyplot as plt

def plot_history(history):
    plt.figure(figsize=(12,6))
    plt.plot(history.history["loss"], color="purple")
    plt.plot(history.history["acc"], color="blue")
    plt.plot(history.history["val_loss"], color="red")
    plt.plot(history.history["val_acc"], color="green")
    plt.xlim(0,)
    plt.ylim(0.5,1.5)
    plt.legend(['loss', 'acc', "val_loss", "val_acc"], loc='upper right')
    plt.show()


# In[ ]:


model = Sequential()
model.add(Bidirectional(CuDNNLSTM(128, return_sequences=True),
                        input_shape=(100, 100)))
model.add(Dropout(0.05))
model.add(Bidirectional(CuDNNLSTM(128)))
model.add(Dense(58))
model.add(Activation('softmax'))

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


# In[ ]:


mg = batch_gen(train_df)
history = model.fit_generator(mg, epochs=90,
                    steps_per_epoch=1000,
                    validation_data=(val_vects, val_y_labels),
                    verbose=True)


# In[ ]:


plot_history(history)


# In[ ]:


history = model.fit_generator(mg, epochs=1,
                    steps_per_epoch=1000,
                    validation_data=(val_vects, val_y_labels),
                    verbose=True)


# # Inference

# In[ ]:


# Make the prediction from the model
batch_size = 256
def batch_gen(test_df):
    n_batches = math.ceil(len(test_df) / batch_size)
    for i in range(n_batches):
        texts = test_df.iloc[i*batch_size:(i+1)*batch_size, 1]
        text_arr = np.array([text_to_array(text) for text in texts])
        yield text_arr

test_df = pd.read_csv("../input/ndsc-beginner/test.csv")
test_df["Supercategory"] = test_df["image_path"].str[0]
supercats = np.array(test_df["Supercategory"])
supercat_dict = {
    "b" : np.array([1]*17 + [0]*14 + [0]*27),
    "f" : np.array([0]*17 + [1]*14 + [0]*27),
    "m" : np.array([0]*17 + [0]*14 + [1]*27)
}

all_preds = []
for x in tqdm(batch_gen(test_df)):
    all_preds.extend(model.predict(x))


# In[ ]:


import matplotlib.pyplot as plt
plt.plot(all_preds[0])
plt.plot(all_preds[int(len(all_preds)/2)])
plt.plot(all_preds[-1])
plt.yscale("log")
plt.show()


# In[ ]:


print(np.shape(all_preds))
y_te = [np.argmax(pred) for pred,supercat in zip(all_preds,supercats)]


# In[ ]:


submit_df = pd.DataFrame({"itemid": test_df["itemid"], "Category": y_te})
submit_df.to_csv("submission.csv", index=False)


# In[ ]:


submit_df.head()


# In[ ]:


submit_df.tail()

