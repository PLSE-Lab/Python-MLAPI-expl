#!/usr/bin/env python
# coding: utf-8

# # Acknowldegements
# 
# - Built upon from: https://www.kaggle.com/xhlulu/disaster-nlp-train-a-universal-sentence-encoder
# - Edits: Varied epochs, learning rate to see their impacts
# - Inference: More epochs (20) improved LB

# # About this kernel
# 
# [Universal Sentence Encoder](https://tfhub.dev/google/universal-sentence-encoder/4) is a model created and publicly made available by Google. Built in Tensorflow, it was trained to embed any type of sentences or short paragraphs so that the meaning is as much preserved as possible; so that it can be finetuned for classification tasks specifically.
# 
# This implementation is extremely pleasant to use, since the input is simply the string, and the output is just the 512-dimensional encoded sentence; no preprocessing is needed. It is also fully-trainable, and uses an almost state-of-the-art architecture (namely, pre-BERT transformers (nice [comparison in this blog post](https://blog.floydhub.com/when-the-best-nlp-model-is-not-the-best-choice/)).
# 
# Since this competition is dedicated for beginners to get started, I feel this is a perfect example of using a novel technology, but packaged in a gentle and correctly abstracted API (as opposed to the horrors of [1000 lines of tensorflow code](https://github.com/google-research/bert/blob/master/modeling.py) that needs to be understood before modifying BERT). Here, instead, **I'm only showing you some 50 lines of codes to get all up and running**; and only ~15 lines to setup the model!
# 
# ## Summary
# 
# This kernel serves as a short and straightforward introduction to the process of:
# 1. Loading a trained model from [Tensorflow hub](https://tfhub.dev/).
# 2. Building a `Sequential` Keras model by using the trained model as a layer.
# 3. Training the newly created Keras model, and perform inference.

# In[ ]:


import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Input, BatchNormalization, Dropout, Concatenate
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import ModelCheckpoint


# # Load data and model

# First load all the CSV files we will need

# In[ ]:


train = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")
test = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")
submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")


# Create convenient names for the variables we will be using for training and inference.

# In[ ]:


train_data = train.text.values
train_labels = train.target.values
test_data = test.text.values


# Finally, load the Universal Sentence Encoder from tfhub.dev (make sure Internet is enabled!).

# In[ ]:


get_ipython().run_cell_magic('time', '', "module_url = 'https://tfhub.dev/google/universal-sentence-encoder-large/4'\nembed = hub.KerasLayer(module_url, trainable=False, name='USE_embedding')")


# # Train the model

# Build a simple sequential model in Keras, with just a few lines. Note that the `Input` here is a tf.string; usually you will see integer inputs followed by an `Embedding` layer; those are needed for RNNs or CNNs, but here it is all taken care of internally by the USE; in other words, the `embed` layer you just loaded is internally tokenizing the strings, convert them to integers, then map them using an embedding.
# 
# If none of those words make any sense, worry not! USE was designed to be easily understood and directly used as is, so you don't have to get into the low-level implementation details, and can focus on using it as a tool in your Keras model, or use it as is.

# In[ ]:


def build_model(embed):
    model = Sequential([
        Input(shape=[], dtype=tf.string),
        embed,
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    
    return model


# Let's check if the model looks the way we want:

# In[ ]:


model = build_model(embed)
model.summary()


# Let's get started with the training step! We'll use 20% of the data to validate the results, and only save the model that has the lowest loss on that 20% data.

# In[ ]:


checkpoint = ModelCheckpoint('model.h5', monitor='val_loss', save_best_only=True)

train_history = model.fit(
    train_data, train_labels,
    validation_split=0.2,
    epochs=20,
    callbacks=[checkpoint],
    batch_size=32
)


# # Inference

# Don't forget that the latest model might not be the best! Instead, the best is the one we saved as `model.h5`; let's load it and run prediction on `test_data`.

# In[ ]:


model.load_weights('model.h5')
test_pred = model.predict(test_data)


# Finally, we round the predictions, set them to integer, update the `submission` dataframe, and save it as CSV... Oof!

# In[ ]:


submission['target'] = test_pred.round().astype(int)
submission.to_csv('submission.csv', index=False)

