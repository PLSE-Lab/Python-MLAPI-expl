#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Problem-Defintion:" data-toc-modified-id="Problem-Defintion:-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Problem Defintion:</a></span></li><li><span><a href="#Data" data-toc-modified-id="Data-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Data</a></span></li><li><span><a href="#Evaluation" data-toc-modified-id="Evaluation-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Evaluation</a></span></li><li><span><a href="#Dealing-with-the-data" data-toc-modified-id="Dealing-with-the-data-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Dealing with the data</a></span><ul class="toc-item"><li><span><a href="#Making-necessary-imports-and-installations" data-toc-modified-id="Making-necessary-imports-and-installations-4.1"><span class="toc-item-num">4.1&nbsp;&nbsp;</span>Making necessary imports and installations</a></span></li><li><span><a href="#Viewing-the-data" data-toc-modified-id="Viewing-the-data-4.2"><span class="toc-item-num">4.2&nbsp;&nbsp;</span>Viewing the data</a></span></li><li><span><a href="#Defining-some-constant-terms-that-we'll-be-using-later" data-toc-modified-id="Defining-some-constant-terms-that-we'll-be-using-later-4.3"><span class="toc-item-num">4.3&nbsp;&nbsp;</span>Defining some constant terms that we'll be using later</a></span></li><li><span><a href="#Loading-the-pretrained-word-embeddings-into-the--notebook-as-a-dictionary" data-toc-modified-id="Loading-the-pretrained-word-embeddings-into-the--notebook-as-a-dictionary-4.4"><span class="toc-item-num">4.4&nbsp;&nbsp;</span>Loading the pretrained word embeddings into the  notebook as a dictionary</a></span></li><li><span><a href="#Preparing-the-data-to-feed-to-model" data-toc-modified-id="Preparing-the-data-to-feed-to-model-4.5"><span class="toc-item-num">4.5&nbsp;&nbsp;</span>Preparing the data to feed to model</a></span></li><li><span><a href="#Converting-sentences-into-numbers-(sequences)" data-toc-modified-id="Converting-sentences-into-numbers-(sequences)-4.6"><span class="toc-item-num">4.6&nbsp;&nbsp;</span>Converting sentences into numbers (sequences)</a></span></li><li><span><a href="#Padding-the-sequences" data-toc-modified-id="Padding-the-sequences-4.7"><span class="toc-item-num">4.7&nbsp;&nbsp;</span>Padding the sequences</a></span></li><li><span><a href="#Preparing-the-embedding-matrix-corresponding-to-our-dataset-using-the-pretrained-embeddings" data-toc-modified-id="Preparing-the-embedding-matrix-corresponding-to-our-dataset-using-the-pretrained-embeddings-4.8"><span class="toc-item-num">4.8&nbsp;&nbsp;</span>Preparing the embedding matrix corresponding to our dataset using the pretrained embeddings</a></span></li><li><span><a href="#Loading-the-embeddings-we-obtained-into-a-keras-Embedding-Layer" data-toc-modified-id="Loading-the-embeddings-we-obtained-into-a-keras-Embedding-Layer-4.9"><span class="toc-item-num">4.9&nbsp;&nbsp;</span>Loading the embeddings we obtained into a keras Embedding Layer</a></span></li></ul></li><li><span><a href="#Modelling" data-toc-modified-id="Modelling-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Modelling</a></span><ul class="toc-item"><li><span><a href="#Baseline-model" data-toc-modified-id="Baseline-model-5.1"><span class="toc-item-num">5.1&nbsp;&nbsp;</span>Baseline model</a></span><ul class="toc-item"><li><span><a href="#Defining-a-1D-Convolutional-Neural-Network" data-toc-modified-id="Defining-a-1D-Convolutional-Neural-Network-5.1.1"><span class="toc-item-num">5.1.1&nbsp;&nbsp;</span>Defining a 1D Convolutional Neural Network</a></span></li><li><span><a href="#Compiling-the-model" data-toc-modified-id="Compiling-the-model-5.1.2"><span class="toc-item-num">5.1.2&nbsp;&nbsp;</span>Compiling the model</a></span></li><li><span><a href="#Fitting-the-model-to-0.8-split-of-total-data-and-validating-on-the-0.2-part" data-toc-modified-id="Fitting-the-model-to-0.8-split-of-total-data-and-validating-on-the-0.2-part-5.1.3"><span class="toc-item-num">5.1.3&nbsp;&nbsp;</span>Fitting the model to 0.8 split of total data and validating on the 0.2 part</a></span></li><li><span><a href="#Reviewing-loss-and-accuracy-path-of-model" data-toc-modified-id="Reviewing-loss-and-accuracy-path-of-model-5.1.4"><span class="toc-item-num">5.1.4&nbsp;&nbsp;</span>Reviewing loss and accuracy path of model</a></span></li></ul></li><li><span><a href="#Conv1D-model-with-adam-optimizer" data-toc-modified-id="Conv1D-model-with-adam-optimizer-5.2"><span class="toc-item-num">5.2&nbsp;&nbsp;</span>Conv1D model with adam optimizer</a></span></li><li><span><a href="#Introducing-Dropout" data-toc-modified-id="Introducing-Dropout-5.3"><span class="toc-item-num">5.3&nbsp;&nbsp;</span>Introducing Dropout</a></span></li><li><span><a href="#Using-callbacks-on-our-model" data-toc-modified-id="Using-callbacks-on-our-model-5.4"><span class="toc-item-num">5.4&nbsp;&nbsp;</span>Using callbacks on our model</a></span></li></ul></li><li><span><a href="#Evaluating-model-on-train-data" data-toc-modified-id="Evaluating-model-on-train-data-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Evaluating model on train data</a></span></li><li><span><a href="#Evaluating-model-on-test-data" data-toc-modified-id="Evaluating-model-on-test-data-7"><span class="toc-item-num">7&nbsp;&nbsp;</span>Evaluating model on test data</a></span></li><li><span><a href="#Creating-submission-file" data-toc-modified-id="Creating-submission-file-8"><span class="toc-item-num">8&nbsp;&nbsp;</span>Creating submission file</a></span></li></ul></div>

# # Problem Defintion:

# * To given a negative comment in English Language, we must be able to classify its toxicity
# * This is a multilabel problem.m
# * So the output for each example should be a six dimensional vector

# #  Data
# The data is available on kaggle at
# 
# https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data
# 
# The data comprises of
# 
# * 159571 comments for train data and
# * 153164 test comments

# # Evaluation

# * The model is evaluated on the mean column-wise ROC AUC. 
# * In other words, the score is the average of the individual AUCs of each predicted column.

# # Dealing with the data

# ## Making necessary imports and installations

# In[ ]:


import sklearn
from sklearn.metrics import roc_auc_score


# In[ ]:


import os
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense,Input,MaxPooling1D,GlobalMaxPooling1D,Conv1D,Embedding
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# ## Viewing the data

# In[ ]:


train=pd.read_csv("train.csv",low_memory=False)
train


# In[ ]:


test=pd.read_csv("test.csv",low_memory=False)
test


# ## Defining some constant terms that we'll be using later

# In[ ]:


MAX_SEQ_LENGTH=100
MAX_VOCAB_SIZE=20000 #This is the maximum number of unique words that will be tokenized
EMBEDDING_DIM=100 # Each word will be represented as 100 dim vector\
VALIDATION_SPLIT=0.2 #Useful while training
BATCH_SIZE=128
EPOCHS=10


# ## Loading the pretrained word embeddings into the  notebook as a dictionary

# In[ ]:


word2vec={}
with open(os.path.join("../large_data/glove.6B/glove.6B.%sd.txt" % EMBEDDING_DIM),encoding="utf-8") as f:
    for line in f:
        values=line.split()
        word=values[0]
        embed=np.asarray(values[1:],dtype="float32")
        word2vec[word]=embed
print("Found ",len(word2vec)," word vectors")   


# ## Preparing the data to feed to model

# In[ ]:


sentences=train["comment_text"].fillna("DUMMY_VALUE").values   #.values returns a numpy array
possible_labels=["toxic","severe_toxic","obscene","threat","insult","identity_hate"]
targets=train[possible_labels].values # returns a one hot encoded label vector for each example in train data
targets.shape


# ## Converting sentences into numbers (sequences)

# In[ ]:


tokenizer=Tokenizer(num_words=MAX_VOCAB_SIZE)
tokenizer.fit_on_texts(sentences)
sequences=tokenizer.texts_to_sequences(sentences)


# In[ ]:


word2idx=tokenizer.word_index
print("Found %s unique tokens " % len(word2idx))


# ## Padding the sequences

# In[ ]:


data=pad_sequences(sequences,maxlen=MAX_SEQ_LENGTH) # padding is pre by default
print("shape of data is ",data.shape)


# ## Preparing the embedding matrix corresponding to our dataset using the pretrained embeddings

# In[ ]:


word2idx


# In[ ]:


num_words=min(MAX_VOCAB_SIZE,len(word2idx)+1) # Num of words should be less than or equal to MAX_VOCAB_SIZE

# The +1 term indicates that the tokenizer indexing begins from 1

embedding_matrix=np.zeros((num_words,EMBEDDING_DIM))
for word,pos_from_start in word2idx.items():
    if pos_from_start<MAX_VOCAB_SIZE:
        embedding_vector=word2vec.get(word) #we use get method instead of indexing because it helps if the word is not present in the dictionary
        if embedding_vector is not None:
            embedding_matrix[pos_from_start]=embedding_vector


# In[ ]:


embedding_matrix.shape


# Note: In the embedding matrix ,vectors corresponding to words that are present in the data but not in the tokenizer are all zeros

# ## Loading the embeddings we obtained into a keras Embedding Layer

# Also we set Trainable as False for this layer as we have already loaded pretrained weights(embeddings)

# In[ ]:


embedding_layer=Embedding(num_words,
                          EMBEDDING_DIM,
                          weights=[embedding_matrix],
                          input_length=MAX_SEQ_LENGTH,
                          trainable=False
                         )


# # Modelling

# ## Baseline model

# ### Defining a 1D Convolutional Neural Network

# In[ ]:


input_=Input(shape=(MAX_SEQ_LENGTH,))
x=embedding_layer(input_)
x=Conv1D(128,3,activation="relu")(x)
x=MaxPooling1D(3)(x)
x=Conv1D(128,3,activation="relu")(x)
x=MaxPooling1D(3)(x)
x=Conv1D(128,3,activation="relu")(x)
x=GlobalMaxPooling1D()(x) # max from every input channel
# it also indicates which timestep value in that sequnce was most influential for classification
x=Dense(128,activation="relu")(x)
output=Dense(len(possible_labels),activation="sigmoid")(x)
# we use sigmoid classifier so that each of the 6 units in the last layer act as a linear classifier(y/n)


# ### Compiling the model

# In[ ]:


model1=Model(input_,output)
model1.compile(loss="binary_crossentropy",
             optimizer="rmsprop",
             metrics=["accuracy"]
              )


# ### Fitting the model to 0.8 split of total data and validating on the 0.2 part

# In[ ]:


history=model1.fit(
                data,
                targets,
                batch_size=BATCH_SIZE,
                epochs=EPOCHS,
                validation_split=VALIDATION_SPLIT
                )


# ### Reviewing loss and accuracy path of model

# In[ ]:


def plot_curves(history):
    fig,(ax0,ax1)=plt.subplots(2,1,figsize=(8,8))
    ax0.plot(history.history["loss"],label="loss")
    ax0.plot(history.history["val_loss"],label="val_loss")
    ax0.legend()
    ax1.plot(history.history["accuracy"],label="accuracy")
    ax1.plot(history.history["val_accuracy"],label="val_accuracy")
    ax1.legend()
    plt.show()


# In[ ]:


plot_curves(history)


# So we see the validation accuracy started dropping later.
# 
# It indicates that the model started overfitting the data

# ## Conv1D model with adam optimizer

# In[ ]:


model2=Model(input_,output)
model2.compile(loss="binary_crossentropy",
             optimizer="adam",
             metrics=["accuracy"]
              )
history=model2.fit(
                data,
                targets,
                batch_size=BATCH_SIZE,
                epochs=EPOCHS,
                validation_split=VALIDATION_SPLIT
                )


# In[ ]:


plot_curves(history)


# ## Introducing Dropout

# In[ ]:


input_=Input(shape=(MAX_SEQ_LENGTH,))
x=embedding_layer(input_)
x=Conv1D(128,3,activation="relu")(x)
x=MaxPooling1D(3)(x)
x=Conv1D(128,3,activation="relu")(x)
x=MaxPooling1D(3)(x)
x=Conv1D(128,3,activation="relu")(x)
x=GlobalMaxPooling1D()(x) # max from every input channel
# it also indicates which timestep value in that sequnce was most influential for classification
x=Dense(128,activation="relu")(x)
x=tf.keras.layers.Dropout(0.3)(x)
output=Dense(len(possible_labels),activation="sigmoid")(x)
# we use sigmoid classifier so that each of the 6 units in the last layer act as a linear classifier(y/n)

model3=Model(input_,output)
model3.compile(loss="binary_crossentropy",
             optimizer="adam",
             metrics=["accuracy"]
              )
history=model3.fit(
                data,
                targets,
                batch_size=BATCH_SIZE,
                epochs=EPOCHS,
                validation_split=VALIDATION_SPLIT
                )


# In[ ]:


plot_curves(history)


# ## Using callbacks on our model

# In[ ]:


early_stopping=tf.keras.callbacks.EarlyStopping(patience=5,monitor="val_accuracy")
model_checkpoint=tf.keras.callbacks.ModelCheckpoint("model3.h5",monitor="val_accuracy",save_best_only=True)


# In[ ]:


EPOCHS=100
history=model3.fit(
                data,
                targets,
                batch_size=BATCH_SIZE,
                epochs=EPOCHS,
                validation_split=VALIDATION_SPLIT,
                callbacks=[early_stopping,model_checkpoint]
                )


# # Evaluating model on train data

# In[ ]:


model=tf.keras.models.load_model("model3.h5")
p=model.predict(data)
aucs = []
for j in range(6):
    auc = roc_auc_score(targets[:,j], p[:,j])
    aucs.append(auc)
print(np.mean(aucs))


# In[ ]:


p.shape


# # Evaluating model on test data

# In[ ]:


test_sentences=test["comment_text"].fillna("DUMMY_VALUE").values
test_sequences=tokenizer.texts_to_sequences(test_sentences)
test_data=pad_sequences(test_sequences,maxlen=MAX_SEQ_LENGTH)


# In[ ]:


pred=model.predict(test_data)


# In[ ]:


pred[:,0].shape


# In[ ]:


possible_labels


# # Creating submission file

# In[ ]:


submit1=pd.DataFrame(columns=["id","toxic","severe_toxic","threat","insult","identity_hate"])
submit1["id"]=test["id"]
i=0
for col in possible_labels:
    submit1[col]=pred[:,i]
    i=i+1


# In[ ]:


submit1


# In[ ]:


submit1.index = submit1.index+1
submit1


# In[ ]:


submit1.to_csv("submission1.csv",index=False)


# In[ ]:


a=pd.read_csv("submission1.csv")
a


# In[ ]:




