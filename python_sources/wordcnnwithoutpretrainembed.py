#!/usr/bin/env python
# coding: utf-8

# Word CNN without pre-trained embeddings. Only using one of the training sets. Relies on translated test and validation data from @bamps53 and @kashnitsky 

# In[ ]:


# # Load packages

# Ignore warnings
import warnings

def warn(*args, **kwargs):
    pass

warnings.warn = warn

import os
import numpy as np
import pandas as pd

import keras
from keras import *
from keras import layers
from keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from keras.models import Model
from keras.preprocessing import *
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import Callback

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score, confusion_matrix

import seaborn as sns
import matplotlib.pyplot as plt

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from io import StringIO


# # Helper functions

# In[ ]:


class RocCallback(Callback):
    def __init__(self,training_data,validation_data):
        self.x = training_data[0]
        self.y = training_data[1]
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]
    
    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        y_pred_train = self.model.predict_proba(self.x)
        roc_train = roc_auc_score(self.y, y_pred_train)
        y_pred_val = self.model.predict_proba(self.x_val)
        roc_val = roc_auc_score(self.y_val, y_pred_val)
        print('\rroc-auc_train: %s - roc-auc_val: %s' % (str(round(roc_train,4)),str(round(roc_val,4))),end=100*' '+'\n')
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return
    

def genAUC(validFeatures,validLabels):
    pred_valid_df = pd.DataFrame(model.predict(validFeatures))   
    auc = roc_auc_score(validLabels, pred_valid_df)
    print('AUC: %.3f' % auc)

    
PLOT_FONT_SIZE = 10    #font size for axis of plots

#define helper function for confusion matrix

def displayConfusionMatrix(confusionMatrix):
    """Confusion matrix plot"""
    
    confusionMatrix = np.transpose(confusionMatrix)
    
    ## calculate class level precision and recall from confusion matrix
    precisionLow = round((confusionMatrix[0][0] / (confusionMatrix[0][0] + confusionMatrix[0][1]))*100, 1)
    precisionHigh = round((confusionMatrix[1][1] / (confusionMatrix[1][0] + confusionMatrix[1][1]))*100, 1)
    recallLow = round((confusionMatrix[0][0] / (confusionMatrix[0][0] + confusionMatrix[1][0]))*100, 1)
    recallHigh = round((confusionMatrix[1][1] / (confusionMatrix[0][1] + confusionMatrix[1][1]))*100, 1)

    ## show heatmap
    plt.imshow(confusionMatrix, interpolation='nearest',cmap=plt.cm.Blues,vmin=0, vmax=100)
    
    ## axis labeling
    xticks = np.array([0,1])
    plt.gca().set_xticks(xticks)
    plt.gca().set_yticks(xticks)
    plt.gca().set_xticklabels(["Not Toxic \n Recall=" + str(recallLow), "Toxic \n Recall=" + str(recallHigh)], fontsize=PLOT_FONT_SIZE)
    plt.gca().set_yticklabels(["Not Toxic \n Precision=" + str(precisionLow), "Toxic \n Precision=" + str(precisionHigh)], fontsize=PLOT_FONT_SIZE)
    plt.ylabel("Predicted Class", fontsize=PLOT_FONT_SIZE)
    plt.xlabel("Actual Class", fontsize=PLOT_FONT_SIZE)
        
    ## add text in heatmap boxes
    addText(xticks, xticks, confusionMatrix)
    
def addText(xticks, yticks, results):
    """Add text in the plot"""
    for i in range(len(yticks)):
        for j in range(len(xticks)):
            text = plt.text(j, i, results[i][j], ha="center", va="center", color="white", size=PLOT_FONT_SIZE) ### size here is the size of text inside a single box in the heatmap

            
def plotConfusion(validFeatures,validLabels):
    
    pred_valid_df = pd.DataFrame(model.predict(validFeatures))
    pred_valid_binary = round(pred_valid_df)

    confusionMatrix = None
    confusionMatrix = confusion_matrix(validLabels, pred_valid_binary)

    plt.rcParams['figure.figsize'] = [3, 3] ## plot size
    displayConfusionMatrix(confusionMatrix)
    plt.title("Confusion Matrix", fontsize=PLOT_FONT_SIZE)
    plt.show()


# In[ ]:


def lemmetize_data(data,field):
    cleaned_texts = []
    for text in data[field]: # Loop through the tokens (the words or symbols) 
        cleaned_text = text.lower()  # Convert the text to lower case
        cleaned_text = ' '.join([word for word in cleaned_text.split() if word not in stopset])  # Keep only words that are not stopwords.
        cleaned_text = ' '.join([wordnet_lemmatizer.lemmatize(word, pos='n') for word in cleaned_text.split()])  # Keep each noun's lemma.
        cleaned_text = ' '.join([wordnet_lemmatizer.lemmatize(word, pos='v') for word in cleaned_text.split()])  # Keep each verb's lemma.
        cleaned_text = re.sub(r"(http\S+)"," ", cleaned_text)  # Remove http links.
        cleaned_text = re.sub("[^a-zA-Z]"," ", cleaned_text)  # Remove numbers and punctuation.
        cleaned_text = ' '.join(cleaned_text.split())  # Remove white space.
        cleaned_texts.append(cleaned_text) 
    data['cleanText'] = cleaned_texts


# In[ ]:


nltk.download('stopwords')
nltk.download('wordnet')

wordnet_lemmatizer = WordNetLemmatizer()
stopset = list(set(stopwords.words('english')))


# # Load data

# In[ ]:


# load training data 1
train_comment=pd.read_csv('../input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv')
train_texts = train_comment['comment_text']

#lemmetize_data(train_comment,'comment_text')
#train_texts = train_comment['cleanText']

train_labels = train_comment['toxic']
train_comment


# In[ ]:


# load validation data
valid=pd.read_csv('../input/val-en-df/validation_en.csv')
valid_texts = valid['comment_text_en']

#lemmetize_data(valid,'comment_text_en')
#valid_texts = valid['cleanText']

valid_labels = valid['toxic']


# In[ ]:


# load testing data (Translated via Google)
test_google=pd.read_csv('../input/test-en-df/test_en.csv')
test_textsGoogle = test_google['content_en']

#lemmetize_data(test_google,'content_en')
#test_textsGoogle = test_google['cleanText']


# In[ ]:


# load testing data (Translated via Yandex)
test_yandex=pd.read_csv('../input/jigsaw-multilingual-toxic-test-translated/jigsaw_miltilingual_test_translated.csv')
test_textsYandex = test_yandex['translated']

#lemmetize_data(test_yandex,'translated')
#test_textsYandex = test_yandex['cleanText']


# In[ ]:


def plotCases(data):
    cases_count = data.value_counts(dropna=False)

    # Plot  results 
    plt.figure(figsize=(6,6))
    sns.barplot(x=cases_count.index, y=cases_count.values)
    plt.ylabel('Texts', fontsize=12)
    plt.xticks(range(len(cases_count.index)), ['Not', 'Toxic'])


# In[ ]:


plotCases(train_labels)


# In[ ]:


plotCases(valid_labels)


# # Prepare data

# In[ ]:


# Define vocabulary size (you can tune this parameter and evaluate model performance)
VOCABULARY_SIZE = 5000


# In[ ]:


# Create input feature arrays
tokenizer = Tokenizer(num_words=VOCABULARY_SIZE)
tokenizer.fit_on_texts(train_texts)


# In[ ]:


# Convert words into word ids
meanLengthTrain = np.mean([len(item.split(" ")) for item in train_texts])
meanLengthValid = np.mean([len(item.split(" ")) for item in valid_texts])
meanLengthTestGoogle = np.mean([len(item.split(" ")) for item in test_textsGoogle])
meanLengthTestYandex = np.mean([len(item.split(" ")) for item in test_textsYandex])

print('Average length - Train:',meanLengthTrain,'Valid:',meanLengthValid,'TestGoogle:',meanLengthTestGoogle,'TestYandex:',meanLengthTestYandex)


# In[ ]:


MAX_SENTENCE_LENGTH = int(meanLengthTrain + 10) # we let a text go 10 words longer than the mean text length (you can also tune this parameter).

# Convert train, validation, and test text into lists with word ids
trainFeatures = tokenizer.texts_to_sequences(train_texts)
trainFeatures = pad_sequences(trainFeatures, MAX_SENTENCE_LENGTH, padding='post')
trainLabels = train_labels.values

validFeatures = tokenizer.texts_to_sequences(valid_texts)
validFeatures = pad_sequences(validFeatures, MAX_SENTENCE_LENGTH, padding='post')
validLabels = valid_labels.values

testFeaturesGoogle = tokenizer.texts_to_sequences(test_textsGoogle)
testFeaturesGoogle = pad_sequences(testFeaturesGoogle, MAX_SENTENCE_LENGTH, padding='post')

testFeaturesYandex = tokenizer.texts_to_sequences(test_textsYandex)
testFeaturesYandex = pad_sequences(testFeaturesYandex, MAX_SENTENCE_LENGTH, padding='post')


# # Train Word CNN without pre-trained embeddings

# In[ ]:


# Define filter and kernel size for CNN (can adjust in tuning model)
FILTERS_SIZE = 16
KERNEL_SIZE = 5


# In[ ]:


# Overfits very quickly, use super low learning rate

# Define embeddings dimensions (columns in matrix fed into CNN and nodes in hidden layer of built-in keras function)
EMBEDDINGS_DIM = 20

# Hyperparameters for model tuning
LEARNING_RATE = 0.0001
BATCH_SIZE = 500
EPOCHS = 8


# In[ ]:


# Word CNN
model = Sequential()

# We use built-in keras funtion to generate embeddings. Another option is pre-trained embeddings with Word2vec or GloVe.
model.add(Embedding(input_dim=VOCABULARY_SIZE + 1, output_dim=EMBEDDINGS_DIM, input_length=len(trainFeatures[0])))
model.add(Conv1D(FILTERS_SIZE, KERNEL_SIZE, activation='relu'))
model.add(Dropout(0.5))
model.add(GlobalMaxPooling1D())
model.add(Dropout(0.5))
model.add(Dense(12, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
            
optimizer = optimizers.Adam(lr=LEARNING_RATE)
model.compile(optimizer=optimizer,loss='binary_crossentropy',metrics=['accuracy'])

print(model.summary())


# In[ ]:


# ratio of non-toxic to toxic in training:
200000/25000


# In[ ]:


# We have a class imbalance, upweight the toxic comments
class_weights = {0: 1,
                 1: 8}


# In[ ]:


roc = RocCallback(training_data=(trainFeatures, trainLabels),
                  validation_data=(validFeatures, validLabels))


# In[ ]:


# train model
history = model.fit(trainFeatures, trainLabels, validation_data = (validFeatures, validLabels), batch_size=BATCH_SIZE, epochs=EPOCHS, class_weight=class_weights, callbacks=[roc])


# In[ ]:


genAUC(validFeatures,validLabels)


# In[ ]:


plotConfusion(validFeatures,validLabels)


# # Make test predictions

# In[ ]:


# make test predictions (average both translations)
predictionsGoogle = pd.DataFrame(model.predict(testFeaturesGoogle))
predictionsYandex = pd.DataFrame(model.predict(testFeaturesYandex))
predictions = (predictionsGoogle+predictionsYandex)/2
predictions


# In[ ]:


# prep for submission
sample = pd.read_csv("../input/jigsaw-multilingual-toxic-comment-classification/sample_submission.csv")
sample['toxic'] = predictions[0]
sample


# In[ ]:


# make submission
sample.to_csv("submission.csv", index=False)

