#!/usr/bin/env python
# coding: utf-8

# # LSTM with pre-trained GloVe embeddings 
# Only using one of the training sets. Relies on translated test and validation data from @bamps53 and @kashnitsky

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
import time

import keras
from keras import *
from keras import layers
from keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from keras.models import Model
from keras.preprocessing import *
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import Callback

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split

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


# In[ ]:


def lemmetize_data(data,field):
    cleaned_texts = []
    for text in data[field]: # Loop through the tokens (the words or symbols) 
        cleaned_text = text.lower()  # Convert the text to lower case
        cleaned_text = ' '.join([word for word in cleaned_text.split() if word not in stopset])  # Keep only words that are not stopwords.
        cleaned_text = ' '.join([wordnet_lemmatizer.lemmatize(word, pos='n') for word in cleaned_text.split()])  # Keep each noun's lemma.
        cleaned_text = ' '.join([wordnet_lemmatizer.lemmatize(word, pos='v') for word in cleaned_text.split()])  # Keep each verb's lemma.
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
lemmetize_data(train_comment,'comment_text')
train_comment


# In[ ]:


# load validation data 1
validGoogle=pd.read_csv('../input/val-en-df/validation_en.csv')
lemmetize_data(validGoogle,'comment_text_en')


# In[ ]:


# load validation data 2
validYandex=pd.read_csv('../input/jigsaw-multilingual-toxic-test-translated/jigsaw_miltilingual_valid_translated.csv')
lemmetize_data(validYandex,'translated')


# In[ ]:


# append, when we calculate AUC it will reflect the average
valid = validGoogle.append(validYandex)


# In[ ]:


# load testing data (Translated via Google)
test_google=pd.read_csv('../input/test-en-df/test_en.csv')
lemmetize_data(test_google,'content_en')


# In[ ]:


# load testing data (Translated via Yandex)
test_yandex=pd.read_csv('../input/jigsaw-multilingual-toxic-test-translated/jigsaw_miltilingual_test_translated.csv')
lemmetize_data(test_yandex,'translated')


# In[ ]:


def plotCases(data):
    cases_count = data.value_counts(dropna=False)

    # Plot  results 
    plt.figure(figsize=(6,6))
    sns.barplot(x=cases_count.index, y=cases_count.values)
    plt.ylabel('Texts', fontsize=12)
    plt.xticks(range(len(cases_count.index)), ['Not', 'Toxic'])


# In[ ]:


train_labels = train_comment['toxic']
valid_labels = valid['toxic']


# In[ ]:


plotCases(train_labels)


# In[ ]:


# the numbers here are doubled from the append above
plotCases(valid_labels)


# # LSTM with pretrained

# Prepare data:

# In[ ]:


# lemmetize
train_texts = train_comment['cleanText']

valid_texts = valid['cleanText']

test_textsGoogle = test_google['cleanText']
test_textsYandex = test_yandex['cleanText']


# In[ ]:


# Define vocabulary size (you can tune this parameter and evaluate model performance)
VOCABULARY_SIZE = 15000


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


MAX_SENTENCE_LENGTH = int(meanLengthTrain + 20) # we let a text go 20 words longer than the mean text length (you can also tune this parameter).

# Convert train, validation, and test text into lists with word ids
trainFeatures = tokenizer.texts_to_sequences(train_texts)
trainFeatures = pad_sequences(trainFeatures, MAX_SENTENCE_LENGTH, padding='post')
trainLabels = train_labels.values

validFeatures = tokenizer.texts_to_sequences(valid_texts)
validFeatures = pad_sequences(validFeatures, MAX_SENTENCE_LENGTH, padding='post')
validLabels = valid_labels.values

testFeaturesGoogleLSTMwith = tokenizer.texts_to_sequences(test_textsGoogle)
testFeaturesGoogleLSTMwith = pad_sequences(testFeaturesGoogleLSTMwith, MAX_SENTENCE_LENGTH, padding='post')

testFeaturesYandexLSTMwith = tokenizer.texts_to_sequences(test_textsYandex)
testFeaturesYandexLSTMwith = pad_sequences(testFeaturesYandexLSTMwith, MAX_SENTENCE_LENGTH, padding='post')


# In[ ]:


#from: https://www.kaggle.com/bertcarremans/using-word-embeddings-for-sentiment-analysis/data
EMBEDDING_FILE='../input/glove-twitter/glove.twitter.27B.25d.txt'
emb_dict = {}
glove = open(EMBEDDING_FILE)
for line in glove:
    values = line.split()
    word = values[0]
    vector = np.asarray(values[1:], dtype='float32')
    emb_dict[word] = vector
glove.close()


# In[ ]:


#from: https://www.kaggle.com/bertcarremans/using-word-embeddings-for-sentiment-analysis/data
embedding_matrix = np.zeros((VOCABULARY_SIZE, 25))

for w, i in tokenizer.word_index.items():
    # The word_index contains a token for all words of the training data so we need to limit that
    if i < VOCABULARY_SIZE:
        vect = emb_dict.get(w)
        # Check if the word from the training data occurs in the GloVe word embeddings
        # Otherwise the vector is kept with only zeros
        if vect is not None:
            embedding_matrix[i] = vect
    else:
        break


# In[ ]:


embedding_matrix.shape


# In[ ]:


# Hyperparameters for model tuning
LEARNING_RATE = 0.001
BATCH_SIZE = 128
EPOCHS = 9


# In[ ]:


#LSTM
LSTMwith = Sequential()

# We use pre-trained embeddings from GloVe. These are fed in as a layer of our network and the weights do not update during the training process.
LSTMwith.add(Embedding(input_dim=VOCABULARY_SIZE, output_dim=embedding_matrix.shape[1], weights=[embedding_matrix], trainable=False, input_length=len(trainFeatures[0])))
LSTMwith.add(Bidirectional(LSTM(24)))
LSTMwith.add(Dropout(0.6))
LSTMwith.add(Dense(12, activation='relu'))
LSTMwith.add(Dropout(0.5))
LSTMwith.add(Dense(1, activation='sigmoid'))
            
optimizer = optimizers.Adam(lr=LEARNING_RATE)
LSTMwith.compile(optimizer=optimizer,loss='binary_crossentropy',metrics=['accuracy'])

print(LSTMwith.summary())


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
start = time.time()    
history = LSTMwith.fit(trainFeatures, trainLabels, validation_data = (validFeatures, validLabels), batch_size=BATCH_SIZE, epochs=EPOCHS, class_weight=class_weights, callbacks=[roc])
print("Training took %d seconds" % (time.time() - start))  


# In[ ]:


pred_valid_LSTMwith = pd.DataFrame(LSTMwith.predict(validFeatures))


# In[ ]:


roc_auc_score(validLabels, pred_valid_LSTMwith)


# In[ ]:


validPred = pred_valid_LSTMwith
pred_valid_binary = round(validPred)

confusionMatrix = None
confusionMatrix = confusion_matrix(validLabels, pred_valid_binary)

plt.rcParams['figure.figsize'] = [3, 3] ## plot size
displayConfusionMatrix(confusionMatrix)
plt.title("Confusion Matrix", fontsize=PLOT_FONT_SIZE)
plt.show()


# # Make predictions

# In[ ]:


# make test predictions (average both translations)
predictionsGoogle = pd.DataFrame(LSTMwith.predict(testFeaturesGoogleLSTMwith))
predictionsYandex = pd.DataFrame(LSTMwith.predict(testFeaturesYandexLSTMwith))
predictions = (predictionsGoogle+predictionsYandex)/2
predictions


# In[ ]:


# prep for submission
sample = pd.read_csv("../input/jigsaw-multilingual-toxic-comment-classification/sample_submission.csv")
sample['toxic'] = predictions
sample


# In[ ]:


# make submission
sample.to_csv("submission.csv", index=False)

