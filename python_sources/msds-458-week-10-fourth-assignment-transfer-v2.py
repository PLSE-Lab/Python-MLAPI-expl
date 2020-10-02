#!/usr/bin/env python
# coding: utf-8

# # References
# 
# https://missinglink.ai/guides/neural-network-concepts/7-types-neural-network-activation-functions-right/
# 
# https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/
# 
# https://towardsdatascience.com/machine-learning-word-embedding-sentiment-classification-using-keras-b83c28087456
# 
# https://github.com/jeffheaton/t81_558_deep_learning
# 
# https://www.quora.com/In-recurrent-neural-networks-like-LSTMs-is-it-possible-to-do-transfer-learning-Has-there-been-any-research-in-this-area
# 
# https://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/
# 
# https://keras.io/examples/cifar10_cnn/
# 
# https://towardsdatascience.com/character-level-cnn-with-keras-50391c3adf33
# 
# https://machinelearningmastery.com/how-to-stop-training-deep-neural-networks-at-the-right-time-using-early-stopping/
# 
# https://www.dlology.com/blog/simple-guide-on-how-to-generate-roc-plot-for-keras-classifier/
# 
# https://www.bmc.com/blogs/keras-neural-network-classification/
# 
# https://www.kdnuggets.com/2020/02/intent-recognition-bert-keras-tensorflow.html

# # Imports

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy
from sklearn.metrics import confusion_matrix, roc_curve, auc

# For Embedding
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

# For Models
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, GRU, Flatten, Dropout, Conv1D, MaxPooling1D, Activation
from keras.layers.embeddings import Embedding

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model

EMBEDDING_DIM = 100    # Set EMBEDDING_DIM
bat_size = 32          # Set batch size
pat = 2                # Set patience
dropout_rate = 0.0     # Set dropout rate


# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # LOAD DATA

# In[ ]:


df1 = pd.DataFrame()
df2 = pd.DataFrame()
df3 = pd.DataFrame()

df1 = pd.read_csv("../input/all-the-news/articles1.csv", encoding = 'utf-8')
df2 = pd.read_csv("../input/all-the-news/articles2.csv", encoding = 'utf-8')
df3 = pd.read_csv("../input/all-the-news/articles3.csv", encoding = 'utf-8')

frames = [df1, df2, df3]
data = pd.concat(frames)

conditions = [
    (data['publication'] == 'Breitbart'),
    (data['publication'] == 'New York Post'),
    (data['publication'] == 'NPR'),
    (data['publication'] == 'CNN'),
    (data['publication'] == 'Washington Post'),
    (data['publication'] == 'Reuters'),
    (data['publication'] == 'Guardian'),
    (data['publication'] == 'New York Times'),
    (data['publication'] == 'Atlantic'),
    (data['publication'] == 'Business Insider'),
    (data['publication'] == 'National Review'),
    (data['publication'] == 'Talking Points Memo'),
    (data['publication'] == 'Vox'),
    (data['publication'] == 'Buzzfeed News'),
    (data['publication'] == 'Fox News')]

# choices = [1, 1, 0, -1, -1, 0, -1, -1, -1, 0, 1, -1, -1, -1, 1]
choices = [1, 1, 2, 0, 0, 2, 0, 0, 0, 2, 1, 0, 0, 0, 1]
data['label'] = np.select(conditions, choices)

article_lengths = []

# Create a column with the number of words in the article

for i in range(0, len(data['content'])):
    article_lengths.append(len(data['content'].iloc[i].split()))
    
data['words'] = article_lengths

#data_pre_trained_model = data[data['publication'].str.contains('New York Times', na=False, regex=True)].append(data[data['publication'].str.contains('National Review', na=False, regex=True)])
#data = data[data['publication'].str.contains('CNN', na=False, regex=True)].append(data[data['publication'].str.contains('Fox News', na=False, regex=True)])

# Filter the articles in the dataset
data = data.query("label != 2")
data = data.query("words <= 300")

# Check word count in the articles after query

article_lengths = []

for i in range(0, len(data['content'])):
    article_lengths.append(len(data['content'].iloc[i].split()))


# # Limited EDA

# In[ ]:


display(data.head(3))

print(np.unique(data[:][['publication']]))
print("\n")
print('Number of articles:',data.shape[0])
print("\n")
print(data.label.value_counts())
print("\n")
print(data.publication.value_counts())

print("Review length: ")
print("Mean %.2f words (%f)" % (numpy.mean(article_lengths), numpy.std(article_lengths)))

# plot review length
plt.title('Total Word Distribution')
plt.ylabel('Number of Words Per Article')
plt.xlabel('News Articles')
plt.boxplot(article_lengths)
plt.show()


# # Word Embedding For Entire Corpus

# In[ ]:


# This class allows to vectorize a text corpus, by turning each text into either a sequence of integers 
# (each integer being the index of a token in a dictionary) or into a vector where the coefficient for each token could be binary, 
# based on word count, based on tf-idf...

# vectorize the text samples into a 2D integer tensor
tokenizer_obj = Tokenizer()
total_artciles = data['content'].values
tokenizer_obj.fit_on_texts(total_artciles)

for i in range(0, len(data['content'])):
    article_lengths.append(len(data['content'].iloc[i].split()))

# Pad Sequences
max_length = 300

# Define volcabulary size
volcabulary_size = len(tokenizer_obj.word_index) + 1


# # Split DataSet to Pre-train Weights

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(data['content'], data['label'], test_size=0.33, random_state=2020)

print("Test Set\n", y_test.value_counts())
print("\nTraining Set\n", y_train.value_counts())

# Vectorize
################################################################################## 

X_train_tokens = tokenizer_obj.texts_to_sequences(X_train)
X_test_tokens  = tokenizer_obj.texts_to_sequences(X_test)

X_train_pad = pad_sequences(X_train_tokens, maxlen = max_length, padding = 'pre')
X_test_pad = pad_sequences(X_test_tokens, maxlen = max_length, padding = 'pre')


# # Build Model to Pre-Train Weights

# In[ ]:


model = Sequential()
model.add(Embedding(volcabulary_size, EMBEDDING_DIM, input_length = max_length))

model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(dropout_rate))

model.add(Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'))
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(dropout_rate))

model.add(Conv1D(filters=128, kernel_size=3, padding='same', activation='relu'))
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(dropout_rate))

model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())


# # Train Model

# In[ ]:


# simple early stopping
es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=pat)
mc = ModelCheckpoint('best_transfer_model.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)

hist = model.fit(X_train_pad, y_train, batch_size = bat_size, epochs=10, validation_data=(X_test_pad, y_test), verbose=1, callbacks=[es, mc])


# # Tranfer Learning NN Model - Training vs. Validation Accuracy

# In[ ]:


num_epochs = len(hist.history['accuracy'])

plt.style.use('seaborn-whitegrid')
plt.plot([x for x in range(1,num_epochs+1)], hist.history['accuracy'], "-c",  marker='o', label='Training Accuracy', linestyle='dashdot')
plt.plot([x for x in range(1,num_epochs+1)], hist.history['val_accuracy'], "-m", marker='o', label='Validation Accuracy', linestyle='dashed')
plt.legend(loc="best")
plt.title('Tranfer Learning NN Model - Training vs. Validation Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.xticks([x for x in range(1,num_epochs+1)])
plt.show()


# # Confusion Matrix

# In[ ]:


pred = model.predict_classes(X_test_pad)

cm = confusion_matrix(y_true = y_test, y_pred = pred)
print(cm)


# # ROC AUC

# In[ ]:


# Generate the probability values
pred = model.predict(X_test_pad)

# Compute ROC curve and ROC area for each class
fpr_transfer = dict()
tpr_transfer = dict()
roc_auc_transfer = dict()

fpr_transfer, tpr_transfer, thresholds_transfer = roc_curve(y_test.values, pred)
roc_auc_transfer = auc(fpr_transfer, tpr_transfer)

print('AUC:', roc_auc_transfer)


# # Create Pre-Trained Model

# In[ ]:


model = load_model('best_transfer_model.h5')

pretrained_model = Sequential()

for i in range(0,14):
    layer = model.layers[i]
    layer.trainable = False
    pretrained_model.add(layer)
    
pretrained_model.summary()


# # Create New Model Using Pre-Trained Model

# In[ ]:


new_model = pretrained_model 

new_model.add(Dense(units=512))
new_model.add(Dropout(0.5))

new_model.add(Dense(1, activation='sigmoid'))

new_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(new_model.summary())


# # Train New Model

# In[ ]:


# simple early stopping
es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=pat)
mc = ModelCheckpoint('best_model.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)

hist = new_model.fit(X_train_pad, y_train, batch_size = bat_size, epochs=10, validation_data=(X_test_pad, y_test), verbose=1, callbacks=[es, mc])


# # New NN Model - Training vs. Validation Accuracy

# In[ ]:


num_epochs = len(hist.history['accuracy'])

plt.style.use('seaborn-whitegrid')
plt.plot([x for x in range(1,num_epochs+1)], hist.history['accuracy'], "-c",  marker='o', label='Training Accuracy'
         , linestyle='dashdot')
plt.plot([x for x in range(1,num_epochs+1)], hist.history['val_accuracy'], "-m", marker='o', label='Validation Accuracy'
         , linestyle='dashed')
plt.legend(loc="best")
plt.title('New NN Model - Training vs. Validation Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.xticks([x for x in range(1,num_epochs+1)])
plt.show()


# # Confusion Model

# In[ ]:


pred = model.predict_classes(X_test_pad)

cm = confusion_matrix(y_true = y_test, y_pred = pred)
print(cm)


# # ROC AUC

# In[ ]:


# Generate the probability values
pred = model.predict(X_test_pad)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
fpr, tpr, thresholds = roc_curve(y_test.values, pred)
roc_auc = auc(fpr, tpr)

plt.figure()
lw = 2

plt.plot(fpr, tpr, color='cornflowerblue',
         lw=lw, label='New Model - ROC curve (area = %0.2f)' % roc_auc)
plt.plot(fpr_transfer, tpr_transfer, color='darkorange',
         lw=lw, label='Transfer Model - ROC curve (area = %0.2f)' % roc_auc_transfer)

plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([-0.05, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (Dropout = 0)')
plt.legend(loc="lower right")
plt.show()

# In general, an AUC of 0.5 suggests no discrimination, 
# 0.7 to 0.8 is considered acceptable, 
# 0.8 to 0.9 is considered excellent, 
# and more than 0.9 is considered outstanding.

