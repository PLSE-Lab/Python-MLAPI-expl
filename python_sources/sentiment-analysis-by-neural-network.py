#!/usr/bin/env python
# coding: utf-8

# 1. [Load Libraries](#1)  
# 1. [Load Train Dataset](#2)
# 1. [Train Dataset Analysis](#3)
#     * [Balanced Dataset](#4)
# 1. [Text Clearing](#5)
#     * [Train Dataset Clearing](#6)
#     * [Test Dataset Clearing](#7)
#     * [Valid Dataset Clearing](#8)
# 1. [Split Dataset](#9)
# 1. [Feature Engineering](#10)
# 1. [Deep Learning](#11)
#     * [Neural Network Model](#12)
#     * [Accuracy](#13)
#     * [Confusion Metrics](#14)
#     * [ROC](#15) 
#     * [Making Prediction](#16) 
#     * [A Positive Comment](#17)
#     * [A Negative Comment](#18)

# <font color = 'red'>
# <h3>Please Upvote, if you like my kernel.<h3>

# <a id = "1"></a><br>
# ## Load Libraries

# In[ ]:


#libraries
from keras.models import Sequential 
from keras.layers import Dense 
import matplotlib.pyplot as plt

#
from textblob import TextBlob
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble
#
from sklearn import metrics
#
import textblob, string
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers
#
#stopwords
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
#
from warnings import filterwarnings
filterwarnings('ignore')

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
seed = 7 
np.random.seed(seed)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# <a id = "2"></a><br>
# ## Load Train Dataset

# In[ ]:


train = pd.read_csv("/kaggle/input/imdb-dataset-sentiment-analysis-in-csv-format/Train.csv")
train.head()


# <a id = "3"></a><br>
# ## Train Dataset Analysis

# In[ ]:


train.info()


# <a id = "4"></a><br>
# ## Balanced Dataset

# In[ ]:


train.label.value_counts()


# In[ ]:


train.groupby("label").count()


# <a id = "5"></a><br>
# ## Text Clearing

# Create a Function for Clearing

# In[ ]:


def transformations(dataframe):
    # upper to lower character
    dataframe['text'] = dataframe['text'].apply(lambda x: " ".join(x.lower() for x in x.split()))
    #punctuations
    dataframe['text'] = dataframe['text'].str.replace('[^\w\s]','')
    #numbers
    dataframe['text'] = dataframe['text'].str.replace('\d','')
    # 
    sw = stopwords.words('english')
    dataframe['text'] = dataframe['text'].apply(lambda x: " ".join(x for x in x.split() if x not in sw))
    #rare characters deleting
    sil = pd.Series(' '.join(dataframe['text']).split()).value_counts()[-1000:]
    dataframe['text'] = dataframe['text'].apply(lambda x: " ".join(x for x in x.split() if x not in sil))
    #lemmi
    from textblob import Word
    #nltk.download('wordnet')
    dataframe['text'] = dataframe['text'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()])) 
    return dataframe


# <a id = "6"></a><br>
# ## Train Dataset Clearing

# In[ ]:


train = transformations(train)
train.head()


# <a id = "7"></a><br>
# ## Validation Dataset Clearing

# In[ ]:


valid = pd.read_csv("/kaggle/input/imdb-dataset-sentiment-analysis-in-csv-format/Valid.csv")
valid = transformations(valid)
valid.head()


# <a id = "8"></a><br>
# ## Test Dataset Clearing

# In[ ]:


test = pd.read_csv("/kaggle/input/imdb-dataset-sentiment-analysis-in-csv-format/Test.csv")
test = transformations(test)
test.head()


# <a id = "9"></a><br>
# ## Split Dataset

# In[ ]:


train_x = train['text']
valid_x = valid["text"]
train_y = train["label"]
valid_y = valid["label"]


# <a id = "10"></a><br>
# ## Feature Engineering
# 
# CountVectorizer is like One-Hot Encoding

# In[ ]:


vectorizer = CountVectorizer()
vectorizer.fit(train_x)


# In[ ]:


x_train_count = vectorizer.transform(train_x)
x_valid_count = vectorizer.transform(valid_x)
x_test_count  = vectorizer.transform(test["text"])


# <a id = "11"></a><br>
# ## Deep Learning

# <a id = "12"></a><br>
# ## Neural Network Model

# In[ ]:


model = Sequential() 
#layers
model.add(Dense(50,input_dim=x_train_count.shape[1], kernel_initializer="uniform", activation="relu")) 
#model.add(Dense(6, kernel_initializer="uniform", activation="relu")) 
model.add(Dense(1, kernel_initializer="uniform", activation="sigmoid")) 
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
# Fit the model
history = model.fit(x_train_count, train_y.values.reshape(-1,1), validation_data=(x_valid_count,valid_y), nb_epoch=2, batch_size=128)


# <a id = "13"></a><br>
# ## Accuracy

# In[ ]:


# evaluate
loss, acc = model.evaluate(x_test_count, test["label"], verbose=0)
print('Test Accuracy: %f' % (acc*100))


# <a id = "14"></a><br>
# ## Confusion Metrics

# In[ ]:


comments = pd.Series(test["text"])
comments = vectorizer.transform(comments)


# In[ ]:


y_pred = model.predict_classes(comments)
nn_cm = metrics.confusion_matrix(test["label"],y_pred)
print(nn_cm)


# <a id = "15"></a><br>
# ## ROC

# In[ ]:


import sklearn.metrics as metrics
# calculate the fpr and tpr for all thresholds of the classification
probs = model.predict_proba(x_valid_count)
preds = probs[:,:]
fpr, tpr, threshold = metrics.roc_curve(test["label"], y_pred)
roc_auc = metrics.auc(fpr, tpr)

# method I: plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.3f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# <a id = "16"></a><br>
# ## Making Prediction

# In[ ]:


comment_1 = pd.Series("this film is very nice and good i like it")
comment_2 = pd.Series("no not good look at that shit very bad")


# In[ ]:


comment_1  = vectorizer.transform(comment_1)
comment_2 = vectorizer.transform(comment_2)


# <a id = "17"></a><br>
# ## A Positive Comment

# In[ ]:


model.predict_classes(comment_1)


# <a id = "18"></a><br>
# ## A Negative Comment

# In[ ]:


model.predict_classes(comment_2)

