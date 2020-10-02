#!/usr/bin/env python
# coding: utf-8

# <p style="color:brown;font-size:20px">Incase you like this notebook do not forget to give an <span style="color:purple;font-size:30px">UPVOTE</span>. Thank you for viewing.

# ![](https://venturebeat.com/wp-content/uploads/2018/09/natural-language-processing-e1572968977211.jpg?w=1200&strip=all)

# <p style="color:red">This notebook is all about NLP which is a very hot topic in the field of machine learning and deeplearning for a while now.The notebook is sure a big longer but please stick with me to learn about NLPs. It'll surely help you a lot if you are beginner in the topic
# If you are just beginning to learn about NLP(Natural Language Processing) and its applications please refer to the link i have provided below to get the very basic knowledge about it :
#     
# https://towardsdatascience.com/a-gentle-introduction-to-natural-language-processing-e716ed3c0863 <---- <span style="color:purple">Introduction to NLP </span>
# 
#                                                                                                        
# <p style="color:red">If you are using python as your language for ML it provides various tools to process text in different ways. Some of them are :
# <ul style="color:green">
#     <li>SkLearn</li>
#     <li>Tensorflow</li> 
#     <li>Spacy</li>
#     <li>NLTK - Natural Language Toolkit </li>
#     <li>Gensim</li>
#     <li>OpenNLP</li>
#     etc..
# </ul>
# 
# <p style="color:red">If you want to get an idea about these tools just refer to the link below :
# 
# https://towardsdatascience.com/5-heroic-tools-for-natural-language-processing-7f3c1f8fc9f0 <--- <span style="color:purple">Tools for NLP</span>
# 
# <p style="color:red">Here is a very interesting article about NLP which shows a bit more about the components of the topic. It is a good read if you are interested 
# 
# https://towardsdatascience.com/natural-language-processing-a1496244c15c#:~:text=Natural%20Language%20Processing%20(NLP)%20is,algorithms%20to%20text%20and%20speech 
# 
# <p style="color:red">Now let us move forward

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# <h2 style="color:orange">Importing Libraries :</h2>
# 
# <p style="color:red">I am going to import the basic necessities required for NLP. If you are not familier with some of them dont worry as you will learn about them sooner or later in this notebook. :)

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
from tqdm import tqdm

from gensim.parsing.preprocessing import remove_stopwords
from bs4 import BeautifulSoup
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from collections import OrderedDict
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split,RandomizedSearchCV
from sklearn.metrics import classification_report,f1_score

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras 
from keras import backend as K
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam

import torch
import transformers


# <h2 style="color:orange">Reading the data :</h2>
# 
# <p style="color:red">I am now gonna read the data here in the .csv files

# In[ ]:


train = pd.read_csv(r'/kaggle/input/nlp-getting-started/train.csv')
test = pd.read_csv(r'/kaggle/input/nlp-getting-started/test.csv')


# In[ ]:


print('Number of datapoints in the train dataset : ',train.shape[0])
print('Number of datapoints in the test dataset : ',test.shape[0])


# In[ ]:


train.head()


# In[ ]:


train.info()


# In[ ]:


test.info()


# In[ ]:


train.describe()


# <h2 style="color:orange">Data Preprocessing :</h2>
# <p style="color:red">Here i am going to write the preprocessing functions that i am going to use in this notebook. If you want to learn about stemming,lemmatization etc just refer to the link given below :</p>
# 
# https://towardsdatascience.com/nlp-text-preprocessing-a-practical-guide-and-template-d80874676e79

# In[ ]:


#removing any shortforms if present
def remove_shortforms(phrase):
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase

def remove_special_char(text):
    text = re.sub('[^A-Za-z0-9]+'," ",text)
    return text

def remove_wordswithnum(text):
    text = re.sub("\S*\d\S*", "", text).strip()
    return text

def lowercase(text):
    text = text.lower()
    return text

def remove_stop_words(text):
    text = remove_stopwords(text)
    return text

st = SnowballStemmer(language='english')
def stemming(text):
    r= []
    for word in text :
        a = st.stem(word)
        r.append(a)
    return r

def listToString(s):  
    str1 = " "   
    return (str1.join(s))

def remove_punctuations(text):
    text = re.sub(r'[^\w\s]','',text)
    return text

def remove_links(text):
    text = re.sub(r'http\S+', '', text)
    return text

lemmatizer = WordNetLemmatizer()
def lemmatize_words(text):
    text = lemmatizer.lemmatize(text)
    return text

def remove_html(text):
    html=re.compile(r'<.*?>')
    return html.sub(r'',text)


# <p style="color:purple">I have seperated the target variables with the dataset here and concatenated the train and test dataset into another dataset using pd.concat which is a function in pandas library in python.

# In[ ]:


Y = train['target']
train = train.drop('target',axis=1)
data = pd.concat([train,test],axis=0).reset_index(drop=True)
data.head()


# <p style="color:red">Here i am converting all the text sentences into str just to make sure my code does not get stuck at any point later on.

# In[ ]:


for i in range(len(data['text'])):
    data['text'][i] = str(data['text'][i])


# In[ ]:


data['text'][1]


# <p style="color:red">Now lets preprocess the data using the functions defined above

# In[ ]:


for i in range(len(data['text'])):
    data['text'][i] = remove_shortforms(data['text'][i])
    data['text'][i] = remove_special_char(data['text'][i])
    data['text'][i] = remove_wordswithnum(data['text'][i])
    data['text'][i] = lowercase(data['text'][i])
    data['text'][i] = remove_stop_words(data['text'][i])
    text = data['text'][i]
    text = text.split()
    data['text'][i] = stemming(text)
    s = data['text'][i]
    data['text'][i] = listToString(s)
    data['text'][i] = lemmatize_words(data['text'][i])


# <p style="color:purple">Look above and see how the sentence changes before and after preprocessing as i have removed stopwords, converted characters to lowercase and applied lemmatization  etc....

# In[ ]:


data['text'][1]


# <h2 style="color:orange">Bag of words :</h2>
# <p style="color:red">The bag-of-words model is a simplifying representation used in natural language processing and information retrieval (IR). In this model, a text (such as a sentence or a document) is represented as the bag (multiset) of its words, disregarding grammar and even word order but keeping multiplicity. 
# If you are not familier with bag of words this article is for you :
# 
# https://machinelearningmastery.com/gentle-introduction-bag-words-model/ <--- <span style="color:purple">Bag of Words</span>
# 
# <p style="color:red">In sklearn which is a very famous lirary in python bag of words is used as the fuction CountVectorizer.For its documentation in sklearb refer to this link below 
# 
# * https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html <--- <span style="color:purple">Documentation</span>

# In[ ]:


cv = CountVectorizer(ngram_range=(1,3))
text_bow = cv.fit_transform(data['text'])
print(text_bow.shape)


# <p style="color:green">Now i am splitting the training and text data with BOW encoding which i had combined earlier to get the dictionary of all words present in the train as well as the text data

# In[ ]:


train_text = text_bow[:train.shape[0]] 
test_text = text_bow[train.shape[0]:] 


# In[ ]:


print(train_text.shape)
print(test_text.shape)


# <h2 style="color:orange">Train test split :</h2>
# 
# <p style="color:purple">The fuction used for splitting the train and test data is a sklearn function. For its documentation refer to :
#     
# https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html <--- <span style="color:red">Train Test split</span>

# In[ ]:


X_train,X_test,Y_train,Y_test = train_test_split(train_text,Y,test_size=0.2)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)


# <p style="color:green">Lets try Logistic Regression with some hyperparameter tuning on the BOW encoded data:
#     
#     
# <p style="color:red">If you want to try hyperparameter tuning on your system just uncomment the code in the next block

# <h2 style="color:orange">Logistic Regression with BOW :

# In[ ]:


# lr = LogisticRegression(max_iter=2000)

# params = {
#     'C' :[0.0001,0.001,0.01,0.1,0.9,0.8,0.7,0.6,0.5,0.4,0.3,1,2,3,4,5,6,7,10,100,1000],
#     'penalty': ['l1','l2']
# }

# clf = RandomizedSearchCV(lr,params,n_jobs=-1,cv=10)
# clf.fit(X_train,Y_train)
# print(clf.best_params_)


# In[ ]:


lr = LogisticRegression(C=10,penalty='l2')
lr.fit(X_train,Y_train)
pred = lr.predict(X_test)
print("F1 score :",f1_score(Y_test,pred))
print("Classification Report \n\n:",classification_report(Y_test,pred))


# <h2 style="color:orange">Predictions and Submission :

# In[ ]:


lr = LogisticRegression(C=10,penalty='l2',max_iter=2000)
lr.fit(train_text,Y)
pred = lr.predict(test_text)
submit = pd.DataFrame(test['id'],columns=['id'])
print(len(pred))
submit.head()


# In[ ]:


submit['target'] = pred
submit.to_csv("realnlp.csv",index=False)


# <h1 style="color:orange">TFIDF encoding of the text data :</h1>
# 
# <p style="color:red">To learn more about TFIDF just follow this link below :
#     
# https://medium.com/analytics-vidhya/tf-idf-term-frequency-technique-easiest-explanation-for-text-classification-in-nlp-with-code-8ca3912e58c3 <--- <span style="color:purple">TFIDF encoding of text</span>
# 
# <p style="color:red">In sklearn TFIDF is applied using the fuction TfidfVectorizer.For its documentation in sklearn refer to this link below :
# 
# https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html <--- <span style="color:purple">Documentation</span>

# In[ ]:


tfidf = TfidfVectorizer(ngram_range=(1,3))
text_tfidf = tfidf.fit_transform(data['text'])
print(text_tfidf.shape)


# In[ ]:


train_text = text_tfidf[:train.shape[0]] 
test_text = text_tfidf[train.shape[0]:] 
print(train_text.shape)
print(test_text.shape)


# In[ ]:


X_train,X_test,Y_train,Y_test = train_test_split(train_text,Y,test_size=0.2)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)


# <h2 style="color:orange">Logistic Regression with TFIDF encoding :

# In[ ]:


# lr = LogisticRegression(max_iter=2000)

# params = {
#     'C' :[0.0001,0.001,0.01,0.1,0.9,0.8,0.7,0.6,0.5,0.4,0.3,1,2,3,4,5,6,7,10,100,1000],
#     'penalty': ['l1','l2']
# }

# clf = RandomizedSearchCV(lr,params,n_jobs=-1,cv=10)
# clf.fit(X_train,Y_train)
# print(clf.best_params_)


# In[ ]:


lr = LogisticRegression(C=100,penalty='l2',max_iter=2000)
lr.fit(X_train,Y_train)
pred = lr.predict(X_test)
print("F1 score :",f1_score(Y_test,pred))
print("Classification Report :",classification_report(Y_test,pred))


# <p style="color:magenta">The feature location is not a useful feature in the data as we have so many missing values in it which cannot be dealt with properly. Therefore i am not gonna perform any calculations using the location feature
# 
# <h2 style="color:turquoise;font-size:30px">2. Lets use the Keywords feature and see if the results change 

# In[ ]:


print("Number of null values in data keywords column : ",data['keyword'].isnull().sum())


# In[ ]:


data.head()


# <p style="color:green">Filling the missing or null values in the keyword feature as 'unknown'

# In[ ]:


data['keyword'] = data['keyword'].fillna("unknown")
data.head()


# <p style="color:red">Combining the text and the keywords togther
# 
# <p style="color:red">Here i have used keyword after the text 3 times to give it more weight as we are predicting the text refers to a disaster or not. Maybe it will help lets see if it does

# In[ ]:


combined_text = [None] * len(data['text'])
for i in range(len(data['text'])):
    if data['keyword'][i] == 'unknown':
        combined_text[i] = data['text'][i]
    else:
        combined_text[i] = data['text'][i] + " " + data['keyword'][i] + " " + data['keyword'][i] + " " + data['keyword'][i]
data['combined_text'] = combined_text


# In[ ]:


data['combined_text'][88]


# In[ ]:


for i in range(len(data['combined_text'])):
    data['combined_text'][i] = str(data['combined_text'][i])


# <p style="color:purple">Preprocessing the combined data :

# In[ ]:


for i in range(len(data['combined_text'])):
    data['combined_text'][i] = remove_shortforms(data['combined_text'][i])
    data['combined_text'][i] = remove_special_char(data['combined_text'][i])
    data['combined_text'][i] = remove_wordswithnum(data['combined_text'][i])
    data['combined_text'][i] = lowercase(data['combined_text'][i])
    data['combined_text'][i] = remove_stop_words(data['combined_text'][i])
    text = data['combined_text'][i]
    text = text.split()
    data['combined_text'][i] = stemming(text)
    s = data['combined_text'][i]
    data['combined_text'][i] = listToString(s)
    data['combined_text'][i] = lemmatize_words(data['combined_text'][i])


# In[ ]:


data['combined_text'][88]


# <h2 style="color:orange">Bag of Words with keywords :</h2>
#     
# <p style="color:red">Applying BOW with the combined text

# In[ ]:


cv = CountVectorizer(ngram_range=(1,3))
text_bow = cv.fit_transform(data['combined_text'])
print(text_bow.shape)


# In[ ]:


train_text = text_bow[:train.shape[0]] 
test_text = text_bow[train.shape[0]:] 


# In[ ]:


X_train,X_test,Y_train,Y_test = train_test_split(train_text,Y,test_size=0.2)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)


# <h2 style="color:orange">Logistic Regression with BOW :

# In[ ]:


# lr = LogisticRegression(max_iter=2000)

# params = {
#     'C' :[0.0001,0.001,0.01,0.1,0.9,0.8,0.7,0.6,0.5,0.4,0.3,1,2,3,4,5,6,7,10,100,1000],
#     'penalty': ['l1','l2']
# }

# clf = RandomizedSearchCV(lr,params,n_jobs=-1,cv=10)
# clf.fit(X_train,Y_train)
# print(clf.best_params_)


# In[ ]:


lr = LogisticRegression(C=1,penalty='l2',max_iter=2000)
lr.fit(X_train,Y_train)
pred = lr.predict(X_test)
print("F1 score :",f1_score(Y_test,pred))
print("Classification Report :",classification_report(Y_test,pred))


# <h2 style="color:red">TFIDF with keyword features :</h2>
# 
# <p style="color:orange">Applying TFIDF with combined text

# In[ ]:


tfidf = TfidfVectorizer(ngram_range=(1,3))
text_tfidf = tfidf.fit_transform(data['combined_text'])
print(text_tfidf.shape)


# In[ ]:


train_text = text_tfidf[:train.shape[0]] 
test_text = text_tfidf[train.shape[0]:] 


# In[ ]:


X_train,X_test,Y_train,Y_test = train_test_split(train_text,Y,test_size=0.2)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)


# <h2 style="color:orange">Logistic Regression with TFIDF encoding :

# In[ ]:


# lr = LogisticRegression(max_iter=2000)

# params = {
#     'C' :[0.0001,0.001,0.01,0.1,0.9,0.8,0.7,0.6,0.5,0.4,0.3,1,2,3,4,5,6,7,10,100,1000],
#     'penalty': ['l1','l2']
# }

# clf = RandomizedSearchCV(lr,params,n_jobs=-1,cv=10)
# clf.fit(X_train,Y_train)
# print(clf.best_params_)


# In[ ]:


lr = LogisticRegression(C=2,penalty='l2',max_iter=2000)
lr.fit(X_train,Y_train)
pred = lr.predict(X_test)
print("F1 score :",f1_score(Y_test,pred))
print("Classification Report :",classification_report(Y_test,pred))


# <h1 style="color:turquoise;font-size:50px">3. Word Embeddings :</h1>
#     
# <p style="color:red">Word embedding are also a very popular way to approach an NLP problem in which words are converted into vectors and used in various ML and deeplearning models.
# 
# <p style="color:red">To know more refer to the link below :
#     
# https://machinelearningmastery.com/what-are-word-embeddings/ <--- <span style="color:purple">Word Embeddings</span>
# 
# <p style="color:red">Here i have used GLOVE vectors dataset for my word embeddings. You can download them either on the internet but i have just used this glove dataset provided on kaggle itself to carry out the task. 
# 
# <p style="color:red">Here is the link :
# 
# https://www.kaggle.com/rtatman/glove-global-vectors-for-word-representation <--- <span style="color:purple">Glove Vectors</span>

# In[ ]:


print('Loading word vectors...')
word2vec = {}
with open(os.path.join('../input/glove-global-vectors-for-word-representation/glove.6B.200d.txt'), encoding = "utf-8") as f:
  # is just a space-separated text file in the format:
  # word vec[0] vec[1] vec[2] ...
    for line in f:
        values = line.split() #split at space
        word = values[0]
        vec = np.asarray(values[1:], dtype='float32') #numpy.asarray()function is used when we want to convert input to an array.
        word2vec[word] = vec
print('Found %s word vectors.' % len(word2vec))


# In[ ]:


train = pd.read_csv(r'/kaggle/input/nlp-getting-started/train.csv')
test = pd.read_csv(r'/kaggle/input/nlp-getting-started/test.csv')


# In[ ]:


Y = train['target']
train = train.drop('target',axis=1)
data = pd.concat([train,test],axis=0).reset_index(drop=True)
text_data = data['text']


# In[ ]:


text_data


# <p style="color:green">For tokenizing the data i have used the keras.preprocessing fuction called Tokenizer.
#     You can search on google for its documentation

# In[ ]:


tokenizer = Tokenizer()
tokenizer.fit_on_texts(text_data)
sequences = tokenizer.texts_to_sequences(text_data)


# In[ ]:


word2index = tokenizer.word_index
print("Number of unique tokens : ",len(word2index))


# <p style="color:red">Here i have done padding on the data itself so as to convert it into equal sized vectors which are processed by the models i am going to use later on these embeddings.
# 
# <p style="color:green">To know more about padding of vectors just refer to the video below :
# 
# https://www.coursera.org/lecture/natural-language-processing-tensorflow/padding-2Cyzs <--- <span style="color:purple">Padding</span>

# In[ ]:


data_padded = pad_sequences(sequences,100)
print(data_padded.shape)


# In[ ]:


data_padded[6]


# In[ ]:


train_pad = data_padded[:train.shape[0]]
test_pad = data_padded[train.shape[0]:]


# In[ ]:


embedding_matrix = np.zeros((len(word2index)+1,200))

embedding_vec=[]
for word, i in tqdm(word2index.items()):
    embedding_vec = word2vec.get(word)
    if embedding_vec is not None:
        embedding_matrix[i] = embedding_vec


# In[ ]:


print(embedding_matrix[1])


# <p style="color:purple">You can use Ml models here but i have used Deeplearning with the word embeddings.Lets see what kind of results we get

# <h1 style="color:magenta;font-size:50px;">RNN :</h1>
# 
# ![](https://cdn-images-1.medium.com/fit/t/1600/480/1*go8PHsPNbbV6qRiwpUQ5BQ.png)
# 
# 

# <h2 style="color:orange">LSTM :</h2> 
# 
# ![](https://miro.medium.com/max/2840/1*0f8r3Vd-i4ueYND1CUrhMA.png)
# 
# <p style="color:red">To learn more about LSTM's and how they actually work please use the links below:
# 
# https://machinelearningmastery.com/gentle-introduction-long-short-term-memory-networks-experts/ <--- <span style="color:purple">LSTM introduction</span>
# 
# https://www.coursera.org/lecture/nlp-sequence-models/long-short-term-memory-lstm-KXoay <--- <span style="color:purple">LSTMs</span>

# In[ ]:


model1 = keras.models.Sequential([
    keras.layers.Embedding(len(word2index)+1,200,weights=[embedding_matrix],input_length=100,trainable=False),
    keras.layers.LSTM(100,return_sequences=True),
    keras.layers.LSTM(200),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1,activation='sigmoid')
])


# In[ ]:


model1.summary()


# In[ ]:


model1.compile(
  loss='binary_crossentropy',
  optimizer='adam',
  metrics=['accuracy'],
)


# In[ ]:


history1 = model1.fit(train_pad,Y,
                    batch_size=64,
                    epochs=10,
                    validation_split=0.2
)


# In[ ]:


plt.figure(figsize=(20,8))
plt.plot(history1.history['loss'], label='train')
plt.plot(history1.history['val_loss'], label='test')
plt.legend()
plt.grid()
plt.show()


# In[ ]:


plt.figure(figsize=(20,8))
plt.plot(history1.history['accuracy'], label='train')
plt.plot(history1.history['val_accuracy'], label='test')
plt.legend()
plt.grid()
plt.show()


# <h2 style="color:orange">GRU :</h2> 
# 
# ![](https://technopremium.com/blog/wp-content/uploads/2019/06/gru-1-1200x600.png)
# 
# <p style="color:red">Let us use GRU's to see if the results change or not
# 
# <p style="color:red">If you want to know more about GRU and how they work please refer to the links below 
# 
# https://towardsdatascience.com/understanding-gru-networks-2ef37df6c9be <--- <span style="color:purple">GRU inroduction</span>
# 
# https://www.coursera.org/lecture/nlp-sequence-models/gated-recurrent-unit-gru-agZiL <--- <span style="color:purple">GRU</span>

# In[ ]:


model2 = keras.models.Sequential([
    keras.layers.Embedding(len(word2index)+1,200,weights=[embedding_matrix],input_length=100,trainable=False),
    keras.layers.GRU(100,return_sequences=True),
    keras.layers.GRU(200),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1,activation='sigmoid')
])


# In[ ]:


model2.summary()


# In[ ]:


model2.compile(
  loss='binary_crossentropy',
  optimizer='adam',
  metrics=['accuracy'],
)


# In[ ]:


history2 = model2.fit(train_pad,Y,
                    batch_size=64,
                    epochs=10,
                    validation_split=0.2
)


# In[ ]:


plt.figure(figsize=(20,8))
plt.plot(history2.history['loss'], label='train')
plt.plot(history2.history['val_loss'], label='test')
plt.legend()
plt.grid()
plt.show()


# In[ ]:


plt.figure(figsize=(20,8))
plt.plot(history2.history['accuracy'], label='train')
plt.plot(history2.history['val_accuracy'], label='test')
plt.legend()
plt.grid()
plt.show()


# <h2 style="color:orange">Bidirectional LSTM :</h2>
# 
# ![](https://www.i2tutorials.com/wp-content/uploads/2019/05/Deep-Dive-into-Bidirectional-LSTM-i2tutorials.jpg)
# 
# <p style="color:red">Let us use Bidirectional LSTM's to see if the results change or not
# 
# <p style="color:red">If you want to know more about Bidirectional LSTM and how they work please refer to the links below 
# 
# https://machinelearningmastery.com/develop-bidirectional-lstm-sequence-classification-python-keras/ <--- <span style="color:purple">Bidirectional LSTM inroduction</span>
# 
# https://www.coursera.org/lecture/nlp-sequence-models/bidirectional-rnn-fyXnn<--- <span style="color:purple">Bidirectional LSTM</span>

# In[ ]:


model3 = keras.models.Sequential([
    keras.layers.Embedding(len(word2index)+1,200,weights=[embedding_matrix],input_length=100,trainable=False),
    keras.layers.Bidirectional(keras.layers.LSTM(100,return_sequences=True)),
    keras.layers.Bidirectional(keras.layers.LSTM(200)),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1,activation='sigmoid')
])


# In[ ]:


model3.summary()


# In[ ]:


model3.compile(
  loss='binary_crossentropy',
  optimizer='adam',
  metrics=['accuracy'],
)


# In[ ]:


history3 = model3.fit(train_pad,Y,
                    batch_size=64,
                    epochs=10,
                    validation_split=0.2
)


# In[ ]:


plt.figure(figsize=(20,8))
plt.plot(history3.history['loss'], label='train')
plt.plot(history3.history['val_loss'], label='test')
plt.legend()
plt.grid()
plt.show()


# In[ ]:


plt.figure(figsize=(20,8))
plt.plot(history3.history['accuracy'], label='train')
plt.plot(history3.history['val_accuracy'], label='test')
plt.legend()
plt.grid()
plt.show()


# <p style="color:red">You guys can also try Bidirectional GRU if you feel like it

# <h2 style="color:orange">Early Stopping :</h2>
# 
# <p style="color:red">Let me try to show you how  early stopping works to determine the best validation accuracy for the model as we can see that the accuracy is maximun in the starting epochs and decreasing as we go further. Here i kept the monitor for early stopping as val_accuracy.
# 
# https://machinelearningmastery.com/early-stopping-to-avoid-overtraining-neural-network-models/ <--- <span style="color:purple">Early Stopping

# <p style="color:green">Let me use early stopping on the Bidirectional LSTM model to show how it works

# In[ ]:


es = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',mode='max',verbose=1,patience=3)


# In[ ]:


history = model3.fit(train_pad,Y,
                    batch_size=64,
                    epochs=30,
                    validation_split=0.2,
                    callbacks=[es]
)


# In[ ]:


plt.figure(figsize=(20,8))
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.grid()
plt.show()


# In[ ]:


plt.figure(figsize=(20,8))
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='test')
plt.legend()
plt.grid()
plt.show()


# <h2 style="color:orange">Predictions and Submissions :

# In[ ]:


submit = pd.DataFrame(test['id'],columns=['id'])
predictions = model3.predict(test_pad)
submit['target_prob'] = predictions
submit.head()


# In[ ]:


target = [None]*len(submit)
for i in range(len(submit)):
    target[i] = np.round(submit['target_prob'][i]).astype(int)
submit['target'] = target
submit.head()


# In[ ]:


submit = submit.drop('target_prob',axis=1)
submit.to_csv('real-nlp_lstm.csv',index=False)


# <h1 style="color:turquoise"><span style="color:darkblue;font-size:50px">4. BERT</span> (Bidirectional Encoder Representations from Transformers) :</h1>
# 
# ![](https://searchengineland.com/figz/wp-content/seloads/2019/10/GoogleBert_1920.jpg)
# 
# <p style="color:green">It is an open-sourced NLP pre-training model developed by researchers at Google in 2018 and it preforms very well with text data as compared to other models.
# 
# <p style="color:green">Most of us may not have an idea about it as i have recently started learning about it too. I am gonna provide some links which are very helpful in starting with it. Just follow them up for the idea about transformers and BERT models
#  
# http://jalammar.github.io/illustrated-transformer/ <--- <span style="color:purple">Transformers</span>
# 
# http://jalammar.github.io/illustrated-bert/ <--- <span style="color:purple">BERT introduction</span>
# 
# https://www.analyticsvidhya.com/blog/2019/09/demystifying-bert-groundbreaking-nlp-framework/ <--- <span style="color:purple">BERT</span>

# In[ ]:


train = pd.read_csv(r'/kaggle/input/nlp-getting-started/train.csv')
test = pd.read_csv(r'/kaggle/input/nlp-getting-started/test.csv')


# In[ ]:


train.head()


# In[ ]:


Y = train['target']
train = train.drop('target',axis=1)
text_data_train = train['text']
text_data_test = test['text']


# In[ ]:


Y.value_counts()


# In[ ]:


tokenizer = transformers.BertTokenizer.from_pretrained('bert-large-uncased', do_lower_case=True)
bert_model = transformers.TFBertModel.from_pretrained('bert-large-uncased')


# In[ ]:


def bert_encode(data,maximum_length) :
    input_ids = []
    attention_masks = []
  

    for i in range(len(data)):
        encoded = tokenizer.encode_plus(
        
          data[i],
          add_special_tokens=True,
          max_length=maximum_length,
          pad_to_max_length=True,
        
          return_attention_mask=True,
        
        )
      
        input_ids.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_mask'])
    return np.array(input_ids),np.array(attention_masks)


# In[ ]:


train_input_ids,train_attention_masks = bert_encode(text_data_train,100)
test_input_ids,test_attention_masks = bert_encode(text_data_test,100)


# In[ ]:


train_input_ids[1]


# In[ ]:


train_attention_masks[1]


# In[ ]:


def create_model(bert_model):
    input_ids = tf.keras.Input(shape=(100,),dtype='int32')
    attention_masks = tf.keras.Input(shape=(100,),dtype='int32')
  
    output = bert_model([input_ids,attention_masks])
    output = output[1]
    output = tf.keras.layers.Dense(1,activation='sigmoid')(output)
    model = tf.keras.models.Model(inputs = [input_ids,attention_masks],outputs = output)
    model.compile(Adam(lr=6e-6), loss='binary_crossentropy', metrics=['accuracy'])
    return model


# In[ ]:


model = create_model(bert_model)
model.summary()


# In[ ]:


history = model.fit([train_input_ids,train_attention_masks],Y,
                    validation_split=0.2,
                    epochs=3,
                    batch_size=5)


# <p style="color:teal">I just ran it for 3 epochs because i got an okayish accuracy but you can change that according to your convenience.
# 
# <p style="color:red">You can also use early stopping here if you'd like

# In[ ]:


# es = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',mode='max',verbose=1,patience=3)


# In[ ]:


# history = model.fit([train_input_ids,train_attention_masks],Y,
#                     batch_size=10,
#                     epochs=10,
#                     validation_split=0.2,
#                     callbacks=[es]
# )


# In[ ]:


plt.figure(figsize=(20,8))
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.grid()
plt.show()


# In[ ]:


plt.figure(figsize=(20,8))
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='test')
plt.legend()
plt.grid()
plt.show()


# <h2 style="color:orange"> Predictions :

# In[ ]:


result = model.predict([test_input_ids,test_attention_masks])
result = np.round(result).astype(int)
submit = pd.DataFrame(test['id'],columns=['id'])
submit['target'] = result
submit.head()


# In[ ]:


submit.to_csv('real_nlp_bert.csv',index=False)


# <p style="color:red">Bert model achieves the best validation accuracy so far.
# 
# <p style="color:darkblue">I am very new at using BERT itself so if you find some errors or mistakes or if you have some suggesstions related to the kernel you are very welcome in the comments section.I'd appreciate it .
# 
# <p style="color:darkblue">Thanks for viewing the kernel I hope it may have helped you :)
#   Please do <span style="color:brown;font-size:20px">UPVOTE</span> as a token of appreciation if you liked it or learned from it in any way
