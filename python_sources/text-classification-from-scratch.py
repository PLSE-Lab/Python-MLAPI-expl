#!/usr/bin/env python
# coding: utf-8

# <div class="list-group" id="list-tab" role="tablist">
#   <h3 class="list-group-item list-group-item-action active" data-toggle="list"  role="tab" aria-controls="home">Notebook Content</h3>
#   <a class="list-group-item list-group-item-action" data-toggle="list" href="##Library-and-Data" role="tab" aria-controls="profile">Library and Data<span class="badge badge-primary badge-pill">1</span></a>
#   <a class="list-group-item list-group-item-action" data-toggle="list" href="#Reading-Data" role="tab" aria-controls="messages">Reading Data<span class="badge badge-primary badge-pill">2</span></a>
#   <a class="list-group-item list-group-item-action"  data-toggle="list" href="#Logistic-Regression-Classifier" role="tab" aria-controls="settings">Logistic Regression Classifier<span class="badge badge-primary badge-pill">3</span></a>
#   <a class="list-group-item list-group-item-action" data-toggle="list" href="#Support-Vector-Classifier" role="tab" aria-controls="settings">Support Vector Classifier<span class="badge badge-primary badge-pill">4</span></a> 
#   <a class="list-group-item list-group-item-action" data-toggle="list" href="#Multinomial-Naive-Bayes-Classifier" role="tab" aria-controls="settings">Multinomial Naive Bayes Classifier<span class="badge badge-primary badge-pill">5</span></a>
#     <a class="list-group-item list-group-item-action" data-toggle="list" href="#Bernoulli-Naive-Bayes-Classifier" role="tab" aria-controls="settings">Bernoulli Naive Bayes Classifier<span class="badge badge-primary badge-pill">6</span></a>
#     <a class="list-group-item list-group-item-action" data-toggle="list" href="#Gradient-Boost-Classifier" role="tab" aria-controls="settings">Gradient Boost Classifier<span class="badge badge-primary badge-pill">7</span></a>
#     <a class="list-group-item list-group-item-action" data-toggle="list" href="#XGBoost-Classifier" role="tab" aria-controls="settings">XGBoost Classifier<span class="badge badge-primary badge-pill">8</span></a>  
#     <a class="list-group-item list-group-item-action" data-toggle="list" href="#Stochastic-Gradient-Descent" role="tab" aria-controls="settings">Stochastic Gradient Descent<span class="badge badge-primary badge-pill">9</span></a>
#      <a class="list-group-item list-group-item-action" data-toggle="list" href="#Decision-Tree" role="tab" aria-controls="settings">Decision Tree<span class="badge badge-primary badge-pill">10</span></a>
#      <a class="list-group-item list-group-item-action" data-toggle="list" href="#Random-Forest-Classifier" role="tab" aria-controls="settings">Random Forest Classifier<span class="badge badge-primary badge-pill">11</span></a>
#      <a class="list-group-item list-group-item-action" data-toggle="list" href="#KNN-Classifier" role="tab" aria-controls="settings">KNN Classifier<span class="badge badge-primary badge-pill">12</span></a>
#     <a class="list-group-item list-group-item-action" data-toggle="list" href="#LSTM" role="tab" aria-controls="settings">LSTM<span class="badge badge-primary badge-pill">12</span></a>
#     

# # Library and Data

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import feature_extraction, linear_model, model_selection, preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import MultinomialNB

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.gaussian_process import GaussianProcessClassifier
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping

from sklearn.metrics import accuracy_score, confusion_matrix,classification_report

import nltk
import nltk as nlp
import string
import re
true = pd.read_csv("../input/fake-and-real-news-dataset/True.csv")
fake = pd.read_csv("../input/fake-and-real-news-dataset/Fake.csv")


# # Reading Data

# In[ ]:


fake['target'] = 'fake'
true['target'] = 'true'
news = pd.concat([fake, true]).reset_index(drop = True)
news.head()


# # Logistic Regression Classifier

# In[ ]:


x_train,x_test,y_train,y_test = train_test_split(news['text'], news.target, test_size=0.2, random_state=2020)

pipe = Pipeline([('vect', CountVectorizer()),
                 ('tfidf', TfidfTransformer()),
                 ('model', LogisticRegression())])

model = pipe.fit(x_train, y_train)
prediction = model.predict(x_test)
print("accuracy: {}%".format(round(accuracy_score(y_test, prediction)*100,2)))


# In[ ]:


print(confusion_matrix(y_test, prediction))


# In[ ]:


print(classification_report(y_test, prediction))


# # Support Vector Classifier

# In[ ]:


x_train,x_test,y_train,y_test = train_test_split(news['text'], news.target, test_size=0.2, random_state=2020)

pipe = Pipeline([('vect', CountVectorizer()),
                 ('tfidf', TfidfTransformer()),
                 ('model', LinearSVC())])

model = pipe.fit(x_train, y_train)
prediction = model.predict(x_test)
print("accuracy: {}%".format(round(accuracy_score(y_test, prediction)*100,2)))


# In[ ]:


print(confusion_matrix(y_test, prediction))


# In[ ]:


print(classification_report(y_test, prediction))


# # Multinomial Naive Bayes Classifier

# In[ ]:


pipe = Pipeline([('vect', CountVectorizer()),
                 ('tfidf', TfidfTransformer()),
                 ('model', MultinomialNB())])

model = pipe.fit(x_train, y_train)
prediction = model.predict(x_test)
print("accuracy: {}%".format(round(accuracy_score(y_test, prediction)*100,2)))


# In[ ]:


print(confusion_matrix(y_test, prediction))


# In[ ]:


print(classification_report(y_test, prediction))


# # Bernoulli Naive Bayes Classifier

# In[ ]:


pipe = Pipeline([('vect', CountVectorizer()),
                 ('tfidf', TfidfTransformer()),
                 ('model', BernoulliNB())])

model = pipe.fit(x_train, y_train)
prediction = model.predict(x_test)
print("accuracy: {}%".format(round(accuracy_score(y_test, prediction)*100,2)))


# In[ ]:


print(confusion_matrix(y_test, prediction))


# In[ ]:


print(classification_report(y_test, prediction))


# # Gradient Boost Classifier

# In[ ]:


pipe = Pipeline([('vect', CountVectorizer()),
                 ('tfidf', TfidfTransformer()),
                 ('model', GradientBoostingClassifier(loss = 'deviance',
                                                   learning_rate = 0.01,
                                                   n_estimators = 10,
                                                   max_depth = 5,
                                                   random_state=55))])

model = pipe.fit(x_train, y_train)
prediction = model.predict(x_test)
print("accuracy: {}%".format(round(accuracy_score(y_test, prediction)*100,2)))


# In[ ]:


print(confusion_matrix(y_test, prediction))


# In[ ]:


print(classification_report(y_test, prediction))


# # XGBoost Classifier

# In[ ]:


pipe = Pipeline([('vect', CountVectorizer()),
                 ('tfidf', TfidfTransformer()),
                 ('model', XGBClassifier(loss = 'deviance',
                                                   learning_rate = 0.01,
                                                   n_estimators = 10,
                                                   max_depth = 5,
                                                   random_state=2020))])

model = pipe.fit(x_train, y_train)
prediction = model.predict(x_test)
print("accuracy: {}%".format(round(accuracy_score(y_test, prediction)*100,2)))


# In[ ]:


print(confusion_matrix(y_test, prediction))


# In[ ]:


print(classification_report(y_test, prediction))


# # Stochastic Gradient Descent

# In[ ]:


pipe = Pipeline([('vect', CountVectorizer()),
                 ('tfidf', TfidfTransformer()),
                 ('model', SGDClassifier())])

model = pipe.fit(x_train, y_train)
prediction = model.predict(x_test)
print("accuracy: {}%".format(round(accuracy_score(y_test, prediction)*100,2)))


# In[ ]:


print(confusion_matrix(y_test, prediction))


# In[ ]:


print(classification_report(y_test, prediction))


# # Decision Tree

# In[ ]:


pipe = Pipeline([('vect', CountVectorizer()),
                 ('tfidf', TfidfTransformer()),
                 ('model', DecisionTreeClassifier(criterion= 'entropy',
                                           max_depth = 10, 
                                           splitter='best', 
                                           random_state=2020))])

model = pipe.fit(x_train, y_train)
prediction = model.predict(x_test)
print("accuracy: {}%".format(round(accuracy_score(y_test, prediction)*100,2)))


# In[ ]:


print(confusion_matrix(y_test, prediction))


# In[ ]:


print(classification_report(y_test, prediction))


# # Random Forest Classifier

# In[ ]:


pipe = Pipeline([('vect', CountVectorizer()),
                 ('tfidf', TfidfTransformer()),
                 ('model', RandomForestClassifier())])

model = pipe.fit(x_train, y_train)
prediction = model.predict(x_test)
print("accuracy: {}%".format(round(accuracy_score(y_test, prediction)*100,2)))


# In[ ]:


print(confusion_matrix(y_test, prediction))


# In[ ]:


print(classification_report(y_test, prediction))


# # KNN Classifier

# In[ ]:


pipe = Pipeline([('vect', CountVectorizer()),
                 ('tfidf', TfidfTransformer()),
                 ('model', KNeighborsClassifier(n_neighbors = 10,weights = 'distance',algorithm = 'brute'))])

model = pipe.fit(x_train, y_train)
prediction = model.predict(x_test)
print("accuracy: {}%".format(round(accuracy_score(y_test, prediction)*100,2)))


# In[ ]:


print(confusion_matrix(y_test, prediction))


# In[ ]:


print(classification_report(y_test, prediction))


# # LSTM

# In[ ]:


X = news.text
Y = news.target
le = LabelEncoder()
Y = le.fit_transform(Y)
Y = Y.reshape(-1,1)


# In[ ]:


X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.15)
max_words = 500
max_len = 75
tok = Tokenizer(num_words=max_words)
tok.fit_on_texts(X_train)
sequences = tok.texts_to_sequences(X_train)
sequences_matrix = sequence.pad_sequences(sequences,maxlen=max_len)
def RNN():
    inputs = Input(name='inputs',shape=[max_len])
    layer = Embedding(max_words,50,input_length=max_len)(inputs)
    layer = LSTM(64)(layer)
    layer = Dense(256,name='FC1')(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(1,name='out_layer')(layer)
    layer = Activation('sigmoid')(layer)
    model = Model(inputs=inputs,outputs=layer)
    return model
model = RNN()


# In[ ]:


from tensorflow.keras.utils import plot_model 
plot_model(model, to_file='model1.png')
model.compile(loss='binary_crossentropy',optimizer=RMSprop(),metrics=['accuracy'])


# In[ ]:


model.fit(sequences_matrix,Y_train,batch_size=256,epochs=1,
          validation_split=0.2,callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.0001)])


# In[ ]:


test_sequences = tok.texts_to_sequences(X_test)
test_sequences_matrix = sequence.pad_sequences(test_sequences,maxlen=max_len)
accr = model.evaluate(test_sequences_matrix,Y_test)
print('Accuracy: {:0.2f}'.format(accr[1]))

