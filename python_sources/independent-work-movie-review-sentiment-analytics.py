#!/usr/bin/env python
# coding: utf-8

# # Project Introduction

# This Rotten Tomatoes movie review dataset is a corpus of movie reviews used for sentiment analysis, originally collected by Pang and Lee. This competition presents a chance to benchmark your sentiment-analysis ideas on the Rotten Tomatoes dataset. You are asked to label phrases on a scale of five values: negative, somewhat negative, neutral, somewhat positive, positive.<br/>
# The objective of this project is to bulid a appropriate model that classify sentiment of each phrase. 

# # EDA

# In[ ]:


import numpy as np
import pandas as pd


# In[ ]:


train = pd.read_csv('../input/movie-review-sentiment-analysis-kernels-only/train.tsv', sep="\t")
test = pd.read_csv('../input/movie-review-sentiment-analysis-kernels-only/test.tsv', sep="\t")
sub = pd.read_csv('../input/movie-review-sentiment-analysis-kernels-only/sampleSubmission.csv', sep=",")


# In[ ]:


sub.to_csv('submission.csv',index=False)
sub.head()


# **Data Understanding**<br/>
# sentiment id representate different sentiment categories, and the following meaning explain the sentiment id:<br/>
#              The sentiment labels are:<br/>
#                              0 - negative<br/>
#                              1 - somewhat negative<br/>
#                              2 - neutral<br/>
#                              3 - somewhat positive<br/>
#                              4 - positive

# In[ ]:


train.head()


# In[ ]:


train.shape


# In[ ]:


test.head()


# In[ ]:


test.shape


# In[ ]:


sub.head()


# In[ ]:


sub.shape


# In[ ]:


print('Number of phrases in train: {}. Number of sentences in train: {}.'.format(train.shape[0], len(train.SentenceId.unique())))
print('Number of phrases in test: {}. Number of sentences in test: {}.'.format(test.shape[0], len(test.SentenceId.unique())))


# In[ ]:


print('Average count of phrases per sentence in train is {0:.0f}.'.format(train.groupby('SentenceId')['Phrase'].count().mean()))
print('Average count of phrases per sentence in test is {0:.0f}.'.format(test.groupby('SentenceId')['Phrase'].count().mean()))


# **Find Overlapped Phrases Between Train and Test Data**

# In[ ]:


overlapped = pd.merge(train[["Phrase", "Sentiment"]], test, on="Phrase", how="inner")
print(overlapped.shape)


# In[ ]:


overlapped.head()


# In[ ]:


overlap_boolean_mask_test = test['Phrase'].isin(overlapped['Phrase'])


# There are overlapped phrase texts between training and testing data, which should assign training data labels directly instead of getting from prediction.

# In[ ]:


# Check sentiment distribution
import seaborn as sns
sns.countplot(x='Sentiment', data = train)


# **As can be seen from the bar chart above, there is an imbalance issue in the training dataset, and neutral sentiment data significant more than any other class. Therefore, undersampling should be used to solve this kind of problem.**

# In[ ]:


# ramdomly select 40,000 records from neutral sentiment to fit the model
neutral = len(train[train['Sentiment'] == 2])
neutral_indices = train[train.Sentiment == 2].index
random_indices = np.random.choice(neutral_indices,40000, replace=True)
no_neutral_indices = train[train.Sentiment != 2].index
under_sample_indices = np.concatenate([no_neutral_indices,random_indices])
train = train.loc[under_sample_indices]
train.reset_index(inplace = True)
train.head()


# In[ ]:


del train['index']


# In[ ]:


train.head()


# In[ ]:


sns.countplot(x='Sentiment', data = train)


# In[ ]:


train.shape


# # Text Data Preprocessing

# **Thoughts on feature processing and engineering:**<br/>
# So, we have only phrases as data. And a phrase can contain a single word. And one punctuation mark can cause phrase to receive a different sentiment. Also assigned sentiments can be strange. This means several things:<br/>
# <br/>
# apply stopwords removal methods on all data records can be a bad idea, because some phrases contain one single stopword;<br/>
# and also puntuation could be important, so it should be used;<br/>
# using features like word count or sentence length won't be useful
# 

# In[ ]:


import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer


# **Clean Training Dataset**

# In[ ]:


# missing value check
train = train.replace('',np.NaN)
train = train.replace(' ',np.NaN)
train.isnull().any()


# In[ ]:


# delete missing value
train.dropna(inplace = True)


# In[ ]:


train.isnull().any().sum()


# In[ ]:


# add phrase lenghth colunm 
# and the phrase length is for separating long phrase and short Phrase
train['phrase_length'] = train['Phrase'].apply(lambda x: len(x.split()))
# check value counts
phrase_length = train.phrase_length.value_counts()
phrase_length.plot.bar(figsize=(25,10))


# In[ ]:


# step 1: Normalization
train['Phrase'] = train['Phrase'].apply(lambda x: x.lower())
# separate the dataset 
train1 = train[train['phrase_length'] >5]
train2 = train[train['phrase_length'] <=5]
# step 2:  stopwords removal 
stop = stopwords.words('english')
train1['Phrase'] = train1['Phrase'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
# step 3: Tokenization and punctuation removal
tokenizer = RegexpTokenizer(r'\w+')
train1['Phrase'] = train1['Phrase'].apply(lambda x: tokenizer.tokenize(x))
train2['Phrase'] = train2['Phrase'].apply(lambda x: word_tokenize(x))
# train1['Phrase'] = train1['Phrase'].apply(lambda x: remove_punctuations(x))
# merge data
frames = [train1, train2]
train = pd.concat(frames)
train.sort_index(inplace=True)
# step 4: Lemmatization
wl=WordNetLemmatizer()
train['Phrase'] = train['Phrase'].apply(lambda x: [wl.lemmatize(w,pos = 'v') for w in x])
train['Phrase'] = train['Phrase'].apply(lambda x: [wl.lemmatize(w) for w in x])
# re-calculate the phrase length
train['phrase_length'] = train['Phrase'].apply(lambda x: len(x))
train.head()


# In[ ]:


train.phrase_length.max()


# **Clean Testing dataset**

# In[ ]:


# same data cleaning process on test data set
# add phrase length column
test['phrase_length'] = test['Phrase'].apply(lambda x: len(x.split()))
# cleaning process:
# step 1: lowercase
test['Phrase'] = test['Phrase'].apply(lambda x: x.lower())
# separate dataset
test1 = test[test['phrase_length'] >5]
test2 = test[test['phrase_length'] <=5]
# step 2: stopwords removal 
test1['Phrase'] = test1['Phrase'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
# step 3: Tokenization and punctuation removal
test1['Phrase'] = test1['Phrase'].apply(lambda x: tokenizer.tokenize(x))
test2['Phrase'] = test2['Phrase'].apply(lambda x: word_tokenize(x))
# merge the test dataset
frame = [test1, test2]
test = pd.concat(frame)
test.sort_index(inplace=True)
# step 4: Lemmatization
test['Phrase'] = test['Phrase'].apply(lambda x: [wl.lemmatize(w,pos = 'v') for w in x])
test['Phrase'] = test['Phrase'].apply(lambda x: [wl.lemmatize(w) for w in x])
test['phrase_length'] = test['Phrase'].apply(lambda x: len(x))


# In[ ]:


test.phrase_length.max()


# # Model Buliding

# In[ ]:


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Conv1D, GRU, CuDNNGRU, CuDNNLSTM, BatchNormalization
from keras.layers import Bidirectional, GlobalMaxPool1D, MaxPooling1D, Add, Flatten
from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, SpatialDropout1D
from keras.models import Model, load_model
from keras import initializers, regularizers, constraints, optimizers, layers, callbacks
from keras import backend as K
from keras.engine import InputSpec, Layer
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, TensorBoard, Callback, EarlyStopping


# In[ ]:


import keras
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels


# In[ ]:


y = train['Sentiment']
y = keras.utils.to_categorical(y,num_classes=5)


# In[ ]:


phrase_train, phrase_valid, y_train, y_valid = train_test_split(train['Phrase'], y, test_size=0.2, random_state=1000)


# **1D_CNN model**

# In[ ]:


tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(phrase_train)
tokenizer.fit_on_texts(phrase_valid)
x_train = tokenizer.texts_to_sequences(phrase_train)
x_valid = tokenizer.texts_to_sequences(phrase_valid)
max_len = 32
x_train = pad_sequences(x_train, maxlen = max_len)
x_valid = pad_sequences(x_valid, maxlen = max_len)


# In[ ]:


x_train


# In[ ]:


cnn_model1 = Sequential()
filters = 100
cnn_model1.add(Embedding(input_dim = 15000, output_dim=1000, input_length = 32))
# cnn_model1.add(layers.Flatten())
cnn_model1.add(Dropout(0.1))
cnn_model1.add(Conv1D(filters, 3, strides=1, padding='valid', activation='relu'))

cnn_model1.add(GlobalMaxPool1D())
cnn_model1.add(layers.Dense(10, activation='relu'))
cnn_model1.add(layers.Dense(5, activation='softmax'))
cnn_model1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
cnn_model1.summary()


# In[ ]:


early_stopping = EarlyStopping(monitor='val_loss', patience = 10, verbose=2)
history1 = cnn_model1.fit(x_train, y_train,
                         validation_data=(x_valid, y_valid),
                          epochs=20,
                         batch_size=1000,
                         callbacks=[early_stopping])


# In[ ]:


tokenizer = Tokenizer(num_words=7000)
tokenizer.fit_on_texts(test.Phrase)
x_test = tokenizer.texts_to_sequences(test.Phrase)
max_len = 32
x_test = pad_sequences(x_test, maxlen = max_len)
pred1 = cnn_model1.predict_classes(x_test,verbose=1)


# In[ ]:


sub.Sentiment = pred1
sub.to_csv('sub1.csv',index=False)
sub.head()


# **LSTM**

# In[ ]:


lstm_model = Sequential()
lstm_model.add(Embedding(input_dim = 10000, output_dim=1000, input_length = 32))
lstm_model.add(LSTM(64,dropout=0.4, recurrent_dropout=0.4,return_sequences=True))
lstm_model.add(LSTM(32,dropout=0.5, recurrent_dropout=0.5,return_sequences=False))
lstm_model.add(Dense(5, activation='softmax'))
lstm_model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['acc'])
lstm_model.summary()


# In[ ]:


early_stopping = EarlyStopping(monitor='val_loss', patience = 10, verbose=2)
history2 = lstm_model.fit(x_train, y_train,
                          validation_data=(x_valid, y_valid),
                         epochs=20,
                         batch_size=1000
                    )


# In[ ]:


#tokenizer = Tokenizer(num_words=7000)
#tokenizer.fit_on_texts(test.Phrase)
#x_test = tokenizer.texts_to_sequences(test.Phrase)
#max_len = 32
#x_test = pad_sequences(x_test, maxlen = max_len)
pred2 = lstm_model.predict_classes(x_test,verbose=1)


# In[ ]:


sub.Sentiment = pred2
sub.to_csv('sub2.csv',index=False)
sub.head()


# **CNN+GRU**

# In[ ]:


model3= Sequential()
model3.add(Embedding(10000,1000,input_length = 32))
model3.add(Conv1D(64,kernel_size=3,padding='same',activation='relu'))
model3.add(MaxPooling1D(pool_size=2))
model3.add(Dropout(0.25))
model3.add(GRU(128,return_sequences=True))
model3.add(Dropout(0.3))
model3.add(Flatten())
model3.add(Dense(128,activation='relu'))
model3.add(Dropout(0.5))
model3.add(Dense(5,activation='softmax'))
model3.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.001),metrics=['accuracy'])
model3.summary()


# In[ ]:


get_ipython().run_cell_magic('time', '', "early_stopping = EarlyStopping(monitor='val_loss', patience = 10, verbose=2)\nhistory3 = model3.fit(x_train, y_train, validation_data=(x_valid, y_valid),epochs = 15, batch_size = 1000, verbose=1)")


# In[ ]:


pred3 = model3.predict_classes(x_test,verbose=1)


# In[ ]:


sub.Sentiment = pred3
sub.to_csv('sub3.csv',index=False)
sub.head()


# **NB**

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score


# In[ ]:


train['Phrase'] = train['Phrase'].apply(lambda x: ' '.join(x))
train.head()


# In[ ]:


train.shape


# In[ ]:


test['Phrase'] = test['Phrase'].apply(lambda x: ' '.join(x))
test.head()


# In[ ]:


Tfidf_vect = TfidfVectorizer(max_features=10000)
x_train_tf = Tfidf_vect.fit_transform(train.Phrase)
x_test_tf = Tfidf_vect.transform(test.Phrase)
y_train_tf = train['Sentiment']


# In[ ]:


# print(Tfidf_vect.vocabulary_)


# In[ ]:


# print(x_train_tf)


# In[ ]:


# fit the training dataset on the NB classifier
Naive = naive_bayes.MultinomialNB()
Naive.fit(x_train_tf,y_train_tf)


# In[ ]:


# predict the labels on train dataset
predictions_NB = Naive.predict(x_train_tf)
# Use accuracy_score function to get the accuracy
print("Naive Bayes Accuracy Score -> ",accuracy_score(predictions_NB, y_train_tf))


# In[ ]:


pred4 = Naive.predict(x_test_tf)


# In[ ]:


sub.Sentiment = pred4
sub.to_csv('sub4.csv',index=False)
sub.head()


# In[ ]:


sub_all=pd.DataFrame({'model1':pred1,'model2':pred2,'model3':pred3,'model4':pred4})
pred_mode=sub_all.agg('mode',axis=1)[0].values
sub_all.head()


# In[ ]:


finalpred=(pred1+pred2+pred3+pred4)//4
sub.Sentiment = finalpred
sub.to_csv('sub_all.csv',index=False)
sub.head()


# In[ ]:


overlapped.head()


# In[ ]:


overlapped.shape


# In[ ]:


lapped_id = overlapped.PhraseId


# In[ ]:


sub_al = sub[~sub['PhraseId'].isin(lapped_id)]


# In[ ]:


sub_al.shape


# In[ ]:


overlap_records = overlapped[['PhraseId','Sentiment']]
overlap_records.head()


# In[ ]:


overlap_records.shape


# In[ ]:


sub_final= pd.concat([sub_al,overlap_records])
sub_final.shape


# In[ ]:


sub_final['PhraseId'].duplicated().sum()


# In[ ]:


sub_final.reset_index(inplace=True)
type(sub_final.PhraseId[0])


# In[ ]:


sub = sub_final.reindex(sub_final['PhraseId'].abs().sort_values(ascending=True).index)
sub.head(20)


# In[ ]:


sub.tail()


# In[ ]:


sub = sub[['PhraseId','Sentiment']]
sub.tail()


# In[ ]:


sub.reset_index(inplace=True)
sub.tail()


# In[ ]:


sub = sub[['PhraseId','Sentiment']]
sub.tail()


# In[ ]:


sub.to_csv('submission1.csv',index=False)
sub.head()


# In[ ]:




