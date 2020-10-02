#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# print date and time for given type of representation
def date_time(x):
    if x==1:
        return 'Timestamp: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now())
    if x==2:    
        return 'Timestamp: {:%Y-%b-%d %H:%M:%S}'.format(datetime.datetime.now())
    if x==3:  
        return 'Date now: %s' % datetime.datetime.now()
    if x==4:  
        return 'Date today: %s' % datetime.date.today() 


# In[ ]:


import pandas as pd
import sklearn
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


# In[ ]:


tweet= pd.read_csv("../input/twitter-airline-sentiment/Tweets.csv")
tweet.head()


# In[ ]:


# discover missing data for each column
(len(tweet)-tweet.count())/len(tweet)


# In[ ]:


# remove columns that have high ratio of missing values

tweet.drop(['airline_sentiment_gold', 'negativereason_gold', 'tweet_coord'], axis=1, inplace=True ) 


# In[ ]:


tweet.airline_sentiment.value_counts()


# In[ ]:


Sentiment_mood = tweet['airline_sentiment'].value_counts()


# In[ ]:


plt.bar([1,2,3],Sentiment_mood)
plt.xticks([1,2,3], ['negative','neutral','positive'],rotation=45)
plt.ylabel('Count')
plt.xlabel('Mood')
plt.title('Sentiment Moodel count')


# AS shown from graph, there is a bad mood from airlines users,
# but we need to see which airlines have highest bad mood

# In[ ]:


tweet['airline'].value_counts()


# In[ ]:


def plot_Airline_sentiment(Airline):
    df=tweet[tweet['airline']==Airline]
    count=df['airline_sentiment'].value_counts()
    Index = [1,2,3]
    plt.bar(Index,count)
    plt.xticks(Index,['negative','neutral','positive'])
    plt.ylabel('Mood Count')
    plt.xlabel('Mood')
    plt.title('Count of Sentiment  Moods of '+Airline)


# In[ ]:


plt.figure(1,figsize=(12, 12))
plt.subplot(231)
plot_Airline_sentiment('US Airways')
plt.subplot(232)
plot_Airline_sentiment('United')
plt.subplot(233)
plot_Airline_sentiment('American')
plt.subplot(234)
plot_Airline_sentiment('Southwest')
plt.subplot(235)
plot_Airline_sentiment('Delta')
plt.subplot(236)
plot_Airline_sentiment('Virgin America')


# as shown in graphs all airlines have bad moods, but in first three data skewed towords negative and 
# in last three data more normally distributed 

# In[ ]:


nr_Count=dict(tweet['negativereason'].value_counts(sort=False))


# In[ ]:


def nr_Count():
    df=tweet
    count=dict(df['negativereason'].value_counts())
    Unique_reason=list(tweet['negativereason'].unique())
    Unique_reason=[x for x in Unique_reason if str(x) != 'nan']
    Reason_frame=pd.DataFrame({'Reasons':Unique_reason})
    Reason_frame['count']=Reason_frame['Reasons'].apply(lambda x: count[x])
    return Reason_frame


# In[ ]:


def plot_reason():
    df=nr_Count()
    count=df['count']
    Index = range(1,(len(df)+1))
    plt.bar(Index,count)
    plt.xticks(Index,df['Reasons'],rotation=90)
    plt.ylabel('Count')
    plt.xlabel('Reason')
    plt.title('Count of Reasons ')


# In[ ]:


plot_reason()


# as shown, custmoer service have a bad service

# In[ ]:


# we can see what people say about flights using Wordcloud
from wordcloud import WordCloud,STOPWORDS


# In[ ]:


df=tweet[tweet['airline_sentiment']=='negative']
words = ' '.join(df['text'])
cleaned_word = " ".join([word for word in words.split()
                            if 'http' not in word
                                and not word.startswith('@')
                                and word != 'RT'
                            ])


# In[ ]:


wordcloud = WordCloud(stopwords=STOPWORDS,
                      background_color='black',
                      width=3000,
                      height=2500
                     ).generate(cleaned_word)


# In[ ]:


plt.figure(1,figsize=(12, 12))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()


# # Preprocessing

# In[ ]:


import re
import nltk
from nltk.corpus import stopwords


# In[ ]:


def Cleaning_tweets(tweets):
    non_letters_removed = re.sub("[^a-zA-Z]", " ",tweets) 
    words = non_letters_removed.lower().split()                             
    stop_words = set(stopwords.words("english"))                  
    stop_words_removed = [w for w in words if not w in stop_words] 
    return ( " ".join( stop_words_removed ))
                                                   


# In[ ]:


tweet['sentiment']=tweet['airline_sentiment'].apply(lambda x: 0 if x=='negative' else 1)


# In[ ]:


tweet['cleaned_tweet']=tweet['text'].apply(lambda x: Cleaning_tweets(x))
tweet['Tweet_length']=tweet['cleaned_tweet'].apply(lambda x: len(x.split(' ')))


# In[ ]:


tweet.head()


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


# split data to training and testing
train,test = train_test_split(tweet,test_size=0.2,random_state=42)


# In[ ]:


train_data = train.cleaned_tweet.values
test_data = test.cleaned_tweet.values


# ### Convert data to vectors

# In[ ]:


vector =CountVectorizer(analyzer = "word")
train_features = vector.fit_transform(train_data)
test_features= vector.transform(test_data)


# In[ ]:


##import machine learning classicla models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


# In[ ]:


Classifiers = [
    LogisticRegression(C=0.000000001,solver='liblinear',max_iter=200),
    SVC(kernel="rbf", C=0.025, probability=True),
    DecisionTreeClassifier(),
    RandomForestClassifier(n_estimators=200),
    AdaBoostClassifier(),
    GaussianNB()]


# In[ ]:


# convert data features to array
train_to_array =train_features.toarray()
test_to_array = test_features.toarray()
Accuracy=[]
Model=[]
for classifier in Classifiers:
    try:
        fit = classifier.fit(train_features,train['sentiment'])
        pred = fit.predict(test_features)
    except Exception:
        fit = classifier.fit(train_to_array,train['sentiment'])
        pred = fit.predict(test_to_array)
    accuracy = accuracy_score(pred,test['sentiment'])
    Accuracy.append(accuracy)
    Model.append(classifier.__class__.__name__)
    print('Accuracy of '+classifier.__class__.__name__+'is '+str(accuracy))    


# In[ ]:


Index = [1,2,3,4,5,6]
plt.bar(Index,Accuracy)
plt.xticks(Index, Model,rotation=45)
plt.ylabel('Accuracy')
plt.xlabel('Model')
plt.title('Accuracies of Models') 


# AS we can see RandomForestClassifieris  and AdaBoostClassifieris  do better than others

# ###  2. Move To deep learning

# In[ ]:


from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing import sequence

import time
import datetime

# Packages for modeling
from keras.layers import Embedding, LSTM, Bidirectional
from keras.layers import Input, Dense, Activation, BatchNormalization, Dropout, Flatten
from keras import models
from keras import layers
from keras import regularizers
import collections

MAX_LEN = 40
NUM_WORDS = 10000  # Parameter indicating the number of words we'll put in the dictionary
VAL_SIZE = 1000  # Size of the validation set
EPOCHS = 20  # Number of epochs we usually start to train with
BATCH_SIZE = 512  # Size of the batches used in the mini-batch gradient descent


# ### Split data to train and validation

# In[ ]:


data_x = train.cleaned_tweet.values
data_y =train['sentiment'].values


# In[ ]:


data_y.shape


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(train['cleaned_tweet'], train['sentiment'], test_size=0.1, random_state=37)
print('# Train data samples:', X_train.shape[0])
print('# Test data samples:', X_test.shape[0])


# In[ ]:


X_train.shape


# In[ ]:


tokenizer = Tokenizer(num_words=NUM_WORDS,
               filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
               lower=True,
               split=" ")
tokenizer.fit_on_texts(X_train)

print('Fitted tokenizer on {} documents'.format(tokenizer.document_count))
print('{} words in dictionary'.format(tokenizer.num_words))
print('Top 5 most common words are:', collections.Counter(tokenizer.word_counts).most_common(5))


# In[ ]:


X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)


# In[ ]:


def one_hot_seq(seqs, nb_features = NUM_WORDS):
    ohs = np.zeros((len(seqs), nb_features))
    for i, s in enumerate(seqs):
        ohs[i, s] = 1.
    return ohs

X_train_oh = one_hot_seq(X_train_seq)
X_test_oh = one_hot_seq(X_test_seq)

print('"{}" is converted into {}'.format(X_train_seq[0], X_train_oh[0]))
print('For this example we have {} features with a value of 1.'.format(X_train_oh[0].sum()))


# In[ ]:


X_train_oh.shape


# In[ ]:


from keras.utils import np_utils

#Convert the labels to One hot encoding vector for softmax for neural network

num_labels = len(np.unique(y_train))
Y_oh_train = np_utils.to_categorical(y_train, num_labels)
Y_oh_test = np_utils.to_categorical(y_test, num_labels)
print(Y_oh_train.shape)


# In[ ]:


X_train_2, X_valid, y_train_2, y_valid = train_test_split(X_train_oh, Y_oh_train, test_size=0.1, random_state=37)


print('Shape of validation set:',X_train_2.shape)


# In[ ]:


len(X_train_2[0])


# In[ ]:


# CALLbacks
# Deep Learning Callbacs - Keras
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau


reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=1,
    verbose=1)


callbacks = [reduce_lr]


# In[ ]:


def deep_model(model):
    model.compile(optimizer='rmsprop',
                  loss = 'categorical_crossentropy',
                  metrics=['accuracy'])
    
    history = model.fit(X_train_2
                       , y_train_2
                       , epochs=EPOCHS
                       , batch_size=BATCH_SIZE
                       , callbacks=callbacks
                       , validation_data=(X_valid, y_valid)
                       , verbose=1)
    
    return history


# In[ ]:


def Dense_model():
    model = models.Sequential()
    model.add(layers.Dense(128, activation='relu', input_shape=(NUM_WORDS,)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(2, activation='softmax'))
    model.summary()
    return model


# In[ ]:


model = Dense_model()
model_history = deep_model(model)


# In[ ]:


def eval_metric(history, metric_name):
    metric = history.history[metric_name]
    val_metric = history.history['val_' + metric_name]

    e = range(1, EPOCHS + 1)

    plt.plot(e, metric, 'bo', label='Train ' + metric_name)
    plt.plot(e, val_metric, 'b', label='Validation ' + metric_name)
    plt.legend()
    plt.show()


# In[ ]:


eval_metric(model_history, 'accuracy')


# # Build LSTM MODEL 

# In[ ]:


def plot_performance(history=None, figure_directory=None, ylim_pad=[0, 0]):
    xlabel = 'Epoch'
    legends = ['Training', 'Validation']

    plt.figure(figsize=(20, 5))

    y1 = history.history['accuracy']
    y2 = history.history['val_accuracy']

    min_y = min(min(y1), min(y2))-ylim_pad[0]
    max_y = max(max(y1), max(y2))+ylim_pad[0]


    plt.subplot(121)

    plt.plot(y1)
    plt.plot(y2)

    plt.title('Model Accuracy\n'+date_time(1), fontsize=17)
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel('Accuracy', fontsize=15)
    plt.ylim(min_y, max_y)
    plt.legend(legends, loc='upper left')
    plt.grid()

    y1 = history.history['loss']
    y2 = history.history['val_loss']

    min_y = min(min(y1), min(y2))-ylim_pad[1]
    max_y = max(max(y1), max(y2))+ylim_pad[1]


    plt.subplot(122)

    plt.plot(y1)
    plt.plot(y2)

    plt.title('Model Loss\n'+date_time(1), fontsize=17)
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel('Loss', fontsize=15)
    plt.ylim(min_y, max_y)
    plt.legend(legends, loc='upper left')
    plt.grid()
    if figure_directory:
        plt.savefig(figure_directory+"/history")

    plt.show()


# In[ ]:


def ltsm_model():
    
    model = models.Sequential()
    
    model.add(Embedding(NUM_WORDS, 100, input_length=MAX_LEN))
    
    model.add(LSTM(64, recurrent_dropout=0.2))    

    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    
    model.add(Dense(256, activation='relu'))
    
    model.add(Dropout(0.3))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    

    model.add(Dense(2, activation='softmax'))
  
    model.summary()
    return model


# In[ ]:


# Modify sequence model to max 25 in length
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)
X_train_deep = sequence.pad_sequences(X_train_seq, maxlen=MAX_LEN)
X_test_deep = sequence.pad_sequences(X_test_seq, maxlen=MAX_LEN)


# In[ ]:


# split data for lstm model
X_train_2, X_valid, y_train_2, y_valid = train_test_split(X_train_deep, Y_oh_train, test_size=0.1, random_state=37)


print('Shape of validation set:',X_train_2.shape)


# In[ ]:


model = ltsm_model()
model_history = deep_model(model)


# In[ ]:


plot_performance(history=model_history)


# In[ ]:


as shown in figure above there is overfitting so we need more work like preprossessing data and change some model parameters


# as shown in figure above there is overfitting so we need more work like preprossessing data and change some model parameters

# In[ ]:




