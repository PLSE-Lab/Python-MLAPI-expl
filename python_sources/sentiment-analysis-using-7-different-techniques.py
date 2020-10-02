#!/usr/bin/env python
# coding: utf-8

# # <center> Movie Review Sentiment Analysis </center>
# ## <center> Classify the sentiment of sentences from the Rotten Tomatoes dataset </center>

# ##  Contents
# * [Introduction](#introduction)
# * [EDA](#eda)
# * [Different Machine Learning Models ](#ml)
#   *  [  N-Grams Method ](#N-Grams)
#   *  [GRU model ](#gru)
#   *  [ LSTM model](#lstm)
#   *  [Bidirectional-GRU model](#bgru)
#   *  [ CNN model ](#cnn)
#   *  [ CNN-GRU ](#cgru)
#   *  [GRU-CNN ](#gruc)
# * [Final Ensemble](#en) 
#     

# <a id='introduction'></a>
# ## <center> Introduction </center>
# The Rotten Tomatoes movie review dataset is a corpus of movie reviews used for sentiment analysis, originally collected by Pang and Lee [1]. In their work on sentiment treebanks, Socher et al. [2] used Amazon's Mechanical Turk to create fine-grained labels for all parsed phrases in the corpus. This competition presents a chance to benchmark your sentiment-analysis ideas on the Rotten Tomatoes dataset. You are asked to label phrases on a scale of five values: negative, somewhat negative, neutral, somewhat positive, positive. Obstacles like sentence negation, sarcasm, terseness, language ambiguity, and many others make this task very challenging.

# In[ ]:


# imports
import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# <a id='eda'></a>
# ## <center>EDA</center>
# Basic exploration of data to check labels , the number of phrases for each label and average phrase length in the each sentiment.

# In[ ]:


PATH = '../input/'
os.listdir(PATH)


# In[ ]:


train = pd.read_csv('../input/train.tsv',sep = '\t')
test = pd.read_csv('../input/test.tsv',sep = '\t')
sub = pd.read_csv('../input/sampleSubmission.csv' , sep = ',')


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


class_count = train['Sentiment'].value_counts()
class_count


# ### The sentiment labels are:
# 
# 0 - negative
# 1 - somewhat negative
# 2 - neutral
# 3 - somewhat positive
# 4 - positive

# In[ ]:


x = np.array(class_count.index)
y = np.array(class_count.values)
plt.figure(figsize=(8,5))
sns.barplot(x,y)
plt.xlabel('Sentiment ')
plt.ylabel('Number of reviews ')


# In[ ]:





# In[ ]:


print('Number of sentences in training set:',len(train['SentenceId'].unique()))
print('Number of sentences in test set:',len(test['SentenceId'].unique()))
print('Average words per sentence in train:',train.groupby('SentenceId')['Phrase'].count().mean())
print('Average words per sentence in test:',test.groupby('SentenceId')['Phrase'].count().mean())


# ### Using Word Clouds to see the higher fequency words from each sentiment

# In[ ]:


from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
stopwords = set(STOPWORDS)

def show_wordcloud(data, title = None):
    wordcloud = WordCloud(
        background_color='black',
        stopwords=stopwords,
        max_words=200,
        max_font_size=40, 
        scale=3,
        random_state=1 # chosen at random by flipping a coin; it was heads
).generate(str(data))

    fig = plt.figure(1, figsize=(15, 15))
    plt.axis('off')
    if title: 
        fig.suptitle(title, fontsize=20)
        fig.subplots_adjust(top=2.3)

    plt.imshow(wordcloud)
    plt.show()


# In[ ]:


show_wordcloud(train['Phrase'],'Most Common Words from the whole corpus')


# In[ ]:



show_wordcloud(train[train['Sentiment'] == 0]['Phrase'],'Negative Reviews')


# In[ ]:



show_wordcloud(train[train['Sentiment'] == 1]['Phrase'],'Somewhat Negative Reviews')


# In[ ]:


show_wordcloud(train[train['Sentiment'] == 2]['Phrase'],'Neutral Reviews')


# In[ ]:



show_wordcloud(train[train['Sentiment'] == 3]['Phrase'],'Somewhat Positive Reviews')


# In[ ]:



show_wordcloud(train[train['Sentiment'] == 4]['Phrase'],'Postive Reviews')


# <a id='ml'></a>
# ## <center>  Different Machine Learning Models </center>
# 

# In[ ]:


from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
tokenizer = TweetTokenizer()


# <a id='N-Grams'></a>
# ##  <center>1.N-Grams</center>

# In[ ]:


vectorizer = TfidfVectorizer(ngram_range=(1, 3), tokenizer=tokenizer.tokenize)
full_text = list(train['Phrase'].values) + list(test['Phrase'].values)
vectorizer.fit(full_text)
train_vectorized = vectorizer.transform(train['Phrase'])
test_vectorized = vectorizer.transform(test['Phrase'])


# In[ ]:


y = train['Sentiment']


# In[ ]:


from sklearn.model_selection import train_test_split
x_train , x_val, y_train , y_val = train_test_split(train_vectorized,y,test_size = 0.2)


# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.multiclass import OneVsRestClassifier


# In[ ]:





# In[ ]:


from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score


# ### Training Logistic Reagression model and an SVM.

# In[ ]:


lr = LogisticRegression()
ovr = OneVsRestClassifier(lr)
ovr.fit(x_train,y_train)
print(classification_report( ovr.predict(x_val) , y_val))
print(accuracy_score( ovr.predict(x_val) , y_val ))


# In[ ]:


svm = LinearSVC()
svm.fit(x_train,y_train)
print(classification_report( svm.predict(x_val) , y_val))
print(accuracy_score( svm.predict(x_val) , y_val ))


# In[ ]:


estimators = [ ('svm',svm) , ('ovr' , ovr) ]
clf = VotingClassifier(estimators , voting='hard')
clf.fit(x_train,y_train)
print(classification_report( clf.predict(x_val) , y_val))
print(accuracy_score( clf.predict(x_val) , y_val ))


# In[ ]:


from keras.utils import to_categorical
target=train.Sentiment.values
y=to_categorical(target)
y


# In[ ]:


max_features = 13000
max_words = 50
batch_size = 128
epochs = 3
num_classes=5


# In[ ]:


from sklearn.model_selection import train_test_split
X_train , X_val , Y_train , Y_val = train_test_split(train['Phrase'],y,test_size = 0.20)


# In[ ]:


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense,GRU,LSTM,Embedding
from keras.optimizers import Adam
from keras.layers import SpatialDropout1D,Dropout,Bidirectional,Conv1D,GlobalMaxPooling1D,MaxPooling1D,Flatten
from keras.callbacks import ModelCheckpoint, TensorBoard, Callback, EarlyStopping


# In[ ]:


tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(X_train))
X_train = tokenizer.texts_to_sequences(X_train)
X_val = tokenizer.texts_to_sequences(X_val)


# In[ ]:


X_test = tokenizer.texts_to_sequences(test['Phrase'])
X_test =pad_sequences(X_test, maxlen=max_words)


# In[ ]:


len(X_test)


# In[ ]:


X_train =pad_sequences(X_train, maxlen=max_words)
X_val = pad_sequences(X_val, maxlen=max_words)
X_test =pad_sequences(X_test, maxlen=max_words)


# <a id='gru'></a>
# ## <center>GRU</center>

# In[ ]:


model_GRU=Sequential()
model_GRU.add(Embedding(max_features,100,mask_zero=True))
model_GRU.add(GRU(64,dropout=0.4,return_sequences=True))
model_GRU.add(GRU(32,dropout=0.5,return_sequences=False))
model_GRU.add(Dense(num_classes,activation='softmax'))
model_GRU.compile(loss='categorical_crossentropy',optimizer=Adam(lr = 0.001),metrics=['accuracy'])
model_GRU.summary()


# In[ ]:


get_ipython().run_cell_magic('time', '', 'history1=model_GRU.fit(X_train, Y_train, validation_data=(X_val, Y_val),epochs=epochs, batch_size=batch_size, verbose=1)')


# In[ ]:


y_pred1=model_GRU.predict_classes(X_test, verbose=1)
sub.Sentiment=y_pred1
sub.to_csv('sub1_GRU.csv',index=False)
sub.head()


# In[ ]:





# In[ ]:


model2_GRU=Sequential()
model2_GRU.add(Embedding(max_features,100,mask_zero=True))
model2_GRU.add(GRU(64,dropout=0.4,return_sequences=True))
model2_GRU.add(GRU(32,dropout=0.5,return_sequences=False))
model2_GRU.add(Dense(num_classes,activation='sigmoid'))
model2_GRU.compile(loss='binary_crossentropy',optimizer=Adam(lr = 0.001),metrics=['accuracy'])
model2_GRU.summary()


# In[ ]:


get_ipython().run_cell_magic('time', '', 'history2=model2_GRU.fit(X_train, Y_train, validation_data=(X_val, Y_val),epochs=epochs, batch_size=batch_size, verbose=1)')


# In[ ]:


y_pred2=model2_GRU.predict_classes(X_test, verbose=1)
sub.Sentiment=y_pred2
sub.to_csv('sub2_GRU.csv',index=False)
sub.head()


# <a id='lstm'></a>
# ## <center>LSTM</center>

# In[ ]:


model3_LSTM=Sequential()
model3_LSTM.add(Embedding(max_features,100,mask_zero=True))
model3_LSTM.add(LSTM(64,dropout=0.4,return_sequences=True))
model3_LSTM.add(LSTM(32,dropout=0.5,return_sequences=False))
model3_LSTM.add(Dense(num_classes,activation='sigmoid'))
model3_LSTM.compile(loss='binary_crossentropy',optimizer=Adam(lr = 0.001),metrics=['accuracy'])
model3_LSTM.summary()


# In[ ]:


get_ipython().run_cell_magic('time', '', 'history3=model3_LSTM.fit(X_train, Y_train, validation_data=(X_val, Y_val),epochs=epochs, batch_size=batch_size, verbose=1)')


# In[ ]:


y_pred3=model3_LSTM.predict_classes(X_test, verbose=1)
sub.Sentiment=y_pred3
sub.to_csv('sub3_LSTM.csv',index=False)
sub.head()


# <a id='bgru'></a>
# ## <center>Bidirectional-GRU</center>

# In[ ]:


model4_BGRU = Sequential()
model4_BGRU.add(Embedding(max_features, 100, input_length=max_words))
model4_BGRU.add(SpatialDropout1D(0.25))
model4_BGRU.add(Bidirectional(GRU(64,dropout=0.4,return_sequences = True)))
model4_BGRU.add(Bidirectional(GRU(32,dropout=0.5,return_sequences = False)))
model4_BGRU.add(Dense(5, activation='sigmoid'))
model4_BGRU.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model4_BGRU.summary()


# In[ ]:


get_ipython().run_cell_magic('time', '', 'history4=model4_BGRU.fit(X_train, Y_train, validation_data=(X_val, Y_val),epochs=epochs, batch_size=batch_size, verbose=1)')


# In[ ]:


y_pred4=model4_BGRU.predict_classes(X_test, verbose=1)
sub.Sentiment=y_pred4
sub.to_csv('sub4_BGRU.csv',index=False)
sub.head()


# <a id='cnn'></a>
# ## <center>CNN</center>

# In[ ]:


model5_CNN= Sequential()
model5_CNN.add(Embedding(max_features,100,input_length=max_words))
model5_CNN.add(Dropout(0.2))
model5_CNN.add(Conv1D(64,kernel_size=3,padding='same',activation='relu',strides=1))
model5_CNN.add(GlobalMaxPooling1D())
model5_CNN.add(Dense(128,activation='relu'))
model5_CNN.add(Dropout(0.2))
model5_CNN.add(Dense(num_classes,activation='sigmoid'))
model5_CNN.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model5_CNN.summary()


# In[ ]:


get_ipython().run_cell_magic('time', '', 'early_stop = EarlyStopping(monitor = "val_loss", mode = "min", patience = 3)\n\nhistory5=model5_CNN.fit(X_train, Y_train, validation_data=(X_val, Y_val),epochs=3, batch_size=batch_size, verbose=1,callbacks = [early_stop])')


# In[ ]:


y_pred5=model5_CNN.predict_classes(X_test, verbose=1)
sub.Sentiment=y_pred5
sub.to_csv('sub5_CNN.csv',index=False)
sub.head()


# <a id='cgru'></a>
# ## <center>CNN-GRU</center>

# In[ ]:


model6_CnnGRU= Sequential()
model6_CnnGRU.add(Embedding(max_features,100,input_length=max_words))
model6_CnnGRU.add(Conv1D(64,kernel_size=3,padding='same',activation='relu'))
model6_CnnGRU.add(MaxPooling1D(pool_size=2))
model6_CnnGRU.add(Dropout(0.25))
model6_CnnGRU.add(GRU(128,return_sequences=True))
model6_CnnGRU.add(Dropout(0.3))
model6_CnnGRU.add(Flatten())
model6_CnnGRU.add(Dense(128,activation='relu'))
model6_CnnGRU.add(Dropout(0.5))
model6_CnnGRU.add(Dense(5,activation='sigmoid'))
model6_CnnGRU.compile(loss='binary_crossentropy',optimizer=Adam(lr=0.001),metrics=['accuracy'])
model6_CnnGRU.summary()


# In[ ]:


get_ipython().run_cell_magic('time', '', 'history6=model6_CnnGRU.fit(X_train, Y_train, validation_data=(X_val, Y_val),epochs=3, batch_size=batch_size, verbose=1,callbacks=[early_stop])')


# In[ ]:


y_pred6=model6_CnnGRU.predict_classes(X_test, verbose=1)
sub.Sentiment=y_pred6
sub.to_csv('sub6_CnnGRU.csv',index=False)
sub.head()


# <a id='gruc'></a>
# ## <center>GRU-CNN</center>

# In[ ]:


model7_GruCNN = Sequential()
model7_GruCNN.add(Embedding(max_features,100,input_length=max_words))
model7_GruCNN.add(Dropout(0.2))
model7_GruCNN.add(Bidirectional(GRU(units=128 , return_sequences=True)))
model7_GruCNN.add(Conv1D(32 , kernel_size=3 , padding='same' , activation='relu'))
model7_GruCNN.add(GlobalMaxPooling1D())
model7_GruCNN.add(Dense(units = 64 , activation='relu'))
model7_GruCNN.add(Dropout(0.5))
model7_GruCNN.add(Dense(units=5,activation='sigmoid'))
model7_GruCNN.compile(loss='binary_crossentropy' , optimizer = 'adam' , metrics=['accuracy'])
model7_GruCNN.summary()


# In[ ]:


get_ipython().run_cell_magic('time', '', 'history7 = model7_GruCNN.fit(X_train, Y_train, validation_data=(X_val, Y_val),epochs=4, batch_size=batch_size, verbose=1,callbacks=[early_stop])')


# In[ ]:


y_pred7=model7_GruCNN.predict_classes(X_test, verbose=1)
sub.Sentiment=y_pred7
sub.to_csv('sub7_GruCNN.csv',index=False)
sub.head()


# <a id='en'></a>
# ## <center>Ensembling all the predictions</center>

# In[ ]:


sub_all=pd.DataFrame({'model1':y_pred1,'model2':y_pred2,'model3':y_pred3,'model4':y_pred4,'model5':y_pred5,'model6':y_pred6,'model7':y_pred7})
pred_mode=sub_all.agg('mode',axis=1)[0].values
sub_all.head()


# In[ ]:


pred_mode=[int(i) for i in pred_mode]
sub.Sentiment=pred_mode
sub.to_csv('ensemble_mode.csv',index=False)
sub.head()


# In[ ]:




