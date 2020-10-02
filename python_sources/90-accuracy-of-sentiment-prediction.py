#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# install external library for text augmentation
get_ipython().system('pip install nlpaug')
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# plotting libraries
import seaborn as sn 
import matplotlib.pyplot as plt
import plotly.graph_objects as go # Plotly for interactive data visualizatoins

import re # regex for text cleaning
import nltk # natural language processing tool kit
import pickle # for saving/serializing machine learning models
import nlpaug.augmenter.word as naw # text augmentor
# neural networks and embeddings stuff
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Activation
from keras.layers.embeddings import Embedding

from nltk.stem import WordNetLemmatizer # lemmatizing tool
from sklearn.feature_extraction.text import TfidfVectorizer # tfidf for text representation
from sklearn.model_selection import train_test_split # for data splitting
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score # for classifier evaluation
from sklearn.ensemble import RandomForestClassifier # random forest machine learning model
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


# read csv file into a dataframe
raw_tweets=pd.read_csv('/kaggle/input/twitter-airline-sentiment/Tweets.csv')
# show sample of data
raw_tweets.sample(5)


# **Exploratory Analysis**

# When we notice the nature of our dataset , that most of features are categorical features , 
# so i think that par and pie plots should be a very good visualization techinques for our dataset EDA
# 

# **Stacked par chart to show the distribution of reviews per company**

# In[ ]:


crosstab_sentiments=pd.crosstab(raw_tweets.airline, raw_tweets.airline_sentiment)
companies=list(crosstab_sentiments.index)

fig = go.Figure(data=[
    go.Bar(name=col_name, x=companies, y=list(crosstab_sentiments[col_name]))
for col_name in list(crosstab_sentiments.columns)])
# Change the bar mode
fig.update_layout(barmode='stack',
                  title='Sentiment distribution per company',
                  yaxis=dict(title='Sentiment distribution'),
                 xaxis=dict(title='Companies'))
fig.show()


# we see that most of reviews are negative for most of the companies , to help these companies take better decesions we need to focus on negative reviews
# 
# 
# **Stacked par chart to show negative reasons distributions per company**

# In[ ]:


crosstab_neg_reasons=pd.crosstab(raw_tweets.airline,raw_tweets.negativereason)
companies=list(crosstab_neg_reasons.index)

fig = go.Figure(data=[
    go.Bar(name=col_name, x=companies, y=list(crosstab_neg_reasons[col_name]))
for col_name in list(crosstab_neg_reasons.columns)])
# Change the bar mode
fig.update_layout(barmode='stack',
                  title='Negative reasons distribution per company',
                  yaxis=dict(title='Negative reasons distribution'),
                 xaxis=dict(title='Companies'))
fig.show()


# **Pie plot to check the overall distribution for negative reasons**

# In[ ]:


labels = list(crosstab_neg_reasons.columns)
values = [crosstab_neg_reasons[col_name].sum() for col_name in labels]

# Use `hole` to create a donut-like pie chart
fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])
fig.update_layout(title='Overall distribution for negative reasons')
fig.show()


# **Par plot to show sentiment class distributions**

# In[ ]:


fig = go.Figure(
    [go.Bar(x=list(raw_tweets['airline_sentiment'].unique()),
            y=raw_tweets['airline_sentiment'].value_counts())]
)
# Change the bar mode
fig.update_layout(
                  title='Sentiment distribution',
                  yaxis=dict(title='Count'),
                 xaxis=dict(title='Sentiment'))
fig.show()


# from previous plot , we notice that the dataset may suffer from class imbalance problem , we may try to oversample the minority classes using text augmentation technique , nlpaug is very helpful tool , I will use it to oversample neutral and positive classes , note : nlpaug does not blindly copies data , but it apply random word variations with the same meaninig (high cosine similarity)
# check text augmentation section below after data cleaning and stemming

# Now we shall start second part of the problem , we need to build an efficient text / sentiment classifier to predict customer review sentiment

# **Text cleaning and preprocessing**

# In[ ]:


def clean_text(dataframe):
    '''
    function to clean tweets dataframe text , eg : remove white spaces , special chars and website urls
    '''
    # select nedded columns
    dataframe=dataframe[['airline_sentiment','text']]
    # drop nans
    dataframe=dataframe.dropna()
    # convert review into lower case
    dataframe['text']=dataframe['text'].apply(lambda row : row.lower())
    # remove numbers
    dataframe['text']=dataframe['text'].apply(lambda row : re.sub('\d+','',row))
    # remove tweet account name
    dataframe['text']=dataframe['text'].apply(lambda row:re.sub(r'@\w+', '',row))
    # remove website urls
    dataframe['text']=dataframe['text'].apply(lambda row:re.sub(r'http\S+', '',row))
    # rwmove special characters
    dataframe['text']=dataframe['text'].apply(lambda row:re.sub(r"[^A-Za-z0-9']+", ' ',row))
    # remove white spaces
    dataframe['text']=dataframe['text'].apply(lambda row:re.sub(r'\s+', ' ',row))
    
    return dataframe

traindf=clean_text(raw_tweets)


# In[ ]:


lem = WordNetLemmatizer()
def stem_review(review):
    return ' '.join([lem.lemmatize(word) for word in review.split(' ')])
def normalize_text(dataframe):
    '''
    function used for rooting / stemming tokens
    '''
    dataframe['text']=dataframe['text'].apply(lambda x : stem_review(x))

normalize_text(traindf)


# now we will build a classifier using classical machine learning algorithms , then we will use advanced deep learning techniques , from my experience with text classifiers the combination between Tfidfvectorizer with Random forest algorithm has never failed me :)

# **Text augmentation**
# 
# to solve class imbalance problem we shall augment/oversample minor classes

# In[ ]:


def augment_text(dataframe):
    '''
    function used to augment minor classes in tweets dataset
    '''
    augmented_df = pd.DataFrame(columns=['text', 'airline_sentiment'])
    augmentor=naw.WordNetAug(aug_p=0.5)
    aug_sent={'neutral':3,
             'positive':4}
    for i in range(len(dataframe)):
        current_label=dataframe['airline_sentiment'].iloc[i]
        if current_label in list(aug_sent.keys()):
            current_text=dataframe['text'].iloc[i]
            for j in range(aug_sent[current_label]):
                text = augmentor.augment(current_text)
                label = current_label
                tempdf = pd.DataFrame(list(zip([text], [label])), columns=['text', 'airline_sentiment'])
                augmented_df = augmented_df.append(tempdf)
        else :
                text = dataframe['text'].iloc[i]
                label = current_label
                tempdf = pd.DataFrame(list(zip([text], [label])), columns=['text', 'airline_sentiment'])
                augmented_df = augmented_df.append(tempdf)
    return augmented_df
augmented_traindf=augment_text(traindf)


# **Text Representation**
# 
# i will use TfidfVectorizer for sentiment vectorization , from many experiments i have done before with sentiment/reviews datasets , below configuration gives good results 

# In[ ]:


# convert text into numeric values using tf-idf vectorizer
tfidfconverter = TfidfVectorizer(
                                 min_df=5, 
                                 max_df=0.7,
                                 ngram_range=(1,2),
                                 stop_words='english'
                                )  
X = tfidfconverter.fit_transform(augmented_traindf['text']).toarray()
y = augmented_traindf['airline_sentiment'].values


# **Model training**

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
text_classifier = RandomForestClassifier(n_estimators=400,n_jobs=-1,verbose=True)
text_classifier.fit(X_train, y_train)


# **Model evaluation**

# In[ ]:


# model testing
predictions = text_classifier.predict(X_test)
print("Classification Report :\n {} \n Model Acurracy = {}".format(classification_report(y_test,predictions),
                                                                 accuracy_score(y_test, predictions)))
# confusion matrix
print("\n Confusion Matrix")
df_cm = pd.DataFrame(confusion_matrix(y_test, predictions),['negative','neutral','positive'],['negative','neutral','positive'])
sn.set(font_scale=1.2)#for label size
sn.heatmap(df_cm, annot=True,annot_kws={"size": 20})


# the model shows good performance in the evaluation stage , so we may save it to a use as a base model for future enhancements

# In[ ]:


pkl_filename = "classifier_random_forest.pkl"  
with open(pkl_filename, 'wb') as file:  
    pickle.dump(text_classifier, file)
print("Model Saved")


# **Classification using Deep learning techniques**
# 
# 
# we will train an **embedding layer followed by dense layer of 3 neurons with batch size =32 for 4 epochs** , i choosed this arhitecture to keep the network simple as i tried adding more dense layers and the model was training for 10 epochs but the model has symptoms of overfitting also the train and validation error curves were increasing after 4 epochs .

# train an embedding layer with Fully connected neural network

# In[ ]:


ycat=pd.get_dummies(augmented_traindf['airline_sentiment']).values
X=augmented_traindf['text'].values
tk = Tokenizer()
tk.fit_on_texts(X)
X_seq = tk.texts_to_sequences(X)
X_pad = pad_sequences(X_seq, maxlen=100, padding='post')
X_train, X_test, y_train, y_test = train_test_split(X_pad, ycat, test_size = 0.25, random_state = 1)


# In[ ]:


vocabulary_size = len(tk.word_counts.keys())+1
max_words = 100
embedding_size = 32
model = Sequential()
model.add(Embedding(vocabulary_size, embedding_size, input_length=max_words))
model.add(Flatten())
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()


# In[ ]:


history=model.fit(X_train,y_train,validation_data=(X_test,y_test),batch_size=32,epochs=4,verbose=True)


# visualizing model behavior during training

# In[ ]:


# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[ ]:


# model testing
predictions = [np.argmax(i) for i in model.predict(X_test)]
y_test=[np.argmax(i) for i in y_test]
print("Classification Report :\n {} \n Model Acurracy = {}".format(classification_report(y_test,predictions,
                                                                                         target_names=['negative','neutral','positive']),
                                                                 accuracy_score(y_test, predictions)))
# confusion matrix
print("\n Confusion Matrix")
df_cm = pd.DataFrame(confusion_matrix(y_test, predictions),['negative','neutral','positive'],['negative','neutral','positive'])
sn.set(font_scale=1.2)#for label size
sn.heatmap(df_cm, annot=True,annot_kws={"size": 20})


# the model shows good performance in the evaluation stage , so we may save it to a use as a base model for future enhancements

# In[ ]:


model.save('text_classifier_neural_network.h5')
print('model_saved')


# finally as it was predicted , training an embedding layer with fully connected neural network (deep learning methods) shows better performance than conventional text vectorization techniques and machine learning algorithms
