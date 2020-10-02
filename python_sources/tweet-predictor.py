#!/usr/bin/env python
# coding: utf-8

# # **Tweet Predictor**

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # **Data Loading**

# In[ ]:


df_tweet_train = pd.read_csv('../input/nlp-getting-started/train.csv')
df_tweet_test = pd.read_csv('../input/nlp-getting-started/test.csv')


# In[ ]:


df_tweet_train.shape


# The dataset consists of 7613 records with 5 distinct features.

# In[ ]:


df_tweet_test.shape


# The dataset consists of 3263 records with 4 distinct features.

# In[ ]:


print(df_tweet_train.columns)
print(df_tweet_test.columns)


# In[ ]:


df_tweet_train.info()


# The features has 2 numeric values and 3 categoric values.

# In[ ]:


df_tweet_test.info()


# The features has 1 numeric values and 3 categoric values.

# # **Data Cleaning**

# In[ ]:


print(round(((((df_tweet_train.isnull().sum())/(df_tweet_train.shape[0])).sort_values(ascending=False))*100),3))
print('\n')
print(round(((((df_tweet_test.isnull().sum())/(df_tweet_test.shape[0])).sort_values(ascending=False))*100),3))


# Location and keywords have minor missing values in the training as well as the testing datasets, thus we need to handle these values. 

# In[ ]:


df_tweet_train['keyword'].mode()[0]


# In[ ]:


df_tweet_train['location'] = df_tweet_train['location'].fillna(
                             df_tweet_train['location'].mode()[0])

df_tweet_test['location'] = df_tweet_test['location'].fillna(
                            df_tweet_test['location'].mode()[0])


# In[ ]:


df_tweet_train['keyword'] = df_tweet_train['keyword'].fillna(
                            df_tweet_train['keyword'].mode()[0])

df_tweet_test['keyword'] = df_tweet_test['keyword'].fillna(
                           df_tweet_test['keyword'].mode()[0])


# In[ ]:


print(round(((((df_tweet_train.isnull().sum())/(df_tweet_train.shape[0])).sort_values(ascending=False))*100),3))
print()
print(round(((((df_tweet_test.isnull().sum())/(df_tweet_test.shape[0])).sort_values(ascending=False))*100),3))


# # **Data Visualization**

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_style('whitegrid')


# In[ ]:


sns.countplot(df_tweet_train['target'])


# The number of Fake tweets are more than Real tweets which are denoted as 0 and 1 respectively.

# In[ ]:


fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,5))

tlen=df_tweet_train[df_tweet_train['target']==1]['text'].str.len()
ax1.hist(tlen)
ax1.set_title('Real tweets')

tlen=df_tweet_train[df_tweet_train['target']==0]['text'].str.len()
ax2.hist(tlen)
ax2.set_title('Fake tweets')

plt.show()


# Maximum length of tweets of Disastrous tweets and Non-Disastrous tweets lie between 120-150.

# In[ ]:


plt.figure(figsize=(10,8))
df_tweet_train['keyword'].value_counts()[:20].plot.bar()
plt.xticks(rotation=50)
plt.xlabel("Keywords")
plt.ylabel("Number of Tweets")


# Fatalities is the most used keyword to indicate an accident within the tweets.

# In[ ]:


plt.figure(figsize=(10,8))
df_tweet_train['location'].value_counts()[:20].plot.bar()
plt.xticks(rotation=50)
plt.xlabel("Location")
plt.ylabel("Number of Tweets")


# Maximum tweets have been tweeted from USA. 
# We notice there are inconsistencies with respect to the location as some tweets are labelled with only the country, some along with the state and the rest with only the state. We will furthur look into this inconsistency

# In[ ]:


target = df_tweet_train.groupby(['keyword','target'])

plt.figure(figsize=(15,10))
target.count()['id'][:30].plot.bar()
plt.xticks(rotation=50)
plt.xlabel("Keywords (Fake and Real Tweets) ")
plt.ylabel("Number of Tweets")

plt.show()


# Ignoring certain inconsistencies we can identify that accident, apocalypse and armagedon are the three trending words posted in a tweet to indicate a fatality. Whereas aftershock, army, annihilation are less trending words used. 

# In[ ]:


from spacy.lang.en.stop_words import STOP_WORDS
from wordcloud import WordCloud


# In[ ]:


def word(text):
    
    comment_words = ' '
    stopwords = list(STOP_WORDS) 
    
    for val in text: 

         
        val = str(val)   
        tokens = val.split() 

        
        for i in range(len(tokens)): 
            tokens[i] = tokens[i].lower() 

        for words in tokens: 
            comment_words = comment_words + words + ' '


    wordcloud = WordCloud(width = 500, height = 400, 
                    background_color ='black', 
                    stopwords = stopwords, 
                    min_font_size = 10).generate(comment_words) 

                            
    plt.figure(figsize = (12, 12), facecolor = None ) 
    plt.imshow(wordcloud, interpolation='bilinear') 
    plt.axis("off") 
    plt.tight_layout(pad = 0) 

    plt.show() 


# In[ ]:


text = df_tweet_train.text.values
word(text)


# In[ ]:


text = df_tweet_train.location.values
word(text)


# # **Feature Engineering**

# We notice that our "Text" is very dirty. Thus it is necessary that we filter out all the unwanted signs, symbols, hyperlinks, hastags, etc.

# In[ ]:


import nltk, re
data = [df_tweet_train, df_tweet_test]


# In[ ]:


def html_tag(value):
    
    result = re.sub(r"<[^>]+#>", "", value)
    return result

def hyperlink(value):
    
    result = re.sub(r"https?://\S+|www\.\S+", "", value)
    return result

def hashtag(value):
    
    result = re.sub(r"#", "", value)
    return result

def emoticon(value):
    regrex_pattern = re.compile(pattern = "["
        u"\U0001F600-\U0001F64F"  
        u"\U0001F300-\U0001F5FF"  
        u"\U0001F680-\U0001F6FF"  
        u"\U0001F1E0-\U0001F1FF"  
                           "]+", flags = re.UNICODE)
    return regrex_pattern.sub(r'',value)


# In[ ]:


for dataset in data:
    
    dataset['text'] = dataset['text'].apply(html_tag)
    dataset['text'] = dataset['text'].apply(hyperlink)
    dataset['text'] = dataset['text'].apply(hashtag)
    dataset['text'] = dataset['text'].apply(emoticon)


# During the analysis we notice many inconsitencies in the loction feature. We now try to extract the correct loctation of each tweet with the help of pycountry library.

# In[ ]:


import pycountry

def findcon1(text):
        
    for country in pycountry.countries:
        if country.name in text:   
            a = country.name
            return a
        else:
            try:
                a = pycountry.countries.search_fuzzy(text.split()[-1])[0].name
                return a
            
            except:
                return text


# In[ ]:


for dataset in data:
    
    dataset['nlocation'] = dataset['location'].apply(lambda x : findcon1(x))
    


# In[ ]:


plt.figure(figsize=(10,8))
df_tweet_train['nlocation'].value_counts()[:20].plot.bar()
plt.xticks(rotation=50)
plt.xlabel("Location")
plt.ylabel("Number of Tweets")


# As compared to the previous inconistent locaton output, we get a better view so as to which countries have the highest number of tweets. The highest number of tweets belong to The United States.

# # **Data Modeling**

# In[ ]:


df = df_tweet_train[['target','text']]


# In[ ]:


from sklearn.model_selection import train_test_split
X = df['text']
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# # **Machine Learning**

# * Naive Bayes Classifier

# In[ ]:


from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.metrics import confusion_matrix

pipe1 = Pipeline([('vectorize', CountVectorizer()),('tfidf', TfidfTransformer()),('classifier', MultinomialNB())])

pipe1.fit(X_train, y_train)

prediction1 = pipe1.predict(X_test)

accuracy1 = round((pipe1.score(X_test, y_test)*100),0)
print('Accuracy: ',accuracy1,'%')
print()
print('Confusion Matrix: \n',confusion_matrix(y_test,prediction1))


# * Linear Support Vector Machine

# In[ ]:


from sklearn.linear_model import SGDClassifier

pipe2 = Pipeline([('vectorize', CountVectorizer()),('tfidf', TfidfTransformer()),('classifier', SGDClassifier())])

pipe2.fit(X_train, y_train)

prediction2 = pipe2.predict(X_test)

accuracy2 = round((pipe2.score(X_test, y_test)*100),0)
print('Accuracy: ',accuracy2,'%')
print()
print('Confusion Matrix: \n',confusion_matrix(y_test, prediction2))


# * Support Vector Machine

# In[ ]:


from sklearn.svm import SVC

pipe3 = Pipeline([('vectorize', CountVectorizer()),('tfidf', TfidfTransformer()),('classifier', SVC())])

pipe3.fit(X_train, y_train)

prediction3 = pipe3.predict(X_test)

accuracy3 = round((pipe3.score(X_test, y_test)*100),0)
print('Accuracy: ',accuracy3,'%')
print()
print('Confusion Matrix: \n',confusion_matrix(y_test, prediction3))


# # **SpaCy**

# In[ ]:


import spacy
import string
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English


# In[ ]:


stopwords = list(STOP_WORDS)
punctuations = string.punctuation
parser = English()


# In[ ]:


def tokenizer(sentence):
    tokens = parser(sentence)
    tokens = [ word.lemma_.lower() if word.lemma_ != "-PRON-" else word.lower_ for word in tokens ]
    tokens = [ word for word in tokens if word not in stopwords and word not in punctuations ]
    return tokens


# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import TransformerMixin 
from sklearn.svm import LinearSVC


# In[ ]:


class predictors(TransformerMixin):
    def transform(self, X, **transform_params):
        return [clean_text(text) for text in X]
    def fit(self, X, y=None, **fit_params):
        return self
    def get_params(self, deep=True):
        return {}

 
    def clean_text(text):     
        return text.strip().lower()


# * Count Vectorizer along with Support Vector Classifier

# In[ ]:


vectorizer = CountVectorizer(tokenizer = tokenizer, ngram_range=(1,1)) 
classifier = SVC()


# In[ ]:


pipe4 = Pipeline([('cleaner', predictors()),('vectorizer', vectorizer),('classifier', classifier)])

pipe4.fit(X_train,y_train)


# In[ ]:


prediction4 = pipe4.predict(X_test)

accuracy4 = round((pipe4.score(X_test, y_test)*100),0)
print('Accuracy: ',accuracy4,'%')
print()
print('Confusion Matrix: \n',confusion_matrix(y_test,prediction4))


# * TF-IDF Vectorizer along with Support Vector Classifier

# In[ ]:


tfvectorizer = TfidfVectorizer(tokenizer = tokenizer)


# In[ ]:


pipe5 = Pipeline([('cleaner', predictors()),('vectorizer', tfvectorizer),('classifier', classifier)])

pipe5.fit(X_train,y_train)


# In[ ]:


prediction5 = pipe4.predict(X_test)

accuracy5 = round((pipe5.score(X_test, y_test)*100),0)
print('Accuracy: ',accuracy5,'%')
print()
print('Confusion Matrix: \n',confusion_matrix(y_test,prediction5))


# # **Neural Network**

# In[ ]:


import tensorflow as tf

from keras.models import Sequential
from keras import layers
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


# In[ ]:


sentences = df['text'].values
y = df['target'].values

sentences_train, sentences_test, y_train, y_test = train_test_split(sentences, y, test_size=0.20, random_state=42)


# * Basic Neural Model

# In[ ]:


vectorizer = CountVectorizer()
vectorizer.fit(sentences_train)

X_train = vectorizer.transform(sentences_train)
X_test  = vectorizer.transform(sentences_test)


# In[ ]:


model1 = Sequential()
model1.add(layers.Dense(32, input_dim=X_train.shape[1], activation='relu'))
model1.add(layers.Dense(64,activation='relu'))
model1.add(layers.Dense(1, activation='sigmoid'))

model1.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

model1.summary()


# In[ ]:


history = model1.fit(X_train, y_train,
                    epochs=10,
                    verbose=False,
                    validation_data=(X_test, y_test),
                    batch_size=10)


# In[ ]:


loss, accuracy = model1.evaluate(X_test, y_test, verbose=False)

accuracy6 = round((accuracy*100),0)
print('Accuracy: ',accuracy6,'%')


# * Bag Of Words (BOW)

# In[ ]:


vocab = {}  
word_encoding = 1
def bow(sentence):
    
    text = parser(sentence)
    text = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in text ]
    text = [ word for word in text if word not in stopwords and word not in punctuations ]
         
    global word_encoding
    words = text 
    bag = {}  

    for word in words:
        
        if word in vocab:
            encoding = vocab[word]  
        else:
            vocab[word] = word_encoding
            encoding = word_encoding
            word_encoding += 1
    
        if encoding in bag:
            bag[encoding] += 1
        else:
            bag[encoding] = 1
  
    return bag


# In[ ]:


vectorizer = TfidfVectorizer(tokenizer = bow)
vectorizer.fit(sentences_train)

X_train = vectorizer.transform(sentences_train)
X_test  = vectorizer.transform(sentences_test)


# In[ ]:


model2 = Sequential()
model2.add(layers.Dense(32, input_dim=X_train.shape[1], activation='relu'))
model2.add(layers.Dense(64,activation='relu'))
model2.add(layers.Dense(128,activation='relu'))
model2.add(layers.Dense(256,activation='relu'))
model2.add(layers.Dense(1, activation='sigmoid'))

model2.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

model2.summary()


# In[ ]:


history = model2.fit(X_train, y_train,
                    epochs=10,
                    verbose=False,
                    validation_data=(X_test, y_test),
                    batch_size=10)


# In[ ]:


loss, accuracy = model2.evaluate(X_test, y_test, verbose=False)

accuracy7 = round((accuracy*100),0)
print('Accuracy: ',accuracy7,'%')


# * Word Embedding  

# In[ ]:


tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(sentences_train)

X_train = tokenizer.texts_to_sequences(sentences_train)
X_test = tokenizer.texts_to_sequences(sentences_test)


# In[ ]:


vsize = len(tokenizer.word_index) + 1
edim = 50
maxlen = 100


# In[ ]:


X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)


# In[ ]:


model3 = Sequential()
model3.add(layers.Embedding(input_dim=vsize,output_dim=edim,input_length=maxlen))
model3.add(layers.Flatten())
model3.add(layers.Dense(256, activation='relu'))
model3.add(layers.Dense(128, activation='relu'))
model3.add(layers.Dense(64, activation='relu'))
model3.add(layers.Dense(32, activation='relu'))
model3.add(layers.Dense(1, activation='sigmoid'))

model3.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])

model3.summary()


# In[ ]:


history = model3.fit(X_train, y_train,
                    epochs=10,
                    verbose=False,
                    validation_data=(X_test, y_test),
                    batch_size=32)


# In[ ]:


loss, accuracy = model3.evaluate(X_test, y_test, verbose=False)

accuracy8 = round((accuracy*100),0)
print('Accuracy: ',accuracy8,'%')


# * Long-Short Term Memory (LSTM)

# In[ ]:


model4 = Sequential()
model4.add(layers.Embedding(input_dim=vsize,output_dim=edim,input_length=maxlen))
model4.add(layers.LSTM(50))
model4.add(layers.Dense(10, activation='relu'))
model4.add(layers.Dense(1, activation='sigmoid'))

model4.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model4.summary()


# In[ ]:


history = model4.fit(X_train, y_train,
                    epochs=10,
                    verbose=False,
                    validation_data=(X_test, y_test),
                    batch_size=10)


# In[ ]:


loss, accuracy = model4.evaluate(X_test, y_test, verbose=False)

accuracy9 = round((accuracy*100),0)
print('Accuracy: ',accuracy9,'%')


# * Neural Network with Pooling layers

# In[ ]:


model5 = Sequential()
model5.add(layers.Embedding(input_dim=vsize, output_dim=edim,input_length=maxlen))
model5.add(layers.GlobalMaxPooling1D())
model5.add(layers.Dense(10, activation='relu'))
model5.add(layers.Dense(1, activation='sigmoid'))

model5.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])
model5.summary()


# In[ ]:


history = model5.fit(X_train, y_train,
                    epochs=10,
                    verbose=False,
                    validation_data=(X_test, y_test),
                    batch_size=10)


# In[ ]:


loss, accuracy = model5.evaluate(X_test, y_test, verbose=False)

accuracy10 = round((accuracy*100),0)
print('Accuracy: ',accuracy10,'%')


# # **Conclusion**

# In[ ]:


data = [['Naive Bayes Classifier',accuracy1],
       ['Linear Support Vector Machine',accuracy2],
       ['Support Vector Machine',accuracy3],
       ['Count Vectorizer along with Support Vector Classifier',accuracy4],
       ['TF-IDF Vectorizer along with Support Vector Classifier',accuracy5],
       ['Basic Neural Model',accuracy6],
       ['Bag Of Words (BOW)',accuracy7],
       ['Word Embedding',accuracy8],
       ['Long-Short Term Memory (LSTM)',accuracy9],
       ['Neural Network with Pooling layers',accuracy10]]

final = pd.DataFrame(data,columns=['Algorithm','Precision'],index=[1,2,3,4,5,6,7,8,9,10])


# In[ ]:


print("The results of Data Modeling are as follows:\n ")
print(final)


# # **References**
# 1. SapCy  
# https://github.com/Jcharis/Natural-Language-Processing-Tutorials/blob/master/Text%20Classification%20With%20Machine%20Learning,SpaCy,Sklearn(Sentiment%20Analysis)/Text%20Classification%20&%20Sentiment%20Analysis%20with%20SpaCy,Sklearn.ipynb
# 2. Neural Network  
# https://www.kaggle.com/sanikamal/text-classification-with-python-and-keras
# 3. Neural Network (BOW) https://colab.research.google.com/drive/1ysEKrw_LE2jMndo1snrZUh5w87LQsCxk#forceEdit=true&sandboxMode=true&scrollTo=Fo3WY-e86zX2
