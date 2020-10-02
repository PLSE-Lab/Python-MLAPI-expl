#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
import pandas as pd
import numpy as np


# ## Read Data

# In[ ]:


df = pd.read_csv('../input/Tweets.csv')
df.head()


# # Data Analysis
# ## Some Insights

# In[ ]:


sentiment_counts = df.airline_sentiment.value_counts()
number_of_tweets = df.tweet_id.count()
print(sentiment_counts)


# In[ ]:


dff = df.groupby(["airline", "airline_sentiment" ]).count()["name"]
dff['American']


# ## Converting sentiments of indivual airline into percentages

# In[ ]:


airlines=df.airline.unique()
positive_percentage = []
negative_percentage = []
neutral_percentage = []
for i in airlines:
    positive_percentage.append((dff[i].positive/dff[i].sum())*100)
    negative_percentage.append((dff[i].negative/dff[i].sum())*100)
    neutral_percentage.append((dff[i].neutral/dff[i].sum())*100)
percentage_data = [positive_percentage,negative_percentage,neutral_percentage]
percentage_data = np.array(percentage_data)
percentage_data=percentage_data.reshape(6,3)


# In[ ]:


my_series = pd.DataFrame(data=percentage_data, index =airlines)
my_series[0] = positive_percentage
my_series[1] = negative_percentage
my_series[2] = neutral_percentage
my_series


# ## Chart displays the sentiment of each airline in percentage

# In[ ]:


import matplotlib.pyplot as plt
import matplotlib.style
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.style
from matplotlib.pyplot import subplots

fig, ax = subplots()
my_colors =['blue','red','green']
my_series.plot(kind='bar', stacked=False, ax=ax, color=my_colors, figsize=(14, 7), width=0.8)
ax.legend(["Postive Percentage","Negative Percentage","Neutral Percentage"])
plt.title("Percentages of Sentiments, Tweets Sentiments Analysis Airlines, 2017")
plt.show()


# # Data Preprocessing

# ## As we are interested in only 2 columns for our purpose of classfication, we are taking subset of whole data frame

# In[ ]:


data = df[['text','airline_sentiment']]
data.head()


# ## Converting labels into integers 
# ### neutral = 0
# ### positive = 1
# ### negative = 2

# In[ ]:


data.loc[:,('airline_sentiment')] = data.airline_sentiment.map({'neutral':0, 'positive':1,'negative':2})
data.head()


# ## Seperating rows based on their labels

# In[ ]:


positive_sentiment_words = ''
negative_sentiment_words = ''
neutral_sentiment_words = ''
neutral = data[data.airline_sentiment == 0]
positive = data[data.airline_sentiment ==1]
negative = data[data.airline_sentiment ==2]


# ## Tokenizing, Lematizing and removing stop words from data

# In[ ]:


import nltk,re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words('english'))
wordnet_lemmatizer = WordNetLemmatizer()
for val in neutral.text:
    text = val.lower()
    only_letters = re.sub("[^a-zA-Z]", " ",text) 
    tokens = nltk.word_tokenize(only_letters )[2:]
    for word in tokens:
        if word not in stop_words:
            word = wordnet_lemmatizer.lemmatize(word)
            neutral_sentiment_words =  neutral_sentiment_words + word + ' '
            
for val in positive.text:
    text = val.lower()
    only_letters = re.sub("[^a-zA-Z]", " ",text) 
    tokens = nltk.word_tokenize(only_letters )[2:]
    for word in tokens:
        if word not in stop_words:
            word = wordnet_lemmatizer.lemmatize(word)
            positive_sentiment_words =  positive_sentiment_words + word + ' '
            
for val in negative.text:
    text = val.lower()
    only_letters = re.sub("[^a-zA-Z]", " ",text) 
    tokens = nltk.word_tokenize(only_letters )[2:]
    for word in tokens:
        if word not in stop_words:
            word = wordnet_lemmatizer.lemmatize(word)
            negative_sentiment_words =  negative_sentiment_words + word + ' '
            
            


# # WordCloud

# In[ ]:


from wordcloud import WordCloud
neutral_wordcloud = WordCloud(width=600, height=400).generate(neutral_sentiment_words)
positive_wordcloud = WordCloud(width=600, height=400).generate(positive_sentiment_words)
negative_wordcloud = WordCloud(width=600, height=400).generate(negative_sentiment_words)


# ## Neutral Sentiments Wordcloud

# In[ ]:


plt.figure( figsize=(10,8), facecolor='k')
plt.imshow(neutral_wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()


# ## Positive Sentiments WordCloud

# In[ ]:


plt.figure( figsize=(10,8), facecolor='k')
plt.imshow(positive_wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()


# ## Negative Sentiments WordCloud

# In[ ]:


plt.figure( figsize=(10,8), facecolor='k')
plt.imshow(negative_wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()


# # Classification

# ## Converting sentences into vectors in order to feed it to Naive Bayes and SVM
# ## Splitting data into training and test data ( test_size = 0.2 )

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
data['text'] = data['text'].apply(lambda x: x.lower())
data['text'] = data['text'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))
X_train,X_test,y_train,y_test = train_test_split(data["text"],data["airline_sentiment"], test_size = 0.2, random_state = 10)
print("train tuples",X_train.shape)
print("test tuples",X_test.shape)
print("train labels",y_train.shape)
print("test labels",y_test.shape)
vect = CountVectorizer()
vect.fit(X_train)
X_train_df = vect.transform(X_train)
X_test_df = vect.transform(X_test)


# ## Model 1 - Naive Bayes

# In[ ]:


from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(X_train_df,y_train)
result=model.predict(X_test_df)


# ## Accuracy = 76.09%

# In[ ]:


print("Accuracy Score:",accuracy_score(y_test,result))


# ## Model 2 - SVM

# In[ ]:


from sklearn import svm
clf = svm.SVC(kernel='rbf')
clf.fit(X_train_df,y_train)
result=clf.predict(X_test_df)


# ## Accuracy = 60.82%

# In[ ]:


print("Accuracy Score:",accuracy_score(y_test,result))


# ## Model 3 - LSTM

# In[ ]:


import keras
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, SpatialDropout1D
from keras.callbacks import ModelCheckpoint
import os
from sklearn.metrics import roc_auc_score
from keras.preprocessing.text import Tokenizer
max_fatures = 2000
tokenizer = Tokenizer(num_words=max_fatures, split=' ')
tokenizer.fit_on_texts(data['text'].values)
X = tokenizer.texts_to_sequences(data['text'].values)
X = pad_sequences(X)

embed_dim = 128
lstm_out = 196

model = Sequential()
model.add(Embedding(max_fatures, embed_dim,input_length = X.shape[1]))
model.add(Dropout(0.5))
model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(3,activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
print(model.summary())

Y = pd.get_dummies(data['airline_sentiment']).values
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state = 42)


# ## Fitting Model
# ### Accuracy Achieved = 81.11%

# In[ ]:


batch_size = 256
history = model.fit(X_train, 
                    Y_train, 
                    epochs = 10, 
                    batch_size=batch_size, 
                    validation_data=(X_test, Y_test))


# ### Model 3 - Accuracy Graph

# In[ ]:


plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')
plt.show()


# # Results

# In[ ]:


models = ['Naive Bayes','SVM','LSTM']
accuracy = [76.09,60.82,81.11]
result_frame = pd.DataFrame(data = accuracy,index = models)

fig, ax = subplots()
my_colors =['blue','red','green']
result_frame.plot(kind='bar', stacked=False, ax=ax, color=my_colors, figsize=(12, 4), width=0.4)
ax.legend(["Percentage"])
plt.title("Comparison of different models on Twitter Sentiments")
plt.show()


# ### Conclusion
# As we can see from graph that deep learning model (LSTM) outperforms other two models in terms of accuracy
