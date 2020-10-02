#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import re
import string
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


#load the dataset
fake_news = pd.read_csv('/kaggle/input/fake-and-real-news-dataset/Fake.csv')
true_news = pd.read_csv('/kaggle/input/fake-and-real-news-dataset/True.csv')


# In[ ]:


fake_news.head()


# In[ ]:


true_news.head()


# ### We need to merge the two datasets into one. For that we add an addtional columns in both datasets for the class labels

# In[ ]:


fake_news['label'] = 1
true_news['label'] = 0
news_data = pd.concat([fake_news,true_news])
print('No. of examples:',len(news_data))
news_data.head()


# ### Let us check the distribution of classes

# In[ ]:


print(news_data['label'].value_counts())
sns.countplot(news_data['label'])


# ### We can see that the classes are balanced. No need to overform undersampling or oversampling. The subject column contains different values for the fake and true news sets. We remove it. We also remove the date column

# In[ ]:


#drop the subjects and date column
news_data.drop('subject',axis=1,inplace=True)
news_data.drop('date',axis=1,inplace=True)
news_data.head()


# ### Next, we merge the title and text columns into one. Removing the title column might lead to loss of some valuable information. So we use both the title as well as the text

# In[ ]:


#merge the title and text columns into one
news_data['news'] = news_data['title']+" "+news_data['text']
news_data.head()


# In[ ]:


#title and text columns no longer needed
news_data.drop('title',axis=1,inplace=True)
news_data.drop('text',axis=1,inplace=True)
news_data.head()


# ### next we clean our news dataset. We shall do the following:
# #### 1. Remove any links from the text.
# #### 2. Remove any html tags from the text.
# #### 3. Remove punctuation from the text.
# #### 4. Remove numbers from the text.
# #### 5. Remove stopwords.
# #### 6. Convert everything to lower case.
# #### 7. Lemmatize the words.

# In[ ]:


def remove_urls(text):
  return re.sub('https?:\S+','',text)


# In[ ]:


def remove_punctuation(text):
  return text.translate(str.maketrans('','',string.punctuation))


# In[ ]:


def remove_tags(text):
  return re.sub('<.*?>'," ",text)


# In[ ]:


def remove_numbers(text):
  return re.sub('[0-9]+','',text)


# In[ ]:


#remove urls from text
news_data['news'] = news_data['news'].apply(remove_urls)


# In[ ]:


#remove any tags present in the text
news_data['news'] = news_data['news'].apply(remove_tags)


# In[ ]:


#remove punctuation from text
news_data['news'] = news_data['news'].apply(remove_punctuation)


# In[ ]:


#remove numbers from the text
news_data['news'] = news_data['news'].apply(remove_numbers)


# ## Stopwords: these are words that do not contribute much to the semantic meaning of the sentence. eg. 'i','am','are','we','your' etc.

# In[ ]:


import nltk
from nltk.corpus import stopwords
stops = stopwords.words('english')


# In[ ]:


def remove_stopwords(text):
  cleaned = []
  for word in text.split():
    if word not in stops:
      cleaned.append(word)
  return " ".join(cleaned)


# In[ ]:


news_data['news'] = news_data['news'].apply(remove_stopwords)
news_data.head(10)


# In[ ]:


#convert words to lower case
news_data['news'] = news_data['news'].apply(lambda word : word.lower())
news_data.head()


# ## We convert the words to their base or root form. For example: play,playing,plays,played will be converted to play. We can do this in two ways- stemming and lemmatization. Stemming might result in words which are not in the language, for eg: trouble is stemmed to troubl. So we perform lemmatization.

# In[ ]:


#stemming/lemmatization
from nltk.stem import WordNetLemmatizer
#nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

def lemmatize_words(text):
  lemmas = []
  for word in text.split():
    lemmas.append(lemmatizer.lemmatize(word))
  return " ".join(lemmas)


# In[ ]:


news_data['lemmatized_news'] = news_data['news'].apply(lemmatize_words)


# ### Our news data is now ready. Next we split our dataset into X and y sets. We then tokenize the sentences. Tokenization is the process of converting a sentence into a sequence of tokens where each token is the index of the corresponding word in the vocabulary. These tokenized words can then be vectorized

# In[ ]:


#split into X and y sets
#shuffle the dataset
news_data = news_data.sample(frac=1).reset_index(drop=True)
news_x = news_data['lemmatized_news'].values
news_y = news_data['label'].values


# In[ ]:


#tokenization,padding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# In[ ]:


tokenizer = Tokenizer() 
tokenizer.fit_on_texts(news_x)
word_to_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(news_x)


# ### The token sequences will be of different lengths. We set a maximum length for the sequences and pad(or truncate) the sequences accordingly so that each sequence will have the same length

# In[ ]:


vocab_size = len(word_to_index)
max_length = 200
embedding_dim = 100
padded_sequences = pad_sequences(sequences,maxlen=max_length,padding='post',truncating='post')


# ## Vectorization:
# ### now, we vectorize the tokens. Vectorize can be done on several ways. One way is to one-hot encode each vector. But it doesn't reflect the realtionship between the words. For example words 'love' and 'like' have similar meaning but this is not reflected by the one hot vectors. To preserve such relationships we make use of word embeddings. Word embeddings are high dimensional vectors which portray the semantic relationship between words. For example, words 'love' and 'like' will have similar embeddings representation implying that these two words are similar.

# ### We use glove word embeddings for the same. The embeddings can be dowloaded from the official glove website:
# https://nlp.stanford.edu/projects/glove/

# ### Now we construct the embeddings matrix. It is a (nxd) matrix where n is the no. of words in the vocabulary and d is the dimension of the glove vectors

# In[ ]:


embeddings_index = {};
with open('/kaggle/working/glove.6B.100d.txt') as f:
    for line in f:
        values = line.split();
        word = values[0];
        coefs = np.asarray(values[1:], dtype='float32');
        embeddings_index[word] = coefs;

embeddings_matrix = np.zeros((vocab_size+1, embedding_dim));
for word, i in word_to_index.items():
    embedding_vector = embeddings_index.get(word);
    if embedding_vector is not None:
        embeddings_matrix[i] = embedding_vector;


# In[ ]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Embedding,SpatialDropout1D,LSTM,Bidirectional,Dropout
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam


# ### We create the model. We shall use a multilayer LSTM network. We start with an intial learning rate of 0.01 which is then reduced by a factor of 0.5 if there is no improvement in the validation accuracy for 2 epochs.

# In[ ]:


model = Sequential([
    Embedding(vocab_size+1, embedding_dim, input_length=max_length, weights=[embeddings_matrix], trainable=False),
    SpatialDropout1D(0.2),
    Bidirectional(LSTM(128,return_sequences=True)),
    Dropout(0.2),
    LSTM(64),
    Dense(32,activation='relu'),
    Dense(1,activation='sigmoid')
])
optimizer = Adam(learning_rate=0.01)
callbacks = ReduceLROnPlateau(monitor='val_accuracy',patience=2,factor=0.5,min_lr=0.00001)
model.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['accuracy'])
model.summary()


# In[ ]:


#split data into training and testing sets
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(padded_sequences,news_y,test_size=0.15,random_state=1)
print('No. of training samples:',len(X_train))
print('No. of testing samples:',len(X_test))


# In[ ]:


epochs = 10
history = model.fit(X_train,y_train,epochs=epochs,validation_data=(X_test,y_test),batch_size=64,callbacks=[callbacks])


# In[ ]:


#plot the losses and accuracy
fig,axes = plt.subplots(1,2)
fig.set_size_inches(30,10)
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = list(range(10))
axes[0].plot(epochs,acc,label='training accuracy')
axes[0].plot(epochs,val_acc,label='validation accuracy')
axes[0].set_xlabel('epoch no.')
axes[0].set_ylabel('accuracy')
axes[0].legend()
axes[1].plot(epochs,loss,label='training loss')
axes[1].plot(epochs,val_loss,label='validation loss')
axes[1].set_xlabel('epoch no.')
axes[1].set_ylabel('loss')
axes[1].legend()


# In[ ]:


#model evaluation
train_stats = model.evaluate(X_train,y_train)
test_stats = model.evaluate(X_test,y_test)
print('training accuracy:',train_stats[1]*100)
print('testing accuracy:',test_stats[1]*100)


# In[ ]:


#classification report
from sklearn.metrics import classification_report,confusion_matrix
y_pred = model.predict_classes(X_test)
print(classification_report(y_test,y_pred))
print('Confusion matix:\n',confusion_matrix(y_test,y_pred))


# In[ ]:




