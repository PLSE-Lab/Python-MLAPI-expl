#!/usr/bin/env python
# coding: utf-8

# # Library Imports

# In[ ]:


import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import re
from wordcloud import WordCloud


# In[ ]:


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Conv1D, MaxPool1D
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


# # Exploring Fake News

# In[ ]:


fake = pd.read_csv("/kaggle/input/fake-and-real-news-dataset/Fake.csv")


# In[ ]:


fake.head()


# In[ ]:


#Counting by Subjects 
for key,count in fake.subject.value_counts().iteritems():
    print(f"{key}:\t{count}")
    
#Getting Total Rows
print(f"Total Records:\t{fake.shape[0]}")


# In[ ]:


plt.figure(figsize=(8,5))
sns.countplot("subject", data=fake)
plt.show()


# In[ ]:


#Word Cloud
text = ''
for news in fake.text.values:
    text += f" {news}"
wordcloud = WordCloud(
    width = 3000,
    height = 2000,
    background_color = 'black',
    stopwords = set(nltk.corpus.stopwords.words("english"))).generate(text)
fig = plt.figure(
    figsize = (40, 30),
    facecolor = 'k',
    edgecolor = 'k')
plt.imshow(wordcloud, interpolation = 'bilinear')
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()
del text


# # Exploring Real news

# In[ ]:


real = pd.read_csv("/kaggle/input/fake-and-real-news-dataset/True.csv")
real.head()


# ### Difference in Text
# Real news seems to have source of publication which is not present in fake news set
# 
# Looking at the data:
# - most of text contains reuters information such as "**WASHINGTON (Reuters)**".
# - Some text are tweets from Twitter 
# - Few text do not contain any publication info

# # Cleaning Data
# Removing Reuters or Twitter Tweet information from the text 
# 
# - Text can be splitted only once at " - " which is always present after mentioning source of publication, this gives us publication part and text part
# - If we do not get text part, this means publication details was't given for that record
# - The Twitter tweets always have same source, a long text of max 259 characters 

# In[ ]:


#First Creating list of index that do not have publication part
unknown_publishers = []
for index,row in enumerate(real.text.values):
    try:
        record = row.split(" -", maxsplit=1)
        #if no text part is present, following will give error
        record[1]
        #if len of piblication part is greater than 260
        #following will give error, ensuring no text having "-" in between is counted
        assert(len(record[0]) < 260)
    except:
        unknown_publishers.append(index)


# In[ ]:


#Thus we have list of indices where publisher is not mentioned
#lets check
real.iloc[unknown_publishers].text
#true, they do not have text like "WASHINGTON (Reuters)"


# While looking at texts that do not contain publication info such as which reuter, we noticed one thing.
# 
# **Text at index 8970 is empty**

# In[ ]:


real.iloc[8970]
#yep empty
#will remove this soon


# In[ ]:


#Seperating Publication info, from actual text
publisher = []
tmp_text = []
for index,row in enumerate(real.text.values):
    if index in unknown_publishers:
        #Add unknown of publisher not mentioned
        tmp_text.append(row)
        
        publisher.append("Unknown")
        continue
    record = row.split(" -", maxsplit=1)
    publisher.append(record[0])
    tmp_text.append(record[1])


# In[ ]:


#Replace existing text column with new text
#add seperate column for publication info
real["publisher"] = publisher
real["text"] = tmp_text

del publisher, tmp_text, record, unknown_publishers


# In[ ]:


real.head()


# New column called "Publisher" has been added.
# 

# In[ ]:


#checking for rows with empty text like row:8970
[index for index,text in enumerate(real.text.values) if str(text).strip() == '']
#seems only one :)


# In[ ]:


#dropping this record
real = real.drop(8970, axis=0)


# In[ ]:


# checking for the same in fake news
empty_fake_index = [index for index,text in enumerate(fake.text.values) if str(text).strip() == '']
print(f"No of empty rows: {len(empty_fake_index)}")
fake.iloc[empty_fake_index].tail()


# **630 Rows in Fake news with empty text**
# 
# Also noticed fake news have a lot of CPATIAL-CASES. Could preserve Cases of letters, but as we are using Google's pretrained word2vec vectors later on, which haswell-formed lower cases word. We will contert to lower case.
# 
# The text for these rows seems to be present in title itself. Lets merge title and text to solve these cases.

# In[ ]:


#Looking at publication Information
# Checking if Some part of text has been included as publisher info... No such cases it seems :)

# for name,count in real.publisher.value_counts().iteritems():
#     print(f"Name: {name}\nCount: {count}\n")


# In[ ]:


#Getting Total Rows
print(f"Total Records:\t{real.shape[0]}")

#Counting by Subjects 
for key,count in real.subject.value_counts().iteritems():
  print(f"{key}:\t{count}")


# In[ ]:


sns.countplot(x="subject", data=real)
plt.show()


# In[ ]:


#WordCloud For Real News
text = ''
for news in real.text.values:
    text += f" {news}"
wordcloud = WordCloud(
    width = 3000,
    height = 2000,
    background_color = 'black',
    stopwords = set(nltk.corpus.stopwords.words("english"))).generate(str(text))
fig = plt.figure(
    figsize = (40, 30),
    facecolor = 'k',
    edgecolor = 'k')
plt.imshow(wordcloud, interpolation = 'bilinear')
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()
del text


# # Preprocessing Text

# In[ ]:


# Adding class Information
real["class"] = 1
fake["class"] = 0


# In[ ]:


#Combining Title and Text
real["text"] = real["title"] + " " + real["text"]
fake["text"] = fake["title"] + " " + fake["text"]


# In[ ]:


# Subject is diffrent for real and fake thus dropping it
# Aldo dropping Date, title and Publication Info of real
real = real.drop(["subject", "date","title",  "publisher"], axis=1)
fake = fake.drop(["subject", "date", "title"], axis=1)


# In[ ]:


#Combining both into new dataframe
data = real.append(fake, ignore_index=True)
del real, fake


# In[ ]:


# Download following if not downloaded in local machine

# nltk.download('stopwords')
# nltk.download('punkt')


# Removing StopWords, Punctuations and single-character words

# In[ ]:


y = data["class"].values
#Converting X to format acceptable by gensim, removing annd punctuation stopwords in the process
X = []
stop_words = set(nltk.corpus.stopwords.words("english"))
tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
for par in data["text"].values:
    tmp = []
    sentences = nltk.sent_tokenize(par)
    for sent in sentences:
        sent = sent.lower()
        tokens = tokenizer.tokenize(sent)
        filtered_words = [w.strip() for w in tokens if w not in stop_words and len(w) > 1]
        tmp.extend(filtered_words)
    X.append(tmp)

del data


# ### Vectorization -- Word2Vec
# 
# Word2Vec is one of the most popular technique to learn word embeddings using shallow neural network. It was developed by Tomas Mikolov in 2013 at Google.
# 
# Word embedding is the most popular representation of document vocabulary. It is capable of capturing context of a word in a document, semantic and syntactic similarity, relation with other words, etc.
# 
# [Here](https://towardsdatascience.com/introduction-to-word-embedding-and-word2vec-652d0c2060fa) is a nice article about it.
# 
# 
# 

# #### Let's create and check our own Word2Vec model with **gensim**

# In[ ]:


import gensim


# In[ ]:


#Dimension of vectors we are generating
EMBEDDING_DIM = 100

#Creating Word Vectors by Word2Vec Method (takes time...)
w2v_model = gensim.models.Word2Vec(sentences=X, size=EMBEDDING_DIM, window=5, min_count=1)


# In[ ]:


#vocab size
len(w2v_model.wv.vocab)

#We have now represented each of 122248 words by a 100dim vector.


# ### Exploring Vectors
# 
# Lets checkout these vectors

# In[ ]:


#see a sample vector for random word, lets say Corona 
w2v_model["corona"]


# In[ ]:


w2v_model.wv.most_similar("iran")


# In[ ]:


w2v_model.wv.most_similar("fbi")


# In[ ]:


w2v_model.wv.most_similar("facebook")


# In[ ]:


w2v_model.wv.most_similar("computer")


# In[ ]:


#Feeding US Presidents
w2v_model.wv.most_similar(positive=["trump","obama", "clinton"])
#First was Bush


# **Looking at the similar words, vectors are well formed for these words :)**
# 
# 
# These Vectors will be passed to LSTM/GRU instead of words. 1D-CNN can further be used to extract features from the vectors. 
# 
# 
# Keras has implementation called "**Embedding Layer**" which would create word embeddings(vectors). Since we did that with gensim's word2vec, we will load these vectors into embedding layer and make the layer non-trainable.
# 
# 
# 

# We cannot pass string words to embedding layer, thus need some way to represent each words by numbers.
# 
# Tokenizer can represent each word by number

# In[ ]:


# Tokenizing Text -> Repsesenting each word by a number
# Mapping of orginal word to number is preserved in word_index property of tokenizer

#Tokenized applies basic processing like changing it yo lower case, explicitely setting that as False
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X)

X = tokenizer.texts_to_sequences(X)


# In[ ]:


# lets check the first 10 words of first news
#every word has been represented with a number
X[0][:10]


# In[ ]:


#Lets check few word to numerical replesentation
#Mapping is preserved in dictionary -> word_index property of instance
word_index = tokenizer.word_index
for word, num in word_index.items():
    print(f"{word} -> {num}")
    if num == 10:
        break        


# **Notice it starts with 1**
# 

# We can pass numerical representation of words into neural network.
# 
# We can use Many-To-One (Sequence-To-Word) Model of RNN, as we have many words in news as input and one output ie Probability of being Real.
# 
# For Many-To-One model, lets use a fixed size input. 
# 

# In[ ]:


# For determining size of input...

# Making histogram for no of words in news shows that most news article are under 700 words.
# Lets keep each news small and truncate all news to 700 while tokenizing
plt.hist([len(x) for x in X], bins=500)
plt.show()

# Its heavily skewed. There are news with 5000 words? Lets truncate these outliers :) 


# In[ ]:


nos = np.array([len(x) for x in X])
len(nos[nos  < 700])
# Out of 48k news, 44k have less than 700 words


# In[ ]:


#Lets keep all news to 700, add padding to news with less than 700 words and truncating long ones
maxlen = 700 

#Making all news of size maxlen defined above
X = pad_sequences(X, maxlen=maxlen)


# In[ ]:


#all news has 700 words (in numerical form now). If they had less words, they have been padded with 0
# 0 is not associated to any word, as mapping of words started from 1
# 0 will also be used later, if unknows word is encountered in test set
len(X[0])


# In[ ]:


# Adding 1 because of reserved 0 index
# Embedding Layer creates one more vector for "UNKNOWN" words, or padded words (0s). This Vector is filled with zeros.
# Thus our vocab size inceeases by 1
vocab_size = len(tokenizer.word_index) + 1


# In[ ]:


# Function to create weight matrix from word2vec gensim model
def get_weight_matrix(model, vocab):
    # total vocabulary size plus 0 for unknown words
    vocab_size = len(vocab) + 1
    # define weight matrix dimensions with all 0
    weight_matrix = np.zeros((vocab_size, EMBEDDING_DIM))
    # step vocab, store vectors using the Tokenizer's integer mapping
    for word, i in vocab.items():
        weight_matrix[i] = model[word]
    return weight_matrix


# We Create a matrix of mapping between word-index and vectors. We use this as weights in embedding layer
# 
# Embedding layer accepts numecical-token of word and outputs corresponding vercor to inner layer.
# 
# It sends vector of zeros to next layer for unknown words which would be tokenized to 0.
# 
# 
# Input length of Embedding Layer is the length of each news (700 now due to padding and truncating)

# In[ ]:


#Getting embedding vectors from word2vec and usings it as weights of non-trainable keras embedding layer
embedding_vectors = get_weight_matrix(w2v_model, word_index)


# In[ ]:


#Defining Neural Network
model = Sequential()
#Non-trainable embeddidng layer
model.add(Embedding(vocab_size, output_dim=EMBEDDING_DIM, weights=[embedding_vectors], input_length=maxlen, trainable=False))
#LSTM 
model.add(LSTM(units=128))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

del embedding_vectors


# In[ ]:


model.summary()


# In[ ]:


#Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y) 


# In[ ]:


model.fit(X_train, y_train, validation_split=0.3, epochs=6)


# In[ ]:


#Prediction is in probability of news being real, so converting into classes
# Class 0 (Fake) if predicted prob < 0.5, else class 1 (Real)
y_pred = (model.predict(X_test) >= 0.5).astype("int")


# In[ ]:


accuracy_score(y_test, y_pred)


# In[ ]:


print(classification_report(y_test, y_pred))


# In[ ]:


del model


# ### Using Pre-Trained Word2Vec Vectors
# 
# **Needs 12GB RAM and 4GB HardDisk Space **

# Now, instead of creating word vectors, let us use pre-trained vectors trained on part of **Google News dataset** (about 100 billion words). The model contains 300-dimensional vectors for 3 million words and phrases.  Source: https://code.google.com/archive/p/word2vec/
# 
# **Please download model file from**: https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing
# 
# 
# Or add Dataset from https://www.kaggle.com/sandreds/googlenewsvectorsnegative300
# 

# In[ ]:


#invoke garbage collector to free ram
import gc
gc.collect()


# In[ ]:


from gensim.models.keyedvectors import KeyedVectors


# In[ ]:


# Takes RAM 
word_vectors = KeyedVectors.load_word2vec_format('../input/googlenewsvectorsnegative300/GoogleNews-vectors-negative300.bin', binary=True)
EMBEDDING_DIM=300


# ### Exploring these trained Vectors

# In[ ]:


# word_vectors.most_similar('usa')


# In[ ]:


# word_vectors.most_similar('fbi')


# In[ ]:


# word_vectors.most_similar('Republic')


# In[ ]:


embedding_matrix = np.zeros((vocab_size, EMBEDDING_DIM))
for word, i in word_index.items():
    try:
        embedding_vector = word_vectors[word]
        embedding_matrix[i] = embedding_vector
    except KeyError:
        embedding_matrix[i]=np.random.normal(0,np.sqrt(0.25),EMBEDDING_DIM)

del word_vectors 


# In[ ]:


model = Sequential()
model.add(Embedding(vocab_size, output_dim=EMBEDDING_DIM, weights=[embedding_matrix], input_length=maxlen, trainable=False))
model.add(Conv1D(activation='relu', filters=4, kernel_size=4))
model.add(MaxPool1D())
model.add(LSTM(units=128))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

del embedding_matrix


# In[ ]:


model.summary()


# In[ ]:


model.fit(X_train, y_train, validation_split=0.3, epochs=12)


# In[ ]:


y_pred = (model.predict(X_test) > 0.5).astype("int")


# In[ ]:


accuracy_score(y_test, y_pred)


# In[ ]:


print(classification_report(y_test, y_pred))


# **Do Upvote if you find this notebook useful.**
# 
# **Thanks**

# In[ ]:




