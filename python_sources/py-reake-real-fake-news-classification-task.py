#!/usr/bin/env python
# coding: utf-8

# > 

# In[ ]:


import pandas as pd
from nltk.corpus import stopwords
import string


# In[ ]:


fake = pd.read_csv("/kaggle/input/fake-and-real-news-dataset/Fake.csv", names=None,encoding='latin-1',low_memory=False)
fake.tail(2)


# In[ ]:


true = pd.read_csv("/kaggle/input/fake-and-real-news-dataset/True.csv", names=None,encoding='latin-1',low_memory=False)
true.tail(2)


# In[ ]:


fake['label']=1
fake.head(2)


# In[ ]:


true['label']=0
true.head(2)


# In[ ]:


df=pd.concat([fake,true]).reset_index(drop=True)
df.head(2)


# In[ ]:


df.shape


# In[ ]:


df.label.value_counts()


# # PREPROCESSING AND CLEANING DATA

# In[ ]:


STOPWORDS = set(stopwords.words('english'))


# In[ ]:


def clean(text):
    #1. Remove punctuation
    translator1 = str.maketrans(string.punctuation, ' '*len(string.punctuation))
    text = text.translate(translator1)
    
    #2. Convert to lowercase characters
    text = text.lower()
    
    #3. Remove stopwords
    text = ' '.join([word for word in text.split() if word not in STOPWORDS])
    
    return text


# In[ ]:


df['clean_title']=df['title'].apply(clean)


# In[ ]:


df['clean_text']=df['text'].apply(clean)
df.head(2)


# In[ ]:


df['clean_subject']=df['subject'].str.lower()
df.head(2)


# In[ ]:


df['combined']=df['clean_subject']+' '+df['clean_title']+' '+df['clean_text']
df.head(2)


# In[ ]:


from nltk.stem.porter import PorterStemmer


# In[ ]:


stop = set(stopwords.words('english'))
punctuation = list(string.punctuation)
stop.update(punctuation)


# In[ ]:


stemmer = PorterStemmer()

def stem_text(text):
    final_text = []
    for i in text.split():
        if i.strip().lower() not in stop:
            word = stemmer.stem(i.strip())
            final_text.append(word)
    return " ".join(final_text)


# In[ ]:


df['after_stemming'] = df.combined.apply(stem_text)
df.head(2)


# In[ ]:


import numpy as np


# In[ ]:


#visualize word distribution
df['doc_len'] = ''
df['doc_len'] = df['after_stemming'].apply(lambda words: len(words.split(" ")))
max_seq_len = np.round(df['doc_len'].mean() + df['doc_len'].std()).astype(int)
print(max_seq_len)


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(df['after_stemming'].to_list(), df['label'].values, test_size=0.33, random_state=42)
print(len(X_train))
print(len(X_test))
print(len(y_train))
print(len(y_test))


# In[ ]:


from keras.preprocessing.text import Tokenizer


# In[ ]:


MAX_NB_WORDS = 28000

tokenizer = Tokenizer(num_words=MAX_NB_WORDS, lower=True, char_level=False)
tokenizer.fit_on_texts(X_train)

word_index = tokenizer.word_index
print("dictionary size: ", len(word_index))


# In[ ]:


from keras.preprocessing import sequence


# In[ ]:


word_seq_train = tokenizer.texts_to_sequences(X_train)
word_seq_train = sequence.pad_sequences(word_seq_train, maxlen=max_seq_len)


# In[ ]:


word_seq_train.shape


# In[ ]:


gensim_news_desc = []

chunk_data = X_train

for record in range(0,len(chunk_data)):
    news_desc_list = []
    for tok in chunk_data[record].split():
        news_desc_list.append(str(tok))
    gensim_news_desc.append(news_desc_list)

len(gensim_news_desc)


# In[ ]:


from gensim.models import Word2Vec

gensim_model = Word2Vec(gensim_news_desc, min_count=5, size = 200, sg=1)

# summarize the loaded model
print(gensim_model)

# summarize vocabulary
words = list(gensim_model.wv.vocab)


# In[ ]:


len(words)


# In[ ]:


#training params
batch_size = 1024
num_epochs = 10
#model parameters
num_filters = 128
embed_dim = 200 
weight_decay = 1e-4
class_weight = {0: 1,
               1: 1}


# In[ ]:


print('preparing embedding matrix...')

gensim_words_not_found = []
gensim_nb_words = len(gensim_model.wv.vocab)
print("gensim_nb_words : ",gensim_nb_words)

gensim_embedding_matrix = np.zeros((gensim_nb_words, embed_dim))

for word, i in word_index.items():
    #print(word)
    if i >= gensim_nb_words:
        continue
    if word in gensim_model.wv.vocab :
        embedding_vector = gensim_model[word]
        if (embedding_vector is not None) and len(embedding_vector) > 0:
            gensim_embedding_matrix[i] = embedding_vector
    else :
        gensim_words_not_found.append(word)


# In[ ]:


gensim_embedding_matrix.shape


# # TRAIN MODEL 

# In[ ]:


from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Input, Embedding, Conv1D, MaxPooling1D
from keras.models import Model
from keras import regularizers
from keras import optimizers
from keras.callbacks import EarlyStopping
from keras import backend as K


# In[ ]:


model = Sequential()
# gensim word2vec embedding
model.add(Embedding(gensim_nb_words, embed_dim, weights=[gensim_embedding_matrix], input_length=max_seq_len))
model.add(Conv1D(num_filters, 5, activation='relu', padding='same'))
model.add(MaxPooling1D(2))
model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Dropout(0.6))
model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Dropout(0.6))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid')) 

adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
model.summary()


# In[ ]:


#define callbacks
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=4, verbose=1)
callbacks_list = [early_stopping]


# In[ ]:


# gensim model training
hist = model.fit(word_seq_train, y_train, batch_size=batch_size, epochs=num_epochs, callbacks=callbacks_list, validation_split=0.1, shuffle=True, verbose=2,class_weight=class_weight)


# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


# list all data in history

print(hist.history.keys())

# summarize history for accuracy
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[ ]:


word_seq_test = tokenizer.texts_to_sequences(X_test)
word_seq_test = sequence.pad_sequences(word_seq_test, maxlen=max_seq_len)


# In[ ]:


predictions = model.predict(word_seq_test)
pred_labels = predictions.round()


# In[ ]:


unique, counts = np.unique(y_test, return_counts=True)
unique, counts


# In[ ]:


from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, pred_labels, labels=[1,0])
cm

