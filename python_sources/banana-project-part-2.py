#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


buisness = pd.read_csv('../input/yelp-csv/yelp_academic_dataset_business.csv')
review = pd.read_csv('../input/yelp-csv/yelp_academic_dataset_review.csv')


# In[3]:


buisness_n = buisness[buisness['categories'].str.contains('Restaurant') == True]


# In[4]:


buisness_n = buisness_n.fillna(0)


# In[5]:


review_n = review[review.business_id.isin(buisness_n['business_id']) == True]


# In[6]:


review_new = review_n.sample(n = 350000, random_state = 42)


# In[7]:


text = review_new['text']


# In[8]:


texts_n = []
for i in text:
    i = i.replace("\n", " ")
    texts_n.append(i)


# In[9]:


from nltk.tokenize import word_tokenize
token_texts = []
for i in range (len(texts_n)):
    r = word_tokenize(texts_n[i])
    token_texts.append(r)


# In[10]:


from string import punctuation
texts_new = [" ".join([word for word in text if word not in punctuation and not word.isnumeric()                       and len(word) > 1]) for text in token_texts]
texts_new[1]


# In[13]:


from nltk.stem import WordNetLemmatizer
lem = WordNetLemmatizer()
lem_texts = []
token_texts2 = []
for i in range (len(texts_new)):
    r = word_tokenize(texts_new[i])
    token_texts2.append(r)
for text in token_texts2:
    l = ' '.join([lem.lemmatize(w) for w in text])
    lem_texts.append(l)
lem_texts[0]


# In[21]:


review = review.drop(['text'], axis=1)


# In[22]:


review['text'] = lem_texts


# In[23]:


review = review_new[['text','stars']]


# In[24]:


train = review[0:280000]
test = review[280000:]


# In[25]:


train = pd.get_dummies(train, columns = ['stars'])


# In[26]:


test = pd.get_dummies(test, columns = ['stars'])


# In[27]:


train = train.sample(frac = 0.1, random_state = 42)
test = test.sample(frac = 0.1, random_state = 42)


# In[28]:


class_names = ['stars_1', 'stars_2', 'stars_3', 'stars_4', 'stars_5']
y_train = train[class_names].values


# In[29]:


w2v_model = '../input/glove-global-vectors-for-word-representation/glove.twitter.27B.200d.txt'

def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
w2v_index = dict(get_coefs(*o.strip().split()) for o in open(w2v_model))


# In[30]:


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
tokenizer = Tokenizer(num_words=20000)
tokenizer.fit_on_texts(list(train['text'].values))


# In[31]:


X_train = tokenizer.texts_to_sequences(train['text'].values)
X_test = tokenizer.texts_to_sequences(test['text'].values)
X_train = pad_sequences(X_train, maxlen = 200)
X_test = pad_sequences(X_test, maxlen = 200)


# In[32]:


word_index = tokenizer.word_index


# In[34]:


nb_words = min(20000, len(word_index))
w2v_matrix = np.zeros((nb_words, 200))
missed = []
for word, i in word_index.items():
    if i >= 20000: break
    w2v_vector = w2v_index.get(word)
    if w2v_vector is not None:
        w2v_matrix[i] = w2v_vector
    else:
        missed.append(word)


# In[35]:


from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, SpatialDropout1D, GRU
from keras.layers import Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers
embed_size = 200 
max_features = 20000
maxlen = 200

inp = Input(shape = (maxlen,))
x = Embedding(max_features, embed_size, weights = [w2v_matrix], trainable = True)(inp)
x = SpatialDropout1D(0.5)(x)
x = Bidirectional(LSTM(40, return_sequences=True))(x)
x = Bidirectional(GRU(40, return_sequences=True))(x)
avg_pool = GlobalAveragePooling1D()(x)
max_pool = GlobalMaxPooling1D()(x)
conc = concatenate([avg_pool, max_pool])
outp = Dense(5, activation = 'sigmoid')(conc)


# In[36]:


from keras.callbacks import EarlyStopping, ModelCheckpoint
model = Model(inputs = inp, outputs = outp)
# patience is how many epochs to wait to see if val_loss will improve again.
earlystop = EarlyStopping(monitor = 'val_loss', min_delta = 0, patience = 3)
checkpoint = ModelCheckpoint(monitor = 'val_loss', save_best_only = True, filepath = 'yelp_lstm_gru_weights.hdf5')
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])


# In[37]:


model.fit(X_train, y_train, batch_size = 512, epochs = 20, validation_split = .1, callbacks=[earlystop, checkpoint])


# In[38]:


pred = model.predict([X_test], batch_size=1024, verbose = 1)


# In[39]:


model.evaluate(X_test, test[class_names].values, verbose = 1, batch_size=1024)


# In[ ]:


review_new.stars.hist()


# In[ ]:




