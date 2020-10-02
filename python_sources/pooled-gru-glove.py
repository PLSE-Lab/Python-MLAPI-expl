#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

import nltk


# In[ ]:


train = pd.read_csv("../input/jigsaw-toxic-comment-classification-challenge/train.csv")
test = pd.read_csv("../input/jigsaw-toxic-comment-classification-challenge/test.csv")
embeding_file_path = "../input/glove840b300dtxt/glove.840B.300d.txt"



# In[ ]:


train['comment_text'][3]


# preprocessing :
# 1. lowercase
# 2. stopwords
# 3. low frequency words drop out
# 4. lemmatizer

# In[ ]:


from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
wl = WordNetLemmatizer()

filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'

def normalize(text):
    text = text.lower()
    translate_map = str.maketrans(filters, " " * len(filters))
    text = text.translate(translate_map)
    tokens = nltk.word_tokenize(text)
    tags = nltk.pos_tag(tokens)
    stop_words = set(stopwords.words('english'))
    seq= [wl.lemmatize(t[0], pos=get_wordnet_pos(t[1])) for t in tags if t[0] not in stop_words]
    seq= [wl.lemmatize(t[0], pos=get_wordnet_pos(t[1])) for t in tags]

    return seq
#s = normalize(train["comment_text"][3/


# In[ ]:


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
max_features = 50000
embed_size = 300
maxlen = 150

data = pd.concat((train['comment_text'], test['comment_text']))
seqs = [normalize(text) for text in data]

def seq_to_sequence(seq, word_index):
    sequence = []
    for word in seq:
        if not word_index.get(word): continue 
        sequence.append(word_index[word])
    return sequence

def fit_on_sequence(seqs):
    word_counts = dict()
    for seq in seqs:
        for w in seq:
            if w not in word_counts:
                word_counts[w] = 0
            word_counts[w] += 1
    wcounts = list(word_counts.items())
    wcounts.sort(key=lambda x: x[1], reverse=True)
    sorted_voc = [wc[0] for wc in wcounts if wc[1]>=3]
    sorted_voc = [wc[0] for wc in wcounts]
    word_index = dict(list(zip(sorted_voc, list(range(1, len(sorted_voc) + 1)))))
    return word_index
    
word_index = fit_on_sequence(seqs)
train_words = [seq_to_sequence(seq, word_index) for seq in seqs[:train.shape[0]]]
test_words = [seq_to_sequence(seq, word_index) for seq in seqs[train.shape[0]:]]
train_words = pad_sequences(train_words, maxlen=maxlen )
test_words = pad_sequences(test_words, maxlen=maxlen)


# In[ ]:


len(word_index)


# In[ ]:


# char_index = tokenizer_char.word_index
# char_size = len(char_index)
# char_size = min(5000, char_size)


# In[ ]:


def get_coef(word, *coefs):
    return word, np.asarray(coefs, dtype=np.float64)
embeding_dict = dict(get_coef(*s.strip().split(" ")) for s in open(embeding_file_path))


# In[ ]:



max_words = min(max_features, len(word_index))
embeding_matrix = np.zeros((max_words+1, embed_size))
for word,i in word_index.items():
    if word not in embeding_dict: continue
    if i>max_words:break
    embeding_matrix[i] = embeding_dict[word]
# char_matrix = np.random.randn(char_size, embed_size)
# embeding_matrix = np.concatenate((embeding_matrix, char_matrix), axis=0)

#  transform the char_index
# addition = max_words + 1
# train_chars += addition
# test_chars += addition

# train_all = np.concatenate((train_words, train_chars), axis=1)
# test_all = np.concatenate((test_words, test_chars), axis=1)


# In[ ]:


len(embeding_dict)


# In[ ]:


from keras.models import Model
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, concatenate
from keras.layers import GRU, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D,Conv1D
from keras.callbacks import Callback
def get_model():
    inp = Input(shape=(maxlen,)) #maxlen
    x = Embedding(max_words+1, embed_size, weights=[embeding_matrix])(inp)
    x = SpatialDropout1D(0.2)(x)
    x = Bidirectional(GRU(maxlen, return_sequences=True))(x)
    x = Conv1D(64, kernel_size = 2, padding = "valid", kernel_initializer = "glorot_uniform")(x)
    max_pool = GlobalMaxPooling1D()(x)
    avg_pool = GlobalAveragePooling1D()(x)
    conc = concatenate([max_pool, avg_pool])
    oup = Dense(6, activation='sigmoid')(conc)
    
    model = Model(input=inp, output=oup)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

model = get_model()


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from keras.callbacks import Callback

class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            print("\n ROC-AUC - epoch: %d - score: %.6f \n" % (epoch+1, score))

batch_size = 128
epochs = 4
y_train = train[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values
X_tra, X_val, y_tra, y_val = train_test_split(train_words, y_train, train_size=0.95, random_state=233)
RocAuc = RocAucEvaluation(validation_data=(X_val, y_val), interval=1)
hist = model.fit(X_tra, y_tra, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val),
                 callbacks=[RocAuc,], verbose=1)


y_pred = model.predict(test_words, batch_size=1024)
          


# In[ ]:


submission = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv')
submission[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]] = y_pred
submission.to_csv('gru.csv', index=False)           

