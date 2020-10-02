#!/usr/bin/env python
# coding: utf-8

# In this notebook I'll try to use RNN on fasttext embeddings (seems like for this case it'll may be good idea to use char-based embedding - at lear as I can see by mine previous kernel with logisitc regressions over word / chars tf-idfs).
# So you'll need fasttext installed (for Windows I used this build - http://cs.mcgill.ca/~mxia3/FastText-for-Windows/).
# 
# **Also - there is no fasttext installed at Kaggle, so you'll need to run notebook on your machine.**
# 
# # Data import

# In[ ]:


import pandas as pd
import numpy as np
from itertools import chain
from nltk.tokenize import wordpunct_tokenize
from keras.preprocessing import text, sequence
from keras.layers import Dense, Embedding, Dropout, LSTM, Bidirectional, GlobalMaxPool1D, InputLayer, BatchNormalization, Activation
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from subprocess import call
from sklearn.utils import compute_sample_weight
from sklearn.metrics import confusion_matrix, log_loss
from collections import OrderedDict


# In[ ]:


train = pd.read_csv("../input/train.csv")
train.fillna("nan")
train.head()


# In[ ]:


test = pd.read_csv("../input/test.csv")
test.fillna("nan")
test.head()


# In[ ]:


submission = pd.read_csv('../input/sample_submission.csv')
submission.head()


# Some values interpreted as float, so I'll convert it to strings:

# In[ ]:


train['comment_text'] = train['comment_text'].apply(str)
test['comment_text'] = test['comment_text'].apply(str)


# # train/validation split
# Let's split data to train/validation set.
# 
# I used next method so save each class distribution:
# 
# - build all possible labels combinations
# - excluded combination that seen l;east then 2 times
# - replaced label combination with combination index
# - build indices for stratified split based on combination indices

# In[ ]:


targets = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']


# In[ ]:


y = np.array(train[targets])
texts = np.array(train['comment_text'])
texts_test = np.array(test['comment_text'])


# In[ ]:


# Some mappings exlucded because have only 1 sample.
label_mapping = np.array([
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 1, 1],
    [0, 0, 0, 1, 0, 0],
    [0, 0, 0, 1, 0, 1],
    [0, 0, 0, 1, 1, 0],
    [0, 0, 0, 1, 1, 1],
    [0, 0, 1, 0, 0, 0],
    #[0, 0, 1, 0, 0, 1],
    [0, 0, 1, 0, 1, 0],
    [0, 0, 1, 0, 1, 1],
    #[0, 0, 1, 1, 0, 0],
    [0, 0, 1, 1, 0, 1],
    #[0, 0, 1, 1, 1, 0],
    [0, 0, 1, 1, 1, 1],
    [0, 1, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 1],
    [0, 1, 0, 0, 1, 0],
    [0, 1, 0, 0, 1, 1],
    [0, 1, 0, 1, 0, 0],
    [0, 1, 0, 1, 0, 1],
    [0, 1, 0, 1, 1, 0],
    [0, 1, 0, 1, 1, 1],
    [0, 1, 1, 0, 0, 0],
    [0, 1, 1, 0, 0, 1],
    [0, 1, 1, 0, 1, 0],
    [0, 1, 1, 0, 1, 1],
    [0, 1, 1, 1, 0, 0],
    [0, 1, 1, 1, 0, 1],
    [0, 1, 1, 1, 1, 0],
    [0, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 1, 0],
    [1, 0, 0, 0, 1, 1],
    [1, 0, 0, 1, 0, 0],
    [1, 0, 0, 1, 0, 1],
    [1, 0, 0, 1, 1, 0],
    [1, 0, 0, 1, 1, 1],
    [1, 0, 1, 0, 0, 0],
    [1, 0, 1, 0, 0, 1],
    [1, 0, 1, 0, 1, 0],
    [1, 0, 1, 0, 1, 1],
    [1, 0, 1, 1, 0, 0],
    [1, 0, 1, 1, 0, 1],
    [1, 0, 1, 1, 1, 0],
    [1, 0, 1, 1, 1, 1],
    [1, 1, 0, 0, 0, 0],
    [1, 1, 0, 0, 0, 1],
    [1, 1, 0, 0, 1, 0],
    [1, 1, 0, 0, 1, 1],
    [1, 1, 0, 1, 0, 0],
    [1, 1, 0, 1, 0, 1],
    [1, 1, 0, 1, 1, 0],
    [1, 1, 0, 1, 1, 1],
    [1, 1, 1, 0, 0, 0],
    [1, 1, 1, 0, 0, 1],
    [1, 1, 1, 0, 1, 0],
    [1, 1, 1, 0, 1, 1],
    [1, 1, 1, 1, 0, 0],
    [1, 1, 1, 1, 0, 1],
    [1, 1, 1, 1, 1, 0],
    [1, 1, 1, 1, 1, 1],
])
y_converted = np.zeros([len(y)])
for i in range(len(label_mapping)):
    idx = (y == label_mapping[i]).sum(axis=1) == 6
    y_converted[idx] = i
train_indices, val_indices, _, _ = train_test_split(np.fromiter(range(len(y)), dtype=np.int32),
                                                    y_converted,
                                                    test_size=0.1,
                                                    stratify=y_converted)


# In[ ]:


texts_train, texts_val = texts[train_indices], texts[val_indices]
y_train, y_val = y[train_indices], y[val_indices]


# # Embedding training
# 
# Now I'll prepare texts from train subset to use in fasttext train:

# In[ ]:


with open('fasttext-embedding-train.txt', 'w', encoding='utf-8') as target:
    for text in texts_train:
        target.write('__label__0\t{0}\n'.format(text.strip()))


# And - with next command I'll start fasttext model train:
# 
# For linux system similar command will be 
# 
#     fasttext skipgram -input fasttext-embedding-train.txt -output embedding-model > /dev/null 2>&1

# In[ ]:


get_ipython().system('fasttext skipgram -input fasttext-embedding-train.txt -output embedding-model >nul 2>&1')


# Now I need to:
# - prepare list of words from train/validation/test sets
# - calculate vectors for each word
# - load vectors in mine model

# In[ ]:


train_texts_tokenized = map(wordpunct_tokenize, train['comment_text'])
test_texts_tokenized = map(wordpunct_tokenize, train['comment_text'])
train_text_tokens = set(chain(*train_texts_tokenized))
test_text_tokens = set(chain(*test_texts_tokenized))
text_tokens = sorted(train_text_tokens | test_text_tokens)
with open("fasttext-words.txt", "w", encoding="utf-8") as target:
    for word in text_tokens:
        target.write("{0}\n".format(word.strip()))


# In[ ]:


get_ipython().system('fasttext print-word-vectors embedding-model.bin < fasttext-words.txt > fasttext-vectors.txt')


# In[ ]:


embedding_matrix = np.zeros([len(text_tokens) + 1, 100])
word2index = {}
with open("fasttext-vectors.txt", "r", encoding="utf-8") as src:
    for i, line in enumerate(src):
        parts = line.strip().split(' ')
        word = parts[0]
        vector = map(float, parts[1:])
        word2index[word] = len(word2index)
        embedding_matrix[i] = np.fromiter(vector, dtype=np.float)


# And finally I'll replace words in text with embedding vector indices:

# In[ ]:


def text2sequence(text):
    return list(map(lambda token: word2index.get(token, len(word2index) - 1), wordpunct_tokenize(str(text))))


X_train = sequence.pad_sequences(list(map(text2sequence, texts_train)), maxlen=100)
X_val = sequence.pad_sequences(list(map(text2sequence, texts_val)), maxlen=100)
X_test = sequence.pad_sequences(list(map(text2sequence, texts_test)), maxlen=100)


# # Model
# 
# Let's build and train model:

# In[ ]:


embed_size = 100
model = Sequential([
    InputLayer(input_shape=(100,), dtype='int32'),
    Embedding(len(embedding_matrix), embed_size),
    Bidirectional(LSTM(50, return_sequences=True)),
    GlobalMaxPool1D(),
    Dropout(0.3),
    Dense(50, activation='relu'),
    Dropout(0.3),
    Dense(6, activation='sigmoid')
])
embedding = model.layers[1]
embedding.set_weights([embedding_matrix])
embedding.trainable = False
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()


# In[ ]:


model.fit(X_train, y_train, 
          batch_size=64, 
          epochs=10, 
          validation_data=(X_val, y_val), 
          verbose=True, 
          callbacks=[
              ModelCheckpoint('model.h5', save_best_only=True),
              EarlyStopping(patience=3)
          ])


# In[ ]:


model.load_weights('model.h5')


# # Test prediction

# In[ ]:


test_prediction = model.predict(X_test, verbose=True)


# In[ ]:


for i, label in enumerate(['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']):
    submission[label] = test_prediction[:, i]


# In[ ]:


submission.head()


# In[ ]:


submission.to_csv('output.csv', index=None)


# # Validation error analysis
# 
# Let's make prediction on validation set - and see what kind of errors we making with different classes:

# In[ ]:


val_prediction = model.predict(X_val, verbose=True)


# In[ ]:


def show_confustion_matrix(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    df = pd.DataFrame(OrderedDict([
        ("true-class", ["negative", "positive"]),
        ("negative-classified", [tn, fn]),
        ("positive-classified", [fp, tp]),
    ]))
    return df.set_index("true-class")


# ## Toxic

# In[ ]:


log_loss(y_val[:, 0], val_prediction[:, 0])


# In[ ]:


show_confustion_matrix(y_val[:, 0], val_prediction[:, 0] > 0.5)


# ## Severe toxic

# In[ ]:


log_loss(y_val[:, 1], val_prediction[:, 1])


# In[ ]:


show_confustion_matrix(y_val[:, 1], val_prediction[:, 1] > 0.5)


# ## Obscene

# In[ ]:


log_loss(y_val[:, 2], val_prediction[:, 2])


# In[ ]:


show_confustion_matrix(y_val[:, 2], val_prediction[:, 2] > 0.5)


# ## Threat

# In[ ]:


log_loss(y_val[:, 3], val_prediction[:, 3])


# In[ ]:


show_confustion_matrix(y_val[:, 3], val_prediction[:, 3] > 0.5)


# ## Insult

# In[ ]:


log_loss(y_val[:, 4], val_prediction[:, 4])


# In[ ]:


show_confustion_matrix(y_val[:, 4], val_prediction[:, 4] > 0.5)


# ## Identity hate

# In[ ]:


log_loss(y_val[:, 5], val_prediction[:, 5])


# In[ ]:


show_confustion_matrix(y_val[:, 5], val_prediction[:, 5] > 0.5)


# In[ ]:




