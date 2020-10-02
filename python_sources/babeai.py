import pandas as pd
import numpy as np
import sys
import os
import spacy
import json
import tensorflow as tf
import os
import re
from spacy.lang.id import Indonesian
from tensorflow import keras as k
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier


# preprocessing param
USE_LEMMA = True
USE_STOPWORDS = True
DROP_NON_LETTER = True

# word2vec params
W2V_DIM = 300

# classifier model params
RNN_HIDDEN_UNIT = 64
RNN_N_EPOCH = 20
INPUT_LENGTH = 25
LEARNING_RATE = 3 * 10 ** -4

# training params
PATIENCE = 2
BATCH_SIZE = 128
VALIDATION_SPLIT = 0.05

# paths
TRAIN_PATH = '../input/train.json'
TEST_PATH = '../input/test.json'
STOPWORDS_PATH = '../input/stop_words.txt'
MODEL_OUTPUT_PATH = 'model2.h5'
PREDICTION_OUTPUT_PATH = 'submission2.csv'

nlp = Indonesian()

def read_doc_to_df(path):
    with open(path) as f:
        string = ','.join(f.read().split('\n')[:-1])
        docs = '[' + string + ']'
        docs = json.loads(docs)
    df = pd.DataFrame(docs)
    df = df[['id', 'text', 'category']]
    return df

def load_stopwords(path):
    with open(path) as f:
        return f.read().split('\n')
        
def preprocess(x, stopwords=None, use_lemma=False, drop_non_letter=False):
    tokens = nlp(x.lower())
    result = []
    for t in tokens:
        t = t.lemma_ if use_lemma else t.text
        if stopwords is not None:
            if drop_non_letter:
                if not re.search('[a-zA-Z]', t):
                    continue
            if t in stopwords:
                continue
        result.append(t)
    return ' '.join(result)
        
train_df = read_doc_to_df(TRAIN_PATH)
test_df = read_doc_to_df(TEST_PATH)
stopwords = set(load_stopwords(STOPWORDS_PATH)) if USE_STOPWORDS else None

train_df['text'] = train_df['text'].apply(preprocess, stopwords=stopwords, use_lemma=USE_LEMMA, drop_non_letter=DROP_NON_LETTER)
test_df['text'] = test_df['text'].apply(preprocess, stopwords=stopwords, use_lemma=USE_LEMMA, drop_non_letter=DROP_NON_LETTER)
val_df = train_df.sample(frac=0.05)
train_df = train_df.drop(val_df.index)

train_docs = train_df['text'].values
val_docs = val_df['text'].values
test_docs = test_df['text'].values

train_target_dummies = pd.get_dummies(train_df['category'])
val_target_dummies = pd.get_dummies(val_df['category'])

y_train = train_target_dummies.values
y_val = val_target_dummies.values

dummies_cols = train_target_dummies.columns.tolist()

# Input preprocessing
print('preprocessing input as tensor..')
t = k.preprocessing.text.Tokenizer()
t.fit_on_texts(np.concatenate([train_docs, val_docs, test_docs]))
encoded_train_docs = t.texts_to_sequences(train_docs)
encoded_val_docs = t.texts_to_sequences(val_docs)
encoded_test_docs = t.texts_to_sequences(test_docs)

padded_train_docs = k.preprocessing.sequence.pad_sequences(encoded_train_docs, maxlen=INPUT_LENGTH, padding='post')
padded_val_docs = k.preprocessing.sequence.pad_sequences(encoded_val_docs, maxlen=INPUT_LENGTH, padding='post')
padded_test_docs = k.preprocessing.sequence.pad_sequences(encoded_test_docs, maxlen=INPUT_LENGTH, padding='post')

# model training
n_voc = max(t.word_index.values())
def build_model():
    model = k.models.Sequential([
        k.layers.Embedding(n_voc, W2V_DIM, input_length=INPUT_LENGTH),
        k.layers.GRU(RNN_HIDDEN_UNIT),
        k.layers.Dropout(0.25),
        k.layers.Dense(y_train.shape[1], activation='softmax')
    ])
    model.compile(optimizer=k.optimizers.Adam(lr=LEARNING_RATE), loss='categorical_crossentropy', metrics=['acc'])
    return model
    
model = build_model()

# train
print('Training on {} sample..'.format(padded_train_docs.shape[0]))
history = model.fit(padded_train_docs, y_train, epochs=RNN_N_EPOCH, 
          validation_split=0.01, batch_size=BATCH_SIZE, 
          callbacks=[k.callbacks.EarlyStopping(patience=PATIENCE)], verbose=0)
print(pd.DataFrame(history.history))
model.save(MODEL_OUTPUT_PATH)

# evaluation
pred_prob = model.predict(padded_val_docs)
pred = np.zeros((len(y_val), y_val.shape[1]))
max_idx = pred_prob.argmax(1)
pred[np.arange(len(y_val)), max_idx] = 1
print('Validation report')
print('accuracy:', accuracy_score(y_val, pred))
print(classification_report(y_val, pred, target_names=dummies_cols))


# predict test set
pred_dummies = model.predict(padded_test_docs)
pred_dummies = pd.DataFrame(pred_dummies, columns=dummies_cols)
pred = pred_dummies.idxmax(axis=1).values.reshape(-1,)
print('Test report')
print('accuracy:', accuracy_score(test_df['category'], pred))
print(classification_report(test_df['category'], pred, target_names=dummies_cols))

submission = test_df[['id']]
submission = test_df[['text']]
submission['target'] = pred
submission.to_json(PREDICTION_OUTPUT_PATH, index=False)
