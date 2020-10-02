#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from sklearn import model_selection
from keras.models import Model
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, add, concatenate
from keras.layers import CuDNNLSTM, Bidirectional, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.preprocessing import text, sequence
from keras.callbacks import LearningRateScheduler
from sklearn import metrics


# # Load up the data

# In[ ]:


train_df = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv') # nrows=100000
# test_df = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv')

train_df, val_df = model_selection.train_test_split(train_df, test_size=0.2)
print('%d train comments, %d validate comments' % (len(train_df), len(val_df)))

train_x = train_df['comment_text'].astype(str)
train_y = np.where(train_df['target'] >= 0.5, 1, 0)

val_x = val_df['comment_text'].astype(str)
val_y = np.where(val_df['target'] >= 0.5, 1, 0)


# # Tokenization

# In[ ]:


MAX_LEN = 220

tokenizer = text.Tokenizer()
tokenizer.fit_on_texts(list(train_x) + list(val_x))

train_x = tokenizer.texts_to_sequences(train_x)
val_x = tokenizer.texts_to_sequences(val_x)
train_x = sequence.pad_sequences(train_x, maxlen=MAX_LEN)
val_x = sequence.pad_sequences(val_x, maxlen=MAX_LEN)


# # Load embeddings

# In[ ]:


def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')

def load_embeddings(path):
    with open(path) as f:
        return dict(get_coefs(*line.strip().split(' ')) for line in f)

def build_matrix(word_index, path):
    embedding_index = load_embeddings(path)
    embedding_matrix = np.zeros((len(word_index) + 1, 300))
    for word, i in word_index.items():
        try:
            embedding_matrix[i] = embedding_index[word]
        except KeyError:
            pass
    return embedding_matrix


# In[ ]:


embedding_matrix = build_matrix(tokenizer.word_index, '../input/glove840b300dtxt/glove.840B.300d.txt')


# # Create model

# In[ ]:


NUM_MODELS = 2
BATCH_SIZE = 512
LSTM_UNITS = 128
DENSE_HIDDEN_UNITS = 4 * LSTM_UNITS
EPOCHS = 4

def build_model(embedding_matrix): #, num_aux_targets):
    words = Input(shape=(None,))
    x = Embedding(*embedding_matrix.shape, weights=[embedding_matrix], trainable=False)(words)
    x = SpatialDropout1D(0.2)(x)
    # x = CuDNNLSTM(LSTM_UNITS, return_sequences=True)(x)
    # x = CuDNNLSTM(LSTM_UNITS, return_sequences=True)(x)
    x = Bidirectional(CuDNNLSTM(LSTM_UNITS, return_sequences=True))(x)
    x = Bidirectional(CuDNNLSTM(LSTM_UNITS, return_sequences=True))(x)

    hidden = concatenate([GlobalMaxPooling1D()(x), GlobalAveragePooling1D()(x)])
    # hidden = GlobalMaxPooling1D()(x)
    hidden = Dense(DENSE_HIDDEN_UNITS, activation='relu')(hidden)
    # hidden = Dense(DENSE_HIDDEN_UNITS, activation='relu')(hidden)
    # hidden = add([hidden, Dense(DENSE_HIDDEN_UNITS, activation='relu')(hidden)])
    # hidden = add([hidden, Dense(DENSE_HIDDEN_UNITS, activation='relu')(hidden)])
    result = Dense(1, activation='sigmoid')(hidden)
    # aux_result = Dense(num_aux_targets, activation='sigmoid')(hidden)
    
    # model = Model(inputs=words, outputs=[result, aux_result])
    model = Model(inputs=words, outputs=result)
    model.compile(loss='binary_crossentropy', optimizer='adam')

    return model


# # Run model

# In[ ]:


checkpoint_predictions = []
weights = []

model = build_model(embedding_matrix)
for global_epoch in range(EPOCHS):
    model.fit(
        train_x,
        train_y,
        batch_size=BATCH_SIZE,
        epochs=1,
        verbose=2,
        callbacks=[
            LearningRateScheduler(lambda _: 1e-3 * (0.55 ** global_epoch))
        ]
    )
    checkpoint_predictions.append(model.predict(val_x, batch_size=2048)[0].flatten())
    # weights.append(2 ** global_epoch)

# predictions = np.average(checkpoint_predictions, weights=weights, axis=0)


# # Prediction

# In[ ]:


val_yhat = model.predict(val_x, batch_size=2048)


# # Subgroup analysis

# In[ ]:


val_df['prediction'] = val_yhat
val_df['target'] = val_y
df = val_df

groups = ['black', 'white', 'male', 'female','christian', 'jewish', 'muslim','psychiatric_or_mental_illness','homosexual_gay_or_lesbian']
categories = pd.DataFrame(columns = ['SUB', 'BPSN', 'BNSP'], index = groups)

def auc(df):
    y = df['target']
    pred = df['prediction']
    fpr, tpr, thresholds = metrics.roc_curve(y, pred)
    return metrics.auc(fpr, tpr)

def Mp(data, p=-5.0):
    return np.average(data ** p) ** (1/p)

for group in groups:
    df[group] = df[group] >= 0.5
    categories.loc[group,'SUB'] = auc(df[df[group]])
    bpsn = ((~df[group] & df['target'])    #background positive
            | (df[group] & ~df['target'])) #subgroup negative
    categories.loc[group,'BPSN'] = auc(df[bpsn])
    bnsp = ((~df[group] & ~df['target'])   #background negative
            | (df[group] & df['target']))  #subgrooup positive
    categories.loc[group,'BNSP'] = auc(df[bnsp])

categories.loc['Mp',:] = categories.apply(Mp, axis= 0)

print("Overal AUC: " + str(auc(df)))

categories


# In[ ]:




