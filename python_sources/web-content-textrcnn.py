import re
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score,confusion_matrix
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, add, concatenate,BatchNormalization
from keras.layers import LSTM, Bidirectional, GlobalMaxPooling1D, GlobalAveragePooling1D,CuDNNLSTM,Conv1D,MaxPooling1D,Flatten,Dropout
from keras.preprocessing import text, sequence
from keras.callbacks import LearningRateScheduler, EarlyStopping, TensorBoard

EMBEDDING_FILES = [
    '../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec',
    '../input/glove840b300dtxt/glove.840B.300d.txt'
]
NUM_MODELS = 2
BATCH_SIZE = 256
LSTM_UNITS = 64
DENSE_HIDDEN_UNITS = 4 * LSTM_UNITS
EPOCHS = 4
MAX_LEN = 500
conv_size = 10
dense_units = 128
LEARN_RATE= 0.0004




def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')


def load_embeddings(path):
    with open(path, encoding='UTF-8') as f:
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


def build_model(embedding_matrix):
    words = Input(shape=(MAX_LEN,))
    x_Embedding = Embedding(*embedding_matrix.shape, weights=[embedding_matrix], trainable=False)(words)
    x = SpatialDropout1D(0.2)(x_Embedding)
    x = Bidirectional(CuDNNLSTM(LSTM_UNITS, return_sequences=True))(x)
    x = BatchNormalization()(x)
    # x = Bidirectional(CuDNNLSTM(LSTM_UNITS, return_sequences=True))(x)

    
    x = concatenate([x,x_Embedding])
    x = Conv1D(conv_size, 1, activation='relu', padding='same')(x)
    x = MaxPooling1D(10, padding='same')(x)
    x = Flatten()(x)
    
    # x = concatenate([
    #     GlobalMaxPooling1D()(x),
    #     GlobalAveragePooling1D()(x),
    # ])
    
    hidden = Dropout(0.2)(x)
    hidden = Dense(dense_units, activation='relu')(hidden)
    # hidden = BatchNormalization()(hidden)
    # hidden = add([hidden, Dense(DENSE_HIDDEN_UNITS, activation='relu')(hidden)])
    result = Dense(7, activation='softmax')(hidden)


    model = Model(inputs=words, outputs=result)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=LEARN_RATE),metrics=['acc'])
    model.summary()

    return model


def model_train():

    train_df = pd.read_csv('../input/text-classification-20/web_content_clean.csv')
    x_train = train_df['text'].astype(str)
    y_train = train_df['label'].values

    tokenizer = text.Tokenizer()
    tokenizer.fit_on_texts(list(x_train))

    x_train = tokenizer.texts_to_sequences(x_train)
    x_train = sequence.pad_sequences(x_train, maxlen=MAX_LEN,truncating='post')
    embedding_matrix = np.concatenate(
        [build_matrix(tokenizer.word_index, f) for f in EMBEDDING_FILES], axis=-1)

    num_folds = 5
    patience = 20
    folds = KFold(n_splits=num_folds, shuffle=True)
    cross_valid = []
    predict_sparse = np.zeros(len(x_train))
    for fold_n, (train_index, valid_index) in enumerate(folds.split(x_train)):
        model = build_model(embedding_matrix)
        X_valid = x_train[valid_index]
        Y_valid = y_train[valid_index]
        X_train = x_train[train_index]
        Y_train = y_train[train_index]
        earlyStop = EarlyStopping(monitor='val_acc', patience=patience, restore_best_weights=True, verbose=1)
        TB = TensorBoard(log_dir='./log_fold' + str(fold_n), histogram_freq=0, batch_size=BATCH_SIZE,
                         write_graph=True, write_grads=False, write_images=False,
                         embeddings_freq=0, embeddings_layer_names=None,
                         embeddings_metadata=None, embeddings_data=None,
                         update_freq='epoch')

        model.fit(
            X_train,
            Y_train,
            batch_size=BATCH_SIZE,
            epochs=100,
            verbose=2,
            validation_data=(X_valid, Y_valid),
            callbacks=[earlyStop, TB]
        )
        acc = model.evaluate(x=X_valid, y=Y_valid, batch_size=BATCH_SIZE, verbose=0)
        print(acc)
        cross_valid.append(acc)
        predict_valid = model.predict(X_valid, BATCH_SIZE)

        for i in range(len(predict_valid)):
            predict_sparse[valid_index[i]] = predict_valid[i].argmax()
    print(confusion_matrix(y_train, predict_sparse))
    print(accuracy_score(y_train, predict_sparse))
    # print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(cross_valid), np.std(cross_valid)))
    # print(cross_valid)


model_train()

