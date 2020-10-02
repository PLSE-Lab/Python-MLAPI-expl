import numpy as np
np.random.seed(0)
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

from keras import backend as K
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import *

# ----------------------------------------------------------------------

df = pd.read_csv('../input/training_data_set.csv', na_values='na')

fill_values = df.median()

df.fillna(fill_values, inplace=True)

# ----------------------------------------------------------------------

feature_columns = df.columns[2:]
label_column = df.columns[1]

X = df[feature_columns].values.astype(np.float32)
y = df[label_column].values.astype(np.float32)

print('X:', X.shape, X.dtype)
print('y:', y.shape, y.dtype)

# ----------------------------------------------------------------------

scaler = StandardScaler()

scaler.fit(X)

X = scaler.transform(X)

# ----------------------------------------------------------------------

df = pd.read_csv('../input/test_data_set.csv', na_values='na')

df.fillna(fill_values, inplace=True)

X_test = df[feature_columns].values.astype(np.float32)

X_test = scaler.transform(X_test)

print('X_test:', X_test.shape, X_test.dtype)

# ----------------------------------------------------------------------

n0 = np.sum(1.-y)
n1 = np.sum(y)

class_weight = {0.: (n0 + n1) / (2.*n0),
                1.: (n0 + n1) / (2.*n1)}

print('class_weight:', class_weight)

# ----------------------------------------------------------------------

def f_beta(y_true, y_pred):
    beta = 7.07
    beta2 = beta**2
    y_pred = K.round(y_pred)
    tp = K.sum(y_true * y_pred)
    fp = K.sum((1.-y_true) * y_pred)
    fn = K.sum(y_true * (1.-y_pred))
    a = (1. + beta2)*tp
    b = a + beta2*fn + fp
    return a / b

# ----------------------------------------------------------------------

class MyCallback(Callback):

    def on_train_begin(self, logs=None):
        self.best_score = None
        self.best_weights = None
        print('%10s %10s %10s %10s %10s' % ('epoch', 'loss', 'val_loss', 'score', 'val_score'))

    def on_epoch_end(self, epoch, logs=None):
        loss = logs['loss']
        val_loss = logs['val_loss']
        score = logs['f_beta']
        val_score = logs['val_f_beta']
        if (self.best_score == None) or (val_score > self.best_score):
            self.best_score = val_score
            self.best_weights = self.model.get_weights()
            print('%10d %10.6f %10.6f %10.6f %10.6f *' % (epoch, loss, val_loss, score, val_score))
        else:
            print('%10d %10.6f %10.6f %10.6f %10.6f' % (epoch, loss, val_loss, score, val_score))

# ----------------------------------------------------------------------

skf = StratifiedKFold(n_splits=10)

preds = []

for i_train, i_valid in skf.split(X, y):

    X_train = X[i_train]
    y_train = y[i_train]

    X_valid = X[i_valid]
    y_valid = y[i_valid]

    print('X_train:', X_train.shape, X_train.dtype)
    print('y_train:', y_train.shape, y_train.dtype)

    print('X_valid:', X_valid.shape, X_valid.dtype)
    print('y_valid:', y_valid.shape, y_valid.dtype)

    # ----------------------------------------------------------------------

    model = Sequential()

    model.add(Dense(2000, input_shape=X_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(2000))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.summary()

    # ----------------------------------------------------------------------

    optimizer = Adam(lr=1e-4)

    model.compile(optimizer, loss='binary_crossentropy', metrics=[f_beta])

    # ----------------------------------------------------------------------

    batch_size = X_valid.shape[0]

    epochs = 100

    mc = MyCallback()

    model.fit(X_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=0,
              callbacks=[mc],
              validation_data=(X_valid, y_valid),
              class_weight=class_weight)

    # ----------------------------------------------------------------------

    model.set_weights(mc.best_weights)

    y_test = model.predict(X_test, batch_size=batch_size, verbose=0)
    y_test = np.squeeze(y_test)
    
    print('y_test:', y_test.shape, y_test.dtype)
    
    preds.append(y_test)

# ----------------------------------------------------------------------

preds = np.mean(preds, axis=0)

print('preds:', preds.shape, preds.dtype)

# ----------------------------------------------------------------------

df = pd.read_csv('../input/sampleSubmission_data.csv')

df['Predicted'] = np.round(preds).astype(int)

df.to_csv('submission.csv', index=False)