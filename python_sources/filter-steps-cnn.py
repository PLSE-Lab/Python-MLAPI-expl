# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import time
from tensorflow import keras
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier

def CNNbuild():
    clf = keras.models.Sequential([
        keras.layers.Reshape((28, 28, 1), input_shape=(784,)),
        keras.layers.Conv2D(filters=8, kernel_size=(3, 3), padding='same', activation='relu'),
        keras.layers.Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Dropout(0.25),
        keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu'),
        keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Dropout(0.25),
        keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu'),
        keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Dropout(0.25),
        keras.layers.Flatten(),
        keras.layers.Dense(1024, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.25),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.25),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.25),
        keras.layers.Dense(10, activation='softmax')
    ])
    
    clf.compile(optimizer=keras.optimizers.Nadam(),
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    return clf

train = pd.read_csv(os.path.join('..', 'input', r'train.csv'))
test = pd.read_csv(os.path.join('..', 'input', r'test.csv'))

X = train.drop('label', axis=1)
y = train['label']


cnn_model = Pipeline([
    ('scale', MinMaxScaler()),
    ('keras', keras.wrappers.scikit_learn.KerasClassifier(CNNbuild,
                                                          epochs=40,
                                                          batch_size=128,
                                                          validation_split=0.1,
                                                          callbacks=[
                                                              keras.callbacks.EarlyStopping(patience=7),
                                                              keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1, min_lr=0)
                                                                    ],
                                                          verbose=2))
                                  
])

print(CNNbuild().summary())


cnn_model.fit(X, y)

pd.DataFrame({'ImageId': test.index + 1, 'Label': cnn_model.predict(test)}
            ).to_csv(f'sub_{int(time.time())}.csv', index=False)