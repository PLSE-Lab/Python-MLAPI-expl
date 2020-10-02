# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import keras
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from numpy.ma import array
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dropout
from keras.layers.normalization import BatchNormalization
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import pandas as pd

train_data = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
#label = pd.read_csv('Submission.csv')
test_data = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')

#test_data['label'] = label['label']

#train_data = pd.concat([train_data, test_data], sort=True, ignore_index=True)

Prediction = pd.DataFrame()

data_label = array(train_data['label'])

data_label = to_categorical(data_label)

train_data.drop(columns=['label'], inplace=True)

col_names = train_data.columns

test_data = test_data[col_names]

mm = MinMaxScaler()
train_data = mm.fit_transform(train_data)
test_data = mm.transform(test_data)

train_data = array(train_data)
test_data = array(test_data)

train_data = train_data.reshape(train_data.shape[0], 28,28,1)
test_data = test_data.reshape(test_data.shape[0], 28,28,1)

imd = ImageDataGenerator(rotation_range=10, width_shift_range=0.18, 
                         height_shift_range=0.15)


X_train, X_test, Y_train, Y_test = train_test_split(train_data, data_label, test_size=0.2)

batch_data = imd.flow(X_train, Y_train, batch_size=64, seed=2020)
val_data = imd.flow(X_test, Y_test,batch_size=64, seed=2020)

new_model = Sequential()
new_model.add(Convolution2D(32,(3,3), activation='relu', input_shape=(28,28,1)))
new_model.add(BatchNormalization())
#new_model.add(Dropout(0.5))
new_model.add(MaxPooling2D((2,2)))
new_model.add(BatchNormalization())
#new_model.add(Dropout(0.5))
new_model.add(Convolution2D(128,(3,3), activation='relu'))
new_model.add(BatchNormalization())
#new_model.add(Dropout(0.5))
new_model.add(MaxPooling2D((2,2)))
new_model.add(BatchNormalization())
#new_model.add(Dropout(0.5))
new_model.add(Flatten())
new_model.add(BatchNormalization())
new_model.add(Dense(512, activation='relu'))
new_model.add(BatchNormalization())
#new_model.add(Dropout(0.5))
new_model.add(Dense(10, activation='softmax'))

new_model.summary()

lr_check = ReduceLROnPlateau(monitor='loss', factor=0.01, patience=10)
cb = EarlyStopping(patience=5, restore_best_weights=True)

new_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# new_model.fit_generator(generator=batch_data, steps_per_epoch=batch_data.n, epochs=1, callbacks=[callback, lr_check], validation_data=val_data, validation_steps=val_data.n)

train_batch = imd.flow(train_data, data_label, batch_size=64)

new_model.fit_generator(train_batch, steps_per_epoch=train_batch.n, epochs=3, callbacks=[cb, lr_check])

predictions = new_model.predict_classes(test_data, verbose=1)

Prediction['ImageId'] = list(range(1, len(predictions)+1))
Prediction['label'] = predictions

Prediction.to_csv('Submission_New.csv', index=False)