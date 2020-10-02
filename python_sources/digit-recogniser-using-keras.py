# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Lambda, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import ReduceLROnPlateau, EarlyStopping

training_set = pd.read_csv('../input/train.csv')
test_set = pd.read_csv('../input/test.csv').values

from keras.utils.np_utils import to_categorical
y_train = training_set['label'].values
y_train = to_categorical(y_train, 10)

training_set = training_set.iloc[:,1:].values

#reshaping the data
X_train = training_set.reshape(training_set.shape[0], 28,28,1)
test_set = test_set.reshape(test_set.shape[0], 28,28,1)

#model implementation

def Model(n_classes):
    classification = Sequential()
    classification.add(Lambda(lambda x: x/255. - .5, input_shape=(28,28,1)))
    
    classification.add(Conv2D(30, (5,5), activation='relu'))
    classification.add(MaxPooling2D(pool_size=(2,2)))
    
    classification.add(Flatten())
    
    classification.add(Dense(128))
    classification.add(Activation('relu'))
    classification.add(Dropout(0.2))
    
    classification.add(Dense(50))
    classification.add(Activation('relu'))
    classification.add(Dropout(0.2))
    
    classification.add(Dense(n_classes))
    classification.add(Activation('softmax'))

    return classification 

# training

EPOCHS = 10
BATCH_SIZE = 1000

model = Model(10)

reduce_lr =  ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)
earlyStopping = EarlyStopping(monitor = 'val_loss', patience=10, verbose=0)

#compile and learn
model.compile('adam', 'categorical_crossentropy', ['accuracy'])
model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, shuffle=True, validation_split=0.2, callbacks=[reduce_lr, earlyStopping])

# checking sample submission
idx = np.arange(1, test_set.shape[0]+1, 1)
predictions = model.predict_classes(test_set, verbose=0).flatten()
submission = pd.DataFrame({"ImageId": idx, "Label": predictions})
submission.to_csv('benchmark.csv', index=False)