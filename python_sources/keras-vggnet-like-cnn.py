"""
~*~* VGGnet-like CNN using Keras/Tensorflow *~*~

Code for a CNN classifier with VGGnet-like architecture for the 
MNIST handwritten digit data in the context of Kaggle's 'Digit Recognizer' 
competition. See https://arxiv.org/pdf/1409.1556.pdf for more detail on
VGGnet.
"""
import pandas as pd
import numpy as np

from keras.utils.np_utils import to_categorical # for labels
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.optimizers import SGD


from keras import backend as K
K.set_image_dim_ordering('th')

np.random.seed(237)

# Load data
train_orig = pd.read_csv('../input/train.csv')
X_test = pd.read_csv('../input/test.csv')

# Hold out 4200 random images (10%) as a validation set
valid = train_orig.sample(n = 4200, random_state = 555)
train = train_orig.loc[~train_orig.index.isin(valid.index)]

# delete original train set
del train_orig

# separate images & labels
X_train = train.drop(['label'], axis=1)
labels_train = train['label']

X_valid = valid.drop(['label'], axis=1)
labels_valid = valid['label']

# clear more space
del train, valid

# Normalize and reshape
X_train = X_train.astype('float32') / 255.
X_train = X_train.values.reshape(X_train.shape[0], 1, 28, 28).astype('float32')

X_valid = X_valid.astype('float32') / 255.
X_valid = X_valid.values.reshape(X_valid.shape[0], 1, 28, 28).astype('float32')

X_test = X_test.astype('float32') / 255.
X_test = X_test.values.reshape(X_test.shape[0], 1, 28, 28).astype('float32')

# one hot encoding of digit labels
labels_train = to_categorical(labels_train)
labels_valid = to_categorical(labels_valid)

# K = 10 digits classes
K_classes = 10

convnet2 = Sequential()
# [Conv2D]*2 -> MaxPool -> Dropout #1
convnet2.add(Conv2D(32, (3, 3), 
                    input_shape=(1, 28, 28), 
                    padding = 'same', 
                    activation='relu'))
convnet2.add(Conv2D(32, (3, 3), 
                    padding = 'same', 
                    activation='relu')) 
convnet2.add(MaxPooling2D(pool_size=(2, 2)))
convnet2.add(Dropout(0.10)) 

# [Conv2D]*2 -> MaxPool -> Dropout #2
convnet2.add(Conv2D(64, (3, 3), 
                    strides = (2, 2), 
                    padding = 'same', 
                    activation = 'relu'))
convnet2.add(Conv2D(64, (3, 3), 
                    padding = 'same', 
                    activation='relu'))
convnet2.add(MaxPooling2D(pool_size = (2, 2)))
convnet2.add(Dropout(0.10)) 

# [Conv2D]*2 -> MaxPool -> Dropout #3
convnet2.add(Conv2D(64, (3, 3), 
                    strides = (2, 2), 
                    padding = 'same', 
                    activation = 'relu'))
convnet2.add(Conv2D(64, (3, 3), 
                    padding = 'same', 
                    activation='relu'))
convnet2.add(MaxPooling2D(pool_size = (2, 2)))
convnet2.add(Dropout(0.10))

# Flatten -> Dense -> Dense -> Out
convnet2.add(Flatten())
convnet2.add(Dense(256, activation='relu'))
convnet2.add(Dense(128, activation='relu'))
convnet2.add(Dense(K_classes, activation='softmax'))

# Define stochastic gradient descent optimizer parameters & compile
sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True)
convnet2.compile(loss='categorical_crossentropy', 
                 optimizer=sgd, 
                 metrics=['accuracy'])

# Fit the model
convnet2_fit = convnet2.fit(X_train, labels_train, 
                            validation_data=(X_valid, labels_valid), 
                            epochs=10, 
                            batch_size=100, 
                            verbose=2)

# make predictions on test
convnet2_test_preds = convnet2.predict(X_test)

# predict as the class with highest probability
convnet2_test_preds = np.argmax(convnet2_test_preds,axis = 1)

# put prediction in pandas Series
convnet2_test_preds = pd.Series(convnet2_test_preds, name='label')

# Add 'ImageId' columns
convnet2_for_csv = pd.concat([pd.Series(range(1,28001), name = 'ImageId'), 
                              convnet2_test_preds] ,axis = 1)

# write to csv
convnet2_for_csv.to_csv('convnet2_keras.csv', index=False)