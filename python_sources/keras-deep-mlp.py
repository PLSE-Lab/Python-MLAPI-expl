"""
~*~* Deep Multilayer Perceptron using Keras/Tensorflow *~*~

Code for a deep (3 hidden layers) multilayer perceptron classifier for the 
MNIST handwritten digit data in the context of Kaggle's 'Digit Recognizer' 
competition.
"""
import pandas as pd
import numpy as np

from keras.utils.np_utils import to_categorical # for labels
from keras.models import Sequential
from keras.layers import Dense

np.random.seed(237)

# Load data
train_orig = pd.read_csv('../input/train.csv')
X_test = pd.read_csv('../input/test.csv')

# Hold out 4200 random images (10%) as a validation set
valid = train_orig.sample(n = 4200, random_state = 555)
train = train_orig.loc[~train_orig.index.isin(valid.index)]

# delete original train set to clear space
del train_orig

X_train = train.drop(['label'], axis=1)
labels_train = train['label']

X_valid = valid.drop(['label'], axis=1)
labels_valid = valid['label']

# clear more space
del train, valid

# Normalize
X_train = X_train.astype('float32') / 255.
X_valid = X_valid.astype('float32') / 255.
X_test = X_test.astype('float32') / 255.

# one hot encoding of digit labels
labels_train = to_categorical(labels_train)
labels_valid = to_categorical(labels_valid)

# Take values out of pandas dataframes
X_train = X_train.values
X_valid = X_valid.values
X_test = X_test.values

# K = 10 digit classes; 784 px images
K_classes = 10
px = X_train.shape[1]

# construct sequential model
deep_mlp = Sequential()
# Input & hidden layer #1 with 784 nodes
deep_mlp.add(Dense(px, input_dim=px, 
                   kernel_initializer='normal', 
                   activation='relu')) 
# Hidden layer #2 with 784/2 = 392 nodes                 
deep_mlp.add(Dense(int(px/2), input_dim=px, 
                   kernel_initializer='normal', 
                   activation='relu'))    
# Hidden layer with 784/4 = 196 nodes       
deep_mlp.add(Dense(int(px/4), input_dim=int(px/2), 
                   kernel_initializer='normal', 
                   activation='relu'))   
 # Output layer with 10 nodes 
deep_mlp.add(Dense(K_classes, kernel_initializer='normal', 
                   activation='sigmoid'))
# Compile model
deep_mlp.compile(loss='categorical_crossentropy', 
                 optimizer='adam', metrics=['accuracy'])

# Fit the model
deep_mlp_fit = deep_mlp.fit(X_train, labels_train, 
                            validation_data=(X_valid, labels_valid), 
                            epochs=30, 
                            batch_size=100, 
                            verbose=2)

# make predictions on test
deepMLP_test_preds = deep_mlp.predict(X_test)

# predict as the class with highest probability
deepMLP_test_preds = np.argmax(deepMLP_test_preds,axis = 1)

# put predictions into pandas series
deepMLP_test_preds = pd.Series(deepMLP_test_preds,name='label')

# add to dataframe with 'ImageId' column
deepMLP_for_csv = pd.concat([pd.Series(range(1,28001),name = 'ImageId'), 
                             deepMLP_test_preds],axis = 1)

# send to csv
deepMLP_for_csv.to_csv('deep_mlp_keras.csv', index=False)