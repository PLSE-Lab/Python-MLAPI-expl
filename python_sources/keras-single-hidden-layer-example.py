import numpy as np
import pandas as pd 
from random import randint
from sklearn.preprocessing import MinMaxScaler

# Building training dataset

train_labels = []
train_samples = []

# generates random datapoints

for i in range(50):
    random_younger = randint(13, 64)
    train_samples.append(random_younger)
    train_labels.append(1)
    
    random_older = randint(65, 100)
    train_samples.append(random_older)
    train_labels.append(0)

for i in range(1000):
    random_younger = randint(13, 64)
    train_samples.append(random_younger)
    train_labels.append(0)
    
    random_older = randint(65, 100)
    train_samples.append(random_older)
    train_labels.append(1)

train_labels = np.array(train_labels)
train_samples = np.array(train_samples)

# scales ages from 0 to 1

scaler = MinMaxScaler(feature_range=(0,1))
scaled_train_samples = scaler.fit_transform((train_samples).reshape(-1,1))

# Building model and training it on data

import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy

# Sequential model with 1 hidden layer of 32 neurons
model = Sequential([
        Dense(16, input_shape=(1,), activation='relu'),
        Dense(32, activation='relu'),
        Dense(2, activation='softmax')
])

# Using Adam optimizer and categorical cross entropy as loss function
model.compile(Adam(lr=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# validation data split at 10 percent of original data
model.fit(scaled_train_samples, train_labels, validation_split=0.1, batch_size=10, epochs=50, shuffle=True, verbose=2)

# Building test dataset

test_labels = []
test_samples = []

# generates random datapoints

for i in range(10):
    random_younger = randint(13, 64)
    test_samples.append(random_younger)
    test_labels.append(1)
    
    random_older = randint(65, 100)
    test_samples.append(random_older)
    test_labels.append(0)

for i in range(200):
    random_younger = randint(13, 64)
    test_samples.append(random_younger)
    test_labels.append(0)
    
    random_older = randint(65, 100)
    test_samples.append(random_older)
    test_labels.append(1)

test_labels = np.array(test_labels)
test_samples = np.array(test_samples)

# scales ages from 0 to 1

scaler = MinMaxScaler(feature_range=(0,1))
scaled_test_samples = scaler.fit_transform((test_samples).reshape(-1,1))

# Predict

predictions = model.predict(scaled_test_samples, batch_size=10, verbose=0)
rounded_predictions = model.predict_classes(scaled_test_samples, batch_size=10, verbose=0)

# Confusion Matrix

from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt

cm = confusion_matrix(test_labels, rounded_predictions)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

cm_plot_labels = ['no_side_effects', 'had_side_effects']
plot_confusion_matrix(cm, cm_plot_labels, title='Confusion Matrix')

# Save model with complete architecture, weights and optimizer
""" 
model.save('medical_trial_model.h5')

from keras.models import load_model
new_model = load_model('medical_trial_model.h5')

new_model.summary()
new_model.get_weights()
new_model.optimizer() 
"""

# Save model with only architecture
"""
json_string = model.to_json()
yaml_string = model.to_yaml()

json_string

from keras.models import model_from_json
model_architecture = model_from_json(json_string)

from keras.models import model_from_yaml
model_architecture = model_from_yaml(yaml_string)

model_architecture.summary()
"""

# Save model with only weights
"""
model.save_weights('my_model_weights.h5')
model.load_weights('my_model_weights.h5')
"""
