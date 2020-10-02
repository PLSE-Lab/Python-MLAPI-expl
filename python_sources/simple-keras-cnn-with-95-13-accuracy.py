# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

# Method to load data and images
def load_data_and_labels(data, labels):
    data_df = pd.read_csv(data, header=None)
    X = data_df.values.reshape((-1,28,28,4)).clip(0,255).astype(np.uint8)
    labels_df = pd.read_csv(labels, header=None)
    Y = labels_df.values.getfield(dtype=np.int8)
    return X, Y
    
x_train, y_train = load_data_and_labels(data='../input/X_train_sat6.csv',
                                        labels='../input/y_train_sat6.csv')
x_test, y_test = load_data_and_labels(data='../input/X_test_sat6.csv',
                                      labels='../input/y_test_sat6.csv')

print(pd.read_csv('../input/sat6annotations.csv', header=None))

# Print shape of all training, testing data and labels
# Labels are already loaded in one-hot encoded format
print(x_train.shape) # (324000, 28, 28, 4)
print(y_train.shape) # (324000, 6)
print(x_test.shape)  # (81000, 28, 28, 4)
print(y_test.shape)  # (81000, 6)

import tensorflow as tf
from keras.callbacks import TensorBoard
from keras.layers import Conv2D, MaxPool2D, Dense, Dropout, Flatten
from keras.models import Sequential

# Create our model
tf.reset_default_graph()

model = Sequential()
model.add(Conv2D(16, (3,3), activation='relu', input_shape=(28,28,4)))
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.5))
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(6, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())
'''
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_1 (Conv2D)            (None, 26, 26, 16)        592       
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 24, 24, 32)        4640      
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 12, 12, 32)        0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 12, 12, 32)        0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 10, 10, 32)        9248      
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 8, 8, 64)          18496     
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 4, 4, 64)          0         
_________________________________________________________________
dropout_2 (Dropout)          (None, 4, 4, 64)          0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 1024)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 128)               131200    
_________________________________________________________________
dropout_3 (Dropout)          (None, 128)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 6)                 774       
=================================================================
Total params: 164,950
Trainable params: 164,950
Non-trainable params: 0
'''

tbcallback = TensorBoard(log_dir='./Graph/', histogram_freq=1, write_graph=True, write_grads=True)

model.fit(X_train, y_train, batch_size=200, epochs=6, verbose=1, validation_data=(X_test, y_test), callbacks=[tbcallback])
'''
....
....
Epoch 6/6
324000/324000 [==============================] - 287s 886us/step - loss: 0.0638 - acc: 0.9781 - val_loss: 0.1268 - val_acc: 0.9513
'''

# To save our trained model and weights
# Save our model and weights
json_file = model.to_json()
with open('deepsat6-6epochs-model.json', 'w') as f:
    f.write(json_file)
model.save_weights('deepsat6-6epochs-weights.h5')
