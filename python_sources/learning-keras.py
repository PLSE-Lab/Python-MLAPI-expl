import numpy as np
import pandas as pd
from keras.optimizers import SGD
from keras.utils  import to_categorical
from keras.models import Sequential
from keras.layers import Dense

train = pd.read_csv('../input/train.csv')
test  = pd.read_csv('../input/test.csv')

train_x = train.iloc[:,1:].values / 255
train_y = to_categorical(train.iloc[:,0].values, 10)
test_x = test.values / 255

net = Sequential()
net.add(Dense(units=64, activation='relu', input_dim=784))
net.add(Dense(units=10, activation='softmax'))

sgd = SGD(lr=.025, momentum=.5, decay=1e-6)

net.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

history = net.fit(train_x, train_y, epochs=30, batch_size=32, validation_split=0.1)

test_predictions = np.argmax(net.predict(test_x, batch_size=32), axis=1)

df = pd.DataFrame(test_predictions, columns=['Label'])
df.index += 1
df.to_csv('test_predictions.csv', index_label='ImageId')