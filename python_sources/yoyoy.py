import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils.np_utils import to_categorical
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
# The competition datafiles are in the directory ../input
# Read competition data files:
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
train_image = (train.iloc[:,1:].values).astype('float32')
train_labels = (train.iloc[:,0].values).astype('int32')
train_image=train_image/255;
test = test/255

# Write to the log
print("Training set has {0[0]} rows and {0[1]} columns".format(train.shape))
print("Test set has {0[0]} rows and {0[1]} columns".format(test.shape))
print(train.shape)
# Any files you write to the current directory get shown as outputs
model = Sequential()
model.add(Dense(32, activation='relu', input_dim=784))
model.add(Dense(16,activation='relu'))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
labels = to_categorical(train_labels, num_classes=10)
history=model.fit(train_image,labels,validation_split = 0.05, epochs = 25, batch_size = 64)
history_dict = history.history
history_dict.keys()
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, len(loss_values) + 1)
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, len(loss_values) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss_values, 'bo')
# b+ is for "blue crosses"
plt.plot(epochs, val_loss_values, 'b+')
plt.xlabel('Epochs')
plt.ylabel('Loss')

plt.show()