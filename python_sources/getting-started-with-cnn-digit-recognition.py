# Import required packages
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.layers.normalization import BatchNormalization
from keras.utils import to_categorical

# Setting parameters
batch_size = 64
epochs = 20

# Read the csv files
train = pd.read_csv("../input/train.csv")
to_submit = pd.read_csv("../input/test.csv")
sample_submission = pd.read_csv("../input/sample_submission.csv")

# Get X and y for training + one-hot encoding for y
X = train.drop(['label'], axis=1)
y = to_categorical(train['label'])

# Train, Test and Validate Split + Reshape + Rescale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=123)
X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train, test_size=0.25, stratify=y_train, random_state=123)
X_train = X_train.values.reshape(-1,28,28,1) / 255
X_test = X_test.values.reshape(-1,28,28,1) / 255
X_validate = X_validate.values.reshape(-1,28,28,1) / 255
to_submit = to_submit.values.reshape(-1,28,28,1) / 255

# Build the model
model = Sequential()
model.add(Conv2D(32, kernel_size=3, padding='same', activation='relu', input_shape=(28,28,1)))
model.add(MaxPool2D((2,2), padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(64, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPool2D((2,2), padding='same'))
model.add(Conv2D(128, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPool2D((2,2), padding='same'))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(10, activation = 'softmax'))

model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

# Train the model and see result
training_info = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(X_validate, y_validate))
evaluation_result = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', evaluation_result[0])
print('Test accuracy:', evaluation_result[1])

# Table
test_set_prediction = model.predict(X_test)
test_set_prediction = np.argmax(np.round(test_set_prediction), axis=1)
target_names = ["Class {}".format(i) for i in range(10)]
print(classification_report(np.argmax(y_test, axis=1), test_set_prediction, target_names=target_names))

# Get information regarding the training process and plot results
accuracy = training_info.history['acc']
val_accuracy = training_info.history['val_acc']
loss = training_info.history['loss']
val_loss = training_info.history['val_loss']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'bo', label='Training set accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation set accuracy')
plt.title('Training and Validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training set loss')
plt.plot(epochs, val_loss, 'b', label='Validation set loss')
plt.title('Training and Validation loss')
plt.legend()
plt.show()

# Save the model
model.save('model_v1.h5py')

# Predict the classes and save the results
prediction = model.predict(to_submit)
predicted_classes = np.argmax(np.round(prediction), axis=1)
submission = sample_submission.copy()
submission['Label'] = predicted_classes
submission.to_csv('submission.csv', index=False)