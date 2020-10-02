'''
  We will use keras dataset reuters
  the steps are 
    load the data
    encode  the data using one-hot encoding
    build the model 
    compile the model and check it on cross valdiation set
    evaluate the model on test set
    
    
    feel free  to tweak the parameters like activation functions, number of layers, number of epochs, batch_size etc.
    don't reduce the layer dimension to less because the class dimension is 46.
'''


from keras.datasets import reuters
from keras import models
from keras import layers
import numpy as np
import matplotlib.pyplot as plt

def to_one_hot(sequences, dimension):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results
    
(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words = 10000)
x_train = to_one_hot(train_data, 10000)
x_test = to_one_hot(test_data, 10000)
y_train = to_one_hot(train_labels, 46)
y_test = to_one_hot(test_labels, 46)


model = models.Sequential()
model.add(layers.Dense(64, activation='tanh', input_shape=(10000,)))
model.add(layers.Dense(64, activation='tanh'))
model.add(layers.Dense(64, activation='tanh'))
model.add(layers.Dense(46, activation='softmax'))


model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

x_val = x_train[:1000]
partial_x_train = x_train[1000:]
y_val = y_train[:1000]
partial_y_train = y_train[1000:]

history = model.fit(partial_x_train, partial_y_train, epochs=9, batch_size=512, validation_data=(x_val, y_val))


loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, 10)
plt.plot(epochs, loss, 'bo', label='Training_loss')
plt.plot(epochs, val_loss, 'b', label='Validation_loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


plt.clf()
acc = history.history['acc']
val_acc = history.history['val_acc']
plt.plot(epochs, acc, 'bo', label='Training_accuracy')
plt.plot(epochs, val_acc, 'b', label='Validaion_accuracy')
plt.legend()
plt.show()

result = model.evaluate(x_test, y_test)
print(result)