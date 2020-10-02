import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import random as rnd

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras import backend as K

import os

batch_size = 256
num_classes = 36
epochs = 10

# balanced_train_path = '../input/emnist-balanced-train.csv'
# balanced_test_path = '../input/emnist-balanced-test.csv'
# byclass_train_path = '../input/emnist-byclass-train.csv'
# byclass_test_path = '../input/emnist-byclass-test.csv'
# bymerge_train_path = '../input/emnist-bymerge-train.csv'
# bymerge_test_path = '../input/emnist-bymerge-test.csv'
digits_train_path = '../input/emnist-digits-train.csv'
digits_test_path = '../input/emnist-digits-test.csv'
letters_train_path = '../input/emnist-letters-train.csv'
letters_test_path = '../input/emnist-letters-test.csv'
# mnist_train_path = '../input/emnist-mnist-train.csv'
# mnist_test_path = '../input/emnist-mnist-test.csv'

print('Loading data')

# balanced_train_data = pd.read_csv(balanced_train_path)
# balanced_test_data = pd.read_csv(balanced_test_path)
# byclass_train_data = pd.read_csv(byclass_train_path)
# byclass_test_data = pd.read_csv(byclass_test_path)
# bymerge_train_data = pd.read_csv(bymerge_train_path)
# bymerge_test_data = pd.read_csv(bymerge_test_path)
digits_train_data = pd.read_csv(digits_train_path)
digits_test_data = pd.read_csv(digits_test_path)
letters_train_data = pd.read_csv(letters_train_path)
letters_test_data = pd.read_csv(letters_test_path)
# mnist_train_data = pd.read_csv(mnist_train_path)
# mnist_test_data = pd.read_csv(mnist_test_path)

print('data loaded')

d_datas = np.concatenate((digits_train_data.values, digits_test_data.values), axis=0)
rnd.shuffle(d_datas)

l_datas = np.concatenate((letters_train_data.values, letters_test_data.values), axis=0)
rnd.shuffle(l_datas)

print('separaiting data')
split_percent = 85

split_point = len(d_datas) // 100 * split_percent
d_datas_train = d_datas[:split_point]
d_datas_test = d_datas[split_point:]

split_point = len(l_datas) // 100 * split_percent
l_datas_train = l_datas[:split_point]
l_datas_test = l_datas[split_point:]


print(d_datas_train.shape)
print(d_datas_test.shape)
print(l_datas_train.shape)
print(l_datas_test.shape)


# Verifiying we got all the letters for the test set
y_set = set(l_datas_test[:, 0:1].flatten())
print(y_set)

print('Preparing letters labels for concatenation')
l_datas_train[:, 0:1] =  l_datas_train[:, 0:1] + 9
l_datas_test[:, 0:1] =  l_datas_test[:, 0:1] + 9

print('Conconcatening datas')
datas_train = np.concatenate((d_datas_train, l_datas_train), axis=0)
rnd.shuffle(datas_train)

x_train = datas_train[:, 1:].astype('float32')
y_train = datas_train[:, 0:1]
print('x_train shape : ', x_train.shape)

datas_test = np.concatenate((d_datas_test, l_datas_test), axis=0)

x_test = datas_test[:, 1:].astype('float32')
y_test = datas_test[:, 0:1]
print('x_test shape : ', x_test.shape)

print('transposing data')

# for dense
x_train = np.array(list(map(lambda x : x.reshape(28, 28).transpose().flatten(), x_train)))
x_test = np.array(list(map(lambda x : x.reshape(28, 28).transpose().flatten(), x_test)))
'''
# for conv2d
x_train = np.array(list(map(lambda x : x.reshape(28, 28).transpose().reshape(28, 28, 1), x_train)))
x_test = np.array(list(map(lambda x : x.reshape(28, 28).transpose().reshape(28, 28, 1), x_test)))
'''

print('Verifying data')
# Diffrent test to assert manually that the data are correctly prepared to be proceed in the training phase

'''
# Print the i data 
i = 54000
plt.imshow(x_test[i].reshape(28, 28))
print('data ', i, ' : ', y_test[i])
'''
'''
print('First letter of ...')
for i in range(0, y_train.size, rnd.randint(1, 10)):
    if (y_train[i][0] > 9) : 
        print(i, ':' ,y_train[i])
        plt.imshow(x_train[i].reshape(28, 28))
        break
'''
'''
print(y_test.shape)
y_set = set(y_test.flatten())
print('Shape of ... : ', y_set)
'''

x_train /= 255
x_test /= 255

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

print('data are ready')

print('Construction of Keras Model')
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# for dense
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,))) # la taille de la première couche doit toujours être précisée
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'),)
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'),)
model.add(Dense(num_classes, activation='softmax')) # Softmax : fct sortant un vecteur dont la somme fait 1 

'''
# for conv2D
model = Sequential()
model.add(Conv2D(64, (5, 5),input_shape=(28, 28, 1), activation='relu'))
model.add(MaxPooling2D((2, 2), (2, 2)))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'),)
model.add(Dense(num_classes, activation='softmax'))
'''

model.summary()

print('Compiling the model...')

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(lr=0.0001),
              metrics=['accuracy'])
              
print('Let\'s the training  begins !')

history = model.fit(x_train, y_train,
              batch_size=batch_size, # Paquet de données envoyé en une fois dans le NN
              epochs=epochs, # nb de fois que les données de train passe dans le nn
              verbose=1,
              validation_split=0.1) # pourcentage de données utilisé pour la validation
score = model.evaluate(x_test, y_test, verbose=0) # Evaluation du model avec les données de tests
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

model.save('model_name.h5') # Uniquement dispo sous /output & quand est commit






























