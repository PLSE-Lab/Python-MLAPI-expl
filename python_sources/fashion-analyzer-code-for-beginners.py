#Importing the libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Import the dataset

train_set = pd.read_csv("../input/fashionmnist/fashion-mnist_train.csv", sep =',')
test_set = pd.read_csv("../input/fashionmnist/fashion-mnist_test.csv", sep =',')

#Visulizing the dataset

train_set.head()
train_set.tail()
test_set.tail()
train_set.shape
test_set.shape

training = np.array(train_set, dtype = 'float32')
testing = np.array(test_set, dtype = 'float32')

import random
i = random.randint(1, 60000)
plt.imshow(training[i, 1:].reshape(28, 28))
label = training[i, 0]
label


w_grid = 15
l_grid = 15

fig, axes = plt.subplots(l_grid, w_grid, figsize =(17, 17))
axes = axes.ravel()

n_train = len(training)

for i in np.arange(0, w_grid*l_grid):
    index = np.random.randint(0, n_train)
    axes[i].imshow(training[index, 1:].reshape((28, 28)) )
    axes[i].set_title(training[index, 0], fontsize = 8)
    axes[i].axis('off')
    
plt.subplots_adjust(hspace = 0.4)

#Training the model
x_train = training[:, 1:]/255
y_train = training[:, 0]

x_test = testing[:, 1:]/255
y_test = testing[:, 0]

from sklearn.model_selection import train_test_split
x_train, x_validate, y_train, y_validate = train_test_split(x_train, y_train, test_size = 0.2, random_state = 0)

x_train = x_train.reshape(x_train.shape[0], * (28, 28, 1))
x_test = x_test.reshape(x_test.shape[0], * (28, 28, 1))
x_validate = x_validate.reshape(x_validate.shape[0], * (28, 28, 1))


import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.optimizers import Adam
from keras.callbacks import TensorBoard

cnn = Sequential()
cnn.add(Conv2D(32, 3, 3, input_shape = (28, 28, 1), activation = 'relu'))
cnn.add(MaxPooling2D(pool_size =(2, 2)))
cnn.add(Dense(output_dim = 32, activation ='sigmoid'))
cnn.add(Flatten())

cnn.compile(loss = 'sparse_categorical_crossentropy', optimizer = Adam(lr = 0.001), metrics = ['accuracy'])

epochs = 20

cnn.fit(x_train,
       y_train,
       batch_size = 512,
       nb_epoch = epochs,
       verbose = 1,
       validation_data = (x_validate, y_validate))
#Evaluating the result

evaluation = cnn.evaluate(x_test, y_test)
print("Test accuracy: {:.3f}".format(evaluation[1]))


predicted_classes = cnn.predict_classes(x_test)
from sklearn.metrics import classification_report
nb_classes = 10
target_names = ["Class {}".format(i) for i in range(nb_classes)]
print(classification_report(y_test, predicted_classes, target_names = target_names))













+