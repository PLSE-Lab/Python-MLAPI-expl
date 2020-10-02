# %% [code]

import matplotlib.pyplot as plt
import numpy as np
# import keras
# from keras.models import Sequential
# from keras.layers import Dense, Dropout
# from tensorflow.python.keras.models import Sequential
# from tensorflow.python.keras.layers import Dense, Dropout

from keras.models import Sequential
from keras.layers import Dropout,Dense
from keras import regularizers
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Read from dat file

path = '/kaggle/input/coris.dat'
raw_dat = np.genfromtxt(path, skip_header = 3, delimiter = ',', dtype = float)
# Random split
randomize = np.arange(len(raw_dat))
np.random.shuffle(randomize)
raw_dat = raw_dat[randomize]

train_len = int(round(len(raw_dat) * 0.5))  #split training data and testing data
X_train = raw_dat[0:train_len,1:10]
Y_train = raw_dat[0:train_len,10:]
X_test = raw_dat[train_len:,1:10]
Y_test = raw_dat[train_len:,10:]

mean = np.mean(X_train, axis = 0)
std_dev = np.std(X_train, axis = 0)
X_train = np.divide(np.subtract(X_train, mean), std_dev)
X_test = np.divide(np.subtract(X_test, mean), std_dev)

my_lambda = [0.1,
             0.01,
             0.001,
             0.0001,
             0.00001,
             0.000001,
             0.0000001,
             0.00000001,
             0.000000001,
             0.0000000001,
             0.00000000001,
             0.000000000001,
             0.0000000000001,
             0.00000000000001,
             0.000000000000001,
             0.0000000000000001]

train_loss = []
test_loss = []

itera = [1]
for ite in range(len(my_lambda)):
    model = Sequential()
    model.add(Dense(20,
                kernel_regularizer=regularizers.l2(my_lambda[ite]),
                #activity_regularizer=regularizers.l1(my_lambda[ite]),
                activation = 'relu', input_dim = 9))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation = 'sigmoid'))

    model.compile(loss = 'binary_crossentropy',
              optimizer = 'rmsprop',
              metrics = ['accuracy'])
    model.fit(X_train, Y_train,
          epochs = 20,
          batch_size = 40)

    test_score = model.evaluate(X_test, Y_test, batch_size = 40)
    #print("test loss, test accuracy:",test_score[0])
    test_loss.append(test_score[0])
    train_score = model.evaluate(X_train, Y_train, batch_size = 40)
    train_loss.append(train_score[0])
print(test_loss)
print(train_loss)




plt.figure()
plt.semilogx(my_lambda, train_loss, "x-", label="train loss", lw = 2)
plt.semilogx(my_lambda, test_loss, "+-", label="test loss", lw = 2)
plt.ylim(0,1.5)
plt.xlabel('lambda')
plt.ylabel('loss')
plt.grid(True)
plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.)
plt.title("Lambda vs. Empirical Loss")
plt.show()