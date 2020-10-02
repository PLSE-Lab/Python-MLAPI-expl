### Using a neural network to perfectly predict the type of flower

##################
## iris dataset ##
##################

import numpy as np
import pandas as pd

from sklearn import datasets
iris1 = datasets.load_iris()

## I obtained the dataset from sklearn

X2 = iris1.data
Y2 = iris1.target


X1 = pd.DataFrame(iris1.data)
Y1 = pd.DataFrame(iris1.target)

X1.info()
X1.shape

Y1.info()
Y1.shape

## no nans

# sepal_lenght, sepal_width, pedal_lenght, pedal_width

# retreiving numpy arrays

sepal_lenght1 = X2[:,0]
sepal_width1 = X2[:,1] 
pedal_lenght1 = X2[:,2] 
pedal_width1 = X2[:,3]

# renaming the columns in the pandas df

X1["sepal_lenght"] = X1[0]
X1["sepal_width"] = X1[1]
X1["pedal_lenght"] = X1[2]
X1["pedal_width"] = X1[3]

X3 = X1[["sepal_lenght", "sepal_width", "pedal_lenght", "pedal_width"]]

##############
## Analysis ##
##############


## neural networks

import pandas as pd
import numpy as np
import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split


X = X3
Y = Y1


cols = X.shape[1]

model1 = Sequential()

#first layer
model1.add(Dense(20, activation = "relu", input_shape = (cols, )))
# output layer
model1.add(Dense(1))


## predicting

X3_train, X3_test, y3_train, y3_test = train_test_split(X, Y, test_size = 0.3, random_state=42)

cols2 = X3_train.shape[1]

model2 = Sequential()

model2.add(Dense(20, activation = "relu", input_shape = (cols2, )))
# output layer
model2.add(Dense(1))

model2.compile(optimizer = 'adam', loss = 'mean_squared_error')

early_stopping_monitor = EarlyStopping(patience = 20)

model2.fit(X3_train, y3_train, validation_split=0.3, epochs = 400,
          callbacks = [early_stopping_monitor])

pred3 = model2.predict(X3_test)

list_pred3 = []

for a in pred3:
    if a <= 0.5:
        list_pred3.append(0)
    elif a <= 1.5:
        list_pred3.append(1)
    else:
        list_pred3.append(2)

print(list_pred3)

y3_test["list_pred3"] = list_pred3

y3_test["deviation"] = y3_test[0] - y3_test["list_pred3"]

y3_test.head()
print(np.mean(y3_test["deviation"]))
## the model makes no mistakes at all :D!

#########################################################
#########################################################
