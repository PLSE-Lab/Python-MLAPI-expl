

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as tk
from tensorflow.keras.layers import Dense,Dropout
import matplotlib.pyplot as plt

# Gathering Data.....
train = pd.read_csv('../input/titanic/train.csv')
test = pd.read_csv('../input/titanic/test.csv')

#train = train.dropna(axis=1)
#test = test.dropna(axis=1)

y = train['Survived']
x = train.drop(['PassengerId', 'Survived', 'Name', 'Ticket','Embarked', 'Cabin'], axis= 1)
x = np.array(x)
y = np.array(y)
x = x[1:,:]
y = y[1:]

pred_id = test['PassengerId']
x_test = test.drop(['PassengerId', 'Name', 'Ticket','Embarked', 'Cabin'], axis= 1)
x_test = np.array(x_test)

# Preprocessing Data
for i in range(len(x)):
    if x[i,1] =='male':
        x[i,1] = 1
    else:
        x[i,1] = 0
        
for i in range(len(x_test)):
    if x_test[i,1] =='male':
        x_test[i,1] = 1
    else:
        x_test[i,1] = 0
        
# !!Very Important to remove nan values...pd.dropna() or pd.fillna() can also be used instead of loops        
for i in range(x.shape[0]):            
    for j in range(x.shape[1]):
        if np.isnan(x[i,j]) == 1:
            x[i,j] = 0
            
for i in range(x_test.shape[0]):
    for j in range(x_test.shape[1]):
        if np.isnan(x_test[i,j]) == 1:
            x_test[i,j] = 0

x_train = np.asarray(x).astype(np.float32)   # else Tensorflow won't like this array
y_train = np.asarray(y).astype(int)
x_test = np.asarray(x_test).astype(np.float32)

# Training Data
model = tk.models.Sequential([

Dense(4096, activation="relu", input_shape=(x_train.shape[1],)),
Dropout(0.2),

Dense(1024, activation="relu"),
Dropout(0.2),

Dense(256, activation="relu"),
Dropout(0.2),

Dense(1, activation="sigmoid")])

model.compile(loss = 'binary_crossentropy', optimizer= 'adam', metrics= ['accuracy'])

history = model.fit(x_train , y_train, epochs=500, verbose= 2)

model.summary()

# Plotting Data
plt.plot(history.history['accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Prediction
pred_survived = model.predict(x_test)
pred_survived = np.rint(pred_survived).reshape((-1)).astype(int)

# create csv
submission = pd.DataFrame(zip(pred_id, pred_survived), columns=['PassengerId', 'Survived'])
submission.to_csv('submission.csv', index=False)