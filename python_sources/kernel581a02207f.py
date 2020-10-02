# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from keras import regularizers

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
df = pd.read_csv('/kaggle/input/kpid/train.csv')
test = pd.read_csv('/kaggle/input/kpid/test.csv')
sample = pd.read_csv('/kaggle/input/kpid/sample_submission.csv')
df.head()


min_max_scaler = preprocessing.MinMaxScaler()
#Scaling The Training Data
x = df.values
x_scaled = min_max_scaler.fit_transform(x)
df = pd.DataFrame(x_scaled)
#Scaling The Testing Data
y = test.values
y_scaled = min_max_scaler.fit_transform(y)
test = pd.DataFrame(y_scaled)
df.head()

train, dev = train_test_split(df, test_size=0.2)

hidden_units=300
learning_rate=0.005 #Learning rate was quite optimal
hidden_layer_act='tanh'
output_layer_act='sigmoid'
no_epochs=100 #Increasing The epochs would overfit
bsize = 128 #Batch Size Of 128 

model = Sequential()

model.add(Dense(hidden_units, input_dim=8, activation=hidden_layer_act))
model.add(Dense(hidden_units, activation=hidden_layer_act))
model.add(Dense(1, activation=output_layer_act))

#epsilon=None, 

adam = optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, decay=0.0, amsgrad=False)
model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['acc'])

train_x=train.iloc[:,0:8]
train_y=train.iloc[:,8]

model.fit(train_x, train_y, epochs=no_epochs, batch_size= bsize,  verbose=2)

val_loss, val_acc = model.evaluate(dev.iloc[:,0:8], dev.iloc[:,8])
print("Validation Loss : ", val_loss)
print("Validation Acc : ",val_acc)

test_x=test.iloc[:,0:8]
predictions = model.predict(test_x)
print(predictions)

rounded = [int(round(x[0])) for x in predictions]
print(rounded)
sample.diabetes = rounded
sample.to_csv('submission.csv',index = False)

# Any results you write to the current directory are saved as output.