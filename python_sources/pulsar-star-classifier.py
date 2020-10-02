# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import os
import csv
import keras
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.layers import Dense
from keras.models import Sequential

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

def load_dataset(file):
    with open(file, 'r') as work_file:
            reader = csv.reader(work_file)
            next(reader)
            reader = list(reader)
            total = len(reader)
            features = len(reader[0][:8])
            x = np.zeros((len(reader), features))
            y = np.zeros((len(reader), 1))

            for index, val in enumerate(reader):
                x[index] = val[:features]
                y[index] = val[-1]
            return x, y, features
            
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        data_file = os.path.join(dirname, filename)

x, y, features = load_dataset(data_file)
norm = np.linalg.norm(x)
x = x/norm
# hot encoding vectors
y_bin = keras.utils.to_categorical(y)

model = Sequential()
model.add(Dense(15, activation='relu', input_dim=features))
model.add(Dense(11, activation='relu', input_dim=15))
model.add(Dense(5, activation='tanh', input_dim=11))
model.add(Dense(2, activation='softmax', input_dim=5))
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer='adam',
              metrics=['accuracy'])
model.fit(x, y_bin,
          batch_size=128,
          epochs=20,
          validation_split = 0.2)

    

# Any results you write to the current directory are saved as output.