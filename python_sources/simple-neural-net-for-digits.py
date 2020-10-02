# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout
from keras.utils import np_utils

data = pd.read_csv('../input/train.csv')
labels = data.ix[:,0].values.astype('int32')
pixels = data.ix[:,1:].values.astype('float32')
test = pd.read_csv('../input/test.csv').values.astype('float32')

print(labels.shape)
print(pixels.shape)

labels = np_utils.to_categorical(labels)

divide_by = np.max(pixels)
pixels /= divide_by
test /= divide_by

mean = np.mean(pixels)
pixels -= mean
test -= mean 

print (labels.shape)

input_dim = pixels.shape[1]
nb_classes = labels.shape[1]

model = Sequential()
model.add(Dense(128, input_dim=input_dim))
model.add(Activation('relu'))
model.add(Dropout(0.15))

model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.15))

model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',optimizer='rmsprop')

model.fit(pixels, labels, batch_size=10,epochs=10,verbose=1,validation_split=0.1)

preds = model.predict_classes(test, verbose=1)

def write_preds(preds, fname):
    pd.DataFrame({"ImageId":list(range(1,len(preds)+1)),"Label":preds}).to_csv(fname, index=False, header=True)
    
write_preds(preds, "keras.csv")