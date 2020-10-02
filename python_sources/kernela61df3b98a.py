# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

from sklearn import preprocessing
from sklearn.cross_validation import train_test_split


from preprocessing import *

data=[]
labels=[]
for element in os.listdir('data_spectrogram_class_used/6'):
     if os.path.isdir(element):
         print("'%s' un dossier" % element)
     else:
         labels.append(0)
         data.append(np.load(('data_spectrogram_class_used/6/'+element)))

for element in os.listdir('data_spectrogram_class_used/7'):
     if os.path.isdir(element):
         print("'%s' un dossier" % element)
     else:
         #print("'%s' est un fichier" % element)
         labels.append(1)
         data.append(np.load(('data_spectrogram_class_used/7/'+element)))
         
for element in os.listdir('data_spectrogram_class_used/8'):
     if os.path.isdir(element):
         print("'%s' un dossier" % element)
     else:
         #print("'%s' est un fichier" % element)
         labels.append(2)
         data.append(np.load(('data_spectrogram_class_used/8/'+element)))
print(len(data))
print(len(labels))
data = np.array(data)
labels = np.array(labels)

X=np.array(data)
Y=np.array(labels)
lb=preprocessing.LabelEncoder()
yy=np_utils.to_categorical(lb.fit_transform(Y)) #variable encodée des labels
#print(X[1].shape) #chaque matrice est de taille 129*186

num_labels = yy.shape[1] #3
filter_size = 2

# build model
model = Sequential()

model.add(Dense(50, input_shape=(129,186))) #seulement 50 units au lieu de 256
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(num_labels))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

aTrain,aTest,bTrain,bTest=train_test_split(X,yy,test_size=0.2)
model.fit(aTrain, bTrain, batch_size=15, epochs=100, validation_data=(aTest, bTest)) #batch_size=32, epochs=100
#model.summary()