# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

import os
import numpy as np
from tempfile import TemporaryFile
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import model_selection
import keras
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l1
from quiver_engine import server

path = '/home/argus/data/kaggle/MultiSpectralImages/'

Labels = pd.read_csv(path+'Labels.csv')

wrong_FileName = []

X_ = []

input_size = 64
input_channels = 1
num_chanel = 10

for i in range(len(Labels.iloc[:,0])):
    try:
        Xi = pd.read_csv(path+str(Labels.iloc[i,1]))
        for chanel in range(num_chanel):
            img = (np.array(Xi.iloc[:,2:12])[:,chanel]).reshape(350,350)[100:-100,100:-100]
            cv2.imwrite(path+str(Labels.iloc[i,1])[:-4]+'_chanel_'+str(chanel)+'.png' ,img)
            a = cv2.resize(cv2.imread(path+ str(Labels.iloc[i,1])[:-4]+'_chanel_'+'{}.png'.format(chanel)),
                           (input_size, input_size), interpolation = cv2.INTER_AREA)
            cv2.imwrite(path+str(Labels.iloc[i,1])[:-4]+'_chanel_'+str(chanel)+'res.png',a)
            x = np.array(cv2.imread(path+str(Labels.iloc[i,1])[:-4]+'_chanel_'+str(chanel)+'res.png',
                                    cv2.IMREAD_GRAYSCALE))
            X_.append(x)
    except IOError:
        wrong_FileName += [i]

number_of_img = len(Labels.iloc[:,0])-len(wrong_FileName)
X_nparray = np.array(X_) / 255
X = X_nparray.reshape((number_of_img, num_chanel, input_size , input_size ))
Labels_tru = Labels[Labels.loc[:,'FileName'] != str(Labels.iloc[wrong_FileName,1].values)[2:-2]] 
class_le = LabelEncoder()
y = class_le.fit_transform(Labels_tru.iloc[:,0].values)

ohe = OneHotEncoder()
Y = ohe.fit_transform(y.reshape(-1,1)).toarray()

outX = TemporaryFile()
outY = TemporaryFile()
np.save(outX, X)
np.save(outY, Y)

train_data, test_data, train_labels, test_labels = model_selection.train_test_split(X, Y,
                                                                                   test_size = 0.2, random_state = 0)
batch_size = 120
num_classes = 20
epochs = 400
input_shape = (num_chanel, input_size, input_size)
l1_lambda = 0.00003

"""
datagen = ImageDataGenerator(
            width_shift_range=0.1, # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1, # randomly shift images vertically (fraction of total height)
            channel_shift_range = 4)
datagen.fit(train_data)
"""

model = Sequential()
model.add(BatchNormalization(input_shape=input_shape))
model.add(Conv2D(64, (2, 2), W_regularizer=l1(l1_lambda), activation='relu'))
model.add(Conv2D(64, (2, 2), W_regularizer=l1(l1_lambda), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))
model.add(Conv2D(128, (2, 2), W_regularizer=l1(l1_lambda), activation='relu'))
model.add(Conv2D(128, (2, 2), W_regularizer=l1(l1_lambda), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))
model.add(Conv2D(256, (2, 2), W_regularizer=l1(l1_lambda), activation='relu'))
model.add(Conv2D(256, (2, 2), W_regularizer=l1(l1_lambda), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))
model.add(Conv2D(512, (2, 2), W_regularizer=l1(l1_lambda), activation='relu'))
model.add(Conv2D(512, (2, 2), W_regularizer=l1(l1_lambda), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(2048, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2048, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
model.summary()

opt = keras.optimizers.Adam()
model.compile(loss='categorical_crossentropy',
              optimizer=opt, metrics=['accuracy'])

tensorboard = TensorBoard(log_dir='./logs', write_graph=True)
checkpoint = ModelCheckpoint(filepath=os.path.join(path, "model-{epoch:02d}.h5"))
"""

model.fit_generator(datagen.flow(train_data, train_labels,
          batch_size=batch_size),
          epochs=epochs,
          verbose=1,
          validation_data=(test_data, test_labels),
          callbacks=[tensorboard])
"""
model.fit(train_data, train_labels,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(test_data, test_labels),
          callbacks=[tensorboard, checkpoint])
score = model.evaluate(test_data, test_labels, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Any results you write to the current directory are saved as output.