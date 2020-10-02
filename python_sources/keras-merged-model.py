import numpy as np
import pandas as pd

from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Merge
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.core import Dense, Activation, Dropout, Reshape, Flatten
from keras.utils.np_utils import to_categorical

data = pd.read_csv('../input/train.csv')
images = data.iloc[:,1:].values
images = images.astype(np.float)
images = np.multiply(images, 1.0 / 255.0)
print ('Train\'s shape =>({0[0]},{0[1]})'.format(images.shape))

labelsFlat = data[[0]].values
classes = len(np.unique(labelsFlat))
labelsCategorical = to_categorical(labelsFlat,classes)
labelsCategorical = labelsCategorical.astype(np.uint8)
print ('Train\'s classes =>({0})'.format(classes))

test = pd.read_csv('../input/test.csv').values
testX = test
testX = testX.astype(np.float)
testX = np.multiply(testX, 1.0 / 255.0)
print ('Test\'s shape =>({0[0]},{0[1]})'.format(testX.shape))

leftBranch = Sequential()
leftBranch.add(Reshape((1,28,28), input_shape=(784,)))
leftBranch.add(Convolution2D(classes, 3, 1, activation='relu'))
leftBranch.add(MaxPooling2D((2, 2), strides=(2, 2)))
leftBranch.add(Flatten())

rightBranch = Sequential()
rightBranch.add(Reshape((1,28,28), input_shape=(784,)))
rightBranch.add(Convolution2D(classes, 1, 3, activation='relu'))
rightBranch.add(MaxPooling2D((2, 2), strides=(2, 2)))
rightBranch.add(Flatten())

centralBranch = Sequential()
centralBranch.add(Reshape((1,28,28), input_shape=(784,)))
centralBranch.add(Convolution2D(classes, 5, 5, activation='relu'))
centralBranch.add(MaxPooling2D((2, 2), strides=(2, 2)))
centralBranch.add(Flatten())

merged = Merge([leftBranch, centralBranch, rightBranch], mode='concat')

model = Sequential()
model.add(merged)
model.add(Dense(28*3, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(28, activation='relu'))
model.add(Dense(input_dim=10, output_dim=classes))
model.add(Activation("softmax"))

sgd = SGD(lr=0.5, momentum=0.0, decay=0.0, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
model.fit([images,images,images], labelsCategorical,
            nb_epoch=5, batch_size=100, verbose=2)

yPred = model.predict_classes([testX,testX,testX])

np.savetxt('dr.csv', np.c_[range(1,len(yPred)+1),yPred], delimiter=',',
            header = 'ImageId,Label', comments = '', fmt='%d')