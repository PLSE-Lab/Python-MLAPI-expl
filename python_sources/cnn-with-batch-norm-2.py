import pandas as pd
import numpy as np

from keras.utils import np_utils
from keras.layers import *
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator


# The competition datafiles are in the directory ../input
# Read competition data files:
train = pd.read_csv("../input/train.csv")
test  = pd.read_csv("../input/test.csv")

# Write to the log:
print("Training set has {0[0]} rows and {0[1]} columns".format(train.shape))
print("Test set has {0[0]} rows and {0[1]} columns".format(test.shape))
# Any files you write to the current directory get shown as outputs
train_np = np.array(train)

x_train = train_np[:,1:].reshape(train_np.shape[0], 28, 28, 1).astype('float32') / 255
y_train = np_utils.to_categorical(train_np[:,0], 10)


test_np = np.array(test)
x_test = test_np.reshape(test_np.shape[0], 28, 28, 1).astype('float32') / 255


cnn = Sequential()

cnn.add( Convolution2D(32, 5, 5, input_shape=(28, 28, 1)) )
cnn.add( BatchNormalization() )
cnn.add( Activation('relu') )
cnn.add( MaxPooling2D() )

cnn.add( Convolution2D(32, 3, 3) )
cnn.add( BatchNormalization() )
cnn.add( Activation('relu') )
cnn.add( MaxPooling2D() )

cnn.add( Flatten() ) 

cnn.add( Dense(output_dim=128) )
cnn.add( BatchNormalization() )
cnn.add( Activation('relu') )

cnn.add( Dense(output_dim=10, activation='softmax') )

cnn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


image_data_gen = ImageDataGenerator(
		width_shift_range=0.1,
		height_shift_range=0.1,
		zoom_range=0.1
	)


cnn.fit_generator( image_data_gen.flow(x_train, y_train, batch_size=64), len(x_train), nb_epoch=2 )

y = cnn.predict_classes(x_test)
np.savetxt('mnist.csv', np.c_[range(1,len(y)+1),y], delimiter=',', header = 'ImageId,Label', comments = '', fmt='%d')