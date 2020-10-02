import pandas, numpy

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import *
from keras.preprocessing.image import ImageDataGenerator

#=======
train_raw = numpy.array(pandas.read_csv('../input/train.csv'))

train_x = train_raw[:,1:].reshape(train_raw.shape[0], 28, 28, 1).astype('float32') / 255.
train_y = np_utils.to_categorical(train_raw[:,0], 10)


test_raw = numpy.array(pandas.read_csv('../input/test.csv'))
test_x = test_raw.reshape(test_raw.shape[0], 28, 28, 1).astype('float32') / 255.

#=======
model = Sequential()

model.add( Convolution2D(32, 5, 5, activation='relu', input_shape=(28, 28, 1)) )
model.add( MaxPooling2D() )

model.add( Convolution2D(32, 3, 3, activation='relu') )
model.add( MaxPooling2D() )

model.add( Flatten() ) 

model.add( Dense(output_dim=128, activation='relu') )
model.add( Dropout(0.5) )

model.add( Dense(output_dim=10, activation='softmax') )

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#=======
datagen = ImageDataGenerator(
		width_shift_range=0.1,
		height_shift_range=0.1,
		zoom_range=0.1
	)

#=======
for i in range(1): # change to 20
	model.fit_generator(datagen.flow(train_x, train_y, batch_size=64),
		len(train_x), nb_epoch=1) # change to 10

	# model.save('model_mnist.h5') # uncomment to save your network

y = model.predict_classes(test_x)
numpy.savetxt('mnist.csv', np.c_[range(1,len(y)+1), y], fmt='%d', delimiter=',', header='ImageId,Label')