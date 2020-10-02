# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import sys
import os
from numpy import load
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from keras import backend
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.callbacks import ModelCheckpoint
from keras.layers import Dropout
from keras.layers.normalization import BatchNormalization


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
traindir = "../input/new-plant-diseases-dataset/new plant diseases dataset(augmented)/New Plant Diseases Dataset(Augmented)/train"
validdir = "../input/new-plant-diseases-dataset/new plant diseases dataset(augmented)/New Plant Diseases Dataset(Augmented)/valid"
testdir = "../input/new-plant-diseases-dataset/test/test"

# Any results you write to the current directory are saved as output.

# define cnn model
def define_model(in_shape=(224, 224, 3), out_shape=38):
	# load model
	model = VGG16(include_top=False, input_shape=in_shape, weights="../input/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5")
	#model.load_weights('../input/VGG-16/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')
    # mark loaded layers as not trainable
	for layer in model.layers:
		layer.trainable = False
	
	
	# allow last vgg block to be trainable
	model.get_layer('block5_conv1').trainable = True
	model.get_layer('block5_conv2').trainable = True
	model.get_layer('block5_conv3').trainable = True
	model.get_layer('block5_pool').trainable = True
	
	# add new classifier layers
	flat1 = Flatten()(model.layers[-1].output)
	fcon1 = Dense(4096, activation='relu', kernel_initializer='he_uniform')(flat1)
	fdrop1 = Dropout(0.25)(fcon1)
	fbn1 = BatchNormalization()(fdrop1)
	fcon2 = Dense(4096, activation='relu', kernel_initializer='he_uniform')(fbn1)
	fdrop2 = Dropout(0.25)(fcon2)
	fbn2 = BatchNormalization()(fdrop2)
	output = Dense(out_shape, activation='softmax')(fbn2)
	# define new model
	model = Model(inputs=model.inputs, outputs=output)
	# compile model
	opt = SGD(lr=0.01, momentum=0.9,decay=0.005)
	model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
	return model

# plot diagnostic learning curves
def summarize_diagnostics(history):
    sns.set()
    # plot loss
    pyplot.subplot(211)
    pyplot.title('Cross Entropy Loss')
    pyplot.plot(history.history['loss'], color='blue', label='train')
    pyplot.plot(history.history['val_loss'], color='orange', label='valid')
    pyplot.xlabel('Epoch')
    pyplot.ylabel('Loss')
    pyplot.legend()
    # plot accuracy
    pyplot.subplot(212)
    pyplot.title('Classification Accuracy')
    pyplot.plot(history.history['acc'], color='blue', label='train')
    pyplot.plot(history.history['val_acc'], color='orange', label='valid')
    pyplot.xlabel('Epoch')
    pyplot.ylabel('Accuracy')
    pyplot.legend()
    # save plot to file
    filename = sys.argv[0].split('/')[-1]
    pyplot.savefig(filename + '_plot.png')
    pyplot.close()

# load and prepare the image for prediction
def load_image(filename):
	# load the image
	img = load_img(filename, target_size=(224, 224))
	# convert to array
	img = img_to_array(img)
	# reshape into a single sample with 3 channels
	img = img.reshape(1, 224, 224, 3)
    #rescale
	img = img/255
	# center pixel data
	#img = img.astype('float32')
	#img = img - [123.68, 116.779, 103.939]
	return img    

batch_size = 128

train_datagen = ImageDataGenerator(rescale=1./255)
                                   
valid_datagen = ImageDataGenerator(rescale=1./255)

training_iterator = train_datagen.flow_from_directory(traindir,
                                                 target_size=(224, 224),
                                                 batch_size=batch_size,
                                                 class_mode='categorical')

test_iterator = valid_datagen.flow_from_directory(validdir,
                                            target_size=(224, 224),
                                            batch_size=batch_size,
                                            class_mode='categorical')

class_dict = training_iterator.class_indices
print(class_dict)

class_labels = list(class_dict.keys())
print(class_labels)

train_num_samples = training_iterator.samples
valid_num_samples = test_iterator.samples
# define model
model = define_model()
model.summary()

weightsfilepath = "bestweights.hdf5"
checkpoint = ModelCheckpoint(weightsfilepath, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True, mode='max')
callbacks_list = [checkpoint]
# fit model
history = model.fit_generator(training_iterator, steps_per_epoch=len(training_iterator),
		validation_data=test_iterator, validation_steps=len(test_iterator), epochs=8, callbacks=callbacks_list, verbose=2)


# evaluate model
_, acc = model.evaluate_generator(test_iterator, steps=len(test_iterator), verbose=1)
print('> %.3f' % (acc * 100.0))
# learning curves
summarize_diagnostics(history)

#model.save('plantdisease_vgg19model.h5')

# load the image
img = load_image('../input/new-plant-diseases-dataset/test/test/AppleScab3.JPG')
print("Prediction for AppleScab3:")
#prediction will be an array of values; index corresp to max value will be used to get the class label
prediction = model.predict(img)
predicted_class_name = class_labels[np.argmax(prediction)]
print("Detected the leaf as ", predicted_class_name)     

for filename in os.listdir(testdir):
    #print(filename)
    # load the image
    filepath = testdir + '/' + filename
    #print(filepath)
    img = load_image(filepath)
    #prediction will be an array of values; index corresp to max value will be used to get the class label
    prediction = model.predict(img)
    predicted_class_name = class_labels[np.argmax(prediction)]
    print(filename, "  predicted as ", predicted_class_name)  