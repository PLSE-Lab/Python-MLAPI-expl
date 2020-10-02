########################## Part1-Building the CNN #############################

#Importing the keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#Initializing the CNN
classifier = Sequential()

#Step-1 creating convolution layers
classifier.add(Convolution2D(32, 3, 3, input_shape=(64, 64, 3), activation = 'relu')) # 32 feature detector with 3 columns and rows; input images with shapw 64*64 pixels with 3 layers (RBG); Including relu to remove linearity (negative pixels)

#Step-2 Max pooling with stride = 2 (in both column and row)
classifier.add(MaxPooling2D(pool_size = (2, 2)))

#Adding a second convolutional layer to improve performance
classifier.add(Convolution2D(32, 3, 3, activation = 'relu')) 
classifier.add(MaxPooling2D(pool_size = (2, 2)))

#Step-3 Flattening (converting pooled feature map (n*n) into single vector (n*1))
classifier.add(Flatten()) 

#Step-4 Full Connection
classifier.add(Dense(units = 128, activation = 'relu')) #first input layer and first hidden layer; units represents the number of hidden nodes and it is always advisable to choose the number between input nodes and final output nodes.
classifier.add(Dense(units = 1, activation = 'sigmoid'))

#compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

##################### Part2 - Fitting the CNN to images #######################

# importing the required module
from keras.preprocessing.image import ImageDataGenerator

# we are using in total of 10,000 images for both testing and training which is not much. So we are using the technique of image augmentaion to reduce overfitting
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
        )

#transforming pixels (0-255) to 0's and 1's
test_datagen = ImageDataGenerator(rescale=1./255)

#Generating training data
training_set = train_datagen.flow_from_directory(
        "../input/dataset/dataset/training_set",
        target_size=(64, 64), # as per image size 
        batch_size=32,
        class_mode='binary' # here since it is either cat (0) or dog (1))
        ) 

test_set = test_datagen.flow_from_directory(
        "../input/dataset/dataset/test_set",
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        training_set,
        steps_per_epoch=8000, # total inputs in training set
        epochs=25,
        validation_data=test_set,
        validation_steps=2000 # total inputs in test set
        )

###################### Part 3 - Making New Predictions ########################

#For Dog Prediction
#------------------

import numpy as np
from keras.preprocessing import image
test_image = image.load_img("../input/dataset/dataset/single_prediction/cat_or_dog_1.jpg", target_size =(64, 64))
test_image = image.img_to_array(test_image) # to convert the image into a 2D array
test_image = np.expand_dims(test_image, axis = 0) # gives the batch size, in this case it will be one since we are inputting only one image
result = classifier.predict(test_image)
training_set.class_indices # to know whether the 0 or 1 belongs to cats or dogs
if result[0][0] == 1:
    prediction_img_1 = 'dog'
else:
    prediction_img_1 = 'cat'

#For Cat Prediction (just change the location where the cat picture is present)
#------------------

import numpy as np
from keras.preprocessing import image
test_image = image.load_img("../input/dataset/dataset/single_prediction/cat_or_dog_2.jpg", target_size =(64, 64))
test_image = image.img_to_array(test_image) # to convert the image into a 2D array
test_image = np.expand_dims(test_image, axis = 0) # gives the batch size, in this case it will be one since we are inputting only one image
result = classifier.predict(test_image)
training_set.class_indices # to know whether the 0 or 1 belongs to cats or dogs
if result[0][0] == 1:
    prediction_img_2 = 'dog'
else:
    prediction_img_2 = 'cat'

#############################################################################




