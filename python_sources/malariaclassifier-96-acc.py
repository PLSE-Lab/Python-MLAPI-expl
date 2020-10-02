
# for matrix manipulations and data storage
import numpy as np 

# to produce the output csv file
import pandas as pd

# for image loading and resizing
import skimage

# for splitting training data and validation data
import sklearn.model_selection

# artificial neural network framework
import keras as kr

# for finding the directories of the files
import os
print("Imports successful")

# number of infected or uninfected pictures
SIZE = 13779
# resized image size
RESIZED_IMG_SIZE = (32, 32)
# directories and lists of file paths
INFECTED_DIR = "../input/cell_images/cell_images/Parasitized/"
UNINFECTED_DIR = "../input/cell_images/cell_images/Uninfected/"
INFECTED_DIRS = [ x for x in os.listdir("../input/cell_images/cell_images/Parasitized") if x.endswith(".png") ]
UNINFECTED_DIRS = [ x for x in os.listdir("../input/cell_images/cell_images/Uninfected") if x.endswith(".png") ]
print("Constant declaration successful")

# label creation
labels = np.ones(2*SIZE)    # Infected cells label = 1
labels[:SIZE] = 0.0         # Uninfected cells label = 0
print("Label assignment successful")

# loading the (resized) dataset in memory, uninfected cells first, infected cells second
# I am resizing the images to improve speed, performance and to avoid overcluttering the
# network with useless information
data_u = np.array([skimage.transform.resize(skimage.io.imread(UNINFECTED_DIR + t), RESIZED_IMG_SIZE) for t in UNINFECTED_DIRS])
data_i = np.array([skimage.transform.resize(skimage.io.imread(INFECTED_DIR + t), RESIZED_IMG_SIZE) for t in INFECTED_DIRS])
# concatenating the two arrays to get the final dataset.
data = np.concatenate((data_u, data_i))
print("Images loaded successfully")

# these variables are no longer needed, deleting them to free memory
del data_u, data_i, INFECTED_DIRS, UNINFECTED_DIRS
print("Garbage collection of residual variables done successfully")

# shuffling data (we do not want to have all infected and uninfected cell pictures together) and splitting
# it to training set and validation set
x_train, x_valid, y_train, y_valid = sklearn.model_selection.train_test_split(data, labels, test_size=0.125, shuffle=True)

# we do not want the network to memorize black patches in our inputs
# so we are going to provide some data augmentation to avoid such tendancies
# both in the train set and the validation set
train_generator = kr.preprocessing.image.ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=45,
        width_shift_range=0.05,
        height_shift_range=0.05,
        horizontal_flip=True,
        vertical_flip=True
    )
train_generator.fit(x_train)
evaluate_generator = kr.preprocessing.image.ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=45,
        width_shift_range=0.05,
        height_shift_range=0.05,
        horizontal_flip=True,
        vertical_flip=True
    )
evaluate_generator.fit(x_valid)
print("Data split into train and validation sets successfully")

dropout = 0.5

layers = [
        kr.layers.SeparableConv2D(50, 3, padding="same", depth_multiplier=3, input_shape=(32, 32, 3)),
        kr.layers.SpatialDropout2D(dropout),
        kr.layers.BatchNormalization(momentum=0.98),
        kr.layers.LeakyReLU(alpha=0.15),
        kr.layers.AveragePooling2D(),
        kr.layers.SeparableConv2D(50, 3, depth_multiplier=3, padding="same"),
        kr.layers.SpatialDropout2D(dropout),
        kr.layers.BatchNormalization(momentum=0.98),
        kr.layers.LeakyReLU(alpha=0.15),
        kr.layers.AveragePooling2D(),
        kr.layers.SeparableConv2D(50, 3, depth_multiplier=3, padding="same"),
        kr.layers.SpatialDropout2D(dropout),
        kr.layers.BatchNormalization(momentum=0.98),
        kr.layers.LeakyReLU(alpha=0.15),
        kr.layers.AveragePooling2D(8),
        kr.layers.Flatten(),
        #kr.layers.Dense(20),
        #kr.layers.Dropout(dropout),
        #kr.layers.BatchNormalization(momentum=0.98),
        #kr.layers.LeakyReLU(alpha=0.15),
        kr.layers.Dense(1, activation="sigmoid")
    ]

# creating the network
model = kr.models.Sequential(layers)
# Adam optimizer has been proven to work very well for image related classification
# Using binary crossentropy since we are dealing with just one class
model.compile(optimizer=kr.optimizers.Adam(), loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

# fitting the model
history = model.fit_generator(
        train_generator.flow(x_train, y_train, batch_size=16),
        steps_per_epoch=int(2*SIZE / 16),
        epochs=128,
        validation_data=evaluate_generator.flow(x_valid, y_valid, batch_size=32),
        validation_steps=128,
        verbose=2
    )

# saving the results to outputs.csv
pd.DataFrame.from_dict(history.history).to_csv("outputs.csv", float_format="%.5f", index=False)
model.save("michaelb_malaria_classifier.h5")
