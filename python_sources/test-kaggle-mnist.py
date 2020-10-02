# https://www.kaggle.com/kakauandme/digit-recognizer/tensorflow-deep-nn/comments

import pandas as pd
import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Settings
LEARNING_RATE = 1e-4
TRAINING_ITERATIONS = 2500
DROPOUT = 0.5
BATCH_SIZE = 50
VALIDATION_SIZE = 2000
IMAGE_TO_DISPLAY = 10

# The competition datafiles are in the directory ../input
# Read competition data files:
train = pd.read_csv("../input/train.csv")
test  = pd.read_csv("../input/test.csv")

# Write to the log:
print("Training set has {0[0]} rows and {0[1]} columns".format(train.shape))
print("Test set has {0[0]} rows and {0[1]} columns".format(test.shape))
# Any files you write to the current directory get shown as outputs

images = train.iloc[:,1:].values
images = images.astype(np.float)
images = np.multiply(images, 1.0 / 225.0)

print('images({0[0]}, {0[1]})'.format(images.shape))

image_size = images.shape[1]
print ('image_size => {0}'.format(image_size))

# in this case all images are square
image_width = image_height = np.ceil(np.sqrt(image_size)).astype(np.uint8)
print ('image_width => {0}\nimage_height => {1}'.format(image_width,image_height))

# display image
def display(img):
    one_image = img.reshape(image_width, image_height)
    plt.axis('off')
    plt.imshow(one_image, cmap = cm.binary)
    
display(images[IMAGE_TO_DISPLAY])

labels_flat = train[[0]].values.ravel()

print('labels_flat({0})'.format(len(labels_flat)))
print ('labels_flat[{0}] => {1}'.format(IMAGE_TO_DISPLAY,labels_flat[IMAGE_TO_DISPLAY]))