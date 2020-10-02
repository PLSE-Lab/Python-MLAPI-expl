import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from subprocess import check_output

# The competition datafiles are in the directory ../input
# Read competition data files:
train = pd.read_csv("../input/train.csv")
test  = pd.read_csv("../input/test.csv")

# Write to the log:
print("Training set has {0[0]} rows and {0[1]} columns".format(train.shape))
print("Test set has {0[0]} rows and {0[1]} columns".format(test.shape))
# Any files you write to the current directory get shown as outputs

print('Playing around with @daryadedik solution: https://www.kaggle.com/daryadedik/digit-recognizer/mnist-with-cnn-0-99157-score')




# Learning rate for learning algorithm
LEARNING_RATE = 5e-4

# number of training iterations
TRAINING_ITERATIONS = 10000
ITERATIONS_PER_RUN = 100

# dropout. Dropout is an extremely effective and simple regularization technique
# based on keeping a neuron active with some probability p (a hyperparameter, here is DROPOUT), 
# or setting it to zero otherwise.
# More about dropout http://cs231n.github.io/neural-networks-2/ 
DROPOUT = 0.5
# training is made by selecting batches of size 50
BATCH_SIZE = 100

# validation set size. Validation set is a set of examples used to tune the parameters .
# set to 0 to train on all available data
VALIDATION_SIZE = 2000

# image number to output
IMAGE_TO_DISPLAY = 25



# Strutified shuffle is used insted of simple shuffle in order to achieve sample balancing
    # or equal number of examples in each of 10 classes.
# Since there are different number of examples for each 10 classes in the MNIST data you may
    # also use simple shuffle.
    
def stratified_shuffle(labels, num_classes):
    ix = np.argsort(labels).reshape((num_classes,-1))
    for i in range(len(ix)):
        np.random.shuffle(ix[i])
    return ix.T.reshape((-1))
    
# preload train and test data
dataset = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv").values

images = dataset.iloc[:,1:].values
images = images.astype(np.float)

# convert from [0:255] => [0.0:1.0]
images = np.multiply(images, 1.0 / 255.0)
print('data size: (%g, %g)' % images.shape)

# size of one image - 784 values which can be transformed to 28 by 28 image
image_size = images.shape[1]
print ('image_size => {0}'.format(image_size))

# in this case all images are square (28x28)
image_width = image_height = np.ceil(np.sqrt(image_size)).astype(np.uint8)
print ('image_width => {0}\nimage_height => {1}'.format(image_width,image_height))


#Results:
#data size: (42000, 784)
#image_size => 784
#image_width => 28
#image_height => 28

# display an image
def display(img):
    
    # (784) => (28x28)
    one_image = img.reshape(image_width,image_height)
    
    plt.axis('off')
    plt.imshow(one_image, cmap=cm.binary)

# display an image. IMAGE_TO_DISPLAY = 25
display(images[IMAGE_TO_DISPLAY])


# print information about image size, label of image-to-display and number of labels
labels_flat = dataset[[0]].values.ravel()
print('length of one image ({0})'.format(len(labels_flat)))
print ('label of image [{0}] => {1}'.format(IMAGE_TO_DISPLAY,labels_flat[IMAGE_TO_DISPLAY]))

labels_count = np.unique(labels_flat).shape[0]
print('number of labes => {0}'.format(labels_count))

#   RESULTS
#length of one image (42000)
#label of image [25] => 3
#number of labes => 10

# convert class labels from scalars to one-hot vectors
# 0 => [1 0 0 0 0 0 0 0 0 0]
# 1 => [0 1 0 0 0 0 0 0 0 0]
# ...
# 9 => [0 0 0 0 0 0 0 0 0 1]

def dense_to_one_hot(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

labels = dense_to_one_hot(labels_flat, labels_count)
labels = labels.astype(np.uint8)

print('labels({0[0]},{0[1]})'.format(labels.shape))
print ('labels vector for image [{0}] => {1}'.format(IMAGE_TO_DISPLAY,labels[IMAGE_TO_DISPLAY]))
    #Results
#labels(42000,10)
#labels vector for image [25] => [0 0 0 1 0 0 0 0 0 0]

print('Ended after IN[7]')


    