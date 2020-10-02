import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import tensorflow as tf

LEARNING_RATE= 1e-4

TRAINING_ITERATIONS=2500
DROPOUT=0.5
BATCH_SIZE=50

VALIDATION_SIZE=2000

IMAGE_TO_DISPLAY=10
# The competition datafiles are in the directory ../input
# Read competition data files:
data = pd.read_csv("../input/train.csv")

# Write to the log:
print('data{0[0]},{0[1]})'.format(data.shape))
print(data.head())
# Any files you write to the current directory get shown as outputs
images=data.iloc[:,1:].values
images=images.astype(np.float)
images=np.multiply(images,1.0/255.0)

image_size=images.shape[1]
print('image_size => {0}'.format(image_size))
image_width=image_height=np.ceil(np.sqrt(image_size)).astype(np.uint8)
print('image_width => {0}\nimage_height => {1}'.format(image_width, image_height))

def display(img):
    one_image=img.reshape(image_width, image_height)
    plt.axis('off')
    plt.imshow(one_image, cmap=cm.binary)
    plt.savefig('output1.png')
display(images[0])

labels_flat = data[[0]].values.ravel()

print(labels_flat.shape)
labels_count = np.unique(labels_flat).shape[0]

def dense_to_one_hot(labels_dense, num_classes):
    num_labels=labels_dense.shape[0]
    index_offset=np.arange(num_labels)*num_classes
    labels_one_hot=np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset+labels_dense.ravel()]=1
    return labels_one_hot
    
labels=dense_to_one_hot(labels_flat, labels_count)
labels=labels.astype(np.uint8)

print('labels({0[0]}, {0[1]})'.format(labels.shape))
print('labels[{0}]=>{1}'.format(IMAGE_TO_DISPLAY, labels[IMAGE_TO_DISPLAY]))


validation_images=images[:VALIDATION_SIZE]
validation_labels=labels[:VALIDATION_SIZE]

train_images=images[VALIDATION_SIZE:]
train_labels=labels[VALIDATION_SIZE:]

print('train_images({0[0]},{0[1]})'.format(train_images.shape))
print('validation_images({0[0]},{0[1]})'.format(validation_images.shape))