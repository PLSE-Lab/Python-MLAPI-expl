# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
from __future__ import print_function, division
from builtins import range, input
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from glob import glob 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import itertools

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
        
# Create Helper Functions
def mkdir(p):
  if not os.path.exists(p):
    os.mkdir(p)

    
def link(src, dst):
  if not os.path.exists(dst):
    os.symlink(src, dst, target_is_directory=True)

# Plot confusion Matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
  """
  This function prints and plots the confusion matrix.
  Normalization can be applied by setting `normalize=True`.
  """
  if normalize:
      cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
      print("Normalized confusion matrix")
  else:
      print('Confusion matrix, without normalization')

  print(cm)

  plt.imshow(cm, interpolation='nearest', cmap=cmap)
  plt.title(title)
  plt.colorbar()
  tick_marks = np.arange(len(classes))
  plt.xticks(tick_marks, classes, rotation=45)
  plt.yticks(tick_marks, classes)

  fmt = '.2f' if normalize else 'd'
  thresh = cm.max() / 2.
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
      plt.text(j, i, format(cm[i, j], fmt),
               horizontalalignment="center",
               color="white" if cm[i, j] > thresh else "black")

  plt.tight_layout()
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  plt.show()

    
def get_labels():
    class_labels = [x[0] for x in os.walk('/kaggle/input/fruits/fruits-360_dataset/fruits-360/Training')]
    return class_labels


# Any results you write to the current directory are saved as output.

train_path_from = os.path.abspath('/kaggle/input/fruits/fruits-360_dataset/fruits-360/Training')
valid_path_from = os.path.abspath('/kaggle/input/fruits/fruits-360_dataset/fruits-360/Test')

train_path_to = train_path_from
valid_path_to = valid_path_from

# Resize all images to the size below
IMAGE_SIZE = [100, 100]

# Training Config
EPOCHS = 5
BATCH_SIZE = 32

# getting number of files
image_files = glob(train_path_to + '/*/*.jp*g')
valid_image_files = glob(valid_path_to + '/*/*.jp*g')

# getting number of classes
folders = glob(train_path_to + '/')

# look at an image for fun
plt.imshow(image.load_img(np.random.choice(image_files)))
plt.show()

# Add preprocessing layer to the front of VGG
vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

# Don't train existing weights
for layer in vgg.layers:
    layer.trainable=False
    
# Layers
x = Flatten()(vgg.output)
prediction = Dense(len(folders), activation='softmax')(x)

# Create Model
model = Model(inputs=vgg.input, outputs=prediction)

# Generate Summary
model.summary()

# Compile Model with loss and optimization
model.compile(loss='categorical_crossentropy',
             optimizer='rmsprop',
             metrics=['accuracy'])

# Create an Instance of ImageGenerator
train_gen = ImageDataGenerator(rotation_range=20,
                               width_shift_range=0.1,
                               height_shift_range=0.1,
                               shear_range=0.1,
                               zoom_range=0.2,
                               horizontal_flip=True,
                               vertical_flip=True,
                               preprocessing_function=preprocess_input)

# Test Generator
test_gen = train_gen.flow_from_directory(valid_path_to, target_size=IMAGE_SIZE)
print("Test Gen Class Indices:",test_gen.class_indices)
# Get labels for plotting confusion matrix
labels = [None] * len(test_gen.class_indices)
for k,v in test_gen.class_indices.items():
    labels[v] = k
    

# Image
for x, y in test_gen:
    print("min:", x[0].min(), "max:", x[0].max())
    plt.title(labels[np.argmax(y[0])])
    plt.imshow(x[0])
    plt.show()
    break
    
# create generators
train_generator = train_gen.flow_from_directory(
  train_path_to,
  target_size=IMAGE_SIZE,
  shuffle=True,
  batch_size=BATCH_SIZE,
)

valid_generator = train_gen.flow_from_directory(
  valid_path_to,
  target_size=IMAGE_SIZE,
  shuffle=True,
  batch_size=BATCH_SIZE,
)


# fit the model
modelr = model.fit_generator(
  train_generator,
  validation_data=valid_generator,
  epochs=EPOCHS,
  steps_per_epoch=len(image_files) // BATCH_SIZE,
  validation_steps=len(valid_image_files) // BATCH_SIZE,
)


def create_confusion_matrix(data_path, N):
    # Both input and output should be in same order
    print("Generating confusion matrix", N)
    predictions, targets, i = [], [], 0
    for x, y in train_gen.flow_from_directory(data_path, target_size=IMAGE_SIZE, shuffle=False,
                                             batch_size=BATCH_SIZE*2):
        i+=1
        if i%50 ==0:
            print(i)
        p = model.predict(x)
        p = np.argmax(p, axis=1)
        y = np.argmax(y, axis=1)
        predictions = np.concatenate((predictions, p))
        targets = np.concatenate((targets, y))
        if len(targets) >= N:
            break
            
    cm = confusion_matrix(targets, predictions)
    return cm

cm = create_confusion_matrix(train_path_to, len(image_files))
print("Training_ConfusionMatrix:",cm)

valid_cm = create_confusion_matrix(valid_path_to, len(valid_image_files))
print("Validation_ConfusionMatrix:", valid_cm)

# Plot Loss and Accuracy Data

plt.plot(modelr.history['loss'], label='train_loss')
plt.plot(modelr.history['val_loss'], label='val_loss')
plt.legend()
plt.show()
        
plt.plot(modelr.history['accuracy'], label='train_accuracy')
plt.plot(modelr.history['val_accuracy'], label='val_accuracy')
plt.legend()
plt.show()

class_labels= get_labels()
print(class_labels)

plot_confusion_matrix(cm, labels, title='Train confusion matrix')
plot_confusion_matrix(valid_cm, labels, title='Validation confusion matrix')



    
