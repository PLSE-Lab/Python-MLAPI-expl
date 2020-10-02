import numpy as np
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense, Flatten
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt

train_path = '../input/shapes/shapes/train'
valid_path = '../input/shapes/shapes/valid'
test_path = '../input/shapes/shapes/test'

# Sizes images and does one hot encoding

train_batches = ImageDataGenerator().flow_from_directory(train_path, target_size=(224, 224), classes=['circle', 'square', 'star', 'triangle'], batch_size=100)
valid_batches = ImageDataGenerator().flow_from_directory(valid_path, target_size=(224, 224), classes=['circle', 'square', 'star', 'triangle'], batch_size=50)
test_batches = ImageDataGenerator().flow_from_directory(test_path, target_size=(224, 224), classes=['circle', 'square', 'star', 'triangle'], batch_size=50, shuffle=False)

# Plot sample images

def plots(ims, figsize=(12,6), rows=1, interp=False, titles=None):
    if type(ims[0]) is np.ndarray:
        ims = np.array(ims).astype(np.uint8)
        if (ims.shape[-1] != 3):
            ims = ims.transpose((0,2,3,1))
    f = plt.figure(figsize=figsize)
    cols = len(ims)//rows if len(ims) % 2 == 0 else len(ims)//rows + 1
    for i in range(len(ims)):
        sp = f.add_subplot(rows, cols, i+1)
        sp.axis('Off')
        if titles is not None:
            sp.set_title(titles[i], fontsize=16)
        plt.imshow(ims[i], interpolation=None if interp else 'none')

# imgs, labels = next(train_batches)
# plots(imgs, titles=labels)

# Build and train CNN

model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        Flatten(),
        Dense(4, activation='softmax')
])

model.compile(Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit_generator(train_batches, steps_per_epoch=25,
                    validation_data=valid_batches, validation_steps=20, epochs=5, verbose=2)

# Predict
'''
test_imgs, test_labels = next(test_batches)
plots(test_imgs, titles=test_labels)

test_labels = test_labels[:, 1]

predictions = model.predict_generator(test_batches, steps=1, verbose=0)

test_batches.class_indices

cm = confusion_matrix(test_labels, predictions[:,1])

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

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

cm_plot_labels = ['circle', 'square', 'star', 'triangle']
plot_confusion_matrix(cm, cm_plot_labels, title='Confusion Matrix')
'''

# Building fine-tuned VGG16 Model

vgg16_model = keras.applications.vgg16.VGG16()
vgg16_model.summary()
type(vgg16_model)

# iterates through vgg16 model and adds same layers to our model without last convolutional layer with 1000 outputs

model = Sequential()
for layer in vgg16_model.layers[:-1]:
    model.add(layer)

model.summary()

# Iterates through layers so that weights are not updated and model can be fine tuned

for layer in model.layers:
    layer.trainable = False
    
# Adding layer for 4 outputs
model.add(Dense(4, activation='softmax'))
model.summary()

# Training on vgg16
model.compile(Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit_generator(train_batches, steps_per_epoch=25,
                    validation_data=valid_batches, validation_steps=20, epochs=5, verbose=2)
                    
