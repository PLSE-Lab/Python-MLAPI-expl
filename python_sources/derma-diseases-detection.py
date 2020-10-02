#!/usr/bin/env python
# coding: utf-8

# # Derma Diseases Detection
# - Performs fine tuning on Keras VGG16 in order to detect derma diseases such as: nevus, melanoma and seborrheic_keratosis

# In[ ]:


# Import Keras with tensorflow backend
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import optimizers
from keras.applications import VGG16

# Import OpenCV
import cv2

# Utility
import os
import numpy as np
import itertools
import random
from collections import Counter
from glob import iglob

# Ignore warning
import warnings
warnings.filterwarnings('ignore')

# Confusion Matrix & classification report
from sklearn.metrics import confusion_matrix, classification_report

# Plot
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Settings
# Define all settings usefull for next steps

# In[ ]:


# Set dataset folder path
BASE_DATASET_FOLDER = os.path.join("..","input","derma_disease_dataset","dataset")
TRAIN_FOLDER = "train"
VALIDATION_FOLDER = "validation"
TEST_FOLDER = "test"

# ResNet50 image size
IMAGE_SIZE = (224, 224)
INPUT_SHAPE = (224, 224, 3)

# Keras settings
TRAIN_BATCH_SIZE = 64
VAL_BATCH_SIZE = 8
EPOCHS = 50
LEARNING_RATE = 0.0001
MODEL_PATH = os.path.join("derma_diseases_detection.h5")


# ### Data agumentation
# Read images in batches directly from folders and perform data augmentation

# In[ ]:


def percentage_value(pct, allvals):
    absolute = int(pct/100.*np.sum(allvals))
    return "{:.1f}%\n({:d})".format(pct, absolute)

def plot_dataset_description(path, title):
    classes = []
    for filename in iglob(os.path.join(path, "**","*.jpg")):
        classes.append(os.path.split(os.path.split(filename)[0])[-1])

    classes_cnt = Counter(classes)
    values = list(classes_cnt.values())
    labels = list(classes_cnt.keys())

    plt.figure(figsize=(8,8))
    plt.pie(values, labels=labels, autopct=lambda pct: percentage_value(pct, values), 
            shadow=True, startangle=140)

    plt.title(title)    
    plt.show()


# In[ ]:


train_datagen = ImageDataGenerator(
        rescale=1./255,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

train_generator = train_datagen.flow_from_directory(
        os.path.join(BASE_DATASET_FOLDER, TRAIN_FOLDER),
        target_size=IMAGE_SIZE,
        batch_size=TRAIN_BATCH_SIZE,
        class_mode='categorical', 
        shuffle=True)


# In[ ]:


plot_dataset_description(os.path.join(BASE_DATASET_FOLDER, TRAIN_FOLDER), "Train folder description")


# In[ ]:


val_datagen = ImageDataGenerator(rescale=1./255)
val_generator = val_datagen.flow_from_directory(
        os.path.join(BASE_DATASET_FOLDER, VALIDATION_FOLDER),
        target_size=IMAGE_SIZE,
        class_mode='categorical', 
        shuffle=False)


# In[ ]:


plot_dataset_description(os.path.join(BASE_DATASET_FOLDER, VALIDATION_FOLDER), "Validation folder description")


# In[ ]:


test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
        os.path.join(BASE_DATASET_FOLDER, TEST_FOLDER),
        target_size=IMAGE_SIZE,
        batch_size=VAL_BATCH_SIZE,
        class_mode='categorical', 
        shuffle=False)


# In[ ]:


plot_dataset_description(os.path.join(BASE_DATASET_FOLDER, TEST_FOLDER), "Test folder description")


# In[ ]:


classes = {v: k for k, v in train_generator.class_indices.items()}
print(classes)


# ### Load VGG16 model
# Load the pre-trained VGG16 model without the top layer

# In[ ]:


vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=INPUT_SHAPE)


# ### Freeze the layers except the last 4 layers

# In[ ]:


for layer in vgg_model.layers[:-4]:
    layer.trainable = False


# ### Create model

# In[ ]:


# Create the model
model = Sequential()
 
# Add the vgg convolutional base model
model.add(vgg_model)
 
# Add new layers
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(classes), activation='softmax'))


# In[ ]:


# Show a summary of the model. Check the number of trainable parameters
model.summary()


# ### Compile model
# Compile model specifying the optimizer learning rate

# In[ ]:


# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.Adam(lr=LEARNING_RATE),
              metrics=['acc'])


# ### Train model
# train model using validation dataset for validate each steps

# In[ ]:


get_ipython().run_cell_magic('time', '', 'history = model.fit_generator(\n        train_generator,\n        steps_per_epoch=train_generator.samples//train_generator.batch_size,\n        epochs=EPOCHS,\n        validation_data=val_generator,\n        validation_steps=val_generator.samples//val_generator.batch_size)')


# In[ ]:


model.save(MODEL_PATH)


# ### Check Performance
# Plot training and validation accuracy and loss

# In[ ]:


acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
 
epochs = range(len(acc))
 
plt.plot(epochs, acc, 'b', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
 
plt.figure()
 
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
 
plt.show()


# ### Test model
# Evauate model using test dataset

# In[ ]:


get_ipython().run_cell_magic('time', '', 'loss, accuracy = model.evaluate_generator(test_generator,steps=test_generator.samples//test_generator.batch_size)')


# In[ ]:


print("Accuracy: %f\nLoss: %f" % (accuracy,loss))


# ### Confusion Matrix
# Build and plot confusion matrix

# In[ ]:


get_ipython().run_cell_magic('time', '', 'Y_pred = model.predict_generator(test_generator,verbose=1, steps=test_generator.samples//test_generator.batch_size)')


# In[ ]:


y_pred = np.argmax(Y_pred, axis=1)


# In[ ]:


cnf_matrix = confusion_matrix(test_generator.classes, y_pred)


# In[ ]:


def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(12,12))

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=18)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize=8)
    plt.yticks(tick_marks, classes, fontsize=12)

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label', fontsize=16)
    plt.xlabel('Predicted label', fontsize=16)


# In[ ]:


plot_confusion_matrix(cnf_matrix, list(classes.values()))


# ### Classification Report
# Print classification report

# In[ ]:


print(classification_report(test_generator.classes, y_pred, target_names=list(classes.values())))


# ### Random test
# Random sample images from test dataset and predict 

# In[ ]:


def load_image(filename):
    img = cv2.imread(os.path.join(BASE_DATASET_FOLDER, TEST_FOLDER, filename))
    img = cv2.resize(img, (IMAGE_SIZE[0], IMAGE_SIZE[1]) )
    img = img /255
    
    return img


def predict(image):
    probabilities = model.predict(np.asarray([img]))[0]
    class_idx = np.argmax(probabilities)
    
    return {classes[class_idx]: probabilities[class_idx]}


# In[ ]:


for idx, filename in enumerate(random.sample(test_generator.filenames, 10)):
    print("SOURCE: class: %s, file: %s" % (os.path.split(filename)[0], filename))
    
    img = load_image(filename)
    prediction = predict(img)
    print("PREDICTED: class: %s, confidence: %f" % (list(prediction.keys())[0], list(prediction.values())[0]))
    plt.imshow(img)
    plt.figure(idx)    
    plt.show()

