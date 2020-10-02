#!/usr/bin/env python
# coding: utf-8

# # This kernel uses All COVID-19 images availble on Kaggle # Multi image source
# **This is a kernel that shows you how to Apply Conv Nets to classify COVID-19 scans**
# 

# In[ ]:


get_ipython().system('pip install imutils')


# In[ ]:


import shutil
import tqdm
import os
import cv2
from imutils import paths
import random
import shutil


# In[ ]:


from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator,load_img, img_to_array
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D,GlobalAveragePooling2D
from keras.layers import Activation, Dropout, BatchNormalization, Flatten, Dense

from keras.models import Sequential, Model
#rom keras.applications.xception import Xception
#from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.optimizers import Adam, SGD, RMSprop

import tensorflow as tf

import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# # Preparing the dataset 1

# In[ ]:





# In[ ]:





# In[ ]:


dataset_path = './dataset'


# In[ ]:


covid_dataset_path = '../input/covid-chest-xray'

coivd_path='../input/covid-19-x-ray-10000-images/dataset/covid/'


# In[ ]:


get_ipython().run_cell_magic('bash', '', 'rm -rf dataset\nmkdir -p dataset/covid\nmkdir -p dataset/normal\nmkdir -p dataset/pneumonia')


# In[ ]:


# construct the path to the metadata CSV file and load it
csvPath = os.path.sep.join([covid_dataset_path, "metadata.csv"])
df = pd.read_csv(csvPath)

# loop over the rows of the COVID-19 data frame
for (i, row) in df.iterrows():
    # if (1) the current case is not COVID-19 or (2) this is not
    # a 'PA' view, then ignore the row
    if row["finding"] != "COVID-19" or row["view"] != "PA":
        continue

    # build the path to the input image file
    imagePath = os.path.sep.join([covid_dataset_path, "images", row["filename"]])

    # if the input image file does not exist (there are some errors in
    # the COVID-19 metadeta file), ignore the row
    if not os.path.exists(imagePath):
        continue

    # extract the filename from the image path and then construct the
    # path to the copied image file
    filename = row["filename"].split(os.path.sep)[-1]
    outputPath = os.path.sep.join([f"{dataset_path}/covid", filename])

    # copy the image
    shutil.copy2(imagePath, outputPath)


# In[ ]:





# In[ ]:





# # Preparing the dataset 2

# In[ ]:


data5 = '../input/covid19-radiography-database/COVID-19 Radiography Database/'


# In[ ]:


print('NORMAL cases ',len(os.listdir(data5+'NORMAL')))
print('Covid cases ',len(os.listdir(data5+'COVID-19')))
print('Viral Pneumonia cases ',len(os.listdir(data5+'Viral Pneumonia')))


# In[ ]:


basePath = os.path.sep.join([data5,  "NORMAL"])
imagePaths = list(paths.list_images(basePath))

# randomly sample the image paths
random.seed(42)
random.shuffle(imagePaths)
imagePaths = imagePaths[:220]

# loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
    # extract the filename from the image path and then construct the
    # path to the copied image file
    filename = imagePath.split(os.path.sep)[-1]
    outputPath = os.path.sep.join([f"{dataset_path}/normal", filename])

    # copy the image
    shutil.copy2(imagePath, outputPath)


# In[ ]:


basePath = os.path.sep.join([data5,  "COVID-19"])
imagePaths = list(paths.list_images(basePath))

# randomly sample the image paths
random.seed(42)
random.shuffle(imagePaths)
imagePaths = imagePaths[:220]

# loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
    # extract the filename from the image path and then construct the
    # path to the copied image file
    filename = imagePath.split(os.path.sep)[-1]
    outputPath = os.path.sep.join([f"{dataset_path}/covid", filename])

    # copy the image
    shutil.copy2(imagePath, outputPath)


# In[ ]:


basePath = os.path.sep.join([data5, 'Viral Pneumonia'])
imagePaths = list(paths.list_images(basePath))

# randomly sample the image paths
random.seed(42)
random.shuffle(imagePaths)
imagePaths = imagePaths[:220]

# loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
    # extract the filename from the image path and then construct the
    # path to the copied image file
    filename = imagePath.split(os.path.sep)[-1]
    outputPath = os.path.sep.join([f"{dataset_path}/pneumonia", filename])

    # copy the image
    shutil.copy2(imagePath, outputPath)


# Copying the files from dataset 2 to dataset/covid and to Normal

# In[ ]:





# In[ ]:


print('Covid cases ',len(os.listdir('dataset/covid')))
print('normal cases ',len(os.listdir('dataset/normal')))
print('pneumonia cases ',len(os.listdir('dataset/pneumonia')))


# In[ ]:





# In[ ]:


get_ipython().system('cp -a ../input/covid-19-x-ray-10000-images/dataset/normal/. dataset/normal')


# In[ ]:


get_ipython().system('cp -a ../input/covid-19-x-ray-10000-images/dataset/covid/. dataset/covid')


# # Preparing dataset  3  chest-xray-pneumonia**

# In[ ]:


pneumonia_dataset_path ='../input/chest-xray-pneumonia/chest_xray'


# In[ ]:





# In[ ]:


basePath = os.path.sep.join([pneumonia_dataset_path, "train", "NORMAL"])
imagePaths = list(paths.list_images(basePath))

# randomly sample the image paths
random.seed(42)
random.shuffle(imagePaths)
imagePaths = imagePaths[:53]

# loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
    # extract the filename from the image path and then construct the
    # path to the copied image file
    filename = imagePath.split(os.path.sep)[-1]
    outputPath = os.path.sep.join([f"{dataset_path}/normal", filename])

    # copy the image
    shutil.copy2(imagePath, outputPath)


# In[ ]:


basePath = os.path.sep.join([pneumonia_dataset_path, "train", "PNEUMONIA"])
imagePaths = list(paths.list_images(basePath))

# randomly sample the image paths
random.seed(42)
random.shuffle(imagePaths)
imagePaths = imagePaths[:53]

# loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
    # extract the filename from the image path and then construct the
    # path to the copied image file
    filename = imagePath.split(os.path.sep)[-1]
    outputPath = os.path.sep.join([f"{dataset_path}/pneumonia", filename])

    # copy the image
    shutil.copy2(imagePath, outputPath)


# In[ ]:


basePath = os.path.sep.join([pneumonia_dataset_path, "test", "PNEUMONIA"])
imagePaths = list(paths.list_images(basePath))

# randomly sample the image paths
random.seed(42)
random.shuffle(imagePaths)
imagePaths = imagePaths[:53]

# loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
    # extract the filename from the image path and then construct the
    # path to the copied image file
    filename = imagePath.split(os.path.sep)[-1]
    outputPath = os.path.sep.join([f"{dataset_path}/pneumonia", filename])

    # copy the image
    shutil.copy2(imagePath, outputPath)


# # Preparing dataset  4

# In[ ]:


data4 = '../input/covid19-xray-dataset-train-test-sets/xray_dataset_covid19/'


# In[ ]:


print('NORMAL cases ',len(os.listdir('../input/covid19-xray-dataset-train-test-sets/xray_dataset_covid19/train/NORMAL/')))


# In[ ]:


get_ipython().system('cp -a ../input/covid19-xray-dataset-train-test-sets/xray_dataset_covid19/train/NORMAL/. dataset/normal')


# In[ ]:





# In[ ]:


print('Covid cases ',len(os.listdir('dataset/covid')))
print('normal cases ',len(os.listdir('dataset/normal')))
print('pneumonia cases ',len(os.listdir('dataset/pneumonia')))


# In[ ]:





# In[ ]:





# In[ ]:


# grab the list of images in our dataset directory, then initialize
# the list of data (i.e., images) and class images
print("[INFO] loading images...")
imagePaths = list(paths.list_images(dataset_path))


# In[ ]:


im = mpimg.imread(img_path)


# In[ ]:


im.shape[0]


# In[ ]:


import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
get_ipython().run_line_magic('matplotlib', 'inline')

sizes_covid_width = []
sizes_covid_height = []

sizes_normal_width = []
sizes_normal_height = []

sizes_pneum_width = []
sizes_pneum_height = []

covid_images1 = []
for img_path in glob.glob('dataset/covid/*'):
    im = mpimg.imread(img_path)
    covid_images1.append(im)
    sizes_covid_width.append(im.shape[0])
    sizes_covid_width.append(im.shape[1])
        

fig = plt.figure()
fig.suptitle('COVID')
plt.imshow(covid_images1[0], cmap='gray') 

pneumonia_images = []
for img_path in glob.glob('dataset/pneumonia/*'):
    pneumonia_images.append(mpimg.imread(img_path))

fig = plt.figure()
fig.suptitle('Pneumonoia')
plt.imshow(pneumonia_images[0], cmap='gray') 


normal_images = []
for img_path in glob.glob('dataset/normal/*'):
    normal_images.append(mpimg.imread(img_path))

fig = plt.figure()
fig.suptitle('NORMAL')
plt.imshow(normal_images[0], cmap='gray')


# In[ ]:


covid_images1[0][0][0][0]
import statistics
statistics.mean(covid_images1[0][0][0])


# In[ ]:


IMG_W = 224
IMG_H = 224
CHANNELS = 3

INPUT_SHAPE = (IMG_W, IMG_H, CHANNELS)
NB_CLASSES = 3
EPOCHS = 55
BATCH_SIZE = 16


# In[ ]:


train_datagen = ImageDataGenerator(rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.15)

train_generator = train_datagen.flow_from_directory(
    'dataset',
    target_size=(IMG_H, IMG_W),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training')

validation_generator = train_datagen.flow_from_directory(
    'dataset',  
    target_size=(IMG_H, IMG_W),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation')


# In[ ]:





# In[ ]:


model = Sequential()
model.add(Conv2D(80, (3, 3), input_shape=INPUT_SHAPE))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))


model.add(Conv2D(64, (3, 3)))

model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(3))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy',
          optimizer='adam',
          metrics=['accuracy'])


# In[ ]:



history = model.fit_generator(
    train_generator,
    steps_per_epoch = train_generator.samples // BATCH_SIZE,
    validation_data = validation_generator, 
    validation_steps = validation_generator.samples // BATCH_SIZE,
    epochs = 80)


# In[ ]:


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train.', 'Valid.'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[ ]:


x_test, y_test=next(validation_generator)


# In[ ]:


score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[ ]:





# In[ ]:





# In[ ]:


model.save_weights("3classes.h5")


# LAST

# In[ ]:


Classifier = Sequential()
Classifier.add(Conv2D(80, (3, 3), input_shape=(224,224,3)))
Classifier.add(Activation('relu'))
Classifier.add(MaxPooling2D(pool_size=(2, 2)))
Classifier.add(Dropout(0.2))

Classifier.add(Conv2D(64, (3, 3)))
Classifier.add(Activation('relu'))
Classifier.add(MaxPooling2D(pool_size=(2, 2)))
Classifier.add(Dropout(0.5))

Classifier.add(Conv2D(64, (3, 3)))
Classifier.add(Activation('relu'))


Classifier.add(Conv2D(80, (3, 3)))

Classifier.add(Activation('relu'))
Classifier.add(MaxPooling2D(pool_size=(2, 2)))

Classifier.add(Flatten())
Classifier.add(Dense(64))
Classifier.add(Activation('relu'))
Classifier.add(Dropout(0.5))

Classifier.add(Dense(3))
Classifier.add(Activation('softmax'))
Classifier.compile(loss='categorical_crossentropy',
          optimizer='rmsprop',
          metrics=['accuracy'])


# In[ ]:


BATCH_SIZE=8
history2 = Classifier.fit_generator(
    train_generator,
    steps_per_epoch = train_generator.samples // BATCH_SIZE,
    validation_data = validation_generator, 
    validation_steps = validation_generator.samples // BATCH_SIZE,
    epochs = 80)


# In[ ]:


plt.plot(history2.history['accuracy'])
plt.plot(history2.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train.', 'Valid.'], loc='upper left')
plt.show()

plt.plot(history2.history['loss'])
plt.plot(history2.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[ ]:





# In[ ]:


score = Classifier.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[ ]:





# In[ ]:


from keras.models import model_from_json

model_json = Classifier.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)


# In[ ]:


Classifier.save_weights('Classifier.h5')


# In[ ]:





# In[ ]:


score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# # Transfer learning

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16,InceptionV3
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split


# In[ ]:





# In[ ]:





# In[ ]:


baseModel = InceptionV3(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))
#construct the head of the model that will be placed on top of the
#the base model

headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(4, 4))(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(64, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(3, activation="softmax")(headModel)


model3 = Model(inputs=baseModel.input, outputs=headModel)

for layer in baseModel.layers:
    layer.trainable = False


# In[ ]:


# compile our model
print("[INFO] compiling model...")
opt = Adam(lr=1e-3, decay=1e-3 / EPOCHS)
model3.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])


# In[ ]:


BATCH_SIZE=8
history = model3.fit_generator(
    train_generator,
    steps_per_epoch = train_generator.samples // BATCH_SIZE,
    validation_data = validation_generator, 
    validation_steps = validation_generator.samples // BATCH_SIZE,
    epochs = 50)


# In[ ]:


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train.', 'Valid.'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


model = Sequential()
model.add(Conv2D(80, (3, 3), input_shape=INPUT_SHAPE))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(BatchNormalization())

model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('softmax'))


# In[ ]:


model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
#model.compile(Adam(lr=0.0001),loss="binary_crossentropy", metrics=["accuracy"])


# In[ ]:



history=model.fit(X_train, train_y,
                  batch_size=8, 
                  epochs=20,
                  validation_data=(X_test, y_test))


# In[ ]:


from keras.utils.np.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


# In[ ]:


train_y=tf.keras.utils.to_categorical(y_train)


# In[ ]:


ytest=tf.keras.utils.to_categorical(y_test)


# In[ ]:


X_test,ytest=next(validation_generator)


# In[ ]:


import warnings
warnings.filterwarnings("ignore")


# In[ ]:


import pandas as pd


# In[ ]:





# In[ ]:


import random


# In[ ]:


y_test1


# In[ ]:


import numpy


# In[ ]:


x_test.shape


# In[ ]:


labelsFaces =['COVID-19', 'Normal',"pneumonia"]

 

predictedExpression = model.predict(x_test)

figure = plt.figure(figsize=(20, 8))

for i, index in enumerate(numpy.random.choice(x_test.shape[0], size=16, replace=False)):
    ax = figure.add_subplot(5, 10, i + 1, xticks=[], yticks=[])
    # Display each image
    ax.imshow(numpy.squeeze(x_test[index]))
    predict_index = numpy.argmax(predictedExpression[index])
    true_index = numpy.argmax(y_test[index])
    # Set the title for each image
    ax.set_title("{} ({})".format(labelsFaces[predict_index], 
                                  labelsFaces[true_index]),
                                  color=("green" if predict_index == true_index else "red"))
plt.show()
import warnings
warnings.filterwarnings("ignore")


# In[ ]:


labelsFaces =['COVID-19', 'Normal',"pneumonia"]



predictedExpression = Classifier.predict(x_test)

figure = plt.figure(figsize=(20, 8))

for i, index in enumerate(numpy.random.choice(x_test.shape[0], size=16, replace=False)):
   ax = figure.add_subplot(5, 10, i + 1, xticks=[], yticks=[])
   # Display each image
   ax.imshow(numpy.squeeze(x_test[index]))
   predict_index = numpy.argmax(predictedExpression[index])
   true_index = numpy.argmax(y_test[index])
   # Set the title for each image
   ax.set_title("{} ({})".format(labelsFaces[predict_index], 
                                 labelsFaces[true_index]),
                                 color=("green" if predict_index == true_index else "red"))
plt.show()
import warnings
warnings.filterwarnings("ignore")


# In[ ]:


predictedExpression


# In[ ]:





# In[ ]:





# In[ ]:


path = "../input/covid-19-x-ray-10000-images/dataset"


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




