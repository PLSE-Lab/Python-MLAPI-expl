#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        1+1
        #print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


#references
#flow - https://keras.io/preprocessing/image/#flow
#fit_generator - https://keras.io/models/sequential/


# In[ ]:


import warnings
warnings.filterwarnings('ignore')

from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.optimizers import Adam
from keras.optimizers import SGD
import numpy as np
import os
import cv2
from keras.utils import to_categorical
from sklearn.model_selection import StratifiedKFold
import numpy
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.model_selection import ParameterGrid
from keras.applications import VGG19
from keras.applications.vgg19 import preprocess_input as vgg19_preprocess_input
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input as resnet50_preprocess_input
from keras.models import load_model


# In[ ]:


#constants
NUM_CLASSES = 12
WIDTH = 128
HEIGHT = 128
DEPTH = 3
INPUT_SHAPE = (WIDTH, HEIGHT, DEPTH)
PATH = "/kaggle/input/plant-seedlings-classification/"
TRAINING_SAMPLES_LIMIT = 50
RESNET50_MODEL = "resnet50"
VGG19_MODEL = "vgg19"
TRANSFER_LEARNING = "TransferLearning"


# In[ ]:


def class_labels_to_integer(label):
    label = label.strip()
    if label == "Black-grass":  return 0
    if label == "Charlock":  return 1
    if label == "Cleavers":  return 2
    if label == "Common Chickweed":  return 3
    if label == "Common wheat":  return 4
    if label == "Fat Hen":  return 5
    if label == "Loose Silky-bent": return 6
    if label == "Maize":  return 7
    if label == "Scentless Mayweed": return 8
    if label == "Shepherds Purse": return 9
    if label == "Small-flowered Cranesbill": return 10
    if label == "Sugar beet": return 11
    return -1

def integer_to_classe_labels(i):
    if i == 0: return "Black-grass"
    elif i == 1: return "Charlock"
    elif i == 2: return "Cleavers"
    elif i == 3: return "Common Chickweed"
    elif i == 4: return "Common wheat"
    elif i == 5: return "Fat Hen"
    elif i == 6: return "Loose Silky-bent"
    elif i == 7: return "Maize"
    elif i == 8: return "Scentless Mayweed"
    elif i == 9: return "Shepherds Purse"
    elif i == 10: return "Small-flowered Cranesbill"
    elif i == 11: return "Sugar beet"
    return "Invalid Class"


# In[ ]:


#filenames - filename of image
#data - image pixel arrays
def readTestData(testDir):
    data = []
    filenames = []
    images = os.listdir(testDir)
    for imageFileName in images:
        imageFullPath = os.path.join(testDir, imageFileName)
        img = load_img(imageFullPath)
        arr = img_to_array(img)
        arr = cv2.resize(arr, (HEIGHT,WIDTH)) 
        data.append(arr)
        filenames.append(imageFileName)
    return filenames,data

#data - image pixel arrays
#labels - integers
def readTrainData(trainDirectory):
    data = []
    labels = []
    dirs = os.listdir(trainDirectory)
    for dir in dirs:
        absDirPath = os.path.join(os.path.sep,trainDirectory, dir)
        images = os.listdir(absDirPath)
        count = 0
        for imageFileName in images:
            imageFullPath = os.path.join(trainDirectory, dir, imageFileName)
            img = load_img(imageFullPath)
            arr = img_to_array(img)
            arr = cv2.resize(arr, (HEIGHT,WIDTH))
            data.append(arr)
            label = class_labels_to_integer(dir)
            labels.append(label)
            count = count + 1
            if (count > TRAINING_SAMPLES_LIMIT):
                break;
    print("=======================================")
    print("Read Train Data")
    print("Data Length = ", len(data))
    print("Label Lenght = ", len(labels))
    print("=======================================")
    return data, labels


# In[ ]:


#callbacks for keras modal
def get_callbacks(patience):
    print("Get Callbacks")

    lr_reduce = ReduceLROnPlateau(monitor='val_acc', factor=0.1, epsilon=1e-5, patience=patience, verbose=1)
    #msave = ModelCheckpoint(filepath, save_best_only=True)
    return [lr_reduce, EarlyStopping()]


# In[ ]:


#create model from scratch
def createModel(number_of_hidden_layers, activation, optimizer, learning_rate, epochs):
    print("Create Model")

    model = Sequential()

    model.add(Conv2D(20, (5, 5), padding="same", input_shape=INPUT_SHAPE))
    model.add(Activation(activation))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(50, (5, 5), padding="same"))
    model.add(Activation(activation))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Flatten())
    
    for i in range(0,number_of_hidden_layers):
        model.add(Dense(400))
        model.add(Activation(activation))

    model.add(Dense(output_dim=12))
    model.add(Activation("softmax"))

    if optimizer == 'SGD':
        opt = SGD(lr=learning_rate, decay=learning_rate / epochs)
    elif optimizer == 'Adam':
        opt = Adam(lr=learning_rate, decay=learning_rate / epochs)

    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    return model


# In[ ]:


#create modeel from pre trined model
def createModelFromPreTrainedModel(pretrained_model, number_of_hidden_layers, activation, optimizer, learning_rate, epochs):
    #we use pre trined model as the feature extractor
    print("Create Pre Trained Model")

    model = Sequential()
    
    #feature extracting layers
    if pretrained_model == RESNET50_MODEL:
        model.add(ResNet50(weights='imagenet', input_shape=INPUT_SHAPE, include_top=False))
    elif pretrained_model == VGG19_MODEL:
        model.add(VGG19(weights='imagenet', input_shape=INPUT_SHAPE, include_top=False))
    # freeze feature extracting layers
    for layer in model.layers:
        layer.trainable = False
    
    #classifier layers
    model.add(Flatten())
    for i in range(0,number_of_hidden_layers):
        model.add(Dense(400))
        model.add(Activation(activation))

    model.add(Dense(output_dim=12))
    model.add(Activation("softmax"))

    if optimizer == 'SGD':
        opt = SGD(lr=learning_rate, decay=learning_rate / epochs)
    elif optimizer == 'Adam':
        opt = Adam(lr=learning_rate, decay=learning_rate / epochs)

    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    return model


# In[ ]:


#train non-pre trained models
def train(images, labels, epochs, batch_size, learning_rate, cross_validation_folds, steps_mulitplier, activation, number_of_hidden_layers, optimizer):
    print("Train Model")

    print("Data Length = ", len(images))
    print("Label Lenght = ", len(labels))


    images = np.array(images, dtype="float") / 255.0
    labels = np.array(labels)
    labels_vector =  to_categorical(labels, num_classes=12)
    
    aug = ImageDataGenerator(rotation_range=360, width_shift_range=0.3,                             height_shift_range=0.3, shear_range=0.2, zoom_range=0.3,                             horizontal_flip=True, vertical_flip=True, fill_mode="nearest")

    print("Cross validation")
    kfold = StratifiedKFold(n_splits=cross_validation_folds, shuffle=True)
    cvscores = []
    iteration = 1

    for train_index, test_index in kfold.split(images,labels):

        print("======================================")
        print("Iteration = ", iteration)

        iteration = iteration + 1

        trainX,testX=images[train_index],images[test_index]
        trainY,testY=labels_vector[train_index],labels_vector[test_index]

        model = createModel(number_of_hidden_layers, activation, optimizer, learning_rate, epochs)

        #Trains the model on data generated batch-by-batch by a Python generator
        model.fit_generator(aug.flow(trainX, trainY, batch_size=batch_size),                            validation_data = (testX, testY),                             steps_per_epoch=steps_mulitplier*len(trainX) // batch_size,                             epochs=epochs, verbose=1,                             callbacks = get_callbacks(patience=2))
        
        scores = model.evaluate(testX, testY)
        print("Accuarcy %s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
        cvscores.append(scores[1] * 100)

    accuracy = numpy.mean(cvscores);
    std = numpy.std(cvscores);
    print("Accuracy: %.2f%% (+/- %.2f%%)" % (accuracy, std))
    return accuracy, std


# In[ ]:


#train pre-trained models
def trainPreTrainedModel(pretrained_model, images, labels, epochs, batch_size, learning_rate, cross_validation_folds, steps_mulitplier, activation, number_of_hidden_layers, optimizer):
    print("Train Pre Trained Model")

    print("Data Length = ", len(images))
    print("Label Lenght = ", len(labels))

    images = np.array(images, dtype="float") / 255.0
    if pretrained_model == RESNET50_MODEL:
        images = resnet50_preprocess_input(images)
    elif pretrained_model == VGG19_MODEL:
        images = vgg19_preprocess_input(images)
    
    labels = np.array(labels)
    labels_vector =  to_categorical(labels, num_classes=12)

    aug = ImageDataGenerator(rotation_range=360, width_shift_range=0.3,                             height_shift_range=0.3, shear_range=0.2, zoom_range=0.3,                             horizontal_flip=True, vertical_flip=True, fill_mode="nearest")

    print("Cross validation")
    kfold = StratifiedKFold(n_splits=cross_validation_folds, shuffle=True)
    cvscores = []
    iteration = 1

    for train_index, test_index in kfold.split(images,labels):

        print("======================================")
        print("Iteration = ", iteration)

        iteration = iteration + 1

        trainX,testX=images[train_index],images[test_index]
        trainY,testY=labels_vector[train_index],labels_vector[test_index]

        model = createModelFromPreTrainedModel(pretrained_model,number_of_hidden_layers, activation, optimizer, learning_rate, epochs)

        #Trains the model on data generated batch-by-batch by a Python generator
        model.fit_generator(aug.flow(trainX, trainY, batch_size=batch_size),                            validation_data = (testX, testY),                             steps_per_epoch=steps_mulitplier*len(trainX) // batch_size,                             epochs=epochs, verbose=1,                             callbacks = get_callbacks(patience=2))
        
        scores = model.evaluate(testX, testY)
        print("Accuarcy %s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
        cvscores.append(scores[1] * 100)

    accuracy = numpy.mean(cvscores);
    std = numpy.std(cvscores);
    print("Accuracy: %.2f%% (+/- %.2f%%)" % (accuracy, std))
    return accuracy, std


# In[ ]:


#find best params for models
def evaluateModels(learning_type):
    
    images, labels = readTrainData(os.path.join(os.path.sep,PATH, "train"))
        
    if learning_type == TRANSFER_LEARNING:
        print("Transfer Learning")
        parameters = {'pre_trained_model':[RESNET50_MODEL,VGG19_MODEL],
                    'epochs': [10],
                    'batch_size':[15],
                    'learning_rate':[0.001],
                    'activation':[ 'sigmoid'],
                    'number_of_hidden_layers':[1],
                    'optimizer':['SGD']
                    }
        parameterDictList = list(ParameterGrid(parameters))
        print(parameterDictList)
        
        #grid search using parameter grid
        for parameterDict in parameterDictList: 
            accuracy, std = trainPreTrainedModel(parameterDict['pre_trained_model'], images, labels, epochs = parameterDict['epochs'],                  batch_size = parameterDict['batch_size'], learning_rate = parameterDict['learning_rate'],                   cross_validation_folds = 5, steps_mulitplier = 1, activation = parameterDict['activation'],                  number_of_hidden_layers = parameterDict['number_of_hidden_layers'], optimizer = parameterDict['optimizer'])
            print("=======================================================================================")
            print("PreTrained Model: %s, Epochs: %d, Batch Size: %d, Learing Rate: %.3f%%, Activation: %s, Number of Hidden Layers: %d, Optimizer: %s Accuracy: %.2f%% (+/- %.2f%%)"                  % (parameterDict['pre_trained_model'], parameterDict['epochs'], parameterDict['batch_size'], parameterDict['learning_rate'], parameterDict['activation'],                      parameterDict['number_of_hidden_layers'],parameterDict['optimizer'], accuracy, std))
            print("=======================================================================================")
            parameterDict["Accuarcy"] = accuracy;
            
        print(parameterDictList)
    else:
        print("Non Transfer Learning")
        parameters = {'epochs': [5],
          'batch_size':[200],
          'learning_rate':[0.0001],
          'activation':['relu'],
          'number_of_hidden_layers':[2],
          'optimizer':['Adam']
          }
        parameterDictList = list(ParameterGrid(parameters))
        print(parameterDictList)
        
        #grid search using parameter grid
        for parameterDict in parameterDictList: 
            #print("Batch Size: %d, Epochs: %d" % (parameterSet['batch_size'], parameterSet['epochs']))
            accuracy, std = train(images, labels, epochs = parameterDict['epochs'],                  batch_size = parameterDict['batch_size'], learning_rate = parameterDict['learning_rate'],                   cross_validation_folds = 5, steps_mulitplier = 1, activation = parameterDict['activation'],                  number_of_hidden_layers = parameterDict['number_of_hidden_layers'], optimizer = parameterDict['optimizer'])
            print("=======================================================================================")
            print("Epochs: %d, Batch Size: %d, Learing Rate: %.3f%%, Activation: %s, Number of Hidden Layers: %d, Optimizer: %s Accuracy: %.2f%% (+/- %.2f%%)"                  % (parameterDict['epochs'], parameterDict['batch_size'], parameterDict['learning_rate'], parameterDict['activation'],                      parameterDict['number_of_hidden_layers'],parameterDict['optimizer'], accuracy, std))
            print("=======================================================================================")
            parameterDict["Accuarcy"] = accuracy;
            
        print(parameterDictList)


# In[ ]:


def buildModelFromBestParams(pretrained_model, epochs, batch_size, learning_rate, steps_mulitplier, activation, number_of_hidden_layers, optimizer):
    
    print("Build Model from Best Params")
    
    images, labels = readTrainData(os.path.join(os.path.sep,PATH, "train"))
    print("Data Length = ", len(images))
    print("Label Lenght = ", len(labels))

    images = np.array(images, dtype="float") / 255.0
    if pretrained_model == RESNET50_MODEL:
        images = resnet50_preprocess_input(images)
    elif pretrained_model == VGG19_MODEL:
        images = vgg19_preprocess_input(images)
        
    labels = np.array(labels)
    labels_vector =  to_categorical(labels, num_classes=12)
    
    aug = ImageDataGenerator(rotation_range=360, width_shift_range=0.3,                             height_shift_range=0.3, shear_range=0.2, zoom_range=0.3,                             horizontal_flip=True, vertical_flip=True, fill_mode="nearest")
    
    if pretrained_model == RESNET50_MODEL or pretrained_model == VGG19_MODEL:
        createModelFromPreTrainedModel(pretrained_model,number_of_hidden_layers, activation, optimizer, learning_rate, epochs)
    else:
        model = createModel(number_of_hidden_layers, activation, optimizer, learning_rate, epochs)
    
    model.fit_generator(aug.flow(images, labels_vector, batch_size=batch_size),                            steps_per_epoch=steps_mulitplier*len(images) // 32,                             epochs=epochs, verbose=1,                             callbacks = get_callbacks(patience=2))

    model.save("/kaggle/working/best_model")


# In[ ]:


#predict values 
def predict():
    print("Predicting......")
    
    filenames, images = readTestData(os.path.join(os.path.sep,PATH, "test"))
    images = np.array(images, dtype="float") / 255.0
    
    model = load_model('/kaggle/working/best_model')
    predictions = model.predict(images, batch_size=10, verbose=1)
 
    import csv  
    with open('submission.csv', 'w', newline='') as csvfile:
        fieldnames = ['file', 'species']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for index, file in enumerate(filenames):
            classesProbs = predictions[index]
            maxIdx = 0
            maxProb = 0;
            for idx in range(0,11): #select class with maximum probability
                if(classesProbs[idx] > maxProb):
                    maxIdx = idx
                    maxProb = classesProbs[idx]
            writer.writerow({'file': file, 'species': integer_to_classe_labels(maxIdx)})
    print("Prediction Completed")


# In[ ]:


def main():
    
    #evaluate models
    evaluateModels("")
    
    #build model from best params
    buildModelFromBestParams("", epochs = 5, batch_size =200, learning_rate = 0.0001, steps_mulitplier = 1, activation = 'sigmoid', number_of_hidden_layers = 4, optimizer = 'Adam');
    
    #predict
    predict()


# In[ ]:


main();

