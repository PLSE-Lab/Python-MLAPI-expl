# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import time
import os
import glob

import numpy as np # linear algebra
import cv2
from random import shuffle
from PIL import Image # used for loading images

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers. normalization import BatchNormalization
from keras.models import model_from_json


#print(os.listdir("../input/tsl finger spelling/Images"))




# Any results you write to the current directory are saved as output.


"""CONSTANTS"""

IMG_SIZE = 200
IMAGE_INDEX = 0
VALIDATION_IMAGE_IDS = [2, 6, 12, 16, 22, 26, 32, 36, 42, 46, 52, 56, 62, 66, 72, 76, 82, 86, 92, 96]
TEST_IMAGE_IDS = [1, 5, 11, 15, 21, 25, 31, 35, 41, 45, 51, 55, 61, 65, 71, 75, 81, 85, 91, 95]
DATASET_PATH = '../input/turkish-sign-languagefinger-spelling/tsl finger spelling/Images'

def saveModel(model, modelFilePath, weightFilePath):
    with open(modelFilePath, "w") as json_file:
        json_file.write(model.to_json())
    # serialize weights to HDF5
    model.save_weights(weightFilePath)

def loadModel(modelFilePath, weightFilePath):
    json_file = open(modelFilePath, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(weightFilePath)
    return loaded_model

def label_img(name):
  #create a list with 26 zeros
  labels = [0 for i in range(26)]
  
  #get char_code - for example 65 for A
  char_code = ord(name[0])
  
  #get only the files between A and Z
  if char_code > 64 and char_code < 91:
      #mark image label with 1
      labels[char_code - 65] = 1
      #print(labels)
      return np.array(labels)



def load_data(DIR, data_type = 'train'):
    exclude = []
    include = []
    if data_type == 'train':
        exclude.extend(VALIDATION_IMAGE_IDS)
        exclude.extend(TEST_IMAGE_IDS)
    elif data_type == 'validation':
        include.extend(VALIDATION_IMAGE_IDS)
    elif data_type == 'test':
        include.extend(TEST_IMAGE_IDS)

    
    data = []
    for img in glob.glob(DIR + '/*.png'):
        id = int(img[img.find("(")+1 : img.find(")")])
        if exclude.__len__() != 0 and id in exclude:
            continue
        elif include.__len__() != 0 and id not in include:
            continue

        label = label_img(img.split('/')[-1])
        """
        if label is none
        that means image is not between A-Z pass
        """
        if label is None:
            continue
        
        img = Image.open(img)
        img = img.convert('L')
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.ANTIALIAS)
        data.append([np.array(img), label])
            
    shuffle(data)
    return data

def trainModel():
    train_data = load_data(DATASET_PATH, 'train')
    validation_data = load_data(DATASET_PATH, 'validation')
    #print(train_data)
    """
    import matplotlib.pyplot as plt
    plt.imshow(train_data[43][0], cmap = 'gist_gray')
    """
    trainImages = np.array([i[0] for i in train_data]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    trainLabels = np.array([i[1] for i in train_data])
    
    validationImages = np.array([i[0] for i in validation_data]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    validationLabels = np.array([i[1] for i in validation_data])
    
    model = Sequential()
    model.add(Conv2D(32, kernel_size = (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(BatchNormalization())
    model.add(Conv2D(96, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(BatchNormalization())
    model.add(Conv2D(96, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Flatten())
    
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    
    model.add(Dense(128, activation='relu'))
    #model.add(Dropout(0.3))
    model.add(Dense(26, activation = 'softmax'))
    
    
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics = ['accuracy'])
    
    model.fit(trainImages, trainLabels, batch_size = 50, epochs = 10, verbose = 1, validation_data=(validationImages, validationLabels))
    
    return model


def testModel(model):
    test_data = load_data(DATASET_PATH, 'test')
    testImages = np.array([i[0] for i in test_data]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    testLabels = np.array([i[1] for i in test_data])

    totalList = []
    charLists = {}
    for index, test_image in enumerate(testImages):
        test_image = np.expand_dims(test_image,axis=0)
        prediction, labelIndex = predict(model, test_image)
        predicted = labelIndex == np.where(testLabels[index] == 1)[0][0]
        totalList.append(predicted)
        
        if prediction not in charLists:
            charLists[prediction] = {"total": 0, "predicted": 0}
        charLists[prediction]['total'] += 1
        if predicted:
            charLists[prediction]['predicted'] += 1

    total = totalList.__len__()
    predicted = totalList.count(True)
    print (total)
    print (predicted)
    print (100 * predicted /total)
    
    charLists = dict(sorted(charLists.items()))
    for prediction in charLists:
        success = 100 * charLists[prediction]['predicted'] / charLists[prediction]['total']
        print (prediction + ': ' + str(round(success)))
        

def showImage(image, prediction, index):
    #cv2.imwrite('h.jpg', image)
    #if prediction == 'A':
    cv2.imwrite('A_'+ prediction + '_' + str(index) +'.jpg', image)
    

def captureVideoAndTestFrames(model, index):
    
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    
    frame = cv2.resize(frame, dsize=(IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)
    
    frame_test = np.expand_dims(frame, axis=0)
    frame_test = frame_test.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    #cv2.imshow("frame", frame)
    prediction = predict(model, frame_test)
    showImage(frame, prediction, index)

def predict(model, test_image):
    result = model.predict(test_image, batch_size=10, verbose=0)
    maxPosibility = max(result[0])
    classIds = [i for i, j in enumerate(result[0]) if j == maxPosibility]
    """
    print(result)
    print(maxPosibility)
    """
    #print ('prediction result: ')
    for value in classIds:
        return [chr(value + 65), value]
    """
    print ('all possibilities: ')
    for index, value in enumerate(result[0]):
        print (chr(index + 65) + ' -> ' + str(value))
    """

#index = 0

model = trainModel()
saveModel(model, 'model.json', 'model.h5')
#model = loadModel('model.json', 'model.h5')
#loadModelAndTestImages()
testModel(model)
"""
while True:
    captureVideoAndTestFrames(model, index)
    index += 1
    time.sleep(2)
"""