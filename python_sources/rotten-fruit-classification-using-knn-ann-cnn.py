# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
%matplotlib inline
import tensorflow as tf
import keras
import glob
import cv2
import keras as k
from tensorflow.keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense, Dropout
from keras.models import model_from_json
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
#import numpy as np
#import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['axes.titlesize'] = 8
import h5py


import os
print(os.listdir("C:/Users/STSC/Downloads/train-20200424T221454Z-001/train"))


fruit_images = []
labels = [] 
for fruit_dir_path in glob.glob("C:/Users/STSC/Downloads/train-20200424T221454Z-001/train/*"):
    fruit_label = fruit_dir_path.split("/")[-1]
    for image_path in glob.glob(os.path.join(fruit_dir_path, "*.png")):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (110, 110))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 
        fruit_images.append(image)
        labels.append(fruit_label)
fruit_images = np.array(fruit_images)
labels = np.array(labels)
# Preprocessing

label_to_id_dict = {v:i for i,v in enumerate(np.unique(labels))}
id_to_label_dict = {v: k for k, v in label_to_id_dict.items()}
id_to_label_dict
label_ids = np.array([label_to_id_dict[x] for x in labels])
fruit_images.shape, label_ids.shape, labels.shape

#Testing data

validation_fruit_images = []
validation_labels = [] 
for fruit_dir_path in glob.glob("C:/Users/STSC/Downloads/test-20200424T222521Z-001/test/*"):
    fruit_label = fruit_dir_path.split("/")[-1]
    for image_path in glob.glob(os.path.join(fruit_dir_path, "*.png")):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (110, 110))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        validation_fruit_images.append(image)
        validation_labels.append(fruit_label)
validation_fruit_images = np.array(validation_fruit_images)
validation_labels = np.array(validation_labels)

validation_fruit_images.shape
label_to_id_dict = {v:i for i,v in enumerate(np.unique(validation_labels))}
id_to_label_dict = {v: k for k, v in label_to_id_dict.items()}

validation_label_ids = np.array([label_to_id_dict[x] for x in validation_labels])

#Data spliting
X_train, X_test = fruit_images, validation_fruit_images
Y_train, Y_test = label_ids, validation_label_ids

#Normalize color values to between 0 and 1
X_train = X_train/255
X_test = X_test/255

#Make a flattened version for some of our models
X_flat_train = X_train.reshape(X_train.shape[0], 110*110*3)
X_flat_test = X_test.reshape(X_test.shape[0], 110*110*3)

#One Hot Encode the Output what is this 60 
Y_train = keras.utils.to_categorical(Y_train, 60)
Y_test = keras.utils.to_categorical(Y_test, 60)

print('Original Sizes:', X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)
print('Flattened:', X_flat_train.shape, X_flat_test.shape)

#Dataset images

import matplotlib.pyplot as plt

fig = plt.figure(figsize =(30,5))
for i in range(10):
    ax = fig.add_subplot(2,5,i+1,xticks=[],yticks=[])
    ax.imshow(np.squeeze(X_train[i]))
    

#KNN classifier

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers import LSTM, Input, TimeDistributed
from keras.models import Model
from keras.optimizers import RMSprop, SGD

# Import the backend
from keras import backend as K

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier()
knn.fit(X_flat_train,Y_train)
knn.score(X_flat_test,Y_test)

#ANN
model_dense = Sequential()

# Add dense layers to create a fully connected MLP
# Note that we specify an input shape for the first layer, but only the first layer.
# Relu is the activation function used
model_dense.add(Dense(128, activation='relu', input_shape=(X_flat_train.shape[1],)))
# Dropout layers remove features and fight overfitting
model_dense.add(Dropout(0.1))
model_dense.add(Dense(64, activation='relu'))
model_dense.add(Dropout(0.1))
# End with a number of units equal to the number of classes we have for our outcome
model_dense.add(Dense(60, activation='softmax'))
model_dense.summary()
# Compile the model to put it all together.
#categorical_crossentropy loss
model_dense.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

#Changed the batch size here
history_dense = model_dense.fit(X_flat_train, Y_train,
                          batch_size=50,
                          epochs=10,
                          verbose=1,
                          validation_data=(X_flat_test, Y_test))
score = model_dense.evaluate(X_flat_test, Y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

#LOSS train and valid

import matplotlib.pyplot as plt


def plot_history(histories, key='accuracy'):
    plt.figure(figsize=(6,5))
    for name, history in histories:
        val = plt.plot(history.epoch, history.history['val_'+key],
                       '--', label=name.title()+' Val')
        plt.plot(history.epoch, history.history[key], color='orange',
             label=name.title()+' Train')

    plt.xlabel('Epochs')
    plt.ylabel(key.replace('_',' ').title())
    plt.legend()

  #plt.xlim([0,max(history.epoch)])
    plt.xlim([0,10])
    plt.ylim([0,1.1])

plot_history([('model1', history_dense)])

#ANN with 30 epochs
model_deep = Sequential()
# Add dense layers to create a fully connected MLP
# Note that we specify an input shape for the first layer, but only the first layer.
# Relu is the activation function used
model_deep.add(Dense(256, activation='relu', input_shape=(X_flat_train.shape[1],)))
# Dropout layers remove features and fight overfitting
model_deep.add(Dropout(0.05))
model_deep.add(Dense(128, activation='relu'))
model_deep.add(Dropout(0.05))
model_deep.add(Dense(128, activation='relu'))
model_deep.add(Dropout(0.05))
model_deep.add(Dense(128, activation='relu'))
model_deep.add(Dropout(0.05))
model_deep.add(Dense(128, activation='relu'))
model_deep.add(Dropout(0.05))
# End with a number of units equal to the number of classes we have for our outcome
model_deep.add(Dense(60, activation='softmax'))

model_deep.summary()

# Compile the model to put it all together.
model_deep.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

history_deep = model_deep.fit(X_flat_train, Y_train,
                          batch_size=50,
                          epochs=10,
                          verbose=1,
                          validation_data=(X_flat_test, Y_test))
score = model_deep.evaluate(X_flat_test, Y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

#CNN
#there are using maxpool convolution and final dense layer.
model_cnn = Sequential()
# First convolutional layer, note the specification of shape
model_cnn.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(110, 110, 3)))
model_cnn.add(Conv2D(64, (3, 3), activation='relu'))
model_cnn.add(MaxPooling2D(pool_size=(2, 2)))
model_cnn.add(Dropout(0.25))
model_cnn.add(Flatten())
model_cnn.add(Dense(128, activation='relu'))
model_cnn.add(Dropout(0.5))
model_cnn.add(Dense(60, activation='softmax'))

model_cnn.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model_cnn.fit(X_train, Y_train,
          batch_size=128,
          epochs=1,
          verbose=1,
          validation_data=(X_test, Y_test))
score = model_cnn.evaluate(X_test, Y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

model_cnn.fit(X_train, Y_train,
          batch_size=128,
          epochs=30,
          verbose=1,
          validation_data=(X_test, Y_test))
score = model_cnn.evaluate(X_test, Y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])



#Saving the weights of CNN model as we got the best accuracy of 94% out of three classifiers

fname="C:/Users/STSC/Downloads/weights_cnn_classification_augmented_data.hdf5"
model_cnn.save_weights(fname,overwrite=True)
#In future, you can use this model and later you can load this model for prediction


#Defined predicted class
def predictFruitClass(ImagePath, trainedModel, class_dict):
    """
    Perform class prediction on input image and print predicted class.

    Args:
        ImagePath(str): Absolute Path to test image
        trainedModel(object): trained model from method getTrainedModel()
        DictOfClasses(dict): python dict of all image classes.

    Returns:
        Probability of predictions for each class.
    """
    x = cv2.imread(ImagePath,cv2.IMREAD_COLOR)
    x = cv2.resize(x, (110, 110))
    x = cv2.cvtColor(x, cv2.COLOR_RGB2BGR) 
    x = np.array(x)
    # for Display Only
    import matplotlib.pyplot as plt
    plt.imshow((x * 255).astype(np.uint8))
    x = np.expand_dims(x, axis=0)
    prediction_class = trainedModel.predict_classes(x, batch_size=1)
    #prediction_probs = trainedModel.predict_proba(x, batch_size=1)
    #print('probs:',prediction_probs)
    print('Predicted class',prediction_class[0])
    for key, value in class_dict.items():
        if value == prediction_class.item():
            return key
    return None


#Function to save the model
def SaveModelFile(classify_model, save_model_filename):
    """
    Saves trained classification model

    Args:
        ClassifyModel : trained classification Model
        save_model_filename(str): filename, to save trained model,without extension.

    Returns:
        save_model_filename(str): filename with extension.
    """
    #today = date.today()
    #date_str = today.strftime("%d%m%y")
    #save_model_filename = '_'.join([save_model_filename, dat
    save_model_filename = save_model_filename + '.h5'
    classify_model.save(save_model_filename)
    print('Done Saving Model File...')
    return save_model_filename


#Function to save trained model
def getTrainedModel(PATH_TO_TRAINED_MODEL_FILE):
    """
    Loads trained-saved model from file(.h5) and returns as a object.

    Args:
        PATH_TO_TRAINED_MODEL_FILE(str): path to saved model file.

    returns:
        trainedModel(model object): returns a model saved as a <.h5>
    """
    trainedModel = load_model(PATH_TO_TRAINED_MODEL_FILE)
    return trainedModel

SaveModelFile(model_cnn,"savedmodel")
trained_model_path = "C:/Users/STSC/Downloads/savedmodel.h5"
trained_model = getTrainedModel(trained_model_path)


## Uploading new image from the internet to test our model

image_path = "C:/Users/STSC/Downloads/rottenbanana1.png"
id_labels={0: 'freshapples',
 1: 'freshbanana',
 2: 'freshoranges',
 3: 'rottenapples',
 4: 'rottenbanana',
 5: 'rottenoranges'}
print(id_labels.items())

single_pred = predictFruitClass(image_path,trained_model, id_labels)
print(single_pred)

#And it predicted perfectely fine. But sometimes it do get confused between orrange and apple as features are quite similar to each other. But that can be fixed by 
#tuning the hyperparameters.