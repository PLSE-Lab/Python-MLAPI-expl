# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from PIL import Image

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os, glob
#print(os.listdir("../input/fingers/fingers/train"))

train_img_list = glob.glob("../input/fingers/fingers/train/*.png")
test_img_list = glob.glob("../input/fingers/fingers/test/*.png")
#print(len(train_img_list),
#     len(test_img_list), sep = '\n')
#img = Image.open("../input/fingers/fingers/train/b25805c1-572e-4a9d-ab00-8e4a43a96654_0.png")
#img = np.array(img)
#img = np.reshape(img, (128, 128, -1)) 
#print(img.shape)
#img_read = io.imread("../input/fingers/fingers/train/b25805c1-572e-4a9d-ab00-8e4a43a96654_0.png")     
X_Train = []
Y_Train = []
X_Test = []
Y_Test = []
 
from keras.utils import np_utils
    
for img in train_img_list:
    #print(img)
    #label = np_utils.to_categorical(img[-5], 6)
    Y_Train.append(img[-5])
    img = Image.open(img)
    img = np.array(img)
    #print(img.shape)
    img = np.reshape(img, (128, 128, -1)) 
    #print(img.shape)
    #img_read = transform.resize(img_read, (128,128), mode = 'constant')
    X_Train.append(img)
    
print("Loading Training Data Done")

for img in test_img_list:
    #print(img)
    #label = np_utils.to_categorical(img[-5], 6)
    Y_Test.append(img[-5])
    img = Image.open(img)
    img = np.array(img)
    #print(img.shape)
    img = np.reshape(img, (128, 128, -1)) 
    #print(img.shape)
    #img_read = transform.resize(img_read, (128,128), mode = 'constant')
    X_Test.append(img)    
print("Loading Test Data Done")    


X_Train = np.array(X_Train)
Y_Train = np.array(Y_Train)     
print("Training Data Shape ",X_Train.shape)

X_Test = np.array(X_Test)
#X_Test /= 255
Y_Test = np.array(Y_Test)
#Y_Test /= 255
print("Test Data Shape ",X_Test.shape)

Y_Train = np_utils.to_categorical(Y_Train, 6)
Y_Test = np_utils.to_categorical(Y_Test, 6)


from sklearn.model_selection import train_test_split

X_Train, X_Validation, Y_Train, Y_Validation = train_test_split(X_Train, Y_Train, test_size=0.2, random_state=1)


from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization

model = Sequential()
model.add(Convolution2D(256, (3, 3), padding='same', input_shape=(128,128,1))) 
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2),strides=1))
model.add(Dropout(0.10))

model.add(Convolution2D(128, (3, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2),strides=1))
model.add(Dropout(0.20))

model.add(Convolution2D(64, (3, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2),strides=1))
model.add(Dropout(0.30))

model.add(Convolution2D(32, (3, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2),strides=1))
model.add(Dropout(0.40))

model.add(Flatten()) # No dropout after flattening.
model.add(Dense(100))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Dense(6))
model.add(BatchNormalization())
model.add(Activation('softmax'))

from keras.optimizers import SGD,RMSprop,Adam

opt = SGD(lr=0.01)
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss='categorical_crossentropy', optimizer= adam, metrics = ['accuracy'])

model.fit(X_Train, Y_Train, batch_size=32, epochs=4,verbose=1,shuffle=True, validation_data=(X_Validation, Y_Validation))

from sklearn.metrics import classification_report,confusion_matrix

Y_pred = model.predict(X_Test)
print(Y_pred)
y_pred = np.argmax(Y_pred, axis=1)
print(y_pred)
  

y_pred = model.predict_classes(X_Test)
print(y_pred)

p=model.predict_proba(X_Test) # to predict probability

target_names = ['0', '1', '2','3','4','5']
print(classification_report(np.argmax(Y_Test,axis=1), y_pred,target_names=target_names))
print(confusion_matrix(np.argmax(Y_Test,axis=1), y_pred))

pred = model.evaluate(X_Test,
                      Y_Test,
                    batch_size = 32)

print("Accuracy of model on test data is: ",pred[1]*100)
