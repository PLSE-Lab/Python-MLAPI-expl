# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from imutils import paths
import cv2 as cv
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        #print(os.path.join(dirname, filename))
        pass
#print('')

lis = ['paper','stone','scissors']
fix_path = r'/kaggle/input/rock-paper-scissor/rps/rps'
test_path = r'/kaggle/input/rock-paper-scissor/rps-test-set/rps-test-set'
imgpa =[]
X = []
Y = []
for i in lis:
    #print(test_path + '\\' + i)
    imgpa.append(list(paths.list_images(fix_path + '//' + i)))
    
    
for i,j in enumerate(imgpa):
    for k in j:
        img_t = cv.imread(k)
        img_g = cv.cvtColor(img_t, cv.COLOR_BGR2GRAY)
        img_r = cv.resize(img_g, (28, 28))
        X.append(img_r)
        Y.append(i)
        
x = np.array(X)
X_n = x.reshape(x.shape[0], 28, 28, 1)
y = np.array(Y)
Y_new = to_categorical(y)

x_train, x_test, y_train, y_test = train_test_split(X_n, Y_new, test_size=0.3)
sps_model = models.Sequential()
sps_model.add(layers.Conv2D(64,kernel_size=(3,3),activation='relu',input_shape=(28,28,1)))
sps_model.add(layers.Conv2D(32,kernel_size=(3,3),activation='relu'))
sps_model.add(layers.MaxPool2D(pool_size=(2, 2), strides=1))
sps_model.add(layers.Flatten())
sps_model.add(layers.Dense(128,activation='relu'))
sps_model.add(layers.Dropout(0.2))
sps_model.add(layers.Dense(3,activation='softmax'))
sps_model.compile(optimizer='adam',loss=losses.categorical_crossentropy, metrics=['acc'])
sps_model.fit(x_train,y_train,epochs=10,batch_size=200)
sps_model.evaluate(x_test,y_test)
sps_model.save('sps_model.h5')
        

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

class Build_SPS_Model:
    def __init__(self,x,y):
        self.X = x
        self.Y = y
        

    def get_model(self):
        model_sps = load_model('sps_model.h5')
        return model_sps


    def build_model_keras(self):
        x_train, x_test, y_train, y_test = train_test_split(self.X, self.Y, test_size=0.3)
        sps_model = models.Sequential()
        sps_model.add(layers.Conv2D(64,kernel_size=(3,3),activation='relu',input_shape=(28,28,1)))
        sps_model.add(layers.Conv2D(32,kernel_size=(3,3),activation='relu'))
        sps_model.add(layers.MaxPool2D(pool_size=(2, 2), strides=1))
        sps_model.add(layers.Flatten())
        sps_model.add(layers.Dense(128,activation='relu'))
        sps_model.add(layers.Dropout(0.2))
        sps_model.add(layers.Dense(3,activation='softmax'))
        sps_model.compile(optimizer='adam',loss=losses.categorical_crossentropy, metrics=['acc'])
        sps_model.fit(x_train,y_train,epochs=10,batch_size=200)
        sps_model.evaluate(x_test,y_test)
        sps_model.save('sps_model.h5')


