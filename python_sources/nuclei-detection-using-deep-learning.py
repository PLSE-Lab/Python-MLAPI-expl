# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/content'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Input, Convolution2D,MaxPooling2D,Flatten,Dense,Dropout
from keras.utils import np_utils
import tensorflow


dataframe = pd.read_csv("/kaggle/input/pixelss-intensity-of-positive-and-negative-nuclei/data.csv")

img_rows, img_cols = 20, 20
num_classes = 2

data = np.array(dataframe)
X = data[:,1:]
print(X.shape)
y = data[:,-1]
X = X/255.0
print(X.shape)
print(y)


y.shape
np.unique(y,return_counts=True)

X_train = X.reshape(X.shape[0], img_rows, img_cols, 1)
#X_train = X.reshape((-1,20,20,1))
print(X_train.shape)
input_shape = (img_rows, img_cols, 1)

Y_train = np_utils.to_categorical(y)
Y_train.shape

for i in range(10):
    plt.imshow(X_train[i].reshape(20,20),cmap = 'gray')
    plt.show()
    
    
import numpy as np
import keras
#from keras.models import Sequential
#from keras.datasets import mnist
#from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D
#from tensorflow.python.keras import Sequential
print(keras.__version__)
#from keras import backend as K
#K.common.image_dim_ordering()
#K.common.set_image_dim_ordering('tf')
import tensorflow as tf
from tensorflow.keras.models import load_model

model2 = keras.models.Sequential()
model2.add(keras.layers.Conv2D(32, kernel_size=(3,3), padding='same', activation='relu', input_shape=(20,20,1)))
model2.add(keras.layers.Conv2D(64, kernel_size=(3,3), padding='same', activation='relu' ))
model2.add(keras.layers.MaxPooling2D((2,2)))
model2.add(keras.layers.Conv2D(128, kernel_size=(3,3), padding='same', activation='relu'))
model2.add(keras.layers.MaxPooling2D((2,2)))
model2.add(keras.layers.Conv2D(256, kernel_size=(3,3), padding='same', activation='relu'))
model2.add(keras.layers.MaxPooling2D((2,2)))
model2.add(keras.layers.Conv2D(512, kernel_size=(3,3), padding='same', activation='relu'))
model2.add(keras.layers.MaxPooling2D((2,2)))
model2.add(keras.layers.Flatten())
model2.add(keras.layers.Dropout(0.2))
model2.add(keras.layers.Dense(num_classes, activation='softmax'))
model2.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model2.summary())


model2.compile(loss="categorical_crossentropy",optimizer = 'adam',metrics = ["accuracy"])

model2.fit(X_train, Y_train, epochs=10, batch_size=32, validation_split=0.2)

import cv2
img = cv2.imread('/kaggle/input/test-images/colon_normal_rose100.tif')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(img)

h,w = img.shape

print(h,w)

i = 0

tab = []

n_fois = 0

for i in range(0,w-20):

    j = 0

    for j in range(0,h-20):

        rect = []

        X5 = []

        carreau = img[j:j+20,i:i+20]

        img_array = carreau.flatten()

        X5.append(img_array)

        X5 = np.asarray(X5)

        X5 = X5.astype('float32') / 255

        X5 = X5.reshape(1,img_rows, img_cols, 1)


        y_predits = model2.predict(X5)

        tt = y_predits[0][:]


        if( np.argmax(tt) == 1):

            rect.append(j)

            rect.append(i)

            if(len(tab)==0):

                tab.append(rect)

                n_fois += 1
                
            tab.append(rect)
            n_fois += 1

plt.figure()
for i in range(0,len(tab)):
    
    cv2.circle(img,(tab[i][1]+10,tab[i][0]+10), 1, (0,255,0), 1)
plt.imshow(img,cmap='gray')
plt.show()
cv2.imwrite('Test_micro.png', img)