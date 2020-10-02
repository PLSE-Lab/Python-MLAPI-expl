# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

test=pd.read_csv('../input/test.csv')



from sklearn.model_selection import train_test_split
from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, Dropout


img_rows, img_cols = 28, 28
num_classes = 10

def data_prep(raw):
    out_y = keras.utils.to_categorical(raw.label, num_classes)

    num_images = raw.shape[0]
    x_as_array = raw.values[:,1:]
    x_shaped_array = x_as_array.reshape(num_images, img_rows, img_cols, 1)
    out_x = x_shaped_array / 255
    return out_x, out_y
    
def data_prep_test(raw):
    num_images = raw.shape[0]
    x_as_array = raw.values[:,0:]
    x_shaped_array = x_as_array.reshape(num_images, img_rows, img_cols, 1)
    out_x = x_shaped_array / 255
    return out_x

train_file='../input/train.csv'
raw_data = pd.read_csv(train_file)

x, y = data_prep(raw_data)

model = Sequential()
model.add(Conv2D(20, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(img_rows, img_cols, 1)))
model.add(Conv2D(20, kernel_size=(3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer='adam',
              metrics=['accuracy'])
model.fit(x, y,
          batch_size=128,
          epochs=2,
          validation_split = 0.2)
#print(train.dtypes)
#print(test.dtypes)

#ytrain=train["label"]
#xtrain=train.drop("label", axis=1)
#print(xtrain.shape)
#print(ytrain.shape)

#from math import sqrt
#dim=int(sqrt(train.shape[1]))

#import seaborn as sns
#sns.set(style='white', context='notebook', palette='deep')

#sns.countplot(ytrain)
#print(type(ytrain))
#vals_counts=ytrain.value_counts()
#print(vals_counts)

#xtrain=xtrain/255.0
#test=test/255.0
# reshape of image data to (nimg, img_rows, img_cols, 1)

#nclasses=10
#import keras
#ytrain=keras.utils.to_categorical(ytrain,nclasses)

#from sklearn.model_selection import train_test_split
#xtrain,ytrain,xval,yval=train_test_split(xtrain, ytrain, test_size=0.1, random_state=2, stratify=ytrain)

#from keras import backend as K

# for the architecture
#from keras.models import Sequential
#from keras.layers import Dense, Dropout, Lambda, Flatten, BatchNormalization
#from keras.layers import Conv2D, MaxPool2D, AvgPool2D

# optimizer, data generator and learning rate reductor
#from keras.optimizers import Adam
#from keras.preprocessing.image import ImageDataGenerator
#from keras.callbacks import ReduceLROnPlateau

#model = Sequential()

#dim=28
#def df_reshape(df):
 #  print("Previous shape, pixels are in 1D vector:", df.shape)
  # df = df.values.reshape(-1, dim, dim, 1) 
   # -1 means the dimension doesn't change, so 42000 in the case of xtrain and 28000 in the case of test
   #print("After reshape, pixels are a 28x28x1 3D matrix:", df.shape)
   #return df

#xtrain = df_reshape(xtrain) # numpy.ndarray type
#test = df_reshape(test) # numpy.ndarray type
#dim = 28
#nclasses = 10

#model.add(Conv2D(filters=32, kernel_size=(5,5), padding='valid', activation='relu', input_shape=(dim,dim,1)))
#model.add(Conv2D(filters=32, kernel_size=(5,5), padding='valid', activation='relu',))
#model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
#model.add(Dropout(0.2))

#model.add(Conv2D(filters=64, kernel_size=(3,3), padding='valid', activation='relu'))
#model.add(Conv2D(filters=64, kernel_size=(3,3), padding='valid', activation='relu'))
#model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
#model.add(Dropout(0.2))

#model.add(Flatten())
#model.add(Dense(120, activation='relu'))
#model.add(Dense(84, activation='relu'))
#model.add(Dense(nclasses, activation='softmax'))

#model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
#lr_reduction = ReduceLROnPlateau(monitor='val_acc', 
 #                                patience=3, 
  #                               verbose=1, 
   #                              factor=0.5, 
    #                             min_lr=0.00001)


#model.fit(xtrain,ytrain,epochs=15,batch_size=64)


#history = model.fit_generator(datagen.flow(xtrain, ytrain, batch_size=batch_size),
 #                             epochs=epochs, 
  #                            validation_data=(xval,yval),
   #                           verbose=1, 
    #                          steps_per_epoch=xtrain.shape[0] // batch_size, 
     #                         callbacks=[lr_reduction])


test=data_prep_test(test)

predictions = model.predict_classes(test, verbose=1)


submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)), "Label": predictions})


#print(submissions.to_string(index=False))


print(submissions.to_csv("mnistpriya.csv", index=False, header=True))

#submissions





# Any results you write to the current directory are saved as output.