# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from keras.optimizers import SGD
from sklearn.metrics import accuracy_score
from keras.callbacks import ModelCheckpoint
# fix dimension ordering issue
from keras import backend as K
K.set_image_dim_ordering('tf')

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

#reading train dataset
df_train = pd.read_csv("../input/train.csv")
df_train = df_train.astype('float32')

#dropping label column
y_train = df_train["label"].values
df_train.drop("label",axis=1,inplace=True)

#normalizing train pixel values
df_train = df_train / 255

x_train = df_train.values
x_train = np.reshape(x_train,(-1,28,28,1))

#reading test dataset
df_test = pd.read_csv("../input/test.csv")
df_test = df_test.astype('float32')

#normalizing train pixel values
df_test = df_test / 255
x_test = df_test.values
x_test = np.reshape(x_test,(-1,28,28,1))

#splitting train dataset into train and validation datasets
x_train, x_val, y_train, y_val = train_test_split(x_train,y_train,
                                        test_size=0.2,random_state=7)

#onehot encoding label values
y_train = np.reshape(y_train,(-1,1))
y_train = OneHotEncoder().fit_transform(y_train).toarray()
y_val_singleDigit = y_val
y_val = np.reshape(y_val,(-1,1))
y_val = OneHotEncoder().fit_transform(y_val).toarray()

#epochs = 50
#lrate = 0.01
#decay = lrate/epochs
#sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)

#defining keras model
def baseline_model():
    model = Sequential()
    model.add(Conv2D(32, (5, 5), input_shape=(28, 28, 1), activation='relu',
                     kernel_initializer='normal'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(16, (3, 3), activation='relu',
                     kernel_initializer='normal'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu',kernel_initializer='normal'))
    model.add(Dense(64, activation='relu',kernel_initializer='normal'))
    model.add(Dense(10, activation='softmax',kernel_initializer='normal'))
#    model.load_weights("../working/weights.best.hdf5")
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy'])
    return model

# build the model
model = baseline_model()

#checkpoint file definition
filepath="../working/weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=True,
                             mode='max')
callbacks_list = [checkpoint]

# Fit the model
model.fit(x_train, y_train, epochs=100,validation_data=(x_val,y_val),
          batch_size=200, callbacks=callbacks_list, verbose=2)

#defining keras model
def baseline_model_from_saved():
    model = Sequential()
    model.add(Conv2D(32, (5, 5), input_shape=(28, 28, 1), activation='relu',
                     kernel_initializer='normal'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(16, (3, 3), activation='relu',
                     kernel_initializer='normal'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu',kernel_initializer='normal'))
    model.add(Dense(64, activation='relu',kernel_initializer='normal'))
    model.add(Dense(10, activation='softmax',kernel_initializer='normal'))
    model.load_weights("../working/weights.best.hdf5")
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy'])
    return model
    
# load from the saved model
model_loaded = baseline_model_from_saved()

# Final evaluation of the model based on validation data
scores = model_loaded.evaluate(x_val, y_val, verbose=0)
print("Accuracy: {}".format(scores))

# Evaluating score using sklearn metric
y_val_predicted = model_loaded.predict(x_val)
y_val_predicted_singleDigit = np.argmax(y_val_predicted,axis=1)
print("Sklearn.metrics Accuracy: {}".format(accuracy_score(y_val_singleDigit,y_val_predicted_singleDigit)))

# Evaluating test labels
y_test_predicted = model_loaded.predict(x_test)
y_test_predicted_singleDigit = np.argmax(y_test_predicted,axis=1)

# Producing submission file
df_sub = pd.DataFrame({"ImageId":range(1,y_test_predicted_singleDigit.shape[0]+1),
                      "Label":y_test_predicted_singleDigit})
df_sub.to_csv("submission.csv", index=False)
