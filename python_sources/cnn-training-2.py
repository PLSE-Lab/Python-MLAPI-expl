import numpy as np 
import pandas as pd 
from keras.callbacks import ReduceLROnPlateau
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.utils.np_utils import to_categorical 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
data_train = pd.read_csv('../input/csvdata/training.csv')
data_test = pd.read_csv('../input/csvdata/testing.csv')


#split raw data into features and label
Y_train = data_train["label"] #define training label set
del data_train["label"]
X_train = data_train #define training features set
X_test = data_test
#split tranining data into tranining data for tranining model and valid data for evaluating the performance.
X_training, X_valid, Y_training, Y_valid = train_test_split(X_train, Y_train, test_size=0.2, random_state=0)


#Reshape input data for the CNN training
X_training_RE = X_training.values.reshape(-1, 28, 28, 1)
X_valid_RE = X_valid.values.reshape(-1, 28, 28, 1)
X_training_RE = X_training_RE.astype("float32")/255.
X_valid_RE = X_valid_RE.astype("float32")/255.
Y_training_RE = to_categorical(Y_training.values)
Y_valid_RE = to_categorical(Y_valid.values)


#CNN model train
#add layers
model = Sequential()
model.add(Conv2D(filters = 16, kernel_size = (3, 3), activation='relu', input_shape = (28, 28, 1)))
model.add(Conv2D(filters = 16, kernel_size = (3, 3), activation='relu'))
model.add(MaxPool2D(strides=(2,2)))
model.add(Dropout(0.3))
model.add(Conv2D(filters = 32, kernel_size = (3, 3), activation='relu'))
model.add(Conv2D(filters = 32, kernel_size = (3, 3), activation='relu'))
model.add(MaxPool2D(strides=(2,2)))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(10, activation='softmax'))



reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,patience=5, min_lr=0.001)
#Randomly create new data for Classifier
datagen = ImageDataGenerator(zoom_range = 0.15,height_shift_range = 0.15,width_shift_range = 0.15,rotation_range = 15)                                                                         
#Compile Classifier
model.compile( optimizer = 'SGD',metrics=["accuracy"],loss='categorical_crossentropy',)
#Finally train  Classifier
hist = model.fit_generator(datagen.flow(X_training_RE, Y_training_RE),
                           epochs=150, 
                           verbose=2, 
                           validation_data=(X_valid_RE, Y_valid_RE), 
                           callbacks=[reduce_lr])
#Evaluate the performance                                                                   
final_loss, final_acc = model.evaluate(X_valid_RE, Y_valid_RE, verbose=0)
print("Final loss:")
print(final_loss)
print("Final Accuracy")
print(final_acc)
#Save the model
model.save("model.h5")
