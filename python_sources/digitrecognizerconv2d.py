import numpy
import pandas as pd
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
K.set_image_dim_ordering('th')

def get_data():
    # load data
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    # reshape to be [samples][pixels][width][height]
    X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
    X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')
    # normalize inputs from 0-255 to 0-1
#    X_train = X_train / 255
#    X_test = X_test / 255
    # one hot encode outputs
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    num_classes = y_test.shape[1]
    
    return X_train,X_test,y_train,y_test,num_classes

def get_kaggle_test_data():
    kaggle_test = pd.read_csv('test.csv').values
    kaggle_test = kaggle_test.reshape(kaggle_test.shape[0], 1, 28, 28).astype('float32')
#    kaggle_test = kaggle_test / 255
    
    return kaggle_test

# define the larger model
def larger_model():
  # create model
  model = Sequential()
  model.add(Conv2D(30, (5, 5), input_shape=(1, 28, 28), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Conv2D(15, (3, 3), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.2))
  model.add(Flatten())
  model.add(Dense(128, activation='relu'))
  model.add(Dense(50, activation='relu'))
  model.add(Dense(num_classes, activation='softmax'))
  # Compile model
  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
  return model

####################################################################
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

X_train,X_test,y_train,y_test,num_classes = get_data()
kaggle_test = get_kaggle_test_data()

model = larger_model()
##########################################################use Image Augmentation
datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)
#datagen = ImageDataGenerator(zca_whitening=True)

# build the model
batches = datagen.flow(X_train, y_train, batch_size=200)
val_batches = datagen.flow(X_test, y_test, batch_size=200)
# (std, mean, and principal components if ZCA whitening is applied)
datagen.fit(X_train)
datagen.fit(X_test)
# fits the model on batches with real-time data augmentation:
model.fit_generator(batches, steps_per_epoch=len(X_train) / 200, validation_data=val_batches, validation_steps=len(X_test) / 200, epochs=10)

#########################################################not use Image Augmentation
#model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200)

# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Large CNN Error: %.2f%%" % (100-scores[1]*100))

####################################################################
print("Generating test predictions...")
#datagen.fit(kaggle_test)
preds = model.predict_classes(kaggle_test, verbose=0)
def write_preds(preds, fname):
    pd.DataFrame({"ImageId": list(range(1,len(preds)+1)), "Label": preds}).to_csv(fname, index=False, header=True)

write_preds(preds, "keras-mlp-conv2D.csv")