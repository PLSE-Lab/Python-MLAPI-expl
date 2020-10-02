import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

from skimage.transform import resize, rotate, SimilarityTransform, warp

# Image batch generator
def imageGenerator(X, y, batch_size, imgsz):
    img_rows, img_cols = imgsz[0], imgsz[1]
    resc = 0.05
    rot = 7
    transl = 0.01*img_rows
    while 1: # Infinite loop
        batchX = np.zeros((batch_size, img_rows, img_cols, 1))
        batch_ids = np.random.choice(X.shape[0], batch_size)
        for j in range(batch_ids.shape[0]): # Loop over random images
            imagej = rotate(X[batch_ids[j]], angle =rot*np.random.randn())
            # Rescale and translate
            tf = SimilarityTransform(scale = 1 + resc*np.random.randn(1,2)[0],
                                translation = transl*np.random.randn(1,2)[0]) 
            batchX[j] = warp(imagej, tf)
        yield (batchX, y[batch_ids])

# The competition datafiles are in the directory ../input
# Read competition data files:
# load_data
train_data = pd.read_csv('../input/train.csv')

labels = train_data.label.values.astype('int32')
train_data = train_data.drop('label', axis=1).as_matrix().astype('float32')

test_data = pd.read_csv('../input/test.csv').as_matrix().astype('float32')

# make label vectors
y_train = np_utils.to_categorical(labels) 


# normalize data
X_train = train_data / np.max(train_data)
X_test = test_data / np.max(train_data)

img_size = (28, 28)

if K.image_dim_ordering() == 'th':
    X_train = X_train.reshape(X_train.shape[0], 1, img_size[0], img_size[1])
    X_test = X_test.reshape(X_test.shape[0], 1, img_size[0], img_size[1])
    image_shape = (1, img_size[0], img_size[1])
else:
    X_train = X_train.reshape(X_train.shape[0], img_size[0], img_size[1], 1)
    X_test = X_test.reshape(X_test.shape[0], img_size[0], img_size[1], 1)
    image_shape = (img_size[0], img_size[1], 1)


print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

print("Build our CNN")
model = Sequential()

model.add(Convolution2D(32, 3, 3,  activation="relu", input_shape=(28, 28, 1), border_mode='same'))
model.add(Convolution2D(32, 3, 3, activation="relu", border_mode='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(64, 3, 3, activation="relu", border_mode='same'))
model.add(Convolution2D(64, 3, 3, activation="relu", border_mode='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(84))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(10))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


nb_epoch = 12
batch_size = 40
samples_per_epoch = X_train.shape[0]
nb_classes = 10

print("Training...")
model.fit_generator(imageGenerator(X_train, y_train, batch_size, img_size),  samples_per_epoch = samples_per_epoch,  nb_epoch=nb_epoch, verbose=0)


print("Predictions...")
predictions = model.predict_classes(X_test, verbose=0)


preds = pd.DataFrame({"Label": predictions})
preds.index = preds.index + 1
preds.to_csv("keras_cnn01c.csv",index_label='ImageId')

print ("That's all.")



