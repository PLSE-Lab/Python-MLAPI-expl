import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from keras.optimizers import SGD
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K

aug = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)


class MiniVGGNet:
    @staticmethod
    def build(width, height, depth, classes):
        # initialize the model along with the input shape to be
        # "channels last" and the channels dimension itself
        model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1

        # if we are using "channels first", update the input shape
        # and channels dimension
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1

        # first CONV => RELU => CONV => RELU => POOL layer set
        model.add(Conv2D(32, (3, 3), padding="same", input_shape = inputShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(32, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # second CONV => RELU => CONV => RELU => POOL layer set
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # first (and only) set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        # softmax classifier
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        # return the constructed network architecture
        return model


print("[INFO] loading CIFAR-10 data...")
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


label_names = unpickle("../input/cifar10/batches.meta")
label_names = label_names[b"label_names"]
print(label_names)

lb = LabelBinarizer()

train_X = np.array([])
train_Y = np.array([])

first = True

for i in range(1, 6):
    current_batch = unpickle("../input/cifar10/data_batch_" + str(i))

    current_X = current_batch[b"data"].reshape(10000, 32, 32, 3)
    current_Y = lb.fit_transform(np.array(current_batch[b"labels"]).reshape(10000, 1))

    if(first == True):
        train_X = current_X
        train_Y = current_Y
        first = False

    else:
        train_X = np.concatenate((train_X, current_X), axis=0)
        train_Y = np.concatenate((train_Y, current_Y), axis=0)

train_X = train_X.astype("float") / 255.0

test_batch = unpickle("../input/cifar10/test_batch")

test_X = test_batch[b"data"].reshape(10000, 32, 32, 3)
test_X = test_X.astype("float") / 255.0
test_Y = lb.fit_transform(np.array(test_batch[b"labels"]).reshape(10000, 1))

# initialize the optimizer and model
print("[INFO] compiling model...")
opt = SGD(lr=0.01, decay=0.01 / 40, momentum=0.9, nesterov=True)
model = MiniVGGNet.build(width=32, height=32, depth=3, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# train the network
print("[INFO] training network...")
model.fit_generator(aug.flow(train_X, train_Y, batch_size=32),
validation_data=(test_X, test_Y), steps_per_epoch=len(trainX) // 32,
epochs=40, verbose=1)

# making predictions
print("[INFO] evaluating network...")
predictions = model.predict(test_X, batch_size=64)

#saving predictions
subs = pd.read_csv("../input/cifar10/sample_submission.csv")
subs[["plane", "car", "bird", "cat" ,"deer", "dog", "frog", "horse", "ship", "truck"]] = predictions
subs.to_csv('cifar_doodle_data_aug.csv', index=False)



