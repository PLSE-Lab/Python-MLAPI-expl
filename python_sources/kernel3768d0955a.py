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

# Import packages
#%%
import pandas as pd
import numpy as np
from PIL import Image
import glob
import keras
import ntpath as ntp
import pydot

from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from sklearn import model_selection as ms
#%%
data = pd.read_csv("/kaggle/input/coronahack-chest-xraydataset/Chest_xray_Corona_Metadata.csv")  # represents the summary of the data

label = data['Label'].tolist()
image_name = data['X_ray_image_name'].tolist()

print(data.head())
# display(type(image_name))


dict_data = {}
for i in range(len(image_name)):
    dict_data[image_name[i]] = label[i];

print(len(dict_data))
# Load the test and train data and pre-process and save as npy

#%%
path_train = ['/kaggle/input/coronahack-chest-xraydataset/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/train/*', 'X_train.npy', 'Y_train.npy']
path_test = ['/kaggle/input/coronahack-chest-xraydataset/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/test/*', 'X_test.npy', 'Y_test.npy']

# for filepath in glob.iglob(path_test[0]):
#     display(filepath)


width = 224
height = 224

#%%
def preprocess(path):
    X_data = []
    Y_data = []
    for filepath in glob.iglob(path[0]):
        filename = ntp.basename(filepath)
        if (filename in image_name):

            image = Image.open(filepath)  # type: Image.Image
            image = image.resize((width, height)).convert('LA')

            X_data.append(np.asarray(image))
            if (dict_data[filename] == 'Normal'):
                Y_data.append(0)
            if (dict_data[filename] == 'Pnemonia'):
                Y_data.append(1)

    X_data = np.array(X_data)
    X_data = np.resize(X_data, (len(X_data), width, height, 2))
    X_data = X_data.astype('float32') / 255.

    Y_data = np.array(Y_data)
    Y_data = Y_data.astype('float32')

    # save for later
    np.save(path[2], Y_data)
    np.save(path[1], X_data)



preprocess(path_train)
preprocess(path_test)


# VGG16 model implementation

#%%
model = Sequential()
model.add(Conv2D(input_shape=(224, 224, 2), filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(Conv2D(filters=32, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Conv2D(filters=16, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.4))
model.add(Conv2D(filters=16, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dense(units=100, activation="relu"))
model.add(Dense(units=2, activation="softmax"))
model.add(Dropout(0.4))

# from tensorflow.keras.optimizers import Adam
#
# opt = Adam(lr=0.001)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics = ['accuracy'])

# model.summary()
# plot_model(model,to_file='model.png')

#%%
X_train = np.load(path_train[1])
Y_train = np.load(path_train[2])

X_train, X_test, Y_train, Y_test = ms.train_test_split(X_train, Y_train, test_size=0.4, random_state=42)

# print(type(X_train))
# for i in range(10):
#     print(X_train[0,0,i])
#     print(Y_train[i])


Y_train = keras.utils.to_categorical(Y_train, num_classes=2)
#
# X_test = np.load(path_test[1])
# Y_test = np.load(path_test[2])
Y_test = keras.utils.to_categorical(Y_test, num_classes=2)

testdata = (X_test, Y_test)
print(Y_train)

# print(X_train.shape)
# print(Y_train)
# print(X_test.shape)
# print(Y_test.shape)
# from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

checkpoint = ModelCheckpoint("Model1.hdf5", monitor='val_accuracy', verbose=1, save_best_only=True,
                             save_weights_only=False,
                             mode='auto', )

early = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=30, verbose=1, mode='auto',restore_best_weights=True)

# Use image feature augmentation to preprocess image

# datagen = ImageDataGenerator(
#     featurewise_center=True,
#     featurewise_std_normalization=True
#     # rotation_range=20, zoom_range=0.15,
#     # width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
#     # horizontal_flip=True, fill_mode="nearest"
# )
#
# datagen.fit(X_train)
# iterator=datagen.flow(X_train,Y_train)
# model.fit_generator(generator=datagen.flow(X_train,Y_train),steps_per_epoch=len(X_train)/32, epochs=100,
#                     shuffle=True,validation_data=testdata,callbacks=[checkpoint,early]
#                     )

#%%
hist = model.fit(epochs=100,
                 x=X_train,
                 y=Y_train,
                 batch_size=32,
                 shuffle=True,
                 validation_data=testdata,
                 callbacks=[checkpoint,early]
                 )

# model.fit(X_train, Y_train,
#                     epochs=100,
#                     batch_size=50,
#                     shuffle=True,
#                     validation_data=(X_test, Y_test),
#                     callbacks=[TensorBoard(log_dir='', histogram_freq=0, write_graph=False)])
# model.save("Model.hdf5")
#%%
import matplotlib.pyplot as plt
print(hist.history.keys())
# summarize history for accuracy
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('acc.png')
plt.show()
# summarize history for loss
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("loses.png")
plt.show()