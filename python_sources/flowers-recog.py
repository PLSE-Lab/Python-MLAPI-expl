import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
print("No warnings!")
################################################################################
##-------------------------------Lot of imports-------------------------------##
################################################################################
print("---Import modules---")
import numpy as np 
import pandas as pd
import h5py

import matplotlib.pylab as plt
from matplotlib import cm
%matplotlib inline

from sklearn.model_selection import train_test_split

from keras.utils import to_categorical
from keras.preprocessing import image as keras_image
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.metrics import top_k_categorical_accuracy, categorical_accuracy

from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, GlobalAveragePooling1D, GlobalAveragePooling2D
from keras.layers.advanced_activations import PReLU, LeakyReLU
from keras.layers import Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, GlobalMaxPooling2D

from keras.applications.resnet50 import preprocess_input
from keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing.image import ImageDataGenerator

import os
#print(os.listdir("../input/"))
from os.path import join
print("---Succeded---")

print("Sorts of flowers are:")
types_of_flowers = os.listdir("../input/flowers/flowers")
types_of_flowers.sort()
print(types_of_flowers)

num_classes = len(types_of_flowers)

#pd for easy usage()
img_paths =  {ft: [os.listdir(join("../input/flowers/flowers/",ft))] for ft in types_of_flowers}
frame_flowers = pd.DataFrame(data=img_paths)

print("---Created temp DataFrame---")
################################################################################
##--------------------------Preparing paths before using-------------------------##
################################################################################


#making full path
for i in frame_flowers.columns:
    a=0
    temp_list = frame_flowers[('{0}').format(i)][0]
    for j in temp_list:
        k = join(i, j)
        n = join('../input/flowers/flowers', k)
        frame_flowers[('{0}').format(i)][0][a] = n
        a+=1
        
#inizialize all flowers type
data_daisy = frame_flowers.daisy[0]
data_dandelion = frame_flowers.dandelion[0]
data_rose = frame_flowers.rose[0]
data_sunflower = frame_flowers.sunflower[0]
data_tulip = frame_flowers.tulip[0]

#there are no images values, so we find them and
suffix = '.jpg'
not_image =[]
for i in range(len(data_dandelion)):
    if not data_dandelion[i].endswith(suffix):
        not_image.append(i)

#reversed, because we need
for i in reversed(not_image):
    del data_dandelion[i]

#find all lengths of each kind of flower
len_daisy = len(data_daisy)
len_tulip = len(data_tulip)
len_sunflower = len(data_sunflower)
len_rose = len(data_rose)
len_dandelion = len(data_dandelion)

#inizialize all labels
daisy_label = [0 for x in range(len_daisy)]
dandelion_label = [1 for x in range(len_dandelion)]
rose_label = [2 for x in range(len_rose)]
sunflower_label = [3 for x in range(len_sunflower)]
tulip_label = [4 for x in range(len_tulip)]

#making training data
print("---Creating train & val values---")
train_daisy_paths, val_daisy_paths, train_daisy_labels, val_daisy_labels = train_test_split(data_daisy, 
                                                            daisy_label, test_size=0.2, random_state = 1)
train_dandelion_paths, val_dandelion_paths, train_dandelion_labels, val_dandelion_labels = train_test_split(data_dandelion, 
                                                            dandelion_label, test_size=0.33, random_state = 1)
train_rose_paths, val_rose_paths, train_rose_labels, val_rose_labels = train_test_split(data_rose, 
                                                            rose_label, test_size=0.33, random_state = 1)
train_sunflower_paths, val_sunflower_paths, train_sunflower_labels, val_sunflower_labels = train_test_split(data_sunflower, 
                                                            sunflower_label, test_size=0.33, random_state = 1)
train_tulip_paths, val_tulip_paths, train_tulip_labels, val_tulip_labels = train_test_split(data_tulip, 
                                                            tulip_label, test_size=0.33, random_state = 1)


#making whole paths in 1 list
train_data_paths = train_daisy_paths + train_dandelion_paths + train_rose_paths + train_sunflower_paths + train_tulip_paths
train_data_labels = train_daisy_labels + train_dandelion_labels + train_rose_labels + train_sunflower_labels + train_tulip_labels
#could use zip(train_data_paths, train_data_labels), but not now!
val_data_paths = val_daisy_paths + val_dandelion_paths + val_rose_paths + val_sunflower_paths + val_tulip_paths
val_data_labels = val_daisy_labels + val_dandelion_labels + val_rose_labels + val_sunflower_labels + val_tulip_labels


# One-hot encoding the targets, started from the zero label
train_y = to_categorical(np.array(train_data_labels), 5)
val_y = to_categorical(np.array(val_data_labels), 5)
#
#y_train = np_utils.to_categorical(train_data_labels, num_classes)
#y_test = np_utils.to_categorical(val_data_labels, num_classes)


#############################################################################
##-------------------------------Read IMAGES-------------------------------##
#############################################################################
image_size = 224

def read_and_prep_images(img_paths, img_height=image_size, img_width=image_size):
    imgs = [load_img(img_path, target_size=(img_height, img_width)) for img_path in img_paths]
    img_array = np.array([img_to_array(img) for img in imgs])
    return preprocess_input(img_array)
print("Preparing images")
train_X = read_and_prep_images(train_data_paths)
print("train_data : success")
val_X = read_and_prep_images(val_data_paths)
print("val_data : success")

print("Checking for shapes")
print("train_X.shape : ", train_X.shape)
print("train_y.shape : ", train_y.shape)
print("val_X.shape : ", val_X.shape)
print("val_y.shape: ", val_y.shape)

###############################################################################
#--------------------------------Plot function--------------------------------#
###############################################################################
def history_plot(fit_history, n):
    plt.figure(figsize=(18, 12))
    
    plt.subplot(211)
    plt.plot(fit_history.history['loss'][n:], color='slategray', label = 'train')
    plt.plot(fit_history.history['val_loss'][n:], color='#4876ff', label = 'valid')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title('Loss Function');  
    
    plt.subplot(212)
    plt.plot(fit_history.history['categorical_accuracy'][n:], color='slategray', label = 'train')
    plt.plot(fit_history.history['val_categorical_accuracy'][n:], color='#4876ff', label = 'valid')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")    
    plt.legend()
    plt.title('Accuracy');


def top_3_categorical_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)

# Create callbacks
checkpointer = ModelCheckpoint(filepath='weights.best.model.hdf5', 
                               verbose=2, save_best_only=True)
lr_reduction = ReduceLROnPlateau(monitor='val_loss',
                                 patience=5, verbose=2, factor=0.75)
print("Checkpoints setted")


def model():
    model = Sequential()
    model.add(Conv2D(32, (4, 4), padding='same' ,input_shape=train_X.shape[1:]))

    model.add(Conv2D(32, (3, 3)))

    model.add(Conv2D(64, (3, 3), strides=2, activation='relu'))
    model.add(Flatten())  
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(5))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
    return model

print("Creating model")
model = model()
print("Model created")

history = model.fit(train_X, train_y,
                    epochs=25, batch_size=50, verbose=2,
                    validation_data=(val_X, val_y))
                    
# Plot the training history
history_plot(history, 0)

from IPython.display import Image, display
display(Image(train_data_paths[0]))

# Create a list of symbols
symbols = types_of_flowers
# Model predictions for the testing dataset
y_test_predict = model.predict_classes(val_X)

# Display true labels and predictions
fig = plt.figure(figsize=(14, 14))
for i, idx in enumerate(np.random.choice(val_X.shape[0], size=16, replace=False)):
    ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])
    ax.imshow(np.squeeze(val_X[idx]))
    pred_idx = y_test_predict[idx]
    true_idx = np.argmax(val_y[idx])
    ax.set_title("{} ({})".format(symbols[pred_idx], symbols[true_idx]),
                 color=("#4876ff" if pred_idx == true_idx else "darkred"))







