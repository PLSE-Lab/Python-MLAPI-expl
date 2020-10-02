

# %% [code]
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skimage.io
from skimage.transform import resize
from imgaug import augmenters as iaa
from tqdm import tqdm
import PIL
from PIL import Image, ImageOps
import cv2
from keras import callbacks
from sklearn.utils import class_weight, shuffle
from keras.losses import binary_crossentropy
from keras.applications.resnet50 import preprocess_input
import keras.backend as K
import tensorflow as tf
from sklearn.metrics import f1_score, fbeta_score
from keras.utils import Sequence
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import keras
from keras import layers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, MaxPool2D
from keras.optimizers import SGD
from keras.optimizers import Adam
from numpy import array
from numpy import argmax
from sklearn.metrics import cohen_kappa_score, accuracy_score
import scipy
from keras.utils import to_categorical
from keras.applications import DenseNet121
from keras.applications.xception import Xception
from keras.callbacks import Callback, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from imgaug import augmenters as iaa
import imgaug as ia
#import efficientnet.keras as efn







x_train = np.load('../input/blindness-detection/x_train.npy')
x_val = np.load('../input/blindness-detection/valid_x.npy')
y_train = np.load('../input/blindness-detection/y_train.npy')
y_val = np.load('../input/blindness-detection/valid_y.npy')

class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(y_train),
                                                 y_train)
class_weights = dict(enumerate(class_weights))


y_train = to_categorical(y_train)
y_val = to_categorical(y_val)
print(y_train.shape)


for i in range(3, -1, -1):

    y_train[:, i] = np.logical_or(y_train[:, i], y_train[:, i + 1])
    y_val[:, i] = np.logical_or(y_val[:, i], y_val[:, i + 1])



print("Multilabel version:", y_train.sum(axis=0))







x_train1 = []
for x in x_train:
    i = cv2.resize(x, (224, 224))
    x_train1.append(i)
x_train = np.array(x_train1)

valid_x1 = []
for x in x_val:
    i = cv2.resize(x, (224, 224))
    valid_x1.append(i)
x_val = np.array(valid_x1)


print(x_train.shape)

print(x_val.shape)
plt.imshow(x_train[0], interpolation='nearest')
plt.show()



class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_kappas = []

    def on_epoch_end(self, epoch, logs={}):
        X_val, y_val = self.validation_data[:2]
        y_val = (y_val.sum(axis=1) - 1).clip(0, 4)

        
        y_pred = self.model.predict(X_val) > 0.5
        y_pred = (y_pred.astype(int).sum(axis=1) - 1).clip(0, 4)


        _val_kappa = cohen_kappa_score(
            y_val,
            y_pred, 
            weights='quadratic'
        )

        self.val_kappas.append(_val_kappa)

        print(f"val_kappa: {_val_kappa:.4f}")
        
        if _val_kappa == max(self.val_kappas):
            print("Validation Kappa has improved. Saving model.")
            self.model.save('model.h5')

        return


sometimes = lambda aug: iaa.Sometimes(0.5, aug)
seq = iaa.Sequential(
        [
            # apply the following augmenters to most images
            iaa.Fliplr(0.5), # horizontally flip 50% of all images
            iaa.Flipud(0.2), # vertically flip 20% of all images
            sometimes(iaa.Affine(
                scale={"x": (0.9, 1.1), "y": (0.9, 1.1)}, # scale images to 80-120% of their size, individually per axis
                translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}, # translate by -20 to +20 percent (per axis)
                rotate=(-10, 10), # rotate by -45 to +45 degrees
                shear=(-5, 5), # shear by -16 to +16 degrees
                order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
                cval=(0, 255), # if mode is constant, use a cval between 0 and 255
                mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
            )),
            # execute 0 to 5 of the following (less important) augmenters per image
            # don't execute all of them, as that would often be way too strong
            iaa.SomeOf((0, 5),
                [
                    sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))), # convert images into their superpixel representation
                    iaa.OneOf([
                        iaa.GaussianBlur((0, 1.0)), # blur images with a sigma between 0 and 3.0
                        iaa.AverageBlur(k=(3, 5)), # blur image using local means with kernel sizes between 2 and 7
                        iaa.MedianBlur(k=(3, 5)), # blur image using local medians with kernel sizes between 2 and 7
                    ]),
                    iaa.Sharpen(alpha=(0, 1.0), lightness=(0.9, 1.1)), # sharpen images
                    iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)), # emboss images
                    # search either for all edges or for directed edges,
                    # blend the result with the original image using a blobby mask
                    iaa.SimplexNoiseAlpha(iaa.OneOf([
                        iaa.EdgeDetect(alpha=(0.5, 1.0)),
                        iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
                    ])),
                    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.01*255), per_channel=0.5), # add gaussian noise to images
                    iaa.OneOf([
                        iaa.Dropout((0.01, 0.05), per_channel=0.2), # randomly remove up to 10% of the pixels
                        iaa.CoarseDropout((0.01, 0.03), size_percent=(0.01, 0.02), per_channel=0.2),
                    ]),
                    iaa.Invert(0.01, per_channel=True), # invert color channels
                    iaa.Add((-2, 2), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
                    iaa.AddToHueAndSaturation((-1, 1)), # change hue and saturation
                    # either change the brightness of the whole image (sometimes
                    # per channel) or change the brightness of subareas
                    iaa.OneOf([
                        iaa.Multiply((0.9, 1.1), per_channel=0.5),
                        iaa.FrequencyNoiseAlpha(
                            exponent=(-1, 0),
                            first=iaa.Multiply((0.9, 1.1), per_channel=True),
                            second=iaa.ContrastNormalization((0.9, 1.1))
                        )
                    ]),
                    sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)), # move pixels locally around (with random strengths)
                    sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))), # sometimes move parts of the image around
                    sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
                ],
                random_order=True
            )
        ],
        random_order=True)

BATCH_SIZE = 32

def create_datagen():

    return ImageDataGenerator(
        preprocessing_function=seq.augment_image
    )

data_generator = create_datagen().flow(x_train, y_train, batch_size=BATCH_SIZE)





base_model = Xception(
    weights=None, #'imagenet',
    include_top=False,
    input_shape=(224,224,3)
)

base_model.load_weights("../input/weights/xception_weights_tf_dim_ordering_tf_kernels_notop.h5")


def build_model():
    model = Sequential()
    model.add(base_model)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(5, activation='sigmoid'))
    
    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(lr=0.00005),
        metrics=['accuracy']
    )
    
    return model

model = build_model()
model.summary()



if True:
    kappa_metrics = Metrics()
    rlr = callbacks.ReduceLROnPlateau(factor=0.5, patience=4, verbose=1)
    es  = callbacks.EarlyStopping(patience=10, verbose=1, mode="min")

    history = model.fit_generator(
        data_generator,
        class_weight=class_weights,
        steps_per_epoch=x_train.shape[0] / BATCH_SIZE,
        epochs=5,
        validation_data=(x_val, y_val),
        callbacks=[kappa_metrics, rlr, es]
    )

model.save("model")
def prepare(filepath):
    img_size = 224
    img_array = cv2.imread(filepath)
    new_array = cv2.resize(img_array, (img_size, img_size))
    return new_array.reshape(-1, img_size, img_size, 3)
    

