import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skimage.io
from skimage.transform import resize
from imgaug import augmenters as iaa
from tqdm import tqdm
import PIL
from PIL import Image
import cv2
from sklearn.utils import class_weight, shuffle
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
import gc
from sklearn.model_selection import StratifiedKFold, train_test_split

import warnings
warnings.filterwarnings("ignore")

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.applications.densenet import DenseNet201, preprocess_input
from keras.callbacks import ModelCheckpoint
from keras import metrics
from keras.optimizers import Adam 
from keras import backend as K
import keras
from keras.layers.normalization import BatchNormalization
from keras.models import *
from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler, ReduceLROnPlateau
from keras.regularizers import *

from functools import reduce
import math
import tqdm



ID = 'image_id'
CLASS = 'category'
PATH_TRAIN = '../input/he_challenge_data/data/train/'
PATH_TEST = '../input/he_challenge_data/data/test/'
TARGET = 'target'


BATCH_SIZE = 32
TARGET_SIZE = (224, 224)
INITIAL_LR = 1e-6
EPCOH_DROP = 10
DROP = 0.5
BATCH_SIZE_VAL = 64
NUM_TTA = 10
NUM_LABELS = 102


DATA_TRAIN = pd.read_csv('../input/he_challenge_data/data/train.csv')


DATA_TRAIN.head()
le = LabelEncoder()
DATA_TRAIN[TARGET] = le.fit_transform(DATA_TRAIN[CLASS])



train_dataset_info = []
for name, labels in zip(DATA_TRAIN[ID], DATA_TRAIN[TARGET]):
    train_dataset_info.append({
        'path':os.path.join(PATH_TRAIN, str(name) + '.jpg'),
        'labels':labels})
train_dataset_info = np.array(train_dataset_info)



class data_generator(object):
    
    def create_train(dataset_info, batch_size, shape, augument=True):
        assert shape[2] == 3
        while True:
            dataset_info = shuffle(dataset_info)
            for start in range(0, len(dataset_info), batch_size):
                end = min(start + batch_size, len(dataset_info))
                batch_images = []
                X_train_batch = dataset_info[start:end]
                batch_labels = np.zeros((len(X_train_batch), NUM_LABELS))
                for i in range(len(X_train_batch)):
                    image = data_generator.load_image(X_train_batch[i]['path'], shape, augument=augument)
                    batch_images.append(image)
                    batch_labels[i][X_train_batch[i]['labels']] = 1
                yield np.array(batch_images, np.float32), batch_labels

    def load_image(path, shape,augument=True):
        image = Image.open(path)
        image_data = np.asarray(image).astype(np.float32)
        image = cv2.resize(image_data, (shape[0], shape[1]))
        image = preprocess_input(image)
        if augument:
            image = data_generator.augment(image)
        return image

    def augment(image):
        augment_img = iaa.Sequential([
            iaa.OneOf([
                iaa.Affine(rotate=0),
                iaa.Affine(rotate=90),
                iaa.Affine(rotate=180),
                iaa.Affine(rotate=270),
                iaa.Fliplr(0.5),
                iaa.Flipud(0.5),
            ])], random_order=True)

        image_aug = augment_img.augment_image(image)
        return image_aug
    

cv_split = StratifiedKFold(n_splits=10, shuffle=True, random_state=123456)
contar = 0
for train_index, test_index in cv_split.split(DATA_TRAIN[TARGET], DATA_TRAIN[TARGET]):
    contar += 1
    if contar == 1:
        break

train_generator = data_generator.create_train(train_dataset_info[train_index], BATCH_SIZE, tuple(list(TARGET_SIZE) + [3]), augument=True)
validation_generator = data_generator.create_train(train_dataset_info[test_index], BATCH_SIZE_VAL, tuple(list(TARGET_SIZE) + [3]), augument=False)


basic_model = DenseNet201(include_top=False, weights='imagenet', pooling='avg')

for layer in basic_model.layers:
    layer.trainable = False

input_tensor = basic_model.input
# build top
x = basic_model.output
x = BatchNormalization()(x)
x = Dropout(.5)(x)
x = Dense(NUM_LABELS, activation='softmax')(x)

model = Model(inputs=input_tensor, outputs=x)
model.compile(optimizer=Adam(lr=1e-3),
              loss='categorical_crossentropy',
              metrics=['accuracy'])


model.fit_generator(train_generator,
                    validation_data=validation_generator,
                    epochs=5,
                    steps_per_epoch=len(train_index)//BATCH_SIZE,
                    validation_steps=len(test_index)//BATCH_SIZE,
                    verbose=1,
                    shuffle=True)


for layer in model.layers:
    layer.W_regularizer = l2(1e-3)
    layer.trainable = True

model = Model(inputs=input_tensor, outputs=x)
model.compile(optimizer=Adam(lr=1e-6, decay=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])


earlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=2, mode='auto', min_delta=1e-4)


def step_decay(epoch):
    initial_lrate = INITIAL_LR
    drop = DROP
    epochs_drop = EPCOH_DROP
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate

lrate = LearningRateScheduler(step_decay)


model.fit_generator(train_generator,
                    epochs=15,
                    validation_data=validation_generator,
                    steps_per_epoch=len(train_index)//BATCH_SIZE,
                    validation_steps=len(test_index)//BATCH_SIZE,
                    callbacks=[earlyStopping, lrate],
                    verbose=1,
                    initial_epoch=5)


DATA_TEST = pd.read_csv('../input/he_challenge_data/data/sample_submission.csv')


predicted = []
for name in tqdm.tqdm(DATA_TEST[ID]):
    try:
        path = os.path.join(PATH_TEST, str(name) + '.jpg')
        image_test_TTA = [data_generator.load_image(path, tuple(list(TARGET_SIZE) + [3]), augument=True) for i in range(NUM_TTA)]
        score_predict = [model.predict(x[np.newaxis])[0] for x in image_test_TTA]
        score_predict = np.argmax(np.array(score_predict).mean(axis=0))
    except:
        print(name)
        label_predict = 10
    predicted.append(score_predict)
    
    
SUBMIT = pd.DataFrame({ID: DATA_TEST[ID].values, TARGET: predicted})
SUBMIT[CLASS] = le.inverse_transform(SUBMIT[TARGET])


SUBMIT.drop(TARGET, axis=1).to_csv('submission.csv', index=False)