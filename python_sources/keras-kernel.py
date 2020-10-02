import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt
import matplotlib.image as mplimg
from matplotlib.pyplot import imshow

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from keras import layers
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout
from keras.models import Model

import keras.backend as K
from keras.models import Sequential

import warnings
warnings.simplefilter("ignore", category=DeprecationWarning)

import shutil
from PIL import Image

Train_labels = "../input/rwl-ml-19/noaa_rwr_ml-19/noaa-right-whale-recognition/train.csv"
Whole_train_set = "../input/rwl-ml-19/noaa_rwr_ml-19/noaa-right-whale-recognition/imgs/"
Cropped_train_set = "../input/whaleheadscropped/imgs_cropped/imgs_cropped/"
Crotated_train_set = "../input/headsrotatedcropped/imgs_rotated_cropped/imgs_rotated_cropped/"
#print(os.listdir(Train_set))
train_df = pd.read_csv(Train_labels)
IdCol = train_df.whaleID
train_df.head()

if 1:    
    os.mkdir("../Train_cropped/")
    os.mkdir("../Train_whole/")
    os.mkdir("../Train_crotated/")
    s = pd.Series(train_df.Image)

    for im in os.listdir(Whole_train_set):
        if im in s.tolist():
            shutil.copy2(Whole_train_set + im, "../Train_whole/")
            
    for im in os.listdir(Cropped_train_set):
        if im in s.tolist():
                shutil.copy2(Cropped_train_set + im, "../Train_cropped/")
                
    for im in os.listdir(Crotated_train_set):
        if im in s.tolist():
            if im == 'w_7531.jpg':
                shutil.copy2(Cropped_train_set + im, "../Train_crotated/")
            else:
                shutil.copy2(Crotated_train_set + im, "../Train_crotated/")       

Whole_train_set =  "../Train_whole/"
Cropped_train_set = "../Train_cropped/" 
Crotated_train_set = "../Train_crotated/"


def prepareImages(datadir, m, dataset):
    print("Preparing images")
    X_train = np.zeros((m, 100, 100, 3))
    count = 0
    
    for fig in os.listdir(datadir):
        #load images into images of size 100x100x3
        img = image.load_img(datadir +fig, target_size=(100, 100, 3))
        x = image.img_to_array(img)
        x = preprocess_input(x)

        X_train[count] = x
        if (count%500 == 0):
            print("Processing image: ", count+1, ", ", fig)
        count += 1
    
    return X_train
    
#print(f"There are {len(os.listdir(Train_set))} images in train dataset with {IdCol.nunique()} unique classes.")
#print(f"There are {len(os.listdir(Test_set))} images in test dataset.")

if 0: 
    fig = plt.figure(figsize=(25, 4))
    train_imgs = os.listdir(Train_set)
    for idx, img in enumerate(np.random.choice(train_imgs, 20)):
        ax = fig.add_subplot(2, 20//2, idx+1, xticks=[], yticks=[])
        im = Image.open(Train_set + img)
        plt.imshow(im)
        lab = train_df.loc[train_df.Image == img, 'whaleID'].values[0]
        ax.set_title(f'Label: {lab}')

    IdCol.value_counts().head()
    for i in range(1, 4):
        print(f'There are {IdCol.value_counts()[IdCol.value_counts().values==i].shape[0]} classes with {i} samples in train data.')
    plt.title('Distribution of classes');
    IdCol.value_counts().plot(kind='hist');


def prepare_labels(y):
    values = np.array(y)
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    # print(integer_encoded)

    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    # print(onehot_encoded)

    y = onehot_encoded
    # print(y.shape)
    return y, label_encoder

y, le = prepare_labels(train_df['whaleID'])
y.shape

model = Sequential()

model.add(Conv2D(32, (7, 7), strides = (1, 1), name = 'conv0', input_shape = (100, 100, 3)))

model.add(BatchNormalization(axis = 3, name = 'bn0'))
model.add(Activation('relu'))

model.add(MaxPooling2D((2, 2), name='max_pool'))
model.add(Conv2D(64, (3, 3), strides = (1,1), name="conv1"))
model.add(Activation('relu'))
model.add(AveragePooling2D((3, 3), name='avg_pool'))

model.add(Flatten())
model.add(Dense(500, activation="relu", name='rl'))
model.add(Dropout(0.8))
model.add(Dense(y.shape[1], activation='softmax', name='sm'))

X = prepareImages(Cropped_train_set, train_df.shape[0], "train")
X /= 255

model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
model.summary()
history = model.fit(X, y, epochs=200, batch_size=100, verbose=1)
#gc.collect()

plt.plot(history.history['acc'], label='Re+Cropped')
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.savefig("Cropped_train_set.eps")

predictions = model.predict(np.array(X), verbose=1)

df = pd.DataFrame(data=predictions)
df.to_csv('cropped_model.csv', index=False)

X = prepareImages(Crotated_train_set, train_df.shape[0], "train")
X /= 255

model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
model.summary()
history = model.fit(X, y, epochs=200, batch_size=100, verbose=1)
#gc.collect()

plt.plot(history.history['acc'], label='Re+Cr+Rotated')

plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.savefig("Crotated_train_set.eps")

predictions = model.predict(np.array(X), verbose=1)

df = pd.DataFrame(data=predictions)
df.to_csv('crotated_model.csv', index=False)

X = prepareImages(Whole_train_set, train_df.shape[0], "train")
X /= 255

model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
model.summary()
history = model.fit(X, y, epochs=200, batch_size=100, verbose=1)
#gc.collect()

plt.plot(history.history['acc'], label='Resized')
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.savefig("Whole_train_set.eps")

predictions = model.predict(np.array(X), verbose=1)

df = pd.DataFrame(data=predictions)
df.to_csv('whole_model.csv', index=False)