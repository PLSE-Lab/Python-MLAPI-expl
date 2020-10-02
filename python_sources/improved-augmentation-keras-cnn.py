# This kernel is based on https://www.kaggle.com/gimunu/data-augmentation-with-keras-into-cnn
# with some enhacenment to make you score bigger and life - easier
# 1. Add tqdm to check current for-loop state progress
# 2. Add saving and loading logic for preprocessed image numpy array. You don't need every run to compute it
# 3. Add train-test split to make more obvious and clear validation on images that NN doesn't see while training
# 4. Add param whether you want to generate prediction or you run model experiment
# 5. Add 'he_normal' kernel initializer 
# 6. Replace Adadelta with Adam
# 7. In total the last two points gave me improvement for about 0.5%
# 8. Add submission as another .csv file

import argparse
import os
import warnings
from glob import glob

import keras
import numpy as np
import pandas as pd
from PIL import Image
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from tqdm import tqdm

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--predict", action='store_true', default=False,
                    help="generate prediction or no")

args = parser.parse_args()

train_images = glob("../input/train/*jpg")
test_images = glob("../input/test/*jpg")
df = pd.read_csv("../input/train.csv")
sample_submission_path = '../input/sample_submission.csv'
preprocessed_path = '../input/'
preprocessed_images_path = 'preprocessed_image_array.npy'

df['Image'] = df['Image'].map(lambda x: '../input/train/' + x)
ImageToLabelDict = dict(zip(df['Image'], df['Id']))

IMAGE_SIZE = 64
EPOCHS = 9
BATCH_SIZE = 128


def import_image(filename):
    img = Image.open(filename).convert("LA").resize((IMAGE_SIZE, IMAGE_SIZE))
    return np.array(img)[:, :, 0]


def get_train_images():
    preprocessed_image_array_path = os.path.join(preprocessed_path, preprocessed_images_path)
    if os.path.isfile(preprocessed_image_array_path):
        print('loading preprocessed images')
        return np.load(preprocessed_image_array_path)
    else:
        print('creating preprocessed images')
        preprocessed_images = np.array([import_image(train_image) for train_image in tqdm(train_images)])
        np.save(preprocessed_image_array_path, preprocessed_images)
        return preprocessed_images


x = get_train_images()


class LabelOneHotEncoder:
    def __init__(self):
        self.ohe = OneHotEncoder()
        self.le = LabelEncoder()

    def fit_transform(self, x):
        features = self.le.fit_transform(x)
        return self.ohe.fit_transform(features.reshape(-1, 1))

    def transform(self, x):
        return self.ohe.transform(self.le.transform(x.reshape(-1, 1)))

    def inverse_transform(self, x):
        return self.le.inverse_transform(self.ohe.inverse_tranform(x))

    def inverse_labels(self, x):
        return self.le.inverse_transform(x)


y = list(map(ImageToLabelDict.get, train_images))
label_one_hot_encoder = LabelOneHotEncoder()
y_cat = label_one_hot_encoder.fit_transform(y)

WeightFuction = lambda x: 1. / x ** 0.75
ClassLabel2Index = lambda x: label_one_hot_encoder.le.inverse_transform([[x]])
CountDict = dict(df['Id'].value_counts())
class_weight_dict = {label_one_hot_encoder.le.transform([image])[0]: WeightFuction(count) for image, count in
                     CountDict.items()}

x = x.reshape([-1, IMAGE_SIZE, IMAGE_SIZE, 1])
input_shape = x[0].shape

x_train = x.astype('float32')
y_train = y_cat

num_classes = len(y_cat.toarray()[0])


def get_model():
    model = Sequential()
    model.add(Conv2D(filters=48, kernel_size=(3, 3), activation='relu', input_shape=input_shape,
                     kernel_initializer='he_normal'))
    model.add(Conv2D(filters=49, kernel_size=(3, 3), activation='sigmoid',
                     kernel_initializer='he_normal'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=48, kernel_size=(5, 5), activation='sigmoid',
                     kernel_initializer='he_normal'))
    model.add(MaxPooling2D(pool_size=(3, 3)))

    model.add(Dropout(0.33))
    model.add(Flatten())
    model.add(Dense(36, activation='sigmoid'))

    model.add(Dropout(0.33))
    model.add(Dense(36, activation='sigmoid'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])

    return model


train_image_gen = ImageDataGenerator(rescale=1. / 255)
test_image_gen = ImageDataGenerator(rescale=1. / 255)

train_x, test_x, train_y, test_y = train_test_split(x_train, y_train.toarray(), test_size=0.2)

model = get_model()
model.fit_generator(train_image_gen.flow(x_train, y_train.toarray(), batch_size=BATCH_SIZE),
                    steps_per_epoch=x_train.shape[0] // BATCH_SIZE,
                    epochs=EPOCHS,
                    verbose=2)

evaluation_score = model.evaluate_generator(test_image_gen.flow(test_x, test_y, batch_size=BATCH_SIZE))
print('evaluation ', evaluation_score)


def predict():
    test_files = []
    test_preds = []
    with open(sample_submission_path, 'w') as f:
        with warnings.catch_warnings():
            f.write('Image,Id\n')
            warnings.filterwarnings("ignore", category=DeprecationWarning)

            for image in tqdm(test_images):
                img = import_image(image)
                img = img.astype('float32')

                y = model.predict_proba(img.reshape(1, IMAGE_SIZE, IMAGE_SIZE, 1))
                predicted_args = np.argsort(y)[0][::-1][:5]
                predicted_tags = label_one_hot_encoder.inverse_labels(predicted_args)
                image = os.path.split(image)[-1]
                predicted_tags = ' '.join(predicted_tags)
                test_files.append(image)
                test_preds.append(predicted_tags)

    data = {'Image': test_files, 'Id': test_preds}
    df = pd.DataFrame(data)
    df.to_csv('../submissions/submission.csv', index=None)


if args.predict:
    predict()

# Thank you for reading this
# My thoughts about improvement current flow
# Increase number of layers as well as number of conv2d filters
# Increase image size from 64 to 128
# Try conv-batch norm-activation
# Finetune Adam learning rate
# Make aggressive image augmentation
# Set dropout >= 0.5
# Please share you thought about improvement such model and flow in the comment section :)