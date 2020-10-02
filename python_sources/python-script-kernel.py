import os
import gc
import cv2
import shutil
import operator
import numpy as np
import pandas as pd

from math import ceil
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras import layers, optimizers
from keras.applications import inception_v3, xception, nasnet, inception_resnet_v2
from keras.layers import Dense, GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator


class CarModelClassifier:
    def __init__(self, SEED, TRAIN_SIZE, img_size, input_shape, epochs, batch_size):
        self.SEED = SEED
        self.TRAIN_SIZE = TRAIN_SIZE
        self.img_size = img_size
        self.input_shape = input_shape
        self.epochs = epochs
        self.batch_size = batch_size

        np.random.seed(self.SEED)

        self.DATA_PATH = './input'

        self.TRAIN_IMG_PATH = os.path.join(self.DATA_PATH, 'train')
        self.VALID_IMG_PATH = os.path.join(self.DATA_PATH, 'valid')
        self.TEST_IMG_PATH = os.path.join(self.DATA_PATH, 'test')

        self.TRAIN_CROP_PATH = "./train_crop"
        self.VALID_CROP_PATH = "./valid_crop"
        self.TEST_CROP_PATH = "./test_crop"

    def load_csv(self):
        # load .csv files
        self.df_train = pd.read_csv(os.path.join(self.DATA_PATH, 'train.csv'))
        self.df_valid = pd.read_csv(os.path.join(self.DATA_PATH, 'valid.csv'))
        self.df_test = pd.read_csv(os.path.join(self.DATA_PATH, 'test.csv'))
        self.df_class = pd.read_csv(os.path.join(self.DATA_PATH, 'class.csv'))

    def get_steps(self, num_samples, batch_size):
        return ceil(num_samples / batch_size)

    def move_valid_images(self):
        if not os.path.isdir(self.VALID_IMG_PATH):
            os.mkdir(self.VALID_IMG_PATH)

        files = self.df_valid['img_file']
        for file in files:
            org_path = os.path.join(self.TRAIN_IMG_PATH, file)
            dst_path = self.VALID_IMG_PATH
            shutil.move(org_path, dst_path)

    def train_test_split(self):
        org_train = pd.read_csv(os.path.join(self.DATA_PATH, 'org-train.csv'))

        its = np.arange(org_train.shape[0])
        train_idx, val_idx = train_test_split(its, train_size=self.TRAIN_SIZE, random_state=self.SEED)

        X_train = org_train.iloc[train_idx]
        X_val = org_train.iloc[val_idx]

        X_train.to_csv(os.path.join(self.DATA_PATH, 'train.csv'), index=False)
        X_val.to_csv(os.path.join(self.DATA_PATH, 'valid.csv'), index=False)

    def crop_images(self):
        # Image pre-processing
        self._crop_images(self.TRAIN_IMG_PATH, self.TRAIN_CROP_PATH, self.df_train)
        self._crop_images(self.VALID_IMG_PATH, self.VALID_CROP_PATH, self.df_valid)
        self._crop_images(self.TEST_IMG_PATH, self.TEST_CROP_PATH, self.df_test)

    def _crop_images(self, img_path, crop_path, df):
        if not os.path.isdir(crop_path):
            os.mkdir(crop_path)

        for i, row in tqdm(df.iterrows()):
            cropped = self._crop_boxing_img(row['img_file'], img_path, df)
            if not os.path.isfile(os.path.join(crop_path, row['img_file'])):
                cv2.imwrite(os.path.join(crop_path, row['img_file']), cropped)

    def _crop_boxing_img(self, img_name, img_path, df, margin=5):
        img = cv2.imread(os.path.join(img_path, img_name))
        pos = df.loc[df["img_file"] == img_name, ['bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2']].values.reshape(-1)

        height, width = img.shape[:2]
        x1 = max(0, pos[0] - margin)
        y1 = max(0, pos[1] - margin)
        x2 = min(pos[2] + margin, width)
        y2 = min(pos[3] + margin, height)

        return cv2.resize(img[y1:y2, x1:x2], self.img_size)

    def set_generators(self, model_name):
        self.df_train["class"] = self.df_train["class"].astype('str')
        self.df_valid["class"] = self.df_valid["class"].astype('str')
        self.df_train = self.df_train[['img_file', 'class']]
        self.df_test = self.df_test[['img_file']]

        self.nb_train_samples = len(self.df_train)
        self.nb_validation_samples = len(self.df_valid)
        self.nb_test_samples = len(self.df_test)

        if model_name == "inception_v3":
            preprocess_input = inception_v3.preprocess_input
        elif model_name == "inception_resnet_v2":
            preprocess_input = inception_resnet_v2.preprocess_input
        elif model_name == "xception":
            preprocess_input = xception.preprocess_input
        elif model_name == "nasNet":
            preprocess_input = nasnet.preprocess_input
        else:
            raise ValueError("model name incorrect!")

        # Define Generator config
        train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
        val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
        test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

        # Make Generator
        self.train_generator = train_datagen.flow_from_dataframe(
            dataframe=self.df_train,
            directory=self.TRAIN_CROP_PATH,
            x_col='img_file',
            y_col='class',
            target_size=self.img_size,
            color_mode='rgb',
            class_mode='categorical',
            batch_size=self.batch_size,
            seed=42
        )

        self.validation_generator = val_datagen.flow_from_dataframe(
            dataframe=self.df_valid,
            directory=self.VALID_CROP_PATH,
            x_col='img_file',
            y_col='class',
            target_size=self.img_size,
            color_mode='rgb',
            class_mode='categorical',
            batch_size=self.batch_size,
            shuffle=False
        )

        self.test_generator = test_datagen.flow_from_dataframe(
            dataframe=self.df_test,
            directory=self.TEST_CROP_PATH,
            x_col='img_file',
            y_col=None,
            target_size=self.img_size,
            color_mode='rgb',
            class_mode=None,
            batch_size=self.batch_size,
            shuffle=False
        )

    def create_model(self, model_name):
        """
        :param model_name: model to use (inceptionV3, xception, nasNet)
        :return: selected model
        """
        if model_name == "inception_v3":
            base_model = inception_v3.InceptionV3(include_top=False, weights='imagenet', input_shape=self.input_shape)
        elif model_name == "inception_resnet_v2":
            base_model = inception_resnet_v2.InceptionResNetV2(include_top=False, weights='imagenet', input_shape=self.input_shape)
        elif model_name == "xception":
            base_model = xception.Xception(include_top=False, weights='imagenet', input_shape=self.input_shape)
        elif model_name == "nasNet":
            base_model = nasnet.NASNetLarge(include_top=False, weights='imagenet', input_shape=self.input_shape)
        else:
            raise ValueError("model name incorrect!")

        model = Sequential()
        model.add(base_model)
        model.add(GlobalAveragePooling2D())
        model.add(layers.Dense(1024, activation='relu', kernel_initializer='he_normal'))
        model.add(layers.Dropout(0.5))
        model.add(Dense(self.df_class.shape[0], activation='softmax', kernel_initializer='he_normal'))

        optimizer = optimizers.Adam(lr=0.00001)

        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc'])

        model.summary()

        return model

    def load_model(self, model_name, model_path):
        model = self.create_model(model_name)
        model.load_weights(model_path)

        return model

    def train(self, model, filepath):
        ckpt = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True)
        es = EarlyStopping(monitor='val_acc', patience=3, verbose=1)

        callback_list = [ckpt, es]

        hist = model.fit_generator(
            self.train_generator,
            steps_per_epoch=self.get_steps(self.nb_train_samples, self.batch_size),
            epochs=self.epochs,
            validation_data=self.validation_generator,
            validation_steps=self.get_steps(self.nb_validation_samples, self.batch_size),
            callbacks=callback_list
        )

        gc.collect()

        return hist

    def test(self, model_name, model_path):
        model = self.load_model(model_name, model_path)

        self.test_generator.reset()

        prediction = model.predict_generator(
            generator=self.test_generator,
            steps=self.get_steps(self.nb_test_samples, self.batch_size),
            verbose=1
        )

        return prediction

    def make_probability(self, prediction, out_file_path):
        labels = self.make_labels(self.train_generator.class_indices)
        prediction_T = prediction.T

        predict_dic = {0:np.zeros([len(prediction)], np.float32)}
        for i, row in enumerate(prediction_T):
            predict_dic[int(labels[i])] = row

        sorted_x = dict(sorted(predict_dic.items(), key=operator.itemgetter(0)))
        df_predict_dic = pd.DataFrame(sorted_x)
        df_predict_dic.to_csv(out_file_path, index=False)

    def make_submission(self, prediction, out_file_path):
        predicted_class_indices = np.argmax(prediction, axis=1)

        labels = self.make_labels(self.train_generator.class_indices)

        predictions = [labels[k] for k in predicted_class_indices]

        submission = pd.read_csv(os.path.join(self.DATA_PATH, 'sample_submission.csv'))
        submission["class"] = predictions
        submission.to_csv(out_file_path, index=False)

    def make_labels(self, class_indices):
        labels = dict((v, k) for k, v in class_indices.items())\

        return labels
    

if __name__ == '__main__':
    SEED = 42
    TRAIN_SIZE = 0.8
    img_size = (331, 331)
    input_shape = img_size + (3,)
    epochs = 30
    batch_size = 3

    WEIGHTS_PATH = './weights'
    MODEL_NAME = 'nasNet'
    WEIGHT_FILE_PATH = os.path.join(WEIGHTS_PATH, 'my_{}_model.h5'.format(MODEL_NAME))

    carModelClassifier = CarModelClassifier(SEED, TRAIN_SIZE, img_size, input_shape, epochs, batch_size)

    carModelClassifier.train_test_split()
    carModelClassifier.load_csv()

    carModelClassifier.move_valid_images()
    carModelClassifier.crop_images()

    carModelClassifier.set_generators(MODEL_NAME)

    model = carModelClassifier.create_model(MODEL_NAME)
    # model = carModelClassifier.load_model(MODEL_NAME, WEIGHT_FILE_PATH)
    carModelClassifier.train(model, WEIGHT_FILE_PATH)

    prediction = carModelClassifier.test(MODEL_NAME, WEIGHT_FILE_PATH)
    carModelClassifier.make_probability(prediction, 'submission/predict_dict_{}.csv'.format(MODEL_NAME))
    carModelClassifier.make_submission(prediction, "submission_nasNet.csv")