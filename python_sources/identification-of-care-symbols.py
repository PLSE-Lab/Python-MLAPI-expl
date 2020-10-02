import numpy as np  # linear algebra

import os

import csv
import cv2
import matplotlib.pyplot as plt
from skimage import io
from skimage import filters
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, AveragePooling2D
from keras.callbacks import EarlyStopping
import keras.backend as K


def hamm_loss(y_true, y_pred):
    return K.mean(y_true * (1 - y_pred) + (1 - y_true) * y_pred)

image_ext = ['.jpg', '.jpeg', '.JPG', '.JPEG']


def load_and_filter_image(filename, out_size):
    for ext in image_ext:
        if os.path.exists(filename + ext):
            filename += ext
            break
    image = cv2.imread(filename, 0)
    image = cv2.resize(image, (out_size, out_size))
    image = cv2.medianBlur(image, 3)
    image = filters.prewitt(image, mask = None)
    return image


def load_data(csv_file, image_folder, image_size):
    labels = []
    features = []
    with open(csv_file) as csvfile:
        reader = csv.reader(csvfile)
        image_labels = []
        while True:
            try:
                new_image = True
                _id, image_name, tag, tag_label = reader.__next__()
                image_labels.append(int(tag_label))
                while True:
                    try:
                        _id, image_name_next, tag, tag_label = reader.__next__()
                        if image_name != image_name_next:
                            new_image = False
                            break
                        image_labels.append(int(tag_label))
                    except StopIteration:
                        new_image = False
                        break
                if not new_image:
                    labels.append(image_labels)
                    image_file = os.path.join(image_folder, image_name)
                    image = load_and_filter_image(image_file, image_size)
                    features.append(image)
                    image_labels = []
                    image_labels.append(int(tag_label))
            except StopIteration:
                break
            except ValueError:
                continue
    return np.array(features), np.asarray(labels, dtype=np.float32)


def load_test(test_folder, image_size):
    features = []
    for root, dirs, files in os.walk(test_folder):
        for file in files:
            image_file = os.path.join(root, file)
            image = load_and_filter_image(image_file, image_size)
            features.append(image)
    return np.asarray(features)

# path_caresymbols = "../input/symbol-images/care-symbols/care-symbols/care-symbols"
# caresymbol_features = load_data("../input/symbols/care_symbols.csv", path_caresymbols, 256)

path_train = "../input/identification-care-symbols/train/Train"
path_test = "../input/identification-care-symbols/test/Test"
print(os.listdir(path_train))
image_size = 256
train_bool = True
if train_bool:
    train_features, train_labels = load_data('../input/identification-care-symbols/train_target.csv', path_train, image_size)
    train_features = np.reshape(train_features, (-1, image_size, image_size, 1))
    num_classes = train_labels.shape[1]
    cnn_model = Sequential()
    cnn_model.add(Conv2D(64, 13, padding = "same", activation = 'relu', input_shape=(image_size, image_size, 1)))
    cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
    cnn_model.add(Conv2D(64, 3, activation = 'relu'))
    cnn_model.add(Conv2D(64, 3, activation = 'relu'))
    cnn_model.add(Dropout(0.5))
    cnn_model.add(Conv2D(64, 3, activation = 'relu'))
    cnn_model.add(Conv2D(64, 3, activation = 'relu'))
    cnn_model.add(AveragePooling2D(pool_size=(2, 2)))
    cnn_model.add(Dropout(0.5))
    cnn_model.add(Flatten())
    cnn_model.add(Dense(128, activation = "relu"))
    cnn_model.add(Dropout(0.5))
    cnn_model.add(BatchNormalization())
    cnn_model.add(Dense(num_classes, activation='sigmoid'))
    cnn_model.compile(loss=hamm_loss, optimizer='adam', metrics=['accuracy'])
    hist = cnn_model.fit(train_features, train_labels, batch_size=64, epochs=500, validation_split = 0.20, validation_data = (train_features, train_labels), shuffle=True)
    cnn_model.save('model.model')
else:
    cnn_model = load_model('model.model', custom_objects={'hamm_loss': hamm_loss})

test_features = load_test(path_test, image_size)
test_features = np.reshape(test_features, (-1, image_size, image_size, 1))
test_y = cnn_model.predict(test_features)
test_y[np.where(np.greater(test_y, 0.5))] = 1.0
test_y[np.where(np.less_equal(test_y, 0.5))] = 0.0

with open("../input/identification-care-symbols/sample_submission.csv") as f:
    sub = csv.reader(f)
    test_y = test_y.astype(dtype=np.int8)
    sub = list(sub)[1:]
    ind = 0
    for label in test_y:
        for label_i in label:
            sub[ind].append(label_i)
            ind += 1
    sub.reverse()
    sub.append(['Id', 'ImageName', 'CareSymbolTag', 'PredictedOutcome'])
    sub.reverse()
    with open("output.csv", 'w') as fw:
        w = csv.writer(fw)
        w.writerows(sub)