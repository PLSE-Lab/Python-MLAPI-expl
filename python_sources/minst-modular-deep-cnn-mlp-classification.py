# own approach based on previous work, public kernels, and Hands-on ML book

import pickle
import argparse
from keras.models import Sequential, model_from_yaml
from keras.utils import np_utils
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten, InputLayer
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# fix random seed for reproducibility
seed = 42
np.random.seed(seed)


def read_prepare_data():
    """Read, encode, normalize, reshape, and return data split into X_train, y_train, X_test"""
    train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
    labels = train['label']
    X_train = train.drop(labels=["label"], axis=1)
    X_test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
    del train

    # convert list of labels to binary class matrix
    y_train = np_utils.to_categorical(labels)
    assert X_train.shape[1] == 28*28 and y_train.shape[1] == 10, "X/y dimensions wrong"

    # normalize
    X_train, X_test = normalize(X_train, X_test)

    # Reshape image in 3 dimensions (height = 28px, width = 28px , canal = 1)
    X_train = X_train.values.reshape(-1, 28, 28, 1)
    X_test = X_test.values.reshape(-1, 28, 28, 1)

    return X_train, y_train, X_test


def normalize(X_train, X_test):
    """Normalize: Divide by max, substract mean. Very important for deep NN!"""
    X_train /= 255.0
    X_test /= 255.0
    mean = np.std(X_train)
    X_train -= mean
    X_test -= mean
    return X_train, X_test


def simple_mlp_model(in_shape=(28, 28, 1), num_classes=10):
    """
    Simple MLP with 3 layers and dropbout
    From https://www.kaggle.com/fchollet/simple-deep-mlp-with-keras/code
    """
    model = Sequential()
    model.add(InputLayer(input_shape=in_shape))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.15))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.15))
    model.add(Dense(num_classes, activation='softmax'))
    # we'll use categorical xent for the loss, and RMSprop as the optimizer
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model


def cnn_model(in_shape=(28, 28, 1), num_classes=10):
    """
    In -> [[Conv2D->relu]*2 -> MaxPool2D -> Dropout]*2 -> Flatten -> Dense -> Dropout -> Out
    From https://www.kaggle.com/yassineghouzam/introduction-to-cnn-keras-0-997-top-6
    """
    model = Sequential()

    model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='Same', activation='relu', input_shape=in_shape))
    model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='Same', activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same', activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same', activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation="softmax"))

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model


def save_model(model, name, history=None):
    """save model to yaml and weights to hd5. If available, also save history"""
    print("Saving model and weights to file")
    with open(f'{name}.yaml', 'w') as yaml_file:
        yaml_file.write(model.to_yaml())
    model.save_weights(f'{name}.h5')
    if history is not None:
        with open(f'train_history_{name}.pkl', 'wb') as pkl_file:
            pickle.dump(history.history, pkl_file)


def load_model(name):
    """load model from file and compile"""
    print("Loading model and weights from file")
    with open(f'{name}.yaml', 'r') as yaml_file:
        model = model_from_yaml(yaml_file.read())
    model.load_weights(f'{name}.h5')
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model


def data_gen(X_train):
    """Create, fit, and return data generator for data augmentation. Zoom, rotate, shift by 10%"""
    datagen = ImageDataGenerator(rotation_range=10, zoom_range=0.1, width_shift_range=0.1, height_shift_range=0.1)
    datagen.fit(X_train)
    return datagen

def plot_learning_curve(history):
    """Plot the loss and accuracy curves for training and validation"""
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(history.history['loss'], color='b', label="Training loss")
    ax[0].plot(history.history['val_loss'], color='r', label="validation loss", axes=ax[0])
    legend = ax[0].legend(loc='best', shadow=True)

    ax[1].plot(history.history['acc'], color='b', label="Training accuracy")
    ax[1].plot(history.history['val_acc'], color='r', label="Validation accuracy")
    legend = ax[1].legend(loc='best', shadow=True)


def predict(model, X_test, fname):
    """Make and save predictions"""
    print(f"Generating test predictions and saving in {fname}")
    preds = model.predict_classes(X_test, verbose=0)
    df = pd.DataFrame({"ImageId": list(range(1, len(preds) + 1)), "Label": preds})
    df.to_csv(f"{fname}", index=False, header=True)
    return preds


def run_simple_mlp(X_train, y_train, X_test, epochs=1):
    """Create, train simple MLP and make predictions"""
    model = simple_mlp_model()
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=16, validation_split=0.1, verbose=2)
    save_model(model, name='mlp', history=history)
    y_pred = predict(model, X_test, fname='mlp.csv')


def run_cnn(X_train, y_train, X_test, epochs=1):
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=seed)

    model = cnn_model()
    datagen = data_gen(X_train)
    # reduce learning rate by half if no improvement of accuracy in 3 epochs
    lr_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=3, verbose=1, factor=0.5, min_lr=0.00001)
    batch_size = 86
    # set to 30 epochs for 99+% accuracy. but requires 2,5h...
    history = model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size),
                        validation_data=(X_val, y_val), callbacks=[lr_reduction],
                        epochs=epochs, steps_per_epoch=X_train.shape[0] // batch_size, verbose=2)
    save_model(model, name='cnn', history=history)
    # model = load_model(name='cnn')
    y_pred = predict(model, X_test, fname='cnn.csv')

def parse_args():
    parser = argparse.ArgumentParser(description="Args for training on MINST")
    parser.add_argument('--cnn', required=False, action='store_true', help="Use CNN (default: NN)")
    parser.add_argument('--epochs', '-e', required=False, type=int, default=1, help="Number of training epochs")
    return parser.parse_args()


if __name__ == '__main__':
    #args = parse_args()
    #print(f"CLI Args: {args}")

    X_train, y_train, X_test = read_prepare_data()

    run_cnn(X_train, y_train, X_test, epochs=30)
    #run_simple_mlp(X_train, y_train, X_test)



