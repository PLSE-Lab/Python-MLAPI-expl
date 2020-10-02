import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

x_train = train.drop(labels=["label"], axis=1)
x_train = x_train / 255.0
x_train = x_train.values.reshape(-1, 28, 28, 1)
y_train = train["label"]
y_train = to_categorical(y_train, num_classes=10)
x_test = test / 255.0
x_test = x_test.values.reshape(-1, 28, 28, 1)

datagen = ImageDataGenerator(rotation_range=10,
                             zoom_range=0.1,
                             width_shift_range=0.1,
                             height_shift_range=0.1)

NETS_NUMBER = 16
models = [0] * NETS_NUMBER

for i in range(NETS_NUMBER):
    models[i] = Sequential()
    models[i].add(Conv2D(32, kernel_size=3, activation="linear", padding="same", input_shape=(28, 28, 1)))
    models[i].add(PReLU())
    models[i].add(BatchNormalization())
    models[i].add(Conv2D(32, kernel_size=3, activation="linear", padding="same"))
    models[i].add(PReLU())
    models[i].add(BatchNormalization())
    models[i].add(Conv2D(32, kernel_size=2, activation="linear", padding="valid", strides=2))
    models[i].add(PReLU())
    models[i].add(BatchNormalization())
    models[i].add(Dropout(0.4, seed=i))
    models[i].add(Conv2D(64, kernel_size=3, activation="linear", padding="same"))
    models[i].add(PReLU())
    models[i].add(BatchNormalization())
    models[i].add(Conv2D(64, kernel_size=3, activation="linear", padding="same"))
    models[i].add(PReLU())
    models[i].add(BatchNormalization())
    models[i].add(Conv2D(64, kernel_size=2, activation="linear", padding="valid", strides=2))
    models[i].add(PReLU())
    models[i].add(BatchNormalization())
    models[i].add(Dropout(0.4, seed=i))
    models[i].add(Flatten())
    models[i].add(Dense(128, activation="linear"))
    models[i].add(PReLU())
    models[i].add(BatchNormalization())
    models[i].add(Dropout(0.4, seed=i))
    models[i].add(Dense(10, activation="softmax"))
    models[i].compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    
lr_reduce = ReduceLROnPlateau(monitor="val_acc", factor=0.75, min_delta=0.0001, patience=3, verbose=0)

for i in range(NETS_NUMBER):
    x_train2, x_val2, y_train2, y_val2 = train_test_split(x_train, y_train, test_size=0.2, random_state=i)
    models[i].fit_generator(datagen.flow(x_train2, y_train2, batch_size=60),
                            validation_data=(x_val2, y_val2),
                            epochs=60,
                            steps_per_epoch=700,
                            callbacks=[lr_reduce],
                            verbose=0)
                            
results = np.zeros((x_test.shape[0], 10))
for i in range(NETS_NUMBER):
    results = results + models[i].predict(x_test)
results = np.argmax(results, axis=1)
results = pd.Series(results, name="Label")

submission = pd.concat([pd.Series(range(1, 28001), name="ImageId"), results], axis=1)
submission.to_csv("submission.csv", index=False)