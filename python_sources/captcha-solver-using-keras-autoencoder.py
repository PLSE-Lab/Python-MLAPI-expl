import os
import os.path
import glob
import cv2
import numpy as np
import pandas as pd

from imutils import paths
import cv2

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, Dense, Dropout, Input, MaxPooling2D, UpSampling2D
from keras.models import Model
import keras.optimizers
from keras.callbacks import EarlyStopping, ReduceLROnPlateau


letter_contours = []
counts = {}

# After the trials done in the notebook we can get the contours of each letters.
x, y, w, h = 30, 12, 20, 38
for i in range(5):
    letter_contours.append((x, y, w, h))
    x += w

# Image folders.
CAPTCHA_IMAGE_FOLDER = "../input/samples/samples"
OUTPUT_FOLDER = "../input/samples/letters"

# List of captchas.
captcha_images = glob.glob(os.path.join(CAPTCHA_IMAGE_FOLDER, "*"))

# Go through each image in the folder.
for (i, captcha_image) in enumerate(captcha_images):
    print("Image {}/{}".format(i + 1, len(captcha_image)))

    # Get the letter labels from the imgae names.
    filename = os.path.basename(captcha_image)
    captcha_correct_text = os.path.splitext(filename)[0]

    # Load image and convert to grey.
    img = cv2.imread(captcha_image, 0)

    # From RGB to BW
    # Adaptive thresholding
    th = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 17, 2)

    # Erode and dilate (Because it is black on white, erosion dilates and dilation erodes).
    kernel = np.ones((3, 3), np.uint8)
    dilation = cv2.dilate(th, kernel, iterations=1)

    erosion = cv2.erode(dilation, kernel, iterations=1)

    kernel = np.ones((3, 1), np.uint8)
    dilation = cv2.dilate(erosion, kernel, iterations=1)

    for letter_contour, letter_text in zip(letter_contours, captcha_correct_text):
        # Grab the coordinates of the letter in the image.
        x, y, w, h = letter_contour

        # Extract the letter from the original image with a 2-pixel margin around the edge.
        letter_image = dilation[y - 2:y + h + 2, x - 2:x + w + 2]

        # Get the folder to save the image in
        save_path = os.path.join(OUTPUT_FOLDER, letter_text)

        # If the output directory does not exist, create it
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # Write the letter image to a file
        count = counts.get(letter_text, 1)
        p = os.path.join(save_path, "{}.png".format(str(count).zfill(6)))
        cv2.imwrite(p, letter_image)

        # increment the count for the current key
        counts[letter_text] = count + 1

LETTER_FOLDER = OUTPUT_FOLDER
data = []
target = []
labels = []

# Load data.
for image_file in paths.list_images(LETTER_FOLDER):
    # Load the image.
    image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)

    # Add a third channel dimension to the image.
    image = np.expand_dims(image, axis=2)

    # Get the folder name (ie. the true character value).
    label = image_file.split(os.path.sep)[-2]

    # Add the image and char to the dictionary.
    data.append(image)
    if label == '2':
        img = cv2.imread(LETTER_FOLDER + '/2/000005.png', cv2.IMREAD_GRAYSCALE)
        img = np.expand_dims(img, axis=2)
        target.append(img)
        labels.append(1)
    elif label == '3':
        img = cv2.imread(LETTER_FOLDER +'/3/000127.png', cv2.IMREAD_GRAYSCALE)
        img = np.expand_dims(img, axis=2)
        target.append(img)
        labels.append(2)
    elif label == '4':
        img = cv2.imread(LETTER_FOLDER + '/4/000001.png', cv2.IMREAD_GRAYSCALE)
        img = np.expand_dims(img, axis=2)
        target.append(img)
        labels.append(3)
    elif label == '5':
        img = cv2.imread(LETTER_FOLDER + '/5/000007.png', cv2.IMREAD_GRAYSCALE)
        img = np.expand_dims(img, axis=2)
        target.append(img)
        labels.append(4)
    elif label == '6':
        img = cv2.imread(LETTER_FOLDER + '/6/000006.png', cv2.IMREAD_GRAYSCALE)
        img = np.expand_dims(img, axis=2)
        target.append(img)
        labels.append(5)
    elif label == '7':
        img = cv2.imread(LETTER_FOLDER + '/7/000017.png', cv2.IMREAD_GRAYSCALE)
        img = np.expand_dims(img, axis=2)
        target.append(img)
        labels.append(6)
    elif label == '8':
        img = cv2.imread(LETTER_FOLDER + '/8/000004.png', cv2.IMREAD_GRAYSCALE)
        img = np.expand_dims(img, axis=2)
        target.append(img)
        labels.append(7)
    elif label == 'b':
        img = cv2.imread(LETTER_FOLDER + '/b/000001.png', cv2.IMREAD_GRAYSCALE)
        img = np.expand_dims(img, axis=2)
        target.append(img)
        labels.append(8)
    elif label == 'c':
        img = cv2.imread(LETTER_FOLDER + '/c/000014.png', cv2.IMREAD_GRAYSCALE)
        img = np.expand_dims(img, axis=2)
        target.append(img)
        labels.append(9)
    elif label == 'd':
        img = cv2.imread(LETTER_FOLDER + '/d/000025.png', cv2.IMREAD_GRAYSCALE)
        img = np.expand_dims(img, axis=2)
        target.append(img)
        labels.append(10)
    elif label == 'e':
        img = cv2.imread(LETTER_FOLDER + '/e/000004.png', cv2.IMREAD_GRAYSCALE)
        img = np.expand_dims(img, axis=2)
        target.append(img)
        labels.append(11)
    elif label == 'f':
        img = cv2.imread(LETTER_FOLDER + '/f/000015.png', cv2.IMREAD_GRAYSCALE)
        img = np.expand_dims(img, axis=2)
        target.append(img)
        labels.append(12)
    elif label == 'g':
        img = cv2.imread(LETTER_FOLDER + '/g/000034.png', cv2.IMREAD_GRAYSCALE)
        img = np.expand_dims(img, axis=2)
        target.append(img)
        labels.append(13)
    elif label == 'm':
        img = cv2.imread(LETTER_FOLDER + '/m/000036.png', cv2.IMREAD_GRAYSCALE)
        img = np.expand_dims(img, axis=2)
        target.append(img)
        labels.append(14)
    elif label == 'n':
        img = cv2.imread(LETTER_FOLDER + '/n/000002.png', cv2.IMREAD_GRAYSCALE)
        img = np.expand_dims(img, axis=2)
        target.append(img)
        labels.append(15)
    elif label == 'p':
        img = cv2.imread(LETTER_FOLDER + '/p/000002.png', cv2.IMREAD_GRAYSCALE)
        img = np.expand_dims(img, axis=2)
        target.append(img)
        labels.append(16)
    elif label == 'w':
        img = cv2.imread(LETTER_FOLDER + '/w/000017.png', cv2.IMREAD_GRAYSCALE)
        img = np.expand_dims(img, axis=2)
        target.append(img)
        labels.append(17)
    elif label == 'x':
        img = cv2.imread(LETTER_FOLDER + '/x/000012.png', cv2.IMREAD_GRAYSCALE)
        img = np.expand_dims(img, axis=2)
        target.append(img)
        labels.append(18)
    elif label == 'y':
        img = cv2.imread(LETTER_FOLDER + '/y/000014.png', cv2.IMREAD_GRAYSCALE)
        img = np.expand_dims(img, axis=2)
        target.append(img)
        labels.append(0)

# Normalization.
data = np.array(data, dtype="float") / 255.0
target = np.array(target, dtype="float") / 255.0
labels = np.array(labels)

# Categorization.
labels = to_categorical(labels, num_classes=19)

# Train-test split.
(x_train, x_test, y_train, y_test) = train_test_split(
    data, target, test_size=0.2, random_state=0)

# Data augmentation.
datagen = ImageDataGenerator(
    # randomly rotate images in the range (degrees, 0 to 180).
    rotation_range=15,
    zoom_range=0.1,  # Randomly zoom image.
    # randomly shift images horizontally (fraction of total width).
    width_shift_range=0.1,
    # randomly shift images vertically (fraction of total height).
    height_shift_range=0.1,
)

datagen.fit(x_train)

# Convolutional autoencoder.
input_img = Input(shape=(40, 24, 1))

# Encoder.
x = Conv2D(16, (5, 5), activation='relu', padding='same')(input_img)
x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
x = Conv2D(8, (5, 5), activation='relu', padding='same')(x)
x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
x = Conv2D(8, (5, 5), activation='relu', padding='same')(x)
encoded = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
encoder = Model(input_img, encoded)

# Decoder.
x = Conv2D(8, (5, 5), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(8, (5, 5), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (5, 5), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (5, 5), activation='sigmoid', padding='same')(x)

# Optimizer.
optimizer = keras.optimizers.Adam(
    lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)

# Callbacks.
learning_rate_reduction = ReduceLROnPlateau(
    monitor='val_acc', patience=2, verbose=1, factor=0.4, min_lr=0.000001)
early_stopping = EarlyStopping(
    monitor='val_loss', min_delta=0, patience=3, verbose=1, mode='auto')

# Create model.
autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer=optimizer,
                    loss='binary_crossentropy', metrics=['accuracy'])

# # TERMINAL CMD : tensorboard --logdir=/tmp/autoencoder
# from keras.callbacks import TensorBoard
#
# autoencoder.fit(x_train, x_train,
#                 epochs=1,
#                 batch_size=128,
#                 # shuffle=True,
#                 validation_data=(x_test, x_test),
#                 callbacks=[TensorBoard(log_dir='/tmp/autoencoder'), learning_rate_reduction])

history = autoencoder.fit_generator(datagen.flow(x_train, y_train, batch_size=64),
                                    # history = autoencoder.fit(x_train, y_train, batch_size=64,
                                    epochs=10, validation_data=(x_test, y_test),
                                    verbose=1, steps_per_epoch=x_train.shape[0],
                                    callbacks=[learning_rate_reduction, early_stopping])

# Visualise reconstructed images.
decoded_imgs = autoencoder.predict(x_test)
n = 5
plt.figure(figsize=(20, 4))
for i in range(n):
    i = i + 1
    # display original
    ax = plt.subplot(2, n, i)
    plt.imshow(x_test[i].reshape(40, 24))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + n)
    plt.imshow(decoded_imgs[i].reshape(40, 24))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()


# Salt and peper noise.
noise_factor = 0.5
x_train_noisy = x_train + noise_factor * \
    np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
x_test_noisy = x_test + noise_factor * \
    np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)

# TODO: augmentation ?

x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

history = autoencoder.fit_generator(datagen.flow(x_train, y_train, batch_size=64),
                                    # history = autoencoder.fit(x_train_noisy, y_train, batch_size=64,
                                    epochs=1, validation_data=(x_test_noisy, y_test),
                                    verbose=1, steps_per_epoch=x_train.shape[0],
                                    callbacks=[learning_rate_reduction, early_stopping])
# Save model.
autoencoder.save("denoising.model")

# Visualise decoded images.
decoded_imgs = autoencoder.predict(x_test)
n = 5
plt.figure(figsize=(20, 4))
for i in range(n):
    i = i + 1
    # display original
    ax = plt.subplot(2, n, i)
    plt.imshow(x_test_noisy[i].reshape(40, 24))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # display reconstruction
    ax = plt.subplot(2, n, i + n)
    plt.imshow(decoded_imgs[i].reshape(40, 24))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

# Use denoise images.
decoded_images = autoencoder.predict(data)

# Train-test split.
(x_train, x_test, y_train, y_test) = train_test_split(
    decoded_images, labels, test_size=0.2, random_state=0)

datagen = ImageDataGenerator(
    # randomly rotate images in the range (degrees, 0 to 180).
    rotation_range=15,
    zoom_range=0.1,  # Randomly zoom image.
    # randomly shift images horizontally (fraction of total width).
    width_shift_range=0.1,
    # randomly shift images vertically (fraction of total height).
    height_shift_range=0.1,
)

datagen.fit(x_train)

# Create model for labelling images.
out = Flatten()(encoder.output)
out = Dense(19, activation='softmax')(out)
labeller = Model(encoder.input, out)

labeller.compile(optimizer=optimizer,
                 loss='binary_crossentropy', metrics=['accuracy'])

history = labeller.fit_generator(datagen.flow(x_train, y_train, batch_size=64),
                                 # history = labeller.fit(x_train, y_train, batch_size=64,
                                 epochs=10, validation_data=(x_test, y_test),
                                 verbose=1, steps_per_epoch=x_train.shape[0],
                                 callbacks=[learning_rate_reduction, early_stopping])

scores = labeller.evaluate(x_test, y_test, verbose=1)
print("Accuracy: ", scores[1])

# Save model.
labeller.save("labelling.model")
