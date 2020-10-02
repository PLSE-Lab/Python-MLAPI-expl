import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_data = pd.read_csv('../input/digit-recognizer/train.csv')
train_labels = train_data['label'].values
train_images = train_data.drop(['label'], axis=1).values
test_images = pd.read_csv('../input/digit-recognizer/test.csv').values

train_images = train_images.reshape((-1, 28, 28, 1)) / 255.0
test_images = test_images.reshape((-1, 28, 28, 1)) / 255.0

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(40, 5, activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(80, 5, activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(200, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax'),
    tf.keras.layers.Dropout(0.2)])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])

gen = ImageDataGenerator(horizontal_flip=True, validation_split=0.1)
train_extension = gen.flow(train_images, train_labels,
    batch_size=256,
    subset='training')
val_extension = gen.flow(train_images, train_labels,
    batch_size=256,
    subset='validation')

model.fit(train_extension,
    epochs=150,
    validation_data=val_extension)

predictions = np.argmax(model.predict(test_images), axis=1)
output = pd.DataFrame({'ImageId': range(1, 28001), 'Label': predictions})
output.to_csv('submission_cnn.csv', index=False)