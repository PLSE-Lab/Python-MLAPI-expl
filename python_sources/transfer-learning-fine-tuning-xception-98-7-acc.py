# See full code and description on GitHub
# https://github.com/voandy/monkey-classifier

import os.path

from keras import Model, optimizers
from keras.applications import xception
from keras.layers import GlobalAveragePooling2D, Dense
from keras.preprocessing.image import ImageDataGenerator

img_width = 197
img_height = 197

epochs = 100
epochs_ft = 200
batch_size = 120

train_path = 'training-data/training'
validation_path = 'training-data/validation'

nb_classes = 10

# Counts the number of training and testing samples in the directories
training_samples = sum([len(files) for r, d, files in os.walk(train_path)])
testing_samples = sum([len(files) for r, d, files in os.walk(validation_path)])

train_datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.3,
    height_shift_range=0.2,
    rescale=1.0 / 255,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip=False,
)

validation_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
)

test_generator = validation_datagen.flow_from_directory(
    validation_path,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
)


# Adds new top to base model
def add_top(base):
    x = base.output

    # Global averaging pool layer
    x = GlobalAveragePooling2D()(x)

    # Regular densely connected layer
    x = Dense(512, activation='relu')(x)

    # Output layer
    predictions = Dense(nb_classes, activation='softmax')(x)

    return Model(input=base.input, output=predictions)


# Sets up model for transfer learning
def setup_model(model, base):
    # Freeze the un-trainable layers of the model base
    for layer in base.layers:
        layer.trainable = False

    model.compile(
        loss='categorical_crossentropy',
        optimizer='rmsprop',
        metrics=['accuracy']
    )


# Import the Xception model to use as the base for our model
xception_base = xception.Xception(
    include_top=False,
    weights='imagenet',
    input_shape=(img_width, img_height, 3)
)

model = add_top(xception_base)
setup_model(model, xception_base)

# Train our new top layer
model.fit_generator(
    train_generator,
    steps_per_epoch=training_samples // batch_size,
    epochs=epochs,
    validation_data=test_generator,
    validation_steps=testing_samples // batch_size,
    verbose=1,
)


# Setup model for fine tuning
def setup_model(model, trainable):
    # Freeze the un-trainable layers of the model base
    for layer in model.layers[:(len(model.layers) - trainable)]:
        layer.trainable = False

    for layer in model.layers[(len(model.layers) - trainable):]:
        layer.trainable = True

    model.compile(
        loss='categorical_crossentropy',
        # Slower training rate for fine-tuning
        optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
        metrics=['accuracy']
    )


# Setup model to retrain our top layer plus block 13 and 14 of Xception
setup_model(model, 19)

# Fine-tune the model
model.fit_generator(
    train_generator,
    steps_per_epoch=training_samples // batch_size,
    epochs=epochs_ft,
    validation_data=test_generator,
    validation_steps=testing_samples // batch_size,
    verbose=1,
)