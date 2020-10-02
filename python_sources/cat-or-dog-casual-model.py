import os

# %% [code]
os.system("pip install tensorflowjs")

# %% [code]
import os
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import math

# %% [code]
SAVE_BEST_MODEL = tf.keras.callbacks.ModelCheckpoint(filepath='best_model.hdf5', verbose=1, save_best_only=True, save_weights_only=True)

# %% [code]
class DataManager:
    """
        Creates the image directories, the dataframes, and the generator.
    """
    def __init__(self):
        self.train_directory = "../input/dogs-vs-cats-redux-kernels-edition/train"
        self.test_directory = "../input/dogs-vs-cats-redux-kernels-edition/test"
        
        self.generator = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1/255.,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True
        )
        
        self.DataFrame = pd.DataFrame({
            "id":[filename for filename in os.listdir(self.train_directory) if filename != "train"],
            "label":[filename[:3] for filename in os.listdir(self.train_directory) if filename != "train"]
        })
        self.train_df, self.validation_df = train_test_split(self.DataFrame, test_size=0.25)
      
    """
        Outputs the train dataframe and the validation dataframe. There is no test dataframe to output.
    """
    def get_dataframes(self):
        return (self.train_df, self.validation_df)
    
    """
        Return the 2 image directories. The first dir. is shared by both the training images and the validation images. The second dir. is for just the test images.
    """
    def get_directories(self):
        return (self.train_directory, self.test_directory)
    
    """
        Return the generator used for both train, validation, and test images.
    """
    def get_generator(self):
        return self.generator

# %% [code]
class CasualModel:
    def __init__(self):
        self.vgg = tf.keras.applications.vgg16.VGG16(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3)
        )
        self.vgg.trainable = False
        
    def create_model(self):
        model = tf.keras.models.Sequential([
            self.vgg,
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        return model

# %% [code]
data = DataManager()
generator = data.get_generator()
train_df, validation_df = data.get_dataframes()
train_dir, test_dir = data.get_directories()

# %% [code]
train_generator = generator.flow_from_dataframe(
    dataframe=train_df,
    directory=train_dir,
    x_col='id',
    y_col='label',
    target_size=(224, 224),
    batch_size=128,
    class_mode='binary'
)
validation_generator = generator.flow_from_dataframe(
    dataframe=train_df,
    directory=train_dir,
    x_col='id',
    y_col='label',
    target_size=(224, 224),
    batch_size=128,
    class_mode='binary'
)

# %% [code]
model = CasualModel().create_model()
print(model.summary())
model.compile(optimizer='adam', metrics=['acc'], loss='binary_crossentropy')
history = model.fit_generator(
    train_generator,
    steps_per_epoch=len(train_df)//128,
    epochs=5,
    validation_data=validation_generator,
    validation_steps=len(validation_df)//128,
    callbacks=[SAVE_BEST_MODEL],
    verbose=2
)

# %% [code]
model.save("casual_model.h5")

# %% [code]
os.system("mkdir model/")

# %% [code]
os.system("tensorflowjs_converter --input_format keras casual_model.h5 model/")