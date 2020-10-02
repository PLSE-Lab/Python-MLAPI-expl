#!/usr/bin/env python
# coding: utf-8

# # imports

# In[ ]:


import os
import random
import warnings
import numpy as np
import skimage.io
import skimage.transform

from sklearn.preprocessing import LabelEncoder
from keras import layers, models
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
from keras.preprocessing import image as keras_image
from matplotlib import pyplot as plt
from yellowbrick.target import ClassBalance


# In[ ]:


# warnings.filterwarnings('ignore')
np.random.seed(1337)
np.set_printoptions(suppress=True)


# In[ ]:


get_ipython().run_cell_magic('html', '', '<style>\n.output_wrapper, .output {\n    height:auto !important;\n    max-height:100000px;  /* your desired max-height here */\n}\n.output_scroll {\n    box-shadow:none !important;\n    webkit-box-shadow:none !important;\n}\n</style>')


# # data

# ## data preparation

# In[ ]:


# swaps val and test data because of the number of files
train_dir = '../input/chest_xray/chest_xray/train/'
val_dir = '../input/chest_xray/chest_xray/test/'
test_dir = '../input/chest_xray/chest_xray/val/'


# In[ ]:


train_categories = [x for x in os.listdir(train_dir) if x[0] is not '.']
val_categories = [x for x in os.listdir(val_dir) if x[0] is not '.']
test_categories = [x for x in os.listdir(test_dir) if x[0] is not '.']


# In[ ]:


target_size = (150, 150)


# ## show some images

# In[ ]:


def plot_images_per_category(base_dir, categories, size=(150, 150, 3),
                             images_per_category=1, rows=None):
    """Displays one image from each category.

    :param base_dir: <string> -> The main directory in where the data
        is stored.
    :param categories: <list | tuple> -> This must contain all folders
        (categories) from where the data is to be read.
    :param images_per_category: <int> -> The images to be displayed by
        category. Standard is <1>.
    :parameter rows: int -> number of rows in which the images are to
        be divided. Standard is <len(categories)>.
    :param size: list | tuple -> (width, height, channels). Standard is
        <(150, 150, 3)>.
    :return: <None>
    """

    if not rows:
        rows = len(categories)

    images = []
    categories_list = sorted(
        [category for category in categories * images_per_category])

    for ix, category in enumerate(categories):
        directory = os.path.join(base_dir, category)
        images.extend([os.path.join(directory, x) for x in
                       os.listdir(directory)[:images_per_category]])

    fig = plt.figure(figsize=(12, 8))

    for ix, image in enumerate(images, start=1):
        try:
            image = skimage.io.imread(image)
        except OSError:
            continue
        image = skimage.transform.resize(image, size, mode='reflect')
        fig.add_subplot(rows, (len(images) // rows), ix)
        plt.title(categories_list[ix - 1])
        plt.axis('off')
        plt.imshow(image)
    plt.show()


# In[ ]:


plot_images_per_category(train_dir, train_categories, images_per_category=4)


# ## diversification

# In[ ]:


def files_per_directory(base_dir, categories):
    """Counts the files in the directory.

    :param base_dir: <string> -> The main directory in where the images
        are stored.
    :param categories: <list | tuple> -> This must contain all folders
        (categories) from where the images are to be read.
    :return: <list> -> files_per_directory
    """
    files_per_dir = []

    for category in categories:
        files = os.listdir(os.path.join(base_dir, category))
        files_per_dir.append(len(files))

    return files_per_dir


# In[ ]:


train_files = files_per_directory(train_dir, train_categories)
val_files = files_per_directory(val_dir, val_categories)
test_files = files_per_directory(test_dir, test_categories)


# In[ ]:


def plot_diversification(files_per_dir, categories, title='diversification'):
    """Plot the diversification of the data in the folders.

    :param files_per_dir: <list | tuple> ->  Number of files in each
        category.
    :param categories: <list | tuple> -> This must contain all folders
        (categories) from where the data is to be read.
    :param title: <string> -> Title of the plot.
    :return: <None>
    """

    plt.bar([x for x in range(len(files_per_dir))],
            files_per_dir, tick_label=categories)
    plt.title(title)
    plt.show()


# In[ ]:


plot_diversification(train_files, 
                     train_categories, 
                     title='Train data diversification')


# In[ ]:


plot_diversification(val_files, 
                     val_categories, 
                     title='Val data diversification')


# In[ ]:


plot_diversification(test_files, 
                     test_categories, 
                     title='Test data diversification')


# # models

# In[ ]:


epochs= 150
patience= 5
batch_size = 8


# ## preprocessing

# In[ ]:


train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    # width_shift_range=0.2,
    # height_shift_range=0.2,
    # shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=target_size,
    class_mode='binary',
    batch_size=batch_size,
    seed=1337,
)

validation_datagen = ImageDataGenerator(
    rescale=1./255,
)

validation_generator = validation_datagen.flow_from_directory(
    val_dir,
    target_size=target_size,
    class_mode='binary',
    batch_size=batch_size,
    seed=1337,
)

test_datagen = ImageDataGenerator(
    rescale=1./255,
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=target_size,
    class_mode='binary',
    batch_size=batch_size,
    seed=1337,
)


# ## model

# In[ ]:


model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), 
                        activation='relu', 
                        input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Dropout(0.3))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.Dropout(0.3))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Dropout(0.3))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.summary()


# In[ ]:


def plot_activation_layers(model, image_path, target_size=(150, 150),
                           images_per_row=16, show_pooling=False):
    """Plot the layer activations for ever convolution layer.

    :param model: <keras model>
    :param image_path: <string> -> Path to a image file.
    :param target_size: <list | tuple> -> Standard is <(150, 150)>
    :param images_per_row: <int> -> Number of images to be shown per row
    :param show_pooling: <bool> -> Shows the layer. Default is <False>
    :return: <None>
    """

    img = keras_image.load_img(image_path, target_size=target_size)
    img_tensor = keras_image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.

    layer_outputs = [layer.output for layer in model.layers]
    activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
    activations = activation_model.predict(img_tensor)

    layer_names = []
    for layer in model.layers:
        if 'flatten' in layer.name:
            break
        else:
            layer_names.append(layer.name)

    for layer_name, layer_activation in zip(layer_names, activations):
        if 'dropout' in layer_name:
            continue
        if not show_pooling and 'pooling' in layer_name:
            continue

        n_features = layer_activation.shape[-1]
        size = layer_activation.shape[1]
        n_cols = n_features // images_per_row
        display_grid = np.zeros((size * n_cols, images_per_row * size))

        with np.errstate(divide='ignore', invalid='ignore'):
            for col in range(n_cols):
                for row in range(images_per_row):
                    channel_image = layer_activation[
                                    0, :, :, col * images_per_row + row
                    ]
                    channel_image -= channel_image.mean()
                    channel_image /= channel_image.std()
                    channel_image *= 64
                    channel_image += 128
                    channel_image = np.clip(
                        channel_image, 0, 255
                    ).astype('uint8')
                    display_grid[
                        col * size: (col + 1) * size,
                        row * size: (row + 1) * size
                    ] = channel_image

            scale = 1. / size
            plt.figure(figsize=(scale * display_grid.shape[1],
                                scale * display_grid.shape[0]))
            plt.title(layer_name)
            plt.grid(False)
            plt.axis('off')
            plt.imshow(display_grid, aspect='auto', cmap='viridis')

    plt.show()


# In[ ]:


plot_activation_layers(model, '../input/chest_xray/chest_xray/test/NORMAL/IM-0001-0001.jpeg', images_per_row=16)


# In[ ]:


model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

result = model.fit_generator(
    train_generator, 
    epochs=epochs,
    steps_per_epoch=train_generator.samples / train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples / validation_generator.batch_size,
    callbacks=[
        EarlyStopping(
            monitor='val_loss', 
            mode='auto', 
            verbose=1, 
            patience=patience,
        ), 
        ModelCheckpoint(
            'best_model.h5', 
            monitor='val_loss', 
            mode='auto', 
            save_best_only=True, 
            verbose=1
        )
    ]
)


# # results

# ## validation data

# In[ ]:


def statistic_val_loss_acc(result):
    """Show val_loss | val_acc

    :param result: <keras result> -> Result from a keras model.
    :return: None
    """
    val_loss = min(result.history.get('val_loss'))
    val_acc = max(result.history.get('val_acc'))

    print('val_loss: {} | val_acc: {}'.format(
        val_loss, val_acc
    ))


# In[ ]:


statistic_val_loss_acc(result)


# In[ ]:


def plot_loss_acc(result, linestyle='-', result2=None, linestyle2=':'):
    """Plot the loss | acc and val_loss | val_acc from keras model(s).

    :param result: <keras model>
    :param linestyle: <string> -> See matplotlib documentation.
    :param result2: <keras model>
    :param linestyle2: <string> -> See matplotlib documentation.
    :return: <None>
    """
    acc = result.history['acc']
    val_acc = result.history['val_acc']
    loss = result.history['loss']
    val_loss = result.history['val_loss']
    epochs = range(1, len(acc) + 1)

    if result2:
        acc2 = result2.history['acc']
        val_acc2 = result2.history['val_acc']
        loss2 = result2.history['loss']
        val_loss2 = result2.history['val_loss']
        epochs2 = range(1, len(acc2) + 1)
    else:
        acc2 = 1
        val_acc2 = None
        loss2 = None
        val_loss2 = None
        epochs2 = None

    plt.figure(figsize=(12, 4))
    plt.subplot(121)
    plt.plot(epochs, loss, color='green', linestyle=linestyle, marker='o',
             markersize=2, label='Training')
    plt.plot(epochs, val_loss, color='red', linestyle=linestyle, marker='o',
             markersize=2, label='Validation')
    if result2:
        plt.plot(epochs2, loss2, color='green', linestyle=linestyle2,
                 marker='o', markersize=2, label='Training')
        plt.plot(epochs2, val_loss2, color='red', linestyle=linestyle2,
                 marker='o', markersize=2, label='Validation')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss')
    plt.legend()
    plt.subplot(122)
    plt.plot(epochs, acc, color='green', linestyle=linestyle, marker='o',
             markersize=2, label='Training')
    plt.plot(epochs, val_acc, color='red', linestyle=linestyle, marker='o',
             markersize=2, label='Validation')
    if result2:
        plt.plot(epochs2, acc2, color='green', linestyle=linestyle2,
                 marker='o', markersize=2, label='Training')
        plt.plot(epochs2, val_acc2, color='red', linestyle=linestyle2,
                 marker='o', markersize=2, label='Validation')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy')
    plt.legend()
    plt.show()


# In[ ]:


plot_loss_acc(result)


# ## load best model | evaluate test_set

# In[ ]:


model = load_model('best_model.h5')


# In[ ]:


def generator_error(generator):
    """To catch errors when reading in data.

    :param generator: <generator>
    :return: <None>
    """

    while True:
        try:
            x, y = next(generator)
            yield x, y
        except OSError:
            pass


# In[ ]:


def statistic_test_loss_acc(model, x, y=None, error=False):
    """Displays the result of the test data.

    This does not mean the validation data in Deep Learning! Some say
    you should have three sets of data: Training data, validation data
    and test data.

    :param model: model
    :param x: <array> -> Data
    :param y: <array> -> Labels
    :param error: <bool> -> If an error occurs while reading, you should
    try it with <True>. Maybe the error was already considered in the
    code and intercepted. Standard is <False>

    :return: <None>
    """

    if y is not None:
        loss, acc = model.evaluate(x, y)
        print('test_loss: {} | test_acc: {}'.format(loss, acc))
    else:
        steps = x.samples // x.batch_size
        if error:
            loss, acc = model.evaluate_generator(generator_error(x), steps=steps)
        else:
            loss, acc = model.evaluate_generator(x, steps)

        print('test_loss: {} | test_acc: {}'.format(loss, acc))


# In[ ]:


statistic_test_loss_acc(model, test_generator)

