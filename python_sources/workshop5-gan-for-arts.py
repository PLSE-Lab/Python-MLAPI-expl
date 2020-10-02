#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install tensorflow-gpu==2.0.0-beta1')
import numpy as np # linear algebra
import tensorflow as tf
from sklearn.metrics import classification_report
from tensorflow.python.framework.errors_impl import NotFoundError
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.engine import Layer
from tensorflow.python.keras.layers import Concatenate, Dense, Reshape, LeakyReLU,     BatchNormalization, Flatten, Conv2D, Conv2DTranspose

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

tf.__version__


# # Declare Generator

# In[ ]:


class GatedNonlinearity(Layer):
    def __init__(self, **kwargs):
        super(GatedNonlinearity, self).__init__(**kwargs)

    @tf.function
    def call(self, inputs, **kwargs):
        assert isinstance(inputs, list)
        base, condition = inputs
        a, b = base[:, :, :, ::2], base[:, :, :, 1::2]
        c, d = condition[:, :, :, ::2], condition[:, :, :, 1::2]
        a = a + c
        b = b + d
        result = tf.sigmoid(a) * tf.tanh(b)
        return result


class Generator:
    def __init__(self, num_classes, target_width, bn=False, noise_size=128, min_channels=64, max_channels=1024):
        self.num_classes = num_classes
        self.target_width = target_width
        self.noise_size = noise_size
        self.bn = bn
        self.create_model(min_channels, max_channels)
        self._model_path = '/kaggle/input/gen_weights.h5'

    def create_model(self, min_channels, max_channels):
        gate = GatedNonlinearity()
        _width = 4
        _num_blocks = np.log2(self.target_width / _width).astype(np.uint8)
        _num_channels = min(max_channels, min_channels * (2 ** _num_blocks))
        _num_channels = max(min_channels, _num_channels)
        noise = Input(shape=[self.noise_size], dtype=tf.float32)
        label = Input(shape=[self.num_classes], dtype=tf.float32)
        x = Concatenate()([noise, label])
        output = Dense(_width * _width * _num_channels)(x)
        output = Reshape(target_shape=(_width, _width, _num_channels))(output)

        for i in range(1, _num_blocks + 1):
            if self.bn:
                output = BatchNormalization()(output)
            condition = Dense(_width * _width * _num_channels, use_bias=False)(label)
            condition = Reshape(target_shape=(_width, _width, _num_channels))(condition)
            output = gate([output, condition])
            _num_channels = min(max_channels, min_channels * (2 ** (_num_blocks - i)))
            _num_channels = max(min_channels, _num_channels)
            if i == _num_blocks:
                _num_channels = 3
            _width *= 2
            output = Conv2DTranspose(_num_channels, kernel_size=5, strides=2, padding='same')(output)
        gen_image = tf.tanh(output)
        self.model = Model(inputs=[noise, label], outputs=[gen_image])
    
    def load(self):
        try:
            self.model.load_weights(self._model_path)
        except (NotFoundError, OSError):
            print("[ERROR] loading weights failed: {}".format(self._model_path))
            return False
        return True


# # Declare Discriminator

# In[ ]:


class Discriminator:
    def __init__(self, num_classes, image_width, depth, leak=.2, bn_epsilon=0):
        self.num_classes = num_classes
        self.image_width = image_width
        self.leak = leak
        self.bn_epsilon = bn_epsilon
        self.create_model(depth)
        self._model_path = '/kaggle/input/disc_weights.h5'
    
    def load(self):
        try:
            self.model.load_weights(self._model_path)
        except (NotFoundError, OSError):
            print("[ERROR] loading weights failed: {}".format(self._model_path))
            return False
        return True

    def create_model(self, depth):
        def _conv(d):
            return Conv2D(d, kernel_size=5, strides=2, padding='same')

        _num_blocks = np.log2(self.image_width / 4).astype(np.uint8) - 1
        image = Input(shape=(self.image_width, self.image_width, 3), dtype=tf.float32)
        conv = _conv(depth)(image)
        conv = LeakyReLU(alpha=self.leak)(conv)
        for i in range(_num_blocks):
            depth *= 2
            conv = _conv(depth)(conv)
            if self.bn_epsilon != 0:
                conv = BatchNormalization(epsilon=self.bn_epsilon)(conv)
            conv = LeakyReLU(alpha=self.leak)(conv)
        features = Flatten()(conv)
        logit = Dense(1, name='wasserstein_logit')(features)
        label = Dense(self.num_classes, activation='softmax', name='label')(features)
        self.model = Model(inputs=[image], outputs=[logit, label])


# # Create the generator & discriminator and load trained weights

# In[ ]:


from PIL import Image
from tensorflow.python.keras.utils import to_categorical

idx2label = {
    0: 'abstract',
    1: 'animal-painting',
    2: 'cityscape',
    3: 'figurative',
    4: 'flower-painting',
    5: 'genre-painting',
    6: 'landscape',
    7: 'marina',
    8: 'mythological-painting',
    9: 'nude-painting-nu',
    10: 'portrait',
    11: 'religious-painting',
    12: 'still-life',
    13: 'symbolic-painting'
}
label2idx = {val: key for key, val in idx2label.items()}
num_classes = len(idx2label.keys())
noise_size = 128
image_size = 128

gen = Generator(num_classes, image_size, bn=False)
disc = Discriminator(num_classes, image_size, 64, bn_epsilon=0)
gen.load()
disc.load()


# # Prepare a few helper functions
# 
# a function for displaying arbitrary number of images on the screen

# In[ ]:


def display_image(*images, col=None, width=20):
    from matplotlib import pyplot as plt
    if col is None:
        col = len(images)
    row = np.math.ceil(len(images) / col)
    plt.figure(figsize=(width, (width + 1) * row / col))
    for i, image in enumerate(images):
        plt.subplot(row, col, i + 1)
        plt.axis('off')
        plt.imshow(image, cmap='gray')
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()


# a function to randomly generate one-hot vectors used as classification labels
# 
# and a function to normalize image pixel values to [0 - 255]
# 
# a function to randomly generate a given number of artworks

# In[ ]:


def generate_labels(batch_size, num_classes, condition=None):
    if condition is None:
        labels = np.random.randint(num_classes, size=batch_size).astype(np.float32)
    else:
        labels = np.ones(shape=batch_size).astype(np.float32) * condition
    return to_categorical(labels, num_classes=num_classes)


def denorm_image(x):
    return np.clip((x + 1) * 127.5, 0, 255).astype('uint8')


def generate_images(num_samples, genre=None):
    if genre:
        genre_idx = label2idx[genre]
    else:
        genre_idx = None
    noise = tf.random.normal(shape=[num_samples, noise_size])
    labels = generate_labels(num_samples, num_classes, condition=genre_idx)
    images = gen.model.predict([noise, labels])
    return images


# a function to paste pictures together to make a montage image

# In[ ]:


def create_montage(images, cols, width=1000):
    _width = int(width / cols)
    rows = np.math.ceil(len(images) / cols)
    height = np.ceil(_width * rows).astype('int')
    canvas = Image.new('RGB', (width, height), color=(255, 255, 255))
    for row in range(rows):
        for col in range(cols):
            arr = images[row * cols + col]
            x0, y0 = col * _width, row * _width
            x1, y1 = x0 + _width, y0 + _width
            image = Image.fromarray(arr)
            resized = image.resize((_width, _width))
            canvas.paste(resized, box=(x0, y0, x1, y1))
    return canvas


# ## Let's try generating some random arts!

# In[ ]:


images = denorm_image(generate_images(24))
montage = create_montage(images, cols=6, width=750)
display_image(np.array(montage), width=12)


# ## Let's try generating some arts of a given genre!

# In[ ]:


sample_genre = 'cityscape'
images = denorm_image(generate_images(24, sample_genre))
montage = create_montage(images, cols=6, width=750)
display_image(np.array(montage), width=12)

