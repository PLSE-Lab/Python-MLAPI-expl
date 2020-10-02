#!/usr/bin/env python
# coding: utf-8

# This is the second experiment related to SIREN. You can check out the first experiement [here](https://www.kaggle.com/aakashnain/siren/).
# In this experiment, we will be using [`ResNet20_v2`](https://keras.io/examples/cifar10_resnet/) architecture and will compare the performance of `relu` and `sin` activation. 
# 
# The other thing you will learn in this notebook is how to make `data augmnetation` pipeline a part of the model itself.

# In[ ]:


get_ipython().system('pip install -qq tf-nightly')


# In[ ]:


import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
print("Tensorflow version: ", tf.__version__)


# In[ ]:


seed = 1234
np.random.seed(seed)
tf.random.set_seed(seed)


# In[ ]:


# Training parameters
batch_size = 128
epochs = 100
num_classes = 10

# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


# In[ ]:


# ResNet20 config
n = 2
depth = n * 9 + 2
version = 2
input_shape = (32, 32, 3)

# Model name, depth and version
model_type = 'ResNet%dv%d' % (depth, version)


# ## Augmentations

# In[ ]:


data_augmentation = keras.Sequential(
    [
        layers.experimental.preprocessing.RandomFlip("horizontal"),
        layers.experimental.preprocessing.RandomRotation(0.1),
        layers.experimental.preprocessing.RandomContrast(0.25),
        layers.experimental.preprocessing.RandomCrop(32, 32)
    ]
)


# In[ ]:


# Use tf.data.Dataset for improved performance
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
valid_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

train_dataset = train_dataset.batch(batch_size).shuffle(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
valid_dataset = valid_dataset.batch(batch_size).shuffle(batch_size).prefetch(tf.data.experimental.AUTOTUNE)


# In[ ]:


# Plot some sample images
plt.figure(figsize=(10, 10))
for images, labels in train_dataset.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(int(tf.argmax(labels[i], axis=-1)))
        plt.axis("off")


# In[ ]:


# Let's check if our augmentation pipeline is wokring or not
plt.figure(figsize=(10, 10))
for images, _ in train_dataset.take(1):
    for i in range(9):
        augmented_images = data_augmentation(images)
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(augmented_images[0].numpy().astype("uint8"))
        plt.axis("off")


# ## Resnet20_v2 with ReLU

# In[ ]:


def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation="relu",
                 kernel_initializer="he_uniform",
                 batch_normalization=True,
                 conv_first=True):
    """2D Convolution-Batch Normalization-Activation stack builder

    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)

    # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = layers.Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer=kernel_initializer,
                  kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = layers.BatchNormalization()(x)
        if activation is not None:
            if activation == "relu":
                x = layers.Activation(activation)(x)
            else:
                x = activation(x)
    else:
        if batch_normalization:
            x = layers.BatchNormalization()(x)
        if activation is not None:
            if activation == "relu":
                x = layers.Activation(activation)(x)
            else:
                x = activation(x)
        x = conv(x)
    return x


def resnet_v2(input_shape, depth, num_classes=10, act="relu"):
    """ResNet Version 2 Model builder [b]

    Stacks of (1 x 1)-(3 x 3)-(1 x 1) BN-ReLU-Conv2D or also known as
    bottleneck layer
    First shortcut connection per layer is 1 x 1 Conv2D.
    Second and onwards shortcut connection is identity.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filter maps is
    doubled. Within each stage, the layers have the same number filters and the
    same filter map sizes.
    Features maps sizes:
    conv1  : 32x32,  16
    stage 0: 32x32,  64
    stage 1: 16x16, 128
    stage 2:  8x8,  256

    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)

    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 9 != 0:
        raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
    # Start model definition.
    num_filters_in = 16
    num_res_blocks = int((depth - 2) / 9)

    inputs = layers.Input(shape=input_shape)
    x = data_augmentation(inputs)
    x = layers.experimental.preprocessing.Rescaling(1./255)(x)
    
    x = resnet_layer(inputs=x,
                     num_filters=num_filters_in,
                     activation=act,
                     conv_first=True)

    # Instantiate the stack of residual units
    for stage in range(3):
        for res_block in range(num_res_blocks):
            activation = act
            batch_normalization = True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in * 4
                if res_block == 0:  # first layer and first stage
                    activation = None
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in * 2
                if res_block == 0:  # first layer but not first stage
                    strides = 2    # downsample

            # bottleneck residual unit
            y = resnet_layer(inputs=x,
                             num_filters=num_filters_in,
                             kernel_size=1,
                             strides=strides,
                             activation=activation,
                             batch_normalization=batch_normalization,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_in,
                             activation=activation,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_out,
                             kernel_size=1,
                             activation=activation,
                             conv_first=False)
            if res_block == 0:
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters_out,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = layers.add([x, y])

        num_filters_in = num_filters_out

    # Add classifier on top.
    # v2 has BN-ReLU before Pooling
    x = layers.BatchNormalization()(x)
    if act=="relu":
        x = layers.Activation('relu')(x)
    else:
        x = tf.math.sin(x)
    x = layers.AveragePooling2D(pool_size=8)(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes,
                            activation='softmax',
                            kernel_initializer='he_uniform')(x)

    # Instantiate model.
    model = keras.models.Model(inputs=inputs, outputs=outputs, name=model_type)
    return model


# In[ ]:


model = resnet_v2(input_shape=input_shape, depth=depth)
model.summary()


# In[ ]:


def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 50:
        lr *= 1e-3
    elif epoch > 20:
        lr *= 1e-2
    elif epoch > 10:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr


# In[ ]:


callbacks = keras.callbacks.EarlyStopping(patience=10)
model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.Adam(lr_schedule(0)),
              metrics=['accuracy'])
model.fit(train_dataset, epochs=epochs,validation_data=valid_dataset, callbacks=[callbacks])


# In[ ]:


loss1, acc1 = model.evaluate(valid_dataset, verbose=0)
print("Validation accuracy: ", acc1)
print("Validation loss: ", loss1)


# ### ResNet20_v2 with sine activation

# In[ ]:


model = resnet_v2(input_shape=input_shape, depth=depth, act=tf.math.sin)
model.summary()


# In[ ]:


model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.Adam(lr_schedule(0)),
              metrics=['accuracy'])
model.fit(train_dataset, epochs=epochs,validation_data=valid_dataset, callbacks=[callbacks])


# In[ ]:


loss2, acc2 = model.evaluate(valid_dataset, verbose=0)
print("Validation accuracy: ", acc2)
print("Validation loss: ", loss2)


# In[ ]:




