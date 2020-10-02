#!/usr/bin/env python
# coding: utf-8

# Import neccessary packages

# In[ ]:


import numpy as np
import pickle
import cv2
from os import listdir
from sklearn.preprocessing import LabelBinarizer
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation, Flatten, Dropout, Dense
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# In[ ]:


EPOCHS = 25
INIT_LR = 1e-3
BS = 32
default_image_size = tuple((256, 256))
image_size = 0
directory_root = '../input/plantvillage/'
width=256
height=256
depth=3


# Function to convert images to array

# In[ ]:


def convert_image_to_array(image_dir):
    try:
        image = cv2.imread(image_dir)
        if image is not None :
            image = cv2.resize(image, default_image_size)   
            return img_to_array(image)
        else :
            return np.array([])
    except Exception as e:
        print(f"Error : {e}")
        return None


# Fetch images from directory

# In[ ]:


image_list, label_list = [], []
try:
    print("[INFO] Loading images ...")
    root_dir = listdir(directory_root)
    for directory in root_dir :
        # remove .DS_Store from list
        if directory == ".DS_Store" :
            root_dir.remove(directory)

    for plant_folder in root_dir :
        plant_disease_folder_list = listdir(f"{directory_root}/{plant_folder}")
        
        for disease_folder in plant_disease_folder_list :
            # remove .DS_Store from list
            if disease_folder == ".DS_Store" :
                plant_disease_folder_list.remove(disease_folder)

        for plant_disease_folder in plant_disease_folder_list:
            print(f"[INFO] Processing {plant_disease_folder} ...")
            plant_disease_image_list = listdir(f"{directory_root}/{plant_folder}/{plant_disease_folder}/")
                
            for single_plant_disease_image in plant_disease_image_list :
                if single_plant_disease_image == ".DS_Store" :
                    plant_disease_image_list.remove(single_plant_disease_image)

            for image in plant_disease_image_list[:250]:
                image_directory = f"{directory_root}/{plant_folder}/{plant_disease_folder}/{image}"
                if image_directory.endswith(".jpg") == True or image_directory.endswith(".JPG") == True:
                    image_list.append(convert_image_to_array(image_directory))
                    label_list.append(plant_disease_folder)
    print("[INFO] Image loading completed")  
except Exception as e:
    print(f"Error : {e}")


# Get Size of Processed Image

# In[ ]:


image_size = len(image_list)
image_size


# Transform Image Labels uisng [Scikit Learn](http://scikit-learn.org/)'s LabelBinarizer

# In[ ]:


label_binarizer = LabelBinarizer()
image_labels = label_binarizer.fit_transform(label_list)
pickle.dump(label_binarizer,open('label_transform.pkl', 'wb'))
n_classes = len(label_binarizer.classes_)


# Print the classes

# In[ ]:


print(label_binarizer.classes_)


# In[ ]:


np_image_list = np.array(image_list, dtype=np.float16) / 225.0


# In[ ]:


print("[INFO] Spliting data to train, test")
x_train, x_test, y_train, y_test = train_test_split(np_image_list, image_labels, test_size=0.2, random_state = 42) 


# In[ ]:


aug = ImageDataGenerator(
    rotation_range=25, width_shift_range=0.1,
    height_shift_range=0.1, shear_range=0.2, 
    zoom_range=0.2,horizontal_flip=True, 
    fill_mode="nearest")


# In[ ]:


from __future__ import print_function
import keras
from keras.models import Sequential
from keras import layers
from keras.datasets import mnist
from keras import backend as K


# In[ ]:


from keras.layers import GlobalAveragePooling2D, Reshape, Dense, multiply, Permute
from keras import backend as K


def squeeze_excite_block(input, ratio=16):
    ''' Create a squeeze-excite block
    Args:
        input: input tensor
        filters: number of output filters
        k: width factor

    Returns: a keras tensor
    '''
    init = input
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = init._keras_shape[channel_axis]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    if K.image_data_format() == 'channels_first':
        se = Permute((3, 1, 2))(se)

    x = multiply([init, se])
    return x


# In[ ]:


from keras.layers import GlobalAveragePooling2D, Reshape, Dense, multiply, Permute
from keras import backend as K


def squeeze_excite_block(input, ratio=16):
    ''' Create a squeeze-excite block
    Args:
        input: input tensor
        filters: number of output filters
        k: width factor

    Returns: a keras tensor
    '''
    init = input
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = init._keras_shape[channel_axis]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    if K.image_data_format() == 'channels_first':
        se = Permute((3, 1, 2))(se)

    x = multiply([init, se])
    return x


# In[ ]:


class squeezeExcitation(layers.Layer):
    WEIGHTS_PATH = ''
    WEIGHTS_PATH_NO_TOP = ''


    def _conv2d_bn(self,x,
                   filters,
                   num_row,
                   num_col,
                   padding='same',
                   strides=(1, 1),
                   name=None):
        """Utility function to apply conv + BN.
        # Arguments
            x: input tensor.
            filters: filters in `Conv2D`.
            num_row: height of the convolution kernel.
            num_col: width of the convolution kernel.
            padding: padding mode in `Conv2D`.
            strides: strides in `Conv2D`.
            name: name of the ops; will become `name + '_conv'`
                for the convolution and `name + '_bn'` for the
                batch norm layer.
        # Returns
            Output tensor after applying `Conv2D` and `BatchNormalization`.
        """
        if name is not None:
            bn_name = name + '_bn'
            conv_name = name + '_conv'
        else:
            bn_name = None
            conv_name = None
        if K.image_data_format() == 'channels_first':
            bn_axis = 1
        else:
            bn_axis = 3
        x = Conv2D(
            filters, (num_row, num_col),
            strides=strides,
            padding=padding,
            use_bias=False,
            name=conv_name)(x)
        x = BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
        x = Activation('relu', name=name)(x)
        return x


    def SEInceptionV3(self,include_top=True,
                      weights=None,
                      input_tensor=None,
                      input_shape=None,
                      pooling=None,
                      classes=1000):
        """Instantiates the Squeeze and Excite Inception v3 architecture.
        # Arguments
            include_top: whether to include the fully-connected
                layer at the top of the network.
            weights: one of `None` (random initialization)
                or "imagenet" (pre-training on ImageNet).
            input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
                to use as image input for the model.
            input_shape: optional shape tuple, only to be specified
                if `include_top` is False (otherwise the input shape
                has to be `(299, 299, 3)` (with `channels_last` data format)
                or `(3, 299, 299)` (with `channels_first` data format).
                It should have exactly 3 inputs channels,
                and width and height should be no smaller than 139.
                E.g. `(150, 150, 3)` would be one valid value.
            pooling: Optional pooling mode for feature extraction
                when `include_top` is `False`.
                - `None` means that the output of the model will be
                    the 4D tensor output of the
                    last convolutional layer.
                - `avg` means that global average pooling
                    will be applied to the output of the
                    last convolutional layer, and thus
                    the output of the model will be a 2D tensor.
                - `max` means that global max pooling will
                    be applied.
            classes: optional number of classes to classify images
                into, only to be specified if `include_top` is True, and
                if no `weights` argument is specified.
        # Returns
            A Keras model instance.
        # Raises
            ValueError: in case of invalid argument for `weights`,
                or invalid input shape.
        """
        if weights not in {'imagenet', None}:
            raise ValueError('The `weights` argument should be either '
                             '`None` (random initialization) or `imagenet` '
                             '(pre-training on ImageNet).')

        if weights == 'imagenet' and include_top and classes != 1000:
            raise ValueError('If using `weights` as imagenet with `include_top`'
                             ' as true, `classes` should be 1000')

        # Determine proper input shape
        input_shape = _obtain_input_shape(
            input_shape,
            default_size=299,
            min_size=139,
            data_format=K.image_data_format(),
            require_flatten=include_top)

        if input_tensor is None:
            img_input = Input(shape=input_shape)
        else:
            if not K.is_keras_tensor(input_tensor):
                img_input = Input(tensor=input_tensor, shape=input_shape)
            else:
                img_input = input_tensor

        if K.image_data_format() == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = 3

        x = _conv2d_bn(img_input, 32, 3, 3, strides=(2, 2), padding='valid')
        x = _conv2d_bn(x, 32, 3, 3, padding='valid')
        x = _conv2d_bn(x, 64, 3, 3)
        x = MaxPooling2D((3, 3), strides=(2, 2))(x)

        x = _conv2d_bn(x, 80, 1, 1, padding='valid')
        x = _conv2d_bn(x, 192, 3, 3, padding='valid')
        x = MaxPooling2D((3, 3), strides=(2, 2))(x)

        # mixed 0, 1, 2: 35 x 35 x 256
        branch1x1 = _conv2d_bn(x, 64, 1, 1)

        branch5x5 = _conv2d_bn(x, 48, 1, 1)
        branch5x5 = _conv2d_bn(branch5x5, 64, 5, 5)

        branch3x3dbl = _conv2d_bn(x, 64, 1, 1)
        branch3x3dbl = _conv2d_bn(branch3x3dbl, 96, 3, 3)
        branch3x3dbl = _conv2d_bn(branch3x3dbl, 96, 3, 3)

        branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = _conv2d_bn(branch_pool, 32, 1, 1)
        x = layers.concatenate(
            [branch1x1, branch5x5, branch3x3dbl, branch_pool],
            axis=channel_axis,
            name='mixed0')

        # squeeze and excite block
        x = squeeze_excite_block(x)

        # mixed 1: 35 x 35 x 256
        branch1x1 = _conv2d_bn(x, 64, 1, 1)

        branch5x5 = _conv2d_bn(x, 48, 1, 1)
        branch5x5 = _conv2d_bn(branch5x5, 64, 5, 5)

        branch3x3dbl = _conv2d_bn(x, 64, 1, 1)
        branch3x3dbl = _conv2d_bn(branch3x3dbl, 96, 3, 3)
        branch3x3dbl = _conv2d_bn(branch3x3dbl, 96, 3, 3)

        branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = _conv2d_bn(branch_pool, 64, 1, 1)
        x = layers.concatenate(
            [branch1x1, branch5x5, branch3x3dbl, branch_pool],
            axis=channel_axis,
            name='mixed1')

        # squeeze and excite block
        x = squeeze_excite_block(x)

        # mixed 2: 35 x 35 x 256
        branch1x1 = _conv2d_bn(x, 64, 1, 1)

        branch5x5 = _conv2d_bn(x, 48, 1, 1)
        branch5x5 = _conv2d_bn(branch5x5, 64, 5, 5)

        branch3x3dbl = _conv2d_bn(x, 64, 1, 1)
        branch3x3dbl = _conv2d_bn(branch3x3dbl, 96, 3, 3)
        branch3x3dbl = _conv2d_bn(branch3x3dbl, 96, 3, 3)

        branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = _conv2d_bn(branch_pool, 64, 1, 1)
        x = layers.concatenate(
            [branch1x1, branch5x5, branch3x3dbl, branch_pool],
            axis=channel_axis,
            name='mixed2')

        # squeeze and excite block
        x = squeeze_excite_block(x)

        # mixed 3: 17 x 17 x 768
        branch3x3 = _conv2d_bn(x, 384, 3, 3, strides=(2, 2), padding='valid')

        branch3x3dbl = _conv2d_bn(x, 64, 1, 1)
        branch3x3dbl = _conv2d_bn(branch3x3dbl, 96, 3, 3)
        branch3x3dbl = _conv2d_bn(
            branch3x3dbl, 96, 3, 3, strides=(2, 2), padding='valid')

        branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
        x = layers.concatenate(
            [branch3x3, branch3x3dbl, branch_pool], axis=channel_axis, name='mixed3')

        # squeeze and excite block
        x = squeeze_excite_block(x)

        # mixed 4: 17 x 17 x 768
        branch1x1 = _conv2d_bn(x, 192, 1, 1)

        branch7x7 = _conv2d_bn(x, 128, 1, 1)
        branch7x7 = _conv2d_bn(branch7x7, 128, 1, 7)
        branch7x7 = _conv2d_bn(branch7x7, 192, 7, 1)

        branch7x7dbl = _conv2d_bn(x, 128, 1, 1)
        branch7x7dbl = _conv2d_bn(branch7x7dbl, 128, 7, 1)
        branch7x7dbl = _conv2d_bn(branch7x7dbl, 128, 1, 7)
        branch7x7dbl = _conv2d_bn(branch7x7dbl, 128, 7, 1)
        branch7x7dbl = _conv2d_bn(branch7x7dbl, 192, 1, 7)

        branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = _conv2d_bn(branch_pool, 192, 1, 1)
        x = layers.concatenate(
            [branch1x1, branch7x7, branch7x7dbl, branch_pool],
            axis=channel_axis,
            name='mixed4')

        # squeeze and excite block
        x = squeeze_excite_block(x)

        # mixed 5, 6: 17 x 17 x 768
        for i in range(2):
            branch1x1 = _conv2d_bn(x, 192, 1, 1)

            branch7x7 = _conv2d_bn(x, 160, 1, 1)
            branch7x7 = _conv2d_bn(branch7x7, 160, 1, 7)
            branch7x7 = _conv2d_bn(branch7x7, 192, 7, 1)

            branch7x7dbl = _conv2d_bn(x, 160, 1, 1)
            branch7x7dbl = _conv2d_bn(branch7x7dbl, 160, 7, 1)
            branch7x7dbl = _conv2d_bn(branch7x7dbl, 160, 1, 7)
            branch7x7dbl = _conv2d_bn(branch7x7dbl, 160, 7, 1)
            branch7x7dbl = _conv2d_bn(branch7x7dbl, 192, 1, 7)

            branch_pool = AveragePooling2D(
                (3, 3), strides=(1, 1), padding='same')(x)
            branch_pool = _conv2d_bn(branch_pool, 192, 1, 1)
            x = layers.concatenate(
                [branch1x1, branch7x7, branch7x7dbl, branch_pool],
                axis=channel_axis,
                name='mixed' + str(5 + i))

            # squeeze and excite block
            x = squeeze_excite_block(x)

        # mixed 7: 17 x 17 x 768
        branch1x1 = _conv2d_bn(x, 192, 1, 1)

        branch7x7 = _conv2d_bn(x, 192, 1, 1)
        branch7x7 = _conv2d_bn(branch7x7, 192, 1, 7)
        branch7x7 = _conv2d_bn(branch7x7, 192, 7, 1)

        branch7x7dbl = _conv2d_bn(x, 192, 1, 1)
        branch7x7dbl = _conv2d_bn(branch7x7dbl, 192, 7, 1)
        branch7x7dbl = _conv2d_bn(branch7x7dbl, 192, 1, 7)
        branch7x7dbl = _conv2d_bn(branch7x7dbl, 192, 7, 1)
        branch7x7dbl = _conv2d_bn(branch7x7dbl, 192, 1, 7)

        branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = _conv2d_bn(branch_pool, 192, 1, 1)
        x = layers.concatenate(
            [branch1x1, branch7x7, branch7x7dbl, branch_pool],
            axis=channel_axis,
            name='mixed7')

        # squeeze and excite block
        x = squeeze_excite_block(x)

        # mixed 8: 8 x 8 x 1280
        branch3x3 = _conv2d_bn(x, 192, 1, 1)
        branch3x3 = _conv2d_bn(branch3x3, 320, 3, 3,
                               strides=(2, 2), padding='valid')

        branch7x7x3 = _conv2d_bn(x, 192, 1, 1)
        branch7x7x3 = _conv2d_bn(branch7x7x3, 192, 1, 7)
        branch7x7x3 = _conv2d_bn(branch7x7x3, 192, 7, 1)
        branch7x7x3 = _conv2d_bn(
            branch7x7x3, 192, 3, 3, strides=(2, 2), padding='valid')

        branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
        x = layers.concatenate(
            [branch3x3, branch7x7x3, branch_pool], axis=channel_axis, name='mixed8')

        # squeeze and excite block
        x = squeeze_excite_block(x)

        # mixed 9: 8 x 8 x 2048
        for i in range(2):
            branch1x1 = _conv2d_bn(x, 320, 1, 1)

            branch3x3 = _conv2d_bn(x, 384, 1, 1)
            branch3x3_1 = _conv2d_bn(branch3x3, 384, 1, 3)
            branch3x3_2 = _conv2d_bn(branch3x3, 384, 3, 1)
            branch3x3 = layers.concatenate(
                [branch3x3_1, branch3x3_2], axis=channel_axis, name='mixed9_' + str(i))

            branch3x3dbl = _conv2d_bn(x, 448, 1, 1)
            branch3x3dbl = _conv2d_bn(branch3x3dbl, 384, 3, 3)
            branch3x3dbl_1 = _conv2d_bn(branch3x3dbl, 384, 1, 3)
            branch3x3dbl_2 = _conv2d_bn(branch3x3dbl, 384, 3, 1)
            branch3x3dbl = layers.concatenate(
                [branch3x3dbl_1, branch3x3dbl_2], axis=channel_axis)

            branch_pool = AveragePooling2D(
                (3, 3), strides=(1, 1), padding='same')(x)
            branch_pool = _conv2d_bn(branch_pool, 192, 1, 1)
            x = layers.concatenate(
                [branch1x1, branch3x3, branch3x3dbl, branch_pool],
                axis=channel_axis,
                name='mixed' + str(9 + i))

            # squeeze and excite block
            x = squeeze_excite_block(x)

        if include_top:
            # Classification block
            x = GlobalAveragePooling2D(name='avg_pool')(x)
            x = Dense(classes, activation='softmax', name='predictions')(x)
        else:
            if pooling == 'avg':
                x = GlobalAveragePooling2D()(x)
            elif pooling == 'max':
                x = GlobalMaxPooling2D()(x)

        # Ensure that the model takes into account
        # any potential predecessors of `input_tensor`.
        if input_tensor is not None:
            inputs = get_source_inputs(input_tensor)
        else:
            inputs = img_input
        # Create model.
        model = Model(inputs, x, name='inception_v3')

        return model


    def preprocess_input(self,x):
        x /= 255.
        x -= 0.5
        x *= 2.
        return x


# In[ ]:


model = Sequential()
inputShape = (height, width, depth)
chanDim = -1
if K.image_data_format() == "channels_first":
    inputShape = (depth, height, width)
    chanDim = 1
model.add(Conv2D(32, (3, 3), padding="same",input_shape=inputShape))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(squeezeExcitation())
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(squeezeExcitation())
model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(squeezeExcitation())
model.add(Dropout(0.25))
model.add(Conv2D(128, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(squeezeExcitation())
model.add(Conv2D(128, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(squeezeExcitation())
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(n_classes))
model.add(Activation("softmax"))


# Model Summary

# In[ ]:


model.summary()


# In[ ]:


opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
# distribution
model.compile(loss="binary_crossentropy", optimizer=opt,metrics=["accuracy"])
# train the network
print("[INFO] training network...")


# In[ ]:


history = model.fit_generator(
    aug.flow(x_train, y_train, batch_size=BS),
    validation_data=(x_test, y_test),
    steps_per_epoch=len(x_train) // BS,
    epochs=EPOCHS, verbose=1
    )


# In[ ]:


acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
#Train and validation accuracy
plt.plot(epochs, acc, 'b', label='Training accurarcy')
plt.plot(epochs, val_acc, 'r', label='Validation accurarcy')
plt.title('Training and Validation accurarcy')
plt.legend()

plt.figure()
#Train and validation loss
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and Validation loss')
plt.legend()
plt.show()


# Model Accuracy

# In[ ]:


print("[INFO] Calculating model accuracy")
scores = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {scores[1]*100}")


# Save model using Pickle

# In[ ]:


# save the model to disk
print("[INFO] Saving model...")
pickle.dump(model,open('cnn_model.pkl', 'wb'))


# In[ ]:




