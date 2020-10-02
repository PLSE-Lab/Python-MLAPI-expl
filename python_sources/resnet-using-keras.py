#! coding: utf8
import numpy as np
import pandas as pd
from keras import backend as K
from keras.applications.resnet50 import conv_block, identity_block
from keras.layers import (Activation, BatchNormalization, Convolution2D, Dense,
                          Flatten, Input, MaxPooling2D, ZeroPadding2D)
from keras.models import Model
from keras.optimizers import RMSprop
from keras.utils.np_utils import to_categorical

# This magic thing makes model paralleled among GPUs
# You can get it from
# https://github.com/icyblade/data_mining_tools/blob/master/parallelizer.py
# Credit: https://github.com/kuza55/keras-extras/blob/master/utils/multi_gpu.py
# from parallelizer import Parallelizer

# define some variables
SHAPE = (28, 28, 1)
bn_axis = 3 if K.image_dim_ordering() == 'tf' else 1

# load data
train = pd.read_csv("../input/train.csv")
test  = pd.read_csv("../input/test.csv")
train_x = train.drop('label', axis=1).values.reshape(
    -1, *SHAPE
).astype(float)/255.0
train_y = to_categorical(train.label.values)
test_x = test.values.reshape(
    -1, *SHAPE
).astype(float)/255.0


def build_model(seed=None):
    # We can't use ResNet50 directly, as it might cause a negative dimension
    # error.
    if seed:
        np.random.seed(seed)

    input_layer = Input(shape=SHAPE)

    x = ZeroPadding2D((3, 3))(input_layer)
    x = Convolution2D(64, 7, 7, subsample=(2, 2), name='conv1')(x)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    """
    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')
    print(x)
    x = AveragePooling2D((7, 7), name='avg_pool')(x)
    """

    x = Flatten()(x)
    x = Dense(10, activation='softmax', name='fc10')(x)

    model = Model(input_layer, x)

    return model


# fit
model = build_model()
# model = Parallelizer().transform(model)
model.compile(RMSprop(lr=1e-4), 'categorical_crossentropy', ['accuracy'])
# batch_size = real_batch_size * n_GPUs
# model.fit(train_x, train_y, batch_size=64*2, nb_epoch=20)
model.fit(train_x, train_y, batch_size=64, nb_epoch=20)
# model.save('digit_recognizer_model.h5')

# predict
pred_y = model.predict(test_x).argmax(1)
pd.DataFrame({
    'ImageId': range(1, len(pred_y)+1),
    'Label': pred_y
}).to_csv('test_y.csv', index=False)
