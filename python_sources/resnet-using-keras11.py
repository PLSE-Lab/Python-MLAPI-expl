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
print (train_y.shape, train_x.shape, test_x.shape )