# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

from keras.applications.xception import Xception, preprocess_input
from keras.models import Model
from keras.layers import Input
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.preprocessing.image import ImageDataGenerator

from os import system

system("mkdir -p train/cat train/dog test")
system("for cat in ../input/train/cat.*.jpg; do ln -s $cat train/cat/; done")
system("for dog in ../input/train/dog.*.jpg; do ln -s $dog train/dog/; done")
system("ln -s /kaggle/input/test test/unknown")


batch_size=32
shape=(299,299)
train = ImageDataGenerator().flow_from_directory('train', target_size=shape, class_mode='binary', batch_size=batch_size)
test = ImageDataGenerator().flow_from_directory('test', target_size=shape, class_mode='binary', batch_size=batch_size)


def make_base_model(shape=shape):
    input_shape = shape + (3,)
    inp = Input(shape=input_shape)
    x = Lambda(preprocess_input)(inp)
    x = make_xception_base(input_shape)(x)
    x = GlobalAveragePooling2D(name='avg_pool')(x)
    model = Model(inp, x, name='xception_base')
    return model
def make_xception_base(input_shape):
    model = Xception(weights='imagenet')
    for l in model.layers:
        l.trainable = False
    return model
model = make_base_model()

model.summary()
