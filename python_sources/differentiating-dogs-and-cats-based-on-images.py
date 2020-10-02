# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf # tensorflow
import os # access the directories
import cv2 # computer vision library
import random

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

TRAIN_DIRECTORY = '../input/train'
TEST_DIRECTORY = '../input/test'
# rows and columns to resize the images to
ROWS = 256
COLUMNS = 256

train_dogs =   [TRAIN_DIRECTORY+i for i in os.listdir(TRAIN_DIRECTORY) if 'dog' in i]
train_cats =   [TRAIN_DIRECTORY+i for i in os.listdir(TRAIN_DIRECTORY) if 'cat' in i]
test_images =   [TEST_DIRECTORY+i for i in os.listdir(TEST_DIRECTORY)]


train_images = train_dogs[:1000] + train_cats[:1000]
random.shuffle(train_images)


print(train_images[0])
