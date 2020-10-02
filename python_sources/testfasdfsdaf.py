# -*- coding: utf-8 -*-
from __future__ import print_function

import numpy as np
np.random.seed(1200)

import time
import io, sys, os


import math
from keras.preprocessing.text import *
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.callbacks import ModelCheckpoint,EarlyStopping
from keras.preprocessing.sequence import pad_sequences
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils