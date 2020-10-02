from nltk.corpus import stopwords
from stop_words import get_stop_words


import numpy as np
from chainer import cuda, Function, FunctionSet, gradient_check, Variable, optimizers, utils
import chainer.functions as F
