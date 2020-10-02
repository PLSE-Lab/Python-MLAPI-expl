import pandas as pd
import numpy as np
import tensorflow as tf

# The competition datafiles are in the directory ../input
# Read competition data files:

train = np.genfromtxt("../input/train.csv", delimiter=",")
test  = np.genfromtxt("../input/test.csv", delimiter=",")

train = train[:,:-1]

# Write to the log:
print("Training set has {0[0]} rows and {0[1]} columns".format(train.shape))
print("Test set has {0[0]} rows and {0[1]} columns".format(test.shape))
# Any files you write to the current directory get shown as outputs