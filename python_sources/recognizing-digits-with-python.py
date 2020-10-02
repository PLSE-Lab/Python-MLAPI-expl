# Imports and global stuff
import pandas as pd
import numpy as np
import scipy as sp
import scipy.linalg as spla
import sklearn.neural_network as sknn

# Functions

def down_res(pixel_row):
    """
        Takes a pixel row of 28x28 and diminishes the resolution
        to 7x7 by averaging over 4x4 blocks
    """
    # Construct the averaging matrix
    d_list = [[1.0/4.0,1.0/4.0,1.0/4.0,1.0/4.0]]*7
    d_matrix = spla.block_diag(*d_list)
    # Reconstruct the pixel image (28x28 matrix)
    # from the pixel0 ... pixel783 columns
    


# The competition datafiles are in the directory ../input
# Read competition data files:
train = pd.read_csv("../input/train.csv")
test  = pd.read_csv("../input/test.csv")

# Write to the log:
print("Training set has {0[0]} rows and {0[1]} columns".format(train.shape))
print("Test set has {0[0]} rows and {0[1]} columns".format(test.shape))
# Any files you write to the current directory get shown as outputs

