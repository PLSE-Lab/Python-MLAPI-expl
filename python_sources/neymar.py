import pandas as pd
import numpy as np  
import tensorflow as tf


train = pd.read_csv("../input/train.csv").as_matrix()





test  = pd.read_csv("../input/test.csv")




# Write to the log:
print("Training set has {0[0]} rows and {0[1]} columns".format(train.shape))
print("Test set has {0[0]} rows and {0[1]} columns".format(test.shape))
# Any files you write to the current directory get shown as outputs