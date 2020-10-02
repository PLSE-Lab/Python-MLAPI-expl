import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import keras

train = pd.read_csv("../input/train.csv",header=0).as_matrix()
test = pd.read_csv("../input/test.csv",header=0).as_matrix()


print(train.shape, test.shape)
