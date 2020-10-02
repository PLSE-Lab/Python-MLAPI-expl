import numpy as np
import pandas as pd

#Print you can execute arbitrary python code
train = pd.read_csv("../input/train.csv", header = 0)
#test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64})

train.describe()

