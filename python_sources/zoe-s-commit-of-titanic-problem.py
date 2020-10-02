import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import Series,DataFrame

#Print you can execute arbitrary python code
data_train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
data_test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

data_train