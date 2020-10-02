import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

#Print you can execute arbitrary python code
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )



print(train.head())
