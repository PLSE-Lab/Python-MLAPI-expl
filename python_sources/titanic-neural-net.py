import numpy as np
import pandas as pd
from sklearn import linear_model, datasets

#Print you can execute arbitrary python code
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
print(train)
data = []
for person in train.iterrows():
    print(person)
h = .02  # step size in the mesh
