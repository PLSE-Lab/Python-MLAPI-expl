import numpy as np
import pandas as pd

#Print you can execute arbitrary python code
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )


import matplotlib.pyplot as plt
fig = plt.figure()
fig.set(alpha=0.2)

train.Survived.value_counts().plot(kind='bar')
plt.show()